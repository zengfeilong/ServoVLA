from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from torch.utils.data import Sampler

from servovla.data.batch_mixing import BatchQuotaState, compute_batch_quotas
from servovla.data.dataset_loader import episode_row_bounds, selected_episode_ids
from servovla.data.indexing import PaddedSampleIndex, sample_index_value

SampleIndex = int | PaddedSampleIndex


def _sorted_batch(batch: Sequence[SampleIndex]) -> list[SampleIndex]:
    return sorted(batch, key=sample_index_value)


def _pad_tail_batch(batch: Sequence[int], batch_size: int) -> list[SampleIndex]:
    padded: list[SampleIndex] = list(batch)
    if not padded:
        return padded
    while len(padded) < int(batch_size):
        source = batch[(len(padded) - len(batch)) % len(batch)]
        padded.append(PaddedSampleIndex(source))
    return padded


@dataclass(frozen=True, slots=True)
class DatasetEpisodeIndex:
    dataset_id: int
    offset: int
    episode_to_indices: dict[int, list[int]]

    @classmethod
    def from_lerobot_dataset(
        cls, lerobot_dataset, *, dataset_id: int, offset: int
    ) -> DatasetEpisodeIndex:
        bounds = episode_row_bounds(lerobot_dataset)
        episode_to_indices: dict[int, list[int]] = {}
        for episode_id in selected_episode_ids(lerobot_dataset):
            row_bounds = bounds.get(int(episode_id))
            if row_bounds is None:
                continue
            start_idx, end_idx = row_bounds
            episode_to_indices[int(episode_id)] = list(range(int(start_idx), int(end_idx)))
        return cls(
            dataset_id=int(dataset_id), offset=int(offset), episode_to_indices=episode_to_indices
        )


@dataclass(slots=True)
class _DatasetWindowState:
    episodes: list[int]
    episode_to_indices: dict[int, list[int]]
    active: dict[int, list[int]]
    cursor_by_episode: dict[int, int]
    window_boundary_offset_by_episode: dict[int, int]
    episode_cursor: int = 0


class ExplicitWeightedRawBatchSampler(Sampler[list[SampleIndex]]):
    def __init__(
        self,
        *,
        dataset_indices: Sequence[DatasetEpisodeIndex],
        weights: Sequence[float],
        batch_size: int,
        window_steps: int,
        active_episodes_per_dataset: int,
        seed: int,
        repeat: bool,
        drop_last: bool,
        shuffle: bool = True,
        reactivate_when_remaining_below: int = 0,
        seed_stride: int = 0,
        window_boundary_offsets: bool = False,
        batch_dataset_strategy: str = "quota",
        batch_dataset_burst_batches: int = 1,
        quota_max_datasets_per_batch: int = 0,
        max_episodes_per_batch: int = 0,
        episode_burst_batches: int = 1,
    ) -> None:
        self.dataset_indices = list(dataset_indices)
        self.weights = [float(weight) for weight in weights]
        self.batch_size = int(batch_size)
        self.window_steps = max(int(window_steps), 1)
        self.active_episodes_per_dataset = max(int(active_episodes_per_dataset), 1)
        self.reactivate_when_remaining_below = max(int(reactivate_when_remaining_below), 0)
        self.seed = int(seed)
        self.seed_stride = int(seed_stride)
        self.repeat = bool(repeat)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.window_boundary_offsets = bool(window_boundary_offsets)
        self.batch_dataset_strategy = str(batch_dataset_strategy)
        self.batch_dataset_burst_batches = max(int(batch_dataset_burst_batches), 1)
        self.quota_max_datasets_per_batch = max(int(quota_max_datasets_per_batch), 0)
        self.max_episodes_per_batch = max(int(max_episodes_per_batch), 0)
        self.episode_burst_batches = max(int(episode_burst_batches), 1)
        if self.batch_dataset_strategy not in {"quota", "single_dataset"}:
            raise ValueError(
                "Unsupported batch_dataset_strategy="
                f"{self.batch_dataset_strategy!r}; supported: ['quota', 'single_dataset']"
            )
        if self.max_episodes_per_batch > 0 and self.batch_dataset_strategy != "single_dataset":
            raise ValueError(
                "max_episodes_per_batch requires batch_dataset_strategy='single_dataset'"
            )
        if self.episode_burst_batches > 1 and self.batch_dataset_strategy != "single_dataset":
            raise ValueError(
                "episode_burst_batches requires batch_dataset_strategy='single_dataset'"
            )
        if self.quota_max_datasets_per_batch > 0 and self.batch_dataset_strategy != "quota":
            raise ValueError("quota_max_datasets_per_batch requires batch_dataset_strategy='quota'")

    def _make_state(self, index: DatasetEpisodeIndex, rng: random.Random) -> _DatasetWindowState:
        episodes = sorted(index.episode_to_indices)
        if self.shuffle:
            rng.shuffle(episodes)
        boundary_offsets = {
            episode_id: (
                rng.randrange(self.window_steps)
                if self.window_boundary_offsets
                and self.shuffle
                and self.window_steps > 1
                and len(index.episode_to_indices[episode_id]) > self.window_steps
                else 0
            )
            for episode_id in episodes
        }
        return _DatasetWindowState(
            episodes=episodes,
            episode_to_indices=index.episode_to_indices,
            active={},
            cursor_by_episode={episode_id: 0 for episode_id in episodes},
            window_boundary_offset_by_episode=boundary_offsets,
        )

    def _active_remaining(self, state: _DatasetWindowState) -> int:
        return sum(len(window) for window in state.active.values())

    def _should_activate(self, state: _DatasetWindowState) -> bool:
        if state.episode_cursor >= len(state.episodes):
            return False
        if len(state.active) < self.active_episodes_per_dataset:
            return True
        max_active = self.active_episodes_per_dataset + self.reactivate_when_remaining_below
        return (
            self.reactivate_when_remaining_below > 0
            and len(state.active) < max_active
            and self._active_remaining(state) <= self.reactivate_when_remaining_below
        )

    def _activate(self, state: _DatasetWindowState, rng: random.Random) -> None:
        while self._should_activate(state):
            episode_id = state.episodes[state.episode_cursor]
            state.episode_cursor += 1
            state.active[episode_id] = self._next_window(state, episode_id, rng)

    def _next_window(
        self, state: _DatasetWindowState, episode_id: int, rng: random.Random
    ) -> list[int]:
        cursor = state.cursor_by_episode[episode_id]
        indices = state.episode_to_indices[episode_id]
        boundary_offset = int(state.window_boundary_offset_by_episode.get(episode_id, 0))
        if cursor == 0 and boundary_offset > 0:
            window = indices[:boundary_offset]
        else:
            window = indices[cursor : cursor + self.window_steps]
        state.cursor_by_episode[episode_id] = cursor + len(window)
        if self.shuffle:
            rng.shuffle(window)
        return window

    def _pop_one(
        self, index: DatasetEpisodeIndex, state: _DatasetWindowState, rng: random.Random
    ) -> int | None:
        self._activate(state, rng)
        while state.active:
            episode_ids = sorted(state.active)
            if self.shuffle:
                rng.shuffle(episode_ids)
            for episode_id in episode_ids:
                window = state.active.get(episode_id, [])
                if window:
                    local_idx = window.pop() if self.shuffle else window.pop(0)
                    if not window:
                        replacement = self._next_window(state, episode_id, rng)
                        if replacement:
                            state.active[episode_id] = replacement
                        else:
                            state.active.pop(episode_id, None)
                    self._activate(state, rng)
                    return int(local_idx) + int(index.offset)
            state.active = {
                episode_id: window for episode_id, window in state.active.items() if window
            }
            self._activate(state, rng)
        return None

    def _reset_state(self, dataset_idx: int, rng: random.Random) -> _DatasetWindowState:
        return self._make_state(self.dataset_indices[dataset_idx], rng)

    def _take_from_dataset(
        self,
        *,
        dataset_idx: int,
        states: list[_DatasetWindowState],
        rng: random.Random,
    ) -> int | None:
        sample_idx = self._pop_one(self.dataset_indices[dataset_idx], states[dataset_idx], rng)
        if sample_idx is None and self.repeat:
            states[dataset_idx] = self._reset_state(dataset_idx, rng)
            sample_idx = self._pop_one(self.dataset_indices[dataset_idx], states[dataset_idx], rng)
        return sample_idx

    def _episode_for_sample(self, *, dataset_idx: int, sample_idx: int) -> int | None:
        index = self.dataset_indices[dataset_idx]
        local_idx = int(sample_idx) - int(index.offset)
        for episode_id, indices in index.episode_to_indices.items():
            if indices and int(indices[0]) <= local_idx <= int(indices[-1]):
                return int(episode_id)
        return None

    def _pop_one_from_episode(
        self,
        index: DatasetEpisodeIndex,
        state: _DatasetWindowState,
        *,
        episode_id: int,
        rng: random.Random,
    ) -> int | None:
        self._activate(state, rng)
        window = state.active.get(int(episode_id), [])
        if not window:
            return None
        local_idx = window.pop() if self.shuffle else window.pop(0)
        if not window:
            replacement = self._next_window(state, int(episode_id), rng)
            if replacement:
                state.active[int(episode_id)] = replacement
            else:
                state.active.pop(int(episode_id), None)
        self._activate(state, rng)
        return int(local_idx) + int(index.offset)

    def _take_from_dataset_episode(
        self,
        *,
        dataset_idx: int,
        episode_id: int,
        states: list[_DatasetWindowState],
        rng: random.Random,
    ) -> int | None:
        sample_idx = self._pop_one_from_episode(
            self.dataset_indices[dataset_idx],
            states[dataset_idx],
            episode_id=int(episode_id),
            rng=rng,
        )
        if sample_idx is None and self.repeat:
            sample_idx = self._pop_one_from_episode(
                self.dataset_indices[dataset_idx],
                states[dataset_idx],
                episode_id=int(episode_id),
                rng=rng,
            )
        return sample_idx

    def _single_dataset_order(
        self,
        *,
        emitted_by_dataset: list[int],
        total_emitted: int,
        rng: random.Random,
    ) -> list[int]:
        total_weight = sum(max(weight, 0.0) for weight in self.weights)
        if total_weight <= 0.0:
            order = list(range(len(self.dataset_indices)))
            if self.shuffle:
                rng.shuffle(order)
            return order

        desired_total = int(total_emitted) + int(self.batch_size)
        scored = []
        for dataset_idx, weight in enumerate(self.weights):
            target_fraction = max(float(weight), 0.0) / total_weight
            deficit = desired_total * target_fraction - float(emitted_by_dataset[dataset_idx])
            tie_breaker = rng.random() if self.shuffle else 0.0
            scored.append((deficit, tie_breaker, dataset_idx))
        scored.sort(reverse=True)
        return [dataset_idx for _deficit, _tie_breaker, dataset_idx in scored]

    def _next_quota_batch(
        self,
        *,
        states: list[_DatasetWindowState],
        quota_state: BatchQuotaState,
        emitted_by_dataset: list[int],
        total_emitted: int,
        rng: random.Random,
    ) -> tuple[list[int], list[int], int]:
        if 0 < self.quota_max_datasets_per_batch < len(self.weights):
            selected_dataset_indices = self._single_dataset_order(
                emitted_by_dataset=emitted_by_dataset,
                total_emitted=total_emitted,
                rng=rng,
            )[: self.quota_max_datasets_per_batch]
            selected_weights = [
                self.weights[dataset_idx] for dataset_idx in selected_dataset_indices
            ]
            selected_quotas = compute_batch_quotas(
                batch_size=self.batch_size,
                weights=selected_weights,
                state=BatchQuotaState(),
            )
            quotas = [0 for _ in self.weights]
            for dataset_idx, quota in zip(selected_dataset_indices, selected_quotas, strict=True):
                quotas[dataset_idx] = int(quota)
        else:
            quotas = compute_batch_quotas(
                batch_size=self.batch_size, weights=self.weights, state=quota_state
            )
        batch: list[int] = []
        for dataset_idx, quota in enumerate(quotas):
            for _ in range(int(quota)):
                sample_idx = self._take_from_dataset(
                    dataset_idx=dataset_idx,
                    states=states,
                    rng=rng,
                )
                if sample_idx is not None:
                    batch.append(sample_idx)
                    emitted_by_dataset[dataset_idx] += 1

        if len(batch) < self.batch_size:
            made_progress = True
            while len(batch) < self.batch_size and made_progress:
                made_progress = False
                dataset_order = list(range(len(self.dataset_indices)))
                if self.shuffle:
                    rng.shuffle(dataset_order)
                for dataset_idx in dataset_order:
                    sample_idx = self._take_from_dataset(
                        dataset_idx=dataset_idx,
                        states=states,
                        rng=rng,
                    )
                    if sample_idx is not None:
                        batch.append(sample_idx)
                        emitted_by_dataset[dataset_idx] += 1
                        made_progress = True
                    if len(batch) == self.batch_size:
                        break
        return batch, emitted_by_dataset, total_emitted + len(batch)

    def _next_single_dataset_batch(
        self,
        *,
        states: list[_DatasetWindowState],
        emitted_by_dataset: list[int],
        total_emitted: int,
        rng: random.Random,
        dataset_order: list[int] | None = None,
        episode_hint_by_dataset: dict[int, int] | None = None,
    ) -> tuple[list[int], list[int], int, int | None, int | None]:
        batch: list[int] = []
        if dataset_order is None:
            dataset_order = self._single_dataset_order(
                emitted_by_dataset=emitted_by_dataset,
                total_emitted=total_emitted,
                rng=rng,
            )
        used_dataset_idx: int | None = None
        used_episode_idx: int | None = None
        episode_hint_by_dataset = episode_hint_by_dataset or {}
        for dataset_idx in dataset_order:
            start_len = len(batch)
            preferred_episode_id: int | None = episode_hint_by_dataset.get(int(dataset_idx))
            used_episode_ids: set[int] = set()
            if preferred_episode_id is not None:
                used_episode_ids.add(int(preferred_episode_id))
            while len(batch) < self.batch_size:
                if preferred_episode_id is None:
                    sample_idx = self._take_from_dataset(
                        dataset_idx=dataset_idx,
                        states=states,
                        rng=rng,
                    )
                    if sample_idx is not None:
                        preferred_episode_id = self._episode_for_sample(
                            dataset_idx=dataset_idx,
                            sample_idx=sample_idx,
                        )
                        if preferred_episode_id is not None:
                            used_episode_ids.add(int(preferred_episode_id))
                else:
                    sample_idx = self._take_from_dataset_episode(
                        dataset_idx=dataset_idx,
                        episode_id=preferred_episode_id,
                        states=states,
                        rng=rng,
                    )
                    if sample_idx is None:
                        if (
                            self.max_episodes_per_batch > 0
                            and len(used_episode_ids) >= self.max_episodes_per_batch
                        ):
                            break
                        sample_idx = self._take_from_dataset(
                            dataset_idx=dataset_idx,
                            states=states,
                            rng=rng,
                        )
                        if sample_idx is not None:
                            preferred_episode_id = self._episode_for_sample(
                                dataset_idx=dataset_idx,
                                sample_idx=sample_idx,
                            )
                            if preferred_episode_id is not None:
                                used_episode_ids.add(int(preferred_episode_id))
                if sample_idx is None:
                    break
                batch.append(sample_idx)
                emitted_by_dataset[dataset_idx] += 1
            if len(batch) > start_len and used_dataset_idx is None:
                used_dataset_idx = int(dataset_idx)
                if used_episode_ids:
                    used_episode_idx = sorted(used_episode_ids)[0]
            if len(batch) > start_len and len(batch) < self.batch_size:
                break
            if len(batch) == self.batch_size:
                break
        return (
            batch,
            emitted_by_dataset,
            total_emitted + len(batch),
            used_dataset_idx,
            used_episode_idx,
        )

    def __iter__(self) -> Iterator[list[SampleIndex]]:
        pass_index = 0
        while True:
            rng = random.Random(self.seed + pass_index * self.seed_stride)
            states = [self._make_state(index, rng) for index in self.dataset_indices]
            quota_state = BatchQuotaState()
            emitted_by_dataset = [0 for _ in self.dataset_indices]
            total_emitted = 0
            burst_dataset_idx: int | None = None
            burst_batches_remaining = 0
            burst_episode_id: int | None = None
            burst_episode_batches_remaining = 0
            emitted_any = False
            while True:
                if self.batch_dataset_strategy == "single_dataset":
                    dataset_order = None
                    episode_hint_by_dataset: dict[int, int] | None = None
                    should_continue_dataset = burst_dataset_idx is not None and (
                        burst_batches_remaining > 0 or burst_episode_batches_remaining > 0
                    )
                    if should_continue_dataset:
                        fallback_order = self._single_dataset_order(
                            emitted_by_dataset=emitted_by_dataset,
                            total_emitted=total_emitted,
                            rng=rng,
                        )
                        dataset_order = [burst_dataset_idx] + [
                            dataset_idx
                            for dataset_idx in fallback_order
                            if dataset_idx != burst_dataset_idx
                        ]
                        if burst_episode_id is not None and burst_episode_batches_remaining > 0:
                            episode_hint_by_dataset = {
                                int(burst_dataset_idx): int(burst_episode_id)
                            }
                    (
                        batch,
                        emitted_by_dataset,
                        total_emitted,
                        used_dataset_idx,
                        used_episode_id,
                    ) = self._next_single_dataset_batch(
                        states=states,
                        emitted_by_dataset=emitted_by_dataset,
                        total_emitted=total_emitted,
                        rng=rng,
                        dataset_order=dataset_order,
                        episode_hint_by_dataset=episode_hint_by_dataset,
                    )
                    if len(batch) == self.batch_size and used_dataset_idx is not None:
                        if burst_dataset_idx == used_dataset_idx and burst_batches_remaining > 0:
                            burst_batches_remaining -= 1
                        else:
                            burst_dataset_idx = used_dataset_idx
                            burst_batches_remaining = self.batch_dataset_burst_batches - 1
                        if (
                            used_episode_id is not None
                            and burst_dataset_idx == used_dataset_idx
                            and burst_episode_id == used_episode_id
                            and burst_episode_batches_remaining > 0
                        ):
                            burst_episode_batches_remaining -= 1
                        else:
                            burst_episode_id = used_episode_id
                            burst_episode_batches_remaining = self.episode_burst_batches - 1
                    else:
                        burst_dataset_idx = None
                        burst_batches_remaining = 0
                        burst_episode_id = None
                        burst_episode_batches_remaining = 0
                else:
                    batch, emitted_by_dataset, total_emitted = self._next_quota_batch(
                        states=states,
                        quota_state=quota_state,
                        emitted_by_dataset=emitted_by_dataset,
                        total_emitted=total_emitted,
                        rng=rng,
                    )

                if len(batch) == self.batch_size:
                    emitted_any = True
                    yield _sorted_batch(batch)
                elif batch and not self.drop_last:
                    emitted_any = True
                    yield _sorted_batch(_pad_tail_batch(batch, self.batch_size))
                if len(batch) < self.batch_size:
                    break
            if not self.repeat or not emitted_any:
                break
            pass_index += 1

    def __len__(self) -> int:
        total = sum(
            len(indices)
            for index in self.dataset_indices
            for indices in index.episode_to_indices.values()
        )
        full, tail = divmod(total, self.batch_size)
        return full + (1 if tail and not self.drop_last else 0)


@dataclass(frozen=True, slots=True)
class RawChunkRef:
    dataset_id: int
    episode_id: int
    indices: tuple[int, ...]


@dataclass(slots=True)
class _DatasetChunkState:
    chunks: list[RawChunkRef]
    chunk_cursor: int = 0
    row_cursor: int = 0
    row_cursors: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.row_cursors:
            self.row_cursors = [0 for _ in self.chunks]


def _chunk_remaining(chunk: RawChunkRef, row_cursor: int) -> int:
    return max(len(chunk.indices) - int(row_cursor), 0)


class EpochChunkRawBatchSampler(Sampler[list[SampleIndex]]):
    def __init__(
        self,
        *,
        dataset_indices: Sequence[DatasetEpisodeIndex],
        weights: Sequence[float],
        batch_size: int,
        chunk_steps: int,
        seed: int,
        repeat: bool,
        drop_last: bool,
        shuffle_chunks: bool = True,
        reshuffle_each_epoch: bool = True,
        epoch_boundary_offsets: bool = True,
        seed_stride: int = 0,
        batch_dataset_strategy: str = "quota",
        batch_dataset_burst_batches: int = 1,
        quota_max_datasets_per_batch: int = 0,
        max_episodes_per_batch: int = 0,
        episode_burst_batches: int = 1,
    ) -> None:
        self.dataset_indices = list(dataset_indices)
        self.weights = [float(weight) for weight in weights]
        self.batch_size = int(batch_size)
        self.chunk_steps = max(int(chunk_steps), 1)
        self.seed = int(seed)
        self.seed_stride = int(seed_stride)
        self.repeat = bool(repeat)
        self.drop_last = bool(drop_last)
        self.shuffle_chunks = bool(shuffle_chunks)
        self.reshuffle_each_epoch = bool(reshuffle_each_epoch)
        self.epoch_boundary_offsets = bool(epoch_boundary_offsets)
        self.batch_dataset_strategy = str(batch_dataset_strategy)
        self.batch_dataset_burst_batches = max(int(batch_dataset_burst_batches), 1)
        self.quota_max_datasets_per_batch = max(int(quota_max_datasets_per_batch), 0)
        self.max_episodes_per_batch = max(int(max_episodes_per_batch), 0)
        self.episode_burst_batches = max(int(episode_burst_batches), 1)
        if self.batch_dataset_strategy not in {"quota", "single_dataset"}:
            raise ValueError(
                "Unsupported batch_dataset_strategy="
                f"{self.batch_dataset_strategy!r}; supported: ['quota', 'single_dataset']"
            )
        if self.max_episodes_per_batch > 0 and self.batch_dataset_strategy != "single_dataset":
            raise ValueError(
                "max_episodes_per_batch requires batch_dataset_strategy='single_dataset'"
            )
        if self.episode_burst_batches > 1 and self.batch_dataset_strategy != "single_dataset":
            raise ValueError(
                "episode_burst_batches requires batch_dataset_strategy='single_dataset'"
            )
        if self.quota_max_datasets_per_batch > 0 and self.batch_dataset_strategy != "quota":
            raise ValueError("quota_max_datasets_per_batch requires batch_dataset_strategy='quota'")

    def _episode_chunks(
        self,
        *,
        dataset_id: int,
        episode_id: int,
        indices: list[int],
        rng: random.Random,
    ) -> list[RawChunkRef]:
        if not indices:
            return []
        offset = 0
        if self.epoch_boundary_offsets and self.shuffle_chunks and len(indices) > self.chunk_steps:
            offset = rng.randrange(self.chunk_steps)

        chunks: list[RawChunkRef] = []
        cursor = 0
        if offset > 0:
            chunks.append(
                RawChunkRef(
                    dataset_id=dataset_id,
                    episode_id=episode_id,
                    indices=tuple(indices[:offset]),
                )
            )
            cursor = offset
        while cursor < len(indices):
            chunk = tuple(indices[cursor : cursor + self.chunk_steps])
            if chunk:
                chunks.append(
                    RawChunkRef(dataset_id=dataset_id, episode_id=episode_id, indices=chunk)
                )
            cursor += self.chunk_steps
        return chunks

    def _make_state(self, index: DatasetEpisodeIndex, rng: random.Random) -> _DatasetChunkState:
        chunks: list[RawChunkRef] = []
        for episode_id in sorted(index.episode_to_indices):
            chunks.extend(
                self._episode_chunks(
                    dataset_id=int(index.dataset_id),
                    episode_id=int(episode_id),
                    indices=index.episode_to_indices[int(episode_id)],
                    rng=rng,
                )
            )
        if self.shuffle_chunks:
            rng.shuffle(chunks)
        return _DatasetChunkState(chunks=chunks)

    def _repeat_seed(self, pass_index: int) -> int:
        if not self.reshuffle_each_epoch:
            return self.seed
        return self.seed + int(pass_index) * self.seed_stride

    def _reset_repeating_dataset(
        self,
        *,
        dataset_idx: int,
        states: list[_DatasetChunkState],
        dataset_pass_indices: list[int] | None,
    ) -> bool:
        if not self.repeat or dataset_pass_indices is None:
            return False
        index = self.dataset_indices[dataset_idx]
        if not any(index.episode_to_indices.values()):
            return False

        dataset_pass_indices[dataset_idx] += 1
        rng = random.Random(self._repeat_seed(dataset_pass_indices[dataset_idx]))
        states[dataset_idx] = self._make_state(index, rng)
        return True

    def _advance_chunk_cursor(self, state: _DatasetChunkState) -> None:
        while state.chunk_cursor < len(state.chunks):
            cursor = state.row_cursors[state.chunk_cursor]
            chunk = state.chunks[state.chunk_cursor]
            if _chunk_remaining(chunk, cursor) > 0:
                state.row_cursor = cursor
                return
            state.chunk_cursor += 1
            state.row_cursor = 0

    def _take_from_chunk(
        self,
        *,
        index: DatasetEpisodeIndex,
        state: _DatasetChunkState,
        chunk_idx: int,
    ) -> int:
        chunk = state.chunks[chunk_idx]
        cursor = state.row_cursors[chunk_idx]
        local_idx = chunk.indices[cursor]
        state.row_cursors[chunk_idx] = cursor + 1
        if chunk_idx == state.chunk_cursor:
            self._advance_chunk_cursor(state)
        return int(local_idx) + int(index.offset)

    def _take_from_dataset(
        self,
        *,
        dataset_idx: int,
        states: list[_DatasetChunkState],
        dataset_pass_indices: list[int] | None = None,
    ) -> int | None:
        while True:
            state = states[dataset_idx]
            index = self.dataset_indices[dataset_idx]
            self._advance_chunk_cursor(state)
            if state.chunk_cursor < len(state.chunks):
                return self._take_from_chunk(
                    index=index,
                    state=state,
                    chunk_idx=state.chunk_cursor,
                )
            if not self._reset_repeating_dataset(
                dataset_idx=dataset_idx,
                states=states,
                dataset_pass_indices=dataset_pass_indices,
            ):
                return None

    def _take_from_dataset_episode(
        self,
        *,
        dataset_idx: int,
        episode_id: int,
        states: list[_DatasetChunkState],
        dataset_pass_indices: list[int] | None = None,
    ) -> int | None:
        state = states[dataset_idx]
        index = self.dataset_indices[dataset_idx]
        self._advance_chunk_cursor(state)
        for chunk_idx in range(state.chunk_cursor, len(state.chunks)):
            chunk = state.chunks[chunk_idx]
            cursor = state.row_cursors[chunk_idx]
            if int(chunk.episode_id) == int(episode_id) and _chunk_remaining(chunk, cursor) > 0:
                return self._take_from_chunk(
                    index=index,
                    state=state,
                    chunk_idx=chunk_idx,
                )
        return None

    def _episode_for_sample(self, *, dataset_idx: int, sample_idx: int) -> int | None:
        index = self.dataset_indices[dataset_idx]
        local_idx = int(sample_idx) - int(index.offset)
        for episode_id, indices in index.episode_to_indices.items():
            if indices and int(indices[0]) <= local_idx <= int(indices[-1]):
                return int(episode_id)
        return None

    def _single_dataset_order(
        self,
        *,
        emitted_by_dataset: list[int],
        total_emitted: int,
        rng: random.Random,
    ) -> list[int]:
        total_weight = sum(max(weight, 0.0) for weight in self.weights)
        if total_weight <= 0.0:
            order = list(range(len(self.dataset_indices)))
            if self.shuffle_chunks:
                rng.shuffle(order)
            return order

        desired_total = int(total_emitted) + int(self.batch_size)
        scored = []
        for dataset_idx, weight in enumerate(self.weights):
            target_fraction = max(float(weight), 0.0) / total_weight
            deficit = desired_total * target_fraction - float(emitted_by_dataset[dataset_idx])
            tie_breaker = rng.random() if self.shuffle_chunks else 0.0
            scored.append((deficit, tie_breaker, dataset_idx))
        scored.sort(reverse=True)
        return [dataset_idx for _deficit, _tie_breaker, dataset_idx in scored]

    def _next_quota_batch(
        self,
        *,
        states: list[_DatasetChunkState],
        dataset_pass_indices: list[int] | None,
        quota_state: BatchQuotaState,
        emitted_by_dataset: list[int],
        total_emitted: int,
        rng: random.Random,
    ) -> tuple[list[int], list[int], int]:
        if 0 < self.quota_max_datasets_per_batch < len(self.weights):
            selected_dataset_indices = self._single_dataset_order(
                emitted_by_dataset=emitted_by_dataset,
                total_emitted=total_emitted,
                rng=rng,
            )[: self.quota_max_datasets_per_batch]
            selected_weights = [
                self.weights[dataset_idx] for dataset_idx in selected_dataset_indices
            ]
            selected_quotas = compute_batch_quotas(
                batch_size=self.batch_size,
                weights=selected_weights,
                state=BatchQuotaState(),
            )
            quotas = [0 for _ in self.weights]
            for dataset_idx, quota in zip(selected_dataset_indices, selected_quotas, strict=True):
                quotas[dataset_idx] = int(quota)
        else:
            quotas = compute_batch_quotas(
                batch_size=self.batch_size,
                weights=self.weights,
                state=quota_state,
            )
        batch: list[int] = []
        for dataset_idx, quota in enumerate(quotas):
            for _ in range(int(quota)):
                sample_idx = self._take_from_dataset(
                    dataset_idx=dataset_idx,
                    states=states,
                    dataset_pass_indices=dataset_pass_indices,
                )
                if sample_idx is not None:
                    batch.append(sample_idx)
                    emitted_by_dataset[dataset_idx] += 1

        if len(batch) < self.batch_size:
            made_progress = True
            while len(batch) < self.batch_size and made_progress:
                made_progress = False
                dataset_order = list(range(len(self.dataset_indices)))
                if self.shuffle_chunks:
                    rng.shuffle(dataset_order)
                for dataset_idx in dataset_order:
                    sample_idx = self._take_from_dataset(
                        dataset_idx=dataset_idx,
                        states=states,
                        dataset_pass_indices=dataset_pass_indices,
                    )
                    if sample_idx is not None:
                        batch.append(sample_idx)
                        emitted_by_dataset[dataset_idx] += 1
                        made_progress = True
                    if len(batch) == self.batch_size:
                        break
        return batch, emitted_by_dataset, total_emitted + len(batch)

    def _next_single_dataset_batch(
        self,
        *,
        states: list[_DatasetChunkState],
        dataset_pass_indices: list[int] | None,
        emitted_by_dataset: list[int],
        total_emitted: int,
        rng: random.Random,
        dataset_order: list[int] | None = None,
        episode_hint_by_dataset: dict[int, int] | None = None,
    ) -> tuple[list[int], list[int], int, int | None, int | None]:
        batch: list[int] = []
        if dataset_order is None:
            dataset_order = self._single_dataset_order(
                emitted_by_dataset=emitted_by_dataset,
                total_emitted=total_emitted,
                rng=rng,
            )
        used_dataset_idx: int | None = None
        used_episode_idx: int | None = None
        episode_hint_by_dataset = episode_hint_by_dataset or {}
        for dataset_idx in dataset_order:
            start_len = len(batch)
            preferred_episode_id: int | None = episode_hint_by_dataset.get(int(dataset_idx))
            used_episode_ids: set[int] = set()
            if preferred_episode_id is not None:
                used_episode_ids.add(int(preferred_episode_id))
            while len(batch) < self.batch_size:
                if preferred_episode_id is None:
                    sample_idx = self._take_from_dataset(
                        dataset_idx=dataset_idx,
                        states=states,
                        dataset_pass_indices=dataset_pass_indices,
                    )
                    if sample_idx is not None:
                        preferred_episode_id = self._episode_for_sample(
                            dataset_idx=dataset_idx,
                            sample_idx=sample_idx,
                        )
                        if preferred_episode_id is not None:
                            used_episode_ids.add(int(preferred_episode_id))
                else:
                    sample_idx = self._take_from_dataset_episode(
                        dataset_idx=dataset_idx,
                        episode_id=preferred_episode_id,
                        states=states,
                        dataset_pass_indices=dataset_pass_indices,
                    )
                    if sample_idx is None:
                        if (
                            self.max_episodes_per_batch > 0
                            and len(used_episode_ids) >= self.max_episodes_per_batch
                        ):
                            break
                        sample_idx = self._take_from_dataset(
                            dataset_idx=dataset_idx,
                            states=states,
                            dataset_pass_indices=dataset_pass_indices,
                        )
                        if sample_idx is not None:
                            preferred_episode_id = self._episode_for_sample(
                                dataset_idx=dataset_idx,
                                sample_idx=sample_idx,
                            )
                            if preferred_episode_id is not None:
                                used_episode_ids.add(int(preferred_episode_id))
                if sample_idx is None:
                    break
                batch.append(sample_idx)
                emitted_by_dataset[dataset_idx] += 1
            if len(batch) > start_len and used_dataset_idx is None:
                used_dataset_idx = int(dataset_idx)
                if used_episode_ids:
                    used_episode_idx = sorted(used_episode_ids)[0]
            if len(batch) > start_len and len(batch) < self.batch_size:
                break
            if len(batch) == self.batch_size:
                break
        return (
            batch,
            emitted_by_dataset,
            total_emitted + len(batch),
            used_dataset_idx,
            used_episode_idx,
        )

    def __iter__(self) -> Iterator[list[SampleIndex]]:
        pass_index = 0
        while True:
            epoch_seed = self._repeat_seed(pass_index)
            rng = random.Random(epoch_seed)
            states = [self._make_state(index, rng) for index in self.dataset_indices]
            dataset_pass_indices = (
                [pass_index for _ in self.dataset_indices] if self.repeat else None
            )
            quota_state = BatchQuotaState()
            emitted_by_dataset = [0 for _ in self.dataset_indices]
            total_emitted = 0
            burst_dataset_idx: int | None = None
            burst_batches_remaining = 0
            burst_episode_id: int | None = None
            burst_episode_batches_remaining = 0
            emitted_any = False
            while True:
                if self.batch_dataset_strategy == "single_dataset":
                    dataset_order = None
                    episode_hint_by_dataset: dict[int, int] | None = None
                    should_continue_dataset = burst_dataset_idx is not None and (
                        burst_batches_remaining > 0 or burst_episode_batches_remaining > 0
                    )
                    if should_continue_dataset:
                        fallback_order = self._single_dataset_order(
                            emitted_by_dataset=emitted_by_dataset,
                            total_emitted=total_emitted,
                            rng=rng,
                        )
                        dataset_order = [burst_dataset_idx] + [
                            dataset_idx
                            for dataset_idx in fallback_order
                            if dataset_idx != burst_dataset_idx
                        ]
                        if burst_episode_id is not None and burst_episode_batches_remaining > 0:
                            episode_hint_by_dataset = {
                                int(burst_dataset_idx): int(burst_episode_id)
                            }
                    (
                        batch,
                        emitted_by_dataset,
                        total_emitted,
                        used_dataset_idx,
                        used_episode_id,
                    ) = self._next_single_dataset_batch(
                        states=states,
                        dataset_pass_indices=dataset_pass_indices,
                        emitted_by_dataset=emitted_by_dataset,
                        total_emitted=total_emitted,
                        rng=rng,
                        dataset_order=dataset_order,
                        episode_hint_by_dataset=episode_hint_by_dataset,
                    )
                    if len(batch) == self.batch_size and used_dataset_idx is not None:
                        if burst_dataset_idx == used_dataset_idx and burst_batches_remaining > 0:
                            burst_batches_remaining -= 1
                        else:
                            burst_dataset_idx = used_dataset_idx
                            burst_batches_remaining = self.batch_dataset_burst_batches - 1
                        if (
                            used_episode_id is not None
                            and burst_dataset_idx == used_dataset_idx
                            and burst_episode_id == used_episode_id
                            and burst_episode_batches_remaining > 0
                        ):
                            burst_episode_batches_remaining -= 1
                        else:
                            burst_episode_id = used_episode_id
                            burst_episode_batches_remaining = self.episode_burst_batches - 1
                    else:
                        burst_dataset_idx = None
                        burst_batches_remaining = 0
                        burst_episode_id = None
                        burst_episode_batches_remaining = 0
                else:
                    batch, emitted_by_dataset, total_emitted = self._next_quota_batch(
                        states=states,
                        dataset_pass_indices=dataset_pass_indices,
                        quota_state=quota_state,
                        emitted_by_dataset=emitted_by_dataset,
                        total_emitted=total_emitted,
                        rng=rng,
                    )

                if len(batch) == self.batch_size:
                    emitted_any = True
                    yield _sorted_batch(batch)
                elif batch and not self.drop_last:
                    emitted_any = True
                    yield _sorted_batch(_pad_tail_batch(batch, self.batch_size))
                if len(batch) < self.batch_size:
                    break
            if not self.repeat or not emitted_any:
                break
            pass_index += 1

    def __len__(self) -> int:
        total = sum(
            len(indices)
            for index in self.dataset_indices
            for indices in index.episode_to_indices.values()
        )
        full, tail = divmod(total, self.batch_size)
        return full + (1 if tail and not self.drop_last else 0)

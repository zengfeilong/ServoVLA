from __future__ import annotations

import random
from itertools import islice

import pytest

from servovla.data.indexing import PaddedSampleIndex, sample_index_value
from servovla.data.raw_window_sampler import (
    DatasetEpisodeIndex,
    EpochChunkRawBatchSampler,
    ExplicitWeightedRawBatchSampler,
)


class _FakeDataset:
    def __init__(self, lengths):
        self.num_episodes = len(lengths)
        starts = []
        ends = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += int(length)
            ends.append(cursor)
        self.episode_data_index = {"from": starts, "to": ends}


def test_raw_window_sampler_is_deterministic_for_fixed_seed():
    ds = _FakeDataset([5, 5, 5])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    first = list(
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=3,
            window_steps=2,
            active_episodes_per_dataset=2,
            seed=7,
            repeat=False,
            drop_last=False,
        )
    )
    second = list(
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=3,
            window_steps=2,
            active_episodes_per_dataset=2,
            seed=7,
            repeat=False,
            drop_last=False,
        )
    )

    assert first == second
    assert sorted(item for batch in first for item in batch) == list(range(15))


def test_raw_window_sampler_applies_explicit_dataset_quotas():
    ds_a = _FakeDataset([8])
    ds_b = _FakeDataset([8])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)

    batches = list(
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index_a, index_b],
            weights=[3.0, 1.0],
            batch_size=4,
            window_steps=4,
            active_episodes_per_dataset=1,
            seed=3,
            repeat=False,
            drop_last=True,
        )
    )

    first = batches[0]
    assert sum(0 <= idx < 100 for idx in first) == 3
    assert sum(idx >= 100 for idx in first) == 1


def test_raw_window_quota_can_limit_datasets_per_batch():
    indices = [
        DatasetEpisodeIndex.from_lerobot_dataset(
            _FakeDataset([64]), dataset_id=dataset_id, offset=dataset_id * 100
        )
        for dataset_id in range(4)
    ]

    batches = list(
        islice(
            iter(
                ExplicitWeightedRawBatchSampler(
                    dataset_indices=indices,
                    weights=[4.0, 3.0, 2.0, 1.0],
                    batch_size=8,
                    window_steps=16,
                    active_episodes_per_dataset=1,
                    seed=3,
                    repeat=True,
                    drop_last=True,
                    shuffle=False,
                    batch_dataset_strategy="quota",
                    quota_max_datasets_per_batch=2,
                )
            ),
            12,
        )
    )

    dataset_sets = [{sample_index_value(idx) // 100 for idx in batch} for batch in batches]
    assert all(len(dataset_ids) <= 2 for dataset_ids in dataset_sets)
    assert len(set().union(*dataset_sets)) == 4


def test_epoch_chunk_quota_can_limit_datasets_per_batch():
    indices = [
        DatasetEpisodeIndex.from_lerobot_dataset(
            _FakeDataset([64]), dataset_id=dataset_id, offset=dataset_id * 100
        )
        for dataset_id in range(4)
    ]

    batches = list(
        islice(
            iter(
                EpochChunkRawBatchSampler(
                    dataset_indices=indices,
                    weights=[4.0, 3.0, 2.0, 1.0],
                    batch_size=8,
                    chunk_steps=16,
                    seed=3,
                    repeat=True,
                    drop_last=True,
                    shuffle_chunks=False,
                    batch_dataset_strategy="quota",
                    quota_max_datasets_per_batch=2,
                )
            ),
            12,
        )
    )

    dataset_sets = [{sample_index_value(idx) // 100 for idx in batch} for batch in batches]
    assert all(len(dataset_ids) <= 2 for dataset_ids in dataset_sets)
    assert len(set().union(*dataset_sets)) == 4


def test_raw_window_episode_limit_requires_single_dataset_strategy():
    ds = _FakeDataset([8])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    with pytest.raises(ValueError, match="max_episodes_per_batch"):
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=4,
            window_steps=4,
            active_episodes_per_dataset=1,
            seed=3,
            repeat=False,
            drop_last=True,
            max_episodes_per_batch=1,
        )


def test_raw_window_sampler_can_emit_single_dataset_batches_to_reduce_video_fanout():
    ds_a = _FakeDataset([32])
    ds_b = _FakeDataset([32])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)

    batches = list(
        islice(
            iter(
                ExplicitWeightedRawBatchSampler(
                    dataset_indices=[index_a, index_b],
                    weights=[3.0, 1.0],
                    batch_size=4,
                    window_steps=4,
                    active_episodes_per_dataset=1,
                    seed=3,
                    repeat=True,
                    drop_last=True,
                    shuffle=False,
                    batch_dataset_strategy="single_dataset",
                    batch_dataset_burst_batches=1,
                    max_episodes_per_batch=1,
                )
            ),
            8,
        )
    )

    assert batches
    assert all(
        all(0 <= idx < 100 for idx in batch) or all(idx >= 100 for idx in batch)
        for batch in batches
    )
    assert sum(0 <= idx < 100 for batch in batches for idx in batch) == 24
    assert sum(idx >= 100 for batch in batches for idx in batch) == 8


def test_raw_window_single_dataset_can_burst_dataset_batches_for_decoder_locality():
    ds_a = _FakeDataset([64])
    ds_b = _FakeDataset([64])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)

    batches = list(
        islice(
            iter(
                ExplicitWeightedRawBatchSampler(
                    dataset_indices=[index_a, index_b],
                    weights=[1.0, 1.0],
                    batch_size=4,
                    window_steps=4,
                    active_episodes_per_dataset=1,
                    seed=3,
                    repeat=True,
                    drop_last=True,
                    shuffle=False,
                    batch_dataset_strategy="single_dataset",
                    batch_dataset_burst_batches=3,
                )
            ),
            6,
        )
    )
    dataset_sequence = [0 if all(0 <= idx < 100 for idx in batch) else 1 for batch in batches]

    assert len(set(dataset_sequence[:3])) == 1
    assert len(set(dataset_sequence[3:6])) == 1
    assert dataset_sequence[0] != dataset_sequence[3]


def test_raw_window_single_dataset_can_limit_episodes_per_batch():
    ds = _FakeDataset([5, 5])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    batches = list(
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=8,
            window_steps=5,
            active_episodes_per_dataset=2,
            seed=3,
            repeat=False,
            drop_last=False,
            shuffle=False,
            batch_dataset_strategy="single_dataset",
            max_episodes_per_batch=1,
        )
    )

    assert batches
    assert all(len({sample_index_value(idx) // 5 for idx in batch}) == 1 for batch in batches)


def test_raw_window_sampler_pads_validation_tail_when_drop_last_false():
    ds = _FakeDataset([3])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    batches = list(
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=4,
            window_steps=4,
            active_episodes_per_dataset=1,
            seed=1,
            repeat=False,
            drop_last=False,
            shuffle=False,
        )
    )

    assert len(batches) == 1
    assert len(batches[0]) == 4
    assert [idx for idx in batches[0] if not isinstance(idx, PaddedSampleIndex)] == [0, 1, 2]
    padded = [idx for idx in batches[0] if isinstance(idx, PaddedSampleIndex)]
    assert len(padded) == 1
    assert sample_index_value(padded[0]) in {0, 1, 2}


def test_raw_window_sampler_continues_when_one_dataset_runs_dry():
    ds_a = _FakeDataset([8])
    ds_b = _FakeDataset([2])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)

    batches = list(
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index_a, index_b],
            weights=[1.0, 1.0],
            batch_size=2,
            window_steps=4,
            active_episodes_per_dataset=1,
            seed=1,
            repeat=False,
            drop_last=True,
            shuffle=False,
        )
    )

    flattened = [idx for batch in batches for idx in batch]
    assert len(batches) == 5
    assert sorted(idx for idx in flattened if idx < 100) == list(range(8))
    assert sorted(idx for idx in flattened if idx >= 100) == [100, 101]


def test_raw_window_sampler_advances_repeat_seed_by_stride():
    ds = _FakeDataset([1, 1, 1, 1, 1, 1, 1, 1])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    sampler = ExplicitWeightedRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=2,
        window_steps=1,
        active_episodes_per_dataset=2,
        reactivate_when_remaining_below=0,
        seed=11,
        seed_stride=100003,
        repeat=True,
        drop_last=True,
        shuffle=True,
    )
    batches = list(islice(iter(sampler), 8))

    assert batches[:4] != batches[4:]


def test_raw_window_sampler_sorts_each_batch_for_io_locality_when_shuffled():
    ds = _FakeDataset([4, 4, 4])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    batches = list(
        ExplicitWeightedRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=4,
            window_steps=2,
            active_episodes_per_dataset=3,
            seed=5,
            repeat=False,
            drop_last=True,
            shuffle=True,
        )
    )

    assert batches
    assert all(batch == sorted(batch) for batch in batches)


def test_raw_window_sampler_can_offset_window_boundaries_without_reordering_episode():
    ds = _FakeDataset([10])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)
    sampler = ExplicitWeightedRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=2,
        window_steps=4,
        active_episodes_per_dataset=1,
        seed=5,
        repeat=False,
        drop_last=False,
        shuffle=False,
        window_boundary_offsets=True,
    )
    state = sampler._make_state(index, random.Random(0))
    state.window_boundary_offset_by_episode[0] = 2

    first = sampler._next_window(state, 0, random.Random(0))
    second = sampler._next_window(state, 0, random.Random(0))
    third = sampler._next_window(state, 0, random.Random(0))

    assert first == [0, 1]
    assert second == [2, 3, 4, 5]
    assert third == [6, 7, 8, 9]


def test_raw_window_sampler_assigns_epoch_window_offsets_from_repeat_seed():
    ds = _FakeDataset([12, 12, 12])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)
    sampler = ExplicitWeightedRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=3,
        window_steps=4,
        active_episodes_per_dataset=2,
        seed=5,
        seed_stride=100003,
        repeat=True,
        drop_last=True,
        shuffle=True,
        window_boundary_offsets=True,
    )

    first_state = sampler._make_state(index, random.Random(sampler.seed))
    second_state = sampler._make_state(index, random.Random(sampler.seed + sampler.seed_stride))

    first_offsets = first_state.window_boundary_offset_by_episode
    second_offsets = second_state.window_boundary_offset_by_episode
    assert all(0 <= offset < sampler.window_steps for offset in first_offsets.values())
    assert first_offsets != second_offsets


def test_epoch_chunk_sampler_covers_each_row_once_per_pass():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([5, 7])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    batches = list(
        EpochChunkRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=3,
            chunk_steps=4,
            seed=7,
            repeat=False,
            drop_last=False,
            shuffle_chunks=True,
            epoch_boundary_offsets=True,
        )
    )

    flattened = [idx for batch in batches for idx in batch]
    assert sorted(flattened) == list(range(12))
    assert all(batch == sorted(batch) for batch in batches)


def test_epoch_chunk_sampler_pads_tail_without_losing_real_rows():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([6])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    batches = list(
        EpochChunkRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=4,
            chunk_steps=4,
            seed=7,
            repeat=False,
            drop_last=False,
            shuffle_chunks=False,
            epoch_boundary_offsets=False,
        )
    )

    assert [len(batch) for batch in batches] == [4, 4]
    real_indices = [
        idx for batch in batches for idx in batch if not isinstance(idx, PaddedSampleIndex)
    ]
    padded_indices = [
        idx for batch in batches for idx in batch if isinstance(idx, PaddedSampleIndex)
    ]
    assert real_indices == list(range(6))
    assert len(padded_indices) == 2
    assert all(sample_index_value(idx) in {4, 5} for idx in padded_indices)


def test_epoch_chunk_sampler_reseeds_each_repeat_pass():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([8, 8, 8])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=4,
        chunk_steps=4,
        seed=5,
        seed_stride=100003,
        repeat=True,
        drop_last=True,
        shuffle_chunks=True,
        epoch_boundary_offsets=True,
    )

    batches = list(islice(iter(sampler), 12))
    assert batches[:6] != batches[6:12]
    assert sorted(idx for batch in batches[:6] for idx in batch) == list(range(24))
    assert sorted(idx for batch in batches[6:12] for idx in batch) == list(range(24))


def test_epoch_chunk_sampler_can_disable_repeat_reshuffle():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([8, 8, 8])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=4,
        chunk_steps=4,
        seed=5,
        seed_stride=100003,
        repeat=True,
        drop_last=True,
        shuffle_chunks=True,
        reshuffle_each_epoch=False,
        epoch_boundary_offsets=True,
    )

    batches = list(islice(iter(sampler), 12))
    assert batches[:6] == batches[6:12]


def test_epoch_chunk_sampler_applies_dataset_quotas():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds_a = _FakeDataset([8])
    ds_b = _FakeDataset([8])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)

    batches = list(
        EpochChunkRawBatchSampler(
            dataset_indices=[index_a, index_b],
            weights=[3.0, 1.0],
            batch_size=4,
            chunk_steps=4,
            seed=3,
            repeat=False,
            drop_last=True,
            shuffle_chunks=False,
            epoch_boundary_offsets=False,
        )
    )

    assert batches
    first = batches[0]
    assert sum(0 <= idx < 100 for idx in first) == 3
    assert sum(idx >= 100 for idx in first) == 1


def test_epoch_chunk_episode_limit_requires_single_dataset_strategy():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([8])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    with pytest.raises(ValueError, match="max_episodes_per_batch"):
        EpochChunkRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=4,
            chunk_steps=4,
            seed=3,
            repeat=False,
            drop_last=True,
            max_episodes_per_batch=1,
        )


def test_epoch_chunk_episode_burst_requires_single_dataset_strategy():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([8])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)

    with pytest.raises(ValueError, match="episode_burst_batches"):
        EpochChunkRawBatchSampler(
            dataset_indices=[index],
            weights=[1.0],
            batch_size=4,
            chunk_steps=4,
            seed=3,
            repeat=False,
            drop_last=True,
            episode_burst_batches=2,
        )


def test_epoch_chunk_sampler_can_emit_single_dataset_batches_to_reduce_video_fanout():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds_a = _FakeDataset([32])
    ds_b = _FakeDataset([32])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index_a, index_b],
        weights=[3.0, 1.0],
        batch_size=4,
        chunk_steps=8,
        seed=3,
        repeat=True,
        drop_last=True,
        shuffle_chunks=False,
        epoch_boundary_offsets=False,
        batch_dataset_strategy="single_dataset",
        max_episodes_per_batch=1,
    )

    batches = list(islice(iter(sampler), 8))

    assert batches
    assert all(
        all(0 <= idx < 100 for idx in batch) or all(idx >= 100 for idx in batch)
        for batch in batches
    )
    assert sum(0 <= idx < 100 for batch in batches for idx in batch) == 24
    assert sum(idx >= 100 for batch in batches for idx in batch) == 8


def test_epoch_chunk_single_dataset_can_burst_dataset_batches_for_decoder_locality():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds_a = _FakeDataset([64])
    ds_b = _FakeDataset([64])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index_a, index_b],
        weights=[1.0, 1.0],
        batch_size=4,
        chunk_steps=8,
        seed=3,
        repeat=True,
        drop_last=True,
        shuffle_chunks=False,
        epoch_boundary_offsets=False,
        batch_dataset_strategy="single_dataset",
        batch_dataset_burst_batches=3,
    )

    batches = list(islice(iter(sampler), 6))
    dataset_sequence = [0 if all(0 <= idx < 100 for idx in batch) else 1 for batch in batches]

    assert len(set(dataset_sequence[:3])) == 1
    assert len(set(dataset_sequence[3:6])) == 1
    assert dataset_sequence[0] != dataset_sequence[3]
    assert sum(0 <= idx < 100 for batch in batches for idx in batch) == 12
    assert sum(idx >= 100 for batch in batches for idx in batch) == 12


def test_epoch_chunk_single_dataset_can_burst_episode_batches_for_decoder_locality():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([32, 32, 32])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=4,
        chunk_steps=4,
        seed=9,
        repeat=True,
        drop_last=True,
        shuffle_chunks=True,
        epoch_boundary_offsets=False,
        batch_dataset_strategy="single_dataset",
        max_episodes_per_batch=1,
        episode_burst_batches=2,
    )

    batches = list(islice(iter(sampler), 6))
    episode_sequence = [sample_index_value(batch[0]) // 32 for batch in batches]

    assert batches
    assert all(len({sample_index_value(idx) // 32 for idx in batch}) == 1 for batch in batches)
    assert episode_sequence[0] == episode_sequence[1]


def test_epoch_chunk_single_dataset_can_limit_episodes_per_batch():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([5, 5])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=8,
        chunk_steps=5,
        seed=3,
        repeat=False,
        drop_last=False,
        shuffle_chunks=False,
        epoch_boundary_offsets=False,
        batch_dataset_strategy="single_dataset",
        max_episodes_per_batch=1,
    )

    batches = list(iter(sampler))

    assert batches
    assert all(len({sample_index_value(idx) // 5 for idx in batch}) == 1 for batch in batches)


def test_epoch_chunk_single_dataset_pads_episode_tail_instead_of_crossing_dataset():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds_a = _FakeDataset([2])
    ds_b = _FakeDataset([8])
    index_a = DatasetEpisodeIndex.from_lerobot_dataset(ds_a, dataset_id=0, offset=0)
    index_b = DatasetEpisodeIndex.from_lerobot_dataset(ds_b, dataset_id=1, offset=100)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index_a, index_b],
        weights=[10.0, 1.0],
        batch_size=4,
        chunk_steps=4,
        seed=3,
        repeat=False,
        drop_last=False,
        shuffle_chunks=False,
        epoch_boundary_offsets=False,
        batch_dataset_strategy="single_dataset",
        max_episodes_per_batch=1,
    )

    first_batch = next(iter(sampler))

    assert all(sample_index_value(idx) < 100 for idx in first_batch)
    assert sum(isinstance(idx, PaddedSampleIndex) for idx in first_batch) == 2


def test_epoch_chunk_single_dataset_prefers_same_episode_when_boundary_chunk_is_short():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds = _FakeDataset([16, 16])
    index = DatasetEpisodeIndex.from_lerobot_dataset(ds, dataset_id=0, offset=0)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index],
        weights=[1.0],
        batch_size=4,
        chunk_steps=8,
        seed=0,
        repeat=False,
        drop_last=True,
        shuffle_chunks=True,
        epoch_boundary_offsets=True,
        batch_dataset_strategy="single_dataset",
    )

    batches = list(iter(sampler))

    assert all(len({0 if int(idx) < 16 else 1 for idx in batch}) == 1 for batch in batches)
    assert sorted(idx for batch in batches for idx in batch) == list(range(32))


def test_epoch_chunk_single_dataset_repeat_reuses_exhausted_datasets_by_weight():
    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    ds_small = _FakeDataset([8])
    ds_large = _FakeDataset([64])
    index_small = DatasetEpisodeIndex.from_lerobot_dataset(ds_small, dataset_id=0, offset=0)
    index_large = DatasetEpisodeIndex.from_lerobot_dataset(ds_large, dataset_id=1, offset=100)
    sampler = EpochChunkRawBatchSampler(
        dataset_indices=[index_small, index_large],
        weights=[3.0, 1.0],
        batch_size=4,
        chunk_steps=4,
        seed=3,
        seed_stride=100003,
        repeat=True,
        drop_last=True,
        shuffle_chunks=False,
        epoch_boundary_offsets=False,
        batch_dataset_strategy="single_dataset",
    )

    batches = list(islice(iter(sampler), 16))

    assert all(
        all(0 <= idx < 100 for idx in batch) or all(idx >= 100 for idx in batch)
        for batch in batches
    )
    assert sum(0 <= idx < 100 for batch in batches for idx in batch) == 48
    assert sum(idx >= 100 for batch in batches for idx in batch) == 16


def test_dataset_episode_index_supports_reader_style_hf_dataset_bounds():
    class _HFDataset:
        def __init__(self):
            self.episode_index = [4, 4, 8, 8, 8]

        def __getitem__(self, key):
            if key == "episode_index":
                return self.episode_index
            raise KeyError(key)

    class _MetaWithoutEpisodes:
        pass

    class _Dataset:
        episodes = None
        meta = _MetaWithoutEpisodes()
        hf_dataset = _HFDataset()

    index = DatasetEpisodeIndex.from_lerobot_dataset(
        _Dataset(),
        dataset_id=2,
        offset=100,
    )

    assert index.dataset_id == 2
    assert index.offset == 100
    assert index.episode_to_indices == {
        4: [0, 1],
        8: [2, 3, 4],
    }

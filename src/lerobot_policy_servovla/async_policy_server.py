import gc
import logging
import os
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch
from lerobot.async_inference.helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins

from servovla.config.hf_offline import configure_hf_offline_env

from .server_config_servovla import ServoVLAPolicyServerConfig


class ServoVLAPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "servovla_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: ServoVLAPolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.observation_queue = Queue(maxsize=1)
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps: set[int] = set()
        self.last_processed_obs = None
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self._loaded_policy_key: tuple[str, str, str] | None = None
        self._compiled_policy_key: tuple[str, str, str] | None = None
        self._warmed_policy_key: tuple[str, str, str] | None = None
        self._policy_lock = threading.RLock()
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        self.last_processed_obs = None
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def _policy_key(self, policy_specs: RemotePolicyConfig) -> tuple[str, str, str]:
        return (
            str(policy_specs.policy_type),
            str(policy_specs.pretrained_name_or_path),
            str(policy_specs.device),
        )

    def _release_policy(self, *, reason: str) -> None:
        policy = self.policy
        if policy is None:
            return

        self.logger.info("Releasing loaded policy (%s) before %s", self._loaded_policy_key, reason)
        if hasattr(policy, "shutdown_runtime"):
            policy.shutdown_runtime()
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._loaded_policy_key = None
        self._compiled_policy_key = None
        self._warmed_policy_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _env_flag(name: str, *, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off", "none"}:
            return False
        raise ValueError(f"{name} must be a boolean value, got {value!r}")

    def _policy_head_dtype_from_env(self) -> torch.dtype | None:
        dtype_name = os.environ.get("SERVOVLA_POLICY_HEAD_DTYPE", "").strip().lower()
        if dtype_name in {"", "auto", "none", "off", "float32", "fp32"}:
            return None
        if dtype_name in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype_name in {"float16", "fp16", "half"}:
            return torch.float16
        raise ValueError(
            "SERVOVLA_POLICY_HEAD_DTYPE must be one of: bfloat16, float16, float32, auto, off"
        )

    def _policy_head_attention_dtype_from_env(self) -> torch.dtype | None:
        dtype_name = os.environ.get("SERVOVLA_POLICY_HEAD_ATTN_DTYPE", "").strip().lower()
        if dtype_name in {"", "auto", "none", "off", "float32", "fp32"}:
            return None
        if dtype_name in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype_name in {"float16", "fp16", "half"}:
            return torch.float16
        raise ValueError(
            "SERVOVLA_POLICY_HEAD_ATTN_DTYPE must be one of: bfloat16, float16, float32, auto, off"
        )

    def _policy_head_or_none(self):
        if self.policy is None:
            return None
        policy_model = getattr(self.policy, "model", None)
        return getattr(policy_model, "policy_head", None)

    def _configure_policy_head_dtype(self) -> None:
        target_dtype = self._policy_head_dtype_from_env()
        if target_dtype is None:
            return
        if not str(self.device).startswith("cuda"):
            self.logger.warning(
                "Skipping policy head dtype=%s because policy device is %s",
                target_dtype,
                self.device,
            )
            return

        policy_head = self._policy_head_or_none()
        if policy_head is None:
            self.logger.warning(
                "Skipping policy head dtype conversion: policy.model.policy_head not found"
            )
            return

        policy_head.to(device=self.device, dtype=target_dtype)
        self.logger.info(
            "Policy head moved to %s with dtype=%s for FlashAttention eligibility",
            self.device,
            target_dtype,
        )

    def _configure_policy_head_attention_dtype(self) -> None:
        target_dtype = self._policy_head_attention_dtype_from_env()
        policy_head = self._policy_head_or_none()
        if policy_head is None:
            if target_dtype is not None:
                self.logger.warning(
                    "Skipping policy head attention dtype: policy.model.policy_head not found"
                )
            return

        configured = 0
        for module in policy_head.modules():
            set_dtype = getattr(module, "set_flash_attention_dtype", None)
            if callable(set_dtype):
                set_dtype(target_dtype)
                configured += 1

        if configured:
            self.logger.info(
                "Configured %d policy head attention modules for FlashAttention dtype=%s",
                configured,
                target_dtype,
            )

    def _torch_compile_env_value(self, name: str, *, module_env_stem: str | None = None) -> str:
        if module_env_stem is not None:
            module_value = os.environ.get(f"SERVOVLA_TORCH_COMPILE_{module_env_stem}_{name}")
            if module_value is not None and module_value.strip():
                return module_value.strip()
        return os.environ.get(f"SERVOVLA_TORCH_COMPILE_{name}", "").strip()

    def _torch_compile_kwargs(self, *, module_env_stem: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        backend = self._torch_compile_env_value("BACKEND", module_env_stem=module_env_stem)
        backend_name = backend.lower()
        if backend and backend.lower() not in {"default", "none", "auto"}:
            kwargs["backend"] = backend

        mode = self._torch_compile_env_value("MODE", module_env_stem=module_env_stem)
        mode_supported = backend_name not in {"eager", "aot_eager", "cudagraphs"}
        if mode and mode.lower() not in {"default", "none", "auto"} and mode_supported:
            kwargs["mode"] = mode
        dynamic = self._torch_compile_env_value("DYNAMIC", module_env_stem=module_env_stem)
        if dynamic is not None and dynamic.strip().lower() not in {"", "auto", "none"}:
            flag_name = (
                f"SERVOVLA_TORCH_COMPILE_{module_env_stem}_DYNAMIC"
                if module_env_stem is not None
                and os.environ.get(f"SERVOVLA_TORCH_COMPILE_{module_env_stem}_DYNAMIC")
                else "SERVOVLA_TORCH_COMPILE_DYNAMIC"
            )
            kwargs["dynamic"] = self._env_flag(flag_name)
        fullgraph = self._torch_compile_env_value("FULLGRAPH", module_env_stem=module_env_stem)
        if fullgraph is not None and fullgraph.strip().lower() not in {"", "auto", "none"}:
            flag_name = (
                f"SERVOVLA_TORCH_COMPILE_{module_env_stem}_FULLGRAPH"
                if module_env_stem is not None
                and os.environ.get(f"SERVOVLA_TORCH_COMPILE_{module_env_stem}_FULLGRAPH")
                else "SERVOVLA_TORCH_COMPILE_FULLGRAPH"
            )
            kwargs["fullgraph"] = self._env_flag(flag_name)
        return kwargs

    def _compile_policy_modules(self) -> None:
        if not self._env_flag("SERVOVLA_TORCH_COMPILE", default=False):
            return
        if self.policy is None or self._loaded_policy_key is None:
            return
        if self._compiled_policy_key == self._loaded_policy_key:
            return
        if not hasattr(torch, "compile"):
            self.logger.warning(
                "torch.compile is not available; keeping ServoVLA modules in eager mode"
            )
            self._compiled_policy_key = self._loaded_policy_key
            return

        suppress_errors = self._env_flag("SERVOVLA_TORCH_COMPILE_SUPPRESS_ERRORS", default=True)
        try:
            torch._dynamo.config.suppress_errors = suppress_errors  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.warning(
                "Failed to configure torch.compile suppress_errors=%s (%s)", suppress_errors, exc
            )

        model = getattr(self.policy, "model", None)
        if model is None:
            self.logger.warning("Skipping torch.compile: policy.model not found")
            self._compiled_policy_key = self._loaded_policy_key
            return

        compile_plan = [
            ("vision_encoder", "SERVOVLA_TORCH_COMPILE_VISION_ENCODER", "VISION_ENCODER"),
            ("vlm_encoder", "SERVOVLA_TORCH_COMPILE_VLM_ENCODER", "VLM_ENCODER"),
            ("policy_head", "SERVOVLA_TORCH_COMPILE_POLICY_HEAD", "POLICY_HEAD"),
        ]
        compiled: list[str] = []
        compile_records: list[str] = []
        for attr_name, env_name, env_stem in compile_plan:
            if not self._env_flag(env_name, default=True):
                continue
            module = getattr(model, attr_name, None)
            if module is None:
                self.logger.warning("Skipping torch.compile for %s: module not found", attr_name)
                continue
            compile_kwargs = self._torch_compile_kwargs(module_env_stem=env_stem)
            try:
                setattr(model, attr_name, torch.compile(module, **compile_kwargs))
                compiled.append(attr_name)
                compile_records.append(f"{attr_name}:{compile_kwargs}")
            except Exception as exc:
                self.logger.warning(
                    "torch.compile failed for %s (%s); keeping eager module", attr_name, exc
                )

        self._compiled_policy_key = self._loaded_policy_key
        self.logger.info(
            "torch.compile configured for ServoVLA modules: %s | kwargs=%s | suppress_errors=%s",
            ",".join(compiled) if compiled else "none",
            "; ".join(compile_records) if compile_records else "none",
            suppress_errors,
        )

    def _warmup_policy_after_compile(self) -> None:
        if not self._env_flag("SERVOVLA_TORCH_COMPILE_WARMUP", default=False):
            return
        if self.policy is None or self._loaded_policy_key is None:
            return
        if self._warmed_policy_key == self._loaded_policy_key:
            return

        policy_config = getattr(self.policy, "config", None)
        camera_keys = list(getattr(policy_config, "camera_keys", []) or [])
        if not camera_keys:
            self.logger.warning("Skipping torch.compile warmup: policy config has no camera_keys")
            self._warmed_policy_key = self._loaded_policy_key
            return

        state_dim = int(getattr(policy_config, "state_dim", 6))
        image_size = int(
            max(
                int(getattr(policy_config, "vision_image_size", 256)),
                int(getattr(policy_config, "vlm_image_size", 256)),
            )
        )
        warmup_task = os.environ.get("SERVOVLA_TORCH_COMPILE_WARMUP_TASK", "warmup")
        warmup_fail_ms = int(os.environ.get("SERVOVLA_TORCH_COMPILE_WARMUP_FAIL_MS", "120000"))
        warn_ms = min(int(self.config.semantic_wait_warn_ms), warmup_fail_ms)

        batch: dict[str, Any] = {
            OBS_STATE: torch.zeros(state_dim, dtype=torch.float32),
            "task": warmup_task,
        }
        for camera_key in camera_keys:
            batch[camera_key] = torch.zeros((image_size, image_size, 3), dtype=torch.uint8)

        self.logger.info(
            "Running torch.compile warmup for ServoVLA modules | cameras=%s | image_size=%s | fail_ms=%s",
            camera_keys,
            image_size,
            warmup_fail_ms,
        )
        start = time.perf_counter()
        if hasattr(self.policy, "configure_runtime"):
            self.policy.configure_runtime(
                max_frame_delay=self.config.max_frame_delay,
                semantic_wait_warn_ms=warn_ms,
                semantic_wait_fail_ms=warmup_fail_ms,
            )
        try:
            action_tensor = self.policy.predict_action_chunk(batch, action_step=0)
            if not torch.isfinite(action_tensor).all():
                nonfinite = int((~torch.isfinite(action_tensor)).sum().item())
                raise ValueError(
                    f"torch.compile warmup produced {nonfinite} non-finite action values."
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            if hasattr(self.policy, "configure_runtime"):
                self.policy.configure_runtime(
                    max_frame_delay=self.config.max_frame_delay,
                    semantic_wait_warn_ms=self.config.semantic_wait_warn_ms,
                    semantic_wait_fail_ms=self.config.semantic_wait_fail_ms,
                )

        self._warmed_policy_key = self._loaded_policy_key
        self.logger.info(
            "torch.compile warmup completed in %.2f seconds", time.perf_counter() - start
        )

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info("Client %s connected and ready", client_id)
        self._reset_server()
        with self._policy_lock:
            if self.policy is not None and hasattr(self.policy, "reset"):
                self.policy.reset()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        try:
            policy_class = get_policy_class(policy_specs.policy_type)
        except Exception as exc:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} is not available in this environment. "
                "Install `lerobot_policy_servovla` and make sure it is importable."
            ) from exc

        self.logger.info(
            "Receiving policy instructions from %s | Policy type: %s | Pretrained: %s | "
            "Actions per chunk: %s | Device: %s",
            client_id,
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            policy_specs.actions_per_chunk,
            policy_specs.device,
        )

        configure_hf_offline_env()
        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        start = time.perf_counter()
        with self._policy_lock:
            policy_key = self._policy_key(policy_specs)
            if self.policy is not None and self._loaded_policy_key == policy_key:
                self.logger.info("Reusing already loaded policy %s", policy_key)
            else:
                if self.policy is not None:
                    self._release_policy(reason="loading new policy instructions")
                self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
                self.policy.to(self.device)
                self._loaded_policy_key = policy_key

            self._configure_policy_head_dtype()
            self._configure_policy_head_attention_dtype()
            self._compile_policy_modules()

            if hasattr(self.policy, "configure_runtime"):
                self.policy.configure_runtime(
                    max_frame_delay=self.config.max_frame_delay,
                    semantic_wait_warn_ms=self.config.semantic_wait_warn_ms,
                    semantic_wait_fail_ms=self.config.semantic_wait_fail_ms,
                )
            self._warmup_policy_after_compile()
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            device_override = {"device": self.device}
            # Try to apply device overrides where supported by the saved preprocessor/postprocessor
            # configs. Some exported preprocessor configs may not include a device_processor step,
            # so fall back to calling without that override on failure.
            try:
                self.preprocessor, self.postprocessor = make_pre_post_processors(
                    self.policy.config,
                    pretrained_path=policy_specs.pretrained_name_or_path,
                    preprocessor_overrides={
                        "device_processor": device_override,
                        "rename_observations_processor": {"rename_map": policy_specs.rename_map},
                    },
                    postprocessor_overrides={"device_processor": device_override},
                )
            except Exception as exc:
                # If overrides fail (e.g. override keys don't match saved steps), retry without device override.
                self.logger.warning(
                    "Pre/post-processor override failed (%s). Retrying without device override.",
                    exc,
                )
                self.preprocessor, self.postprocessor = make_pre_post_processors(
                    self.policy.config,
                    pretrained_path=policy_specs.pretrained_name_or_path,
                    preprocessor_overrides={
                        "rename_observations_processor": {"rename_map": policy_specs.rename_map}
                    },
                    postprocessor_overrides={},
                )
        end = time.perf_counter()
        self.logger.info("Time taken to put policy on %s: %.4f seconds", self.device, end - start)
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        client_id = context.peer()
        self.logger.debug("Receiving observations from %s", client_id)

        receive_time = time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            "Received observation #%s | Avg FPS: %.2f | Target: %.2f | One-way latency: %.2fms",
            obs_timestep,
            fps_metrics["avg_fps"],
            fps_metrics["target_fps"],
            (receive_time - obs_timestamp) * 1000,
        )
        self.logger.debug(
            "Server timestamp: %.6f | Client timestamp: %.6f | Deserialization time: %.6fs",
            receive_time,
            obs_timestamp,
            deserialize_time,
        )

        if not self._enqueue_observation(timed_observation):
            self.logger.debug("Observation #%s has been filtered out", obs_timestep)

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.debug("Client %s connected for action streaming", client_id)

        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                "Running inference for observation #%s (must_go: %s)",
                obs.get_timestep(),
                obs.must_go,
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            with self._policy_lock:
                action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                "Action chunk #%s generated | Total time: %.2fms",
                obs.get_timestep(),
                (inference_time + serialize_time) * 1000,
            )

            time.sleep(
                max(
                    0,
                    self.config.inference_latency - max(0, time.perf_counter() - getactions_starts),
                )
            )
            return actions
        except Empty:
            return services_pb2.Empty()
        except Exception as exc:
            self.logger.error("Error in GetActions: %s", exc, exc_info=True)
            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(
                "Skipping observation #%s - timestep predicted already",
                obs.get_timestep(),
            )
            return False
        if observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                "Skipping observation #%s - too similar to last processed obs",
                obs.get_timestep(),
            )
            return False
        return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            if self.observation_queue.full():
                _ = self.observation_queue.get_nowait()
            self.observation_queue.put(obs)
            return True
        return False

    def _time_action_chunk(
        self, t_0: float, action_chunk: list[torch.Tensor], i_0: int
    ) -> list[TimedAction]:
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=action,
            )
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(
        self, observation: dict[str, torch.Tensor], *, action_step: int
    ) -> torch.Tensor:
        chunk = self.policy.predict_action_chunk(observation, action_step=action_step)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        previous_obs = self.last_processed_obs
        previous_timestep = previous_obs.get_timestep() if previous_obs is not None else None

        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(
            observation,
            action_step=int(observation_t.get_timestep()),
        )
        if not torch.isfinite(action_tensor).all():
            nonfinite = int((~torch.isfinite(action_tensor)).sum().item())
            raise ValueError(
                f"Policy produced {nonfinite} non-finite action values before postprocess."
            )
        inference_time = time.perf_counter() - start_inference

        frame_diff = (
            None if previous_timestep is None else observation_t.get_timestep() - previous_timestep
        )
        debug_state = None
        if hasattr(self.policy, "get_last_inference_debug"):
            try:
                debug_state = self.policy.get_last_inference_debug()
            except Exception as exc:
                self.logger.warning("Failed to read inference debug state (%s)", exc)
        if debug_state is not None:
            self.logger.info(
                "Frame diff log | obs_timestep=%s | obs_frame_diff=%s | current_action_step=%s | semantic_action_step=%s | step_delay=%s | action_step_source=%s | semantic_sample_mode=%s | action_queue_size=%s | action_chunk_size=%s | chunk_size_threshold=%s | latest_executed_action_timestep=%s | policy_step_before=%s | policy_step_after=%s | frame_delay=%s | last_vlm_before=%s | last_vlm_after=%s | vlm_refreshed=%s | vlm_latest_submitted=%s | vlm_latest_frame_id=%s | pending_refresh_will_be_supported=%s | task_changed=%s",
                observation_t.get_timestep(),
                frame_diff,
                debug_state.get("current_action_step"),
                debug_state.get("semantic_snapshot_action_step"),
                debug_state.get("step_delay", debug_state.get("frame_delay")),
                debug_state.get("action_step_source"),
                debug_state.get("semantic_sample_mode"),
                debug_state.get("action_queue_size"),
                self.actions_per_chunk,
                debug_state.get("chunk_size_threshold"),
                debug_state.get("latest_executed_action_timestep"),
                debug_state.get("policy_step_before"),
                debug_state.get("policy_step_after"),
                debug_state.get("frame_delay"),
                debug_state.get("last_vlm_update_step_before"),
                debug_state.get("last_vlm_update_step_after"),
                debug_state.get("vlm_refreshed"),
                debug_state.get("vlm_latest_submitted", debug_state.get("vlm_prefetch_submitted")),
                debug_state.get("vlm_latest_frame_id", debug_state.get("vlm_prefetch_frame_id")),
                debug_state.get("pending_refresh_will_be_supported"),
                debug_state.get("task_changed"),
            )

        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        processed_actions = []
        for i in range(chunk_size):
            processed_actions.append(self.postprocessor(action_tensor[:, i, :]))
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu()
        if not torch.isfinite(action_tensor).all():
            nonfinite = int((~torch.isfinite(action_tensor)).sum().item())
            raise ValueError(
                f"Policy produced {nonfinite} non-finite action values after postprocess."
            )
        postprocess_time = time.perf_counter() - start_postprocess

        self.logger.info(
            "Observation %s | Prepare %.2fms | Preprocess %.2fms | Inference %.2fms | Postprocess %.2fms",
            observation_t.get_timestep(),
            prepare_time * 1000,
            preprocessing_time * 1000,
            inference_time * 1000,
            postprocess_time * 1000,
        )

        return self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
        )

    def stop(self):
        self._reset_server()
        with self._policy_lock:
            self._release_policy(reason="server shutdown")
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: ServoVLAPolicyServerConfig):
    register_third_party_plugins()
    logging.info(pformat(asdict(cfg)))
    policy_server = ServoVLAPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info("ServoVLA PolicyServer started on %s:%s", cfg.host, cfg.port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

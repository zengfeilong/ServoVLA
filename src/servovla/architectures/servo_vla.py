import logging

import torch
import torch.nn as nn

from servovla.architectures.action_normalization import ActionNormalizer
from servovla.architectures.policy_head_checkpoint import (
    load_policy_head_state_dict,
    prefixed_policy_head_state_dict,
)
from servovla.architectures.vlm_forward import run_vlm_encoder_forward
from servovla.config.action_mode import normalize_action_delta_state_indices

log = logging.getLogger(__name__)


class ServoVLA(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module | None,
        vlm_encoder: nn.Module | None,
        policy_head: nn.Module,
        fm_solver=None,
        action_is_delta: bool = True,
        action_delta_state_indices: tuple[int | None, ...] | list[int | None] | None = None,
        action_normalizer: ActionNormalizer | None = None,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.vlm_encoder = vlm_encoder
        self.policy_head = policy_head

        self.fm_solver = fm_solver
        self.action_is_delta = bool(action_is_delta)
        self.action_delta_state_indices = (
            None if action_delta_state_indices is None else tuple(action_delta_state_indices)
        )
        self.action_normalizer = action_normalizer or ActionNormalizer(enabled=False)

        self.latest_vlm_feature = None
        self.last_vlm_update_step = 0
        self.current_step = 0

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if args:
            destination = args[0]
        if len(args) > 1:
            prefix = args[1]
        if len(args) > 2:
            keep_vars = args[2]
        return prefixed_policy_head_state_dict(
            self,
            prefix=f"{prefix}policy_head.",
            destination=destination,
            keep_vars=keep_vars,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return load_policy_head_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )

    def encode_observations(
        self,
        pixel_values: torch.Tensor,
        vlm_inputs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vision_encoder is None or self.vlm_encoder is None:
            raise RuntimeError("Online encoding requires both vision_encoder and vlm_encoder.")
        with torch.no_grad():
            f_vision = self.vision_encoder(pixel_values)
            c_sem = run_vlm_encoder_forward(self.vlm_encoder, vlm_inputs)
        return f_vision, c_sem

    def forward(self, x_t, t, pixel_values, vlm_inputs: dict, c_sem_mask, frame_delay, q_current):
        """Run the end-to-end online training path."""
        f_vision, c_sem = self.encode_observations(
            pixel_values=pixel_values,
            vlm_inputs=vlm_inputs,
        )
        policy_param = next(iter(self.policy_head.parameters()), None)
        if policy_param is not None:
            f_vision = f_vision.to(device=policy_param.device, dtype=policy_param.dtype)
            c_sem = c_sem.to(device=policy_param.device, dtype=policy_param.dtype)

        return self.forward_policy(
            x_t=x_t,
            t=t,
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
        )

    def forward_policy(self, x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current):
        """Feature-level helper used by deployment sampling and tests."""
        v_pred = self.policy_head(
            x_t=x_t,
            t=t,
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
        )
        return v_pred

    @torch.inference_mode()
    def update_vlm_feature(self, vlm_inputs: dict):
        """Update the low-frequency VLM feature cache."""
        if self.vlm_encoder is None:
            raise RuntimeError("update_vlm_feature requires vlm_encoder.")
        self.latest_vlm_feature = run_vlm_encoder_forward(self.vlm_encoder, vlm_inputs)
        self.last_vlm_update_step = self.current_step
        log.debug(f"VLM updated at step {self.current_step}")

    @torch.inference_mode()
    def sample_action_chunk_from_features(
        self, f_vision, c_sem, c_sem_mask, frame_delay, q_current, noise=None
    ):
        if self.fm_solver is None:
            raise RuntimeError(
                "sample_action_chunk_from_features() requires fm_solver for action sampling."
            )

        policy_param = next(self.policy_head.parameters())
        policy_device = policy_param.device
        policy_dtype = policy_param.dtype

        f_vision = torch.as_tensor(f_vision, device=policy_device, dtype=policy_dtype)
        c_sem = torch.as_tensor(c_sem, device=policy_device, dtype=policy_dtype)
        c_sem_mask = torch.as_tensor(c_sem_mask, device=policy_device, dtype=torch.bool)
        q_current = torch.as_tensor(q_current, dtype=policy_dtype, device=policy_device)
        frame_delay = torch.as_tensor(frame_delay, dtype=torch.float32, device=policy_device)

        if q_current.ndim == 1:
            q_current = q_current.unsqueeze(0)
        if frame_delay.ndim == 0:
            frame_delay = frame_delay.unsqueeze(0)

        action_trajectory = self.fm_solver.sample(
            policy_head=self.policy_head,
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
            noise=noise,
        )
        delta_action_trajectory = self.action_normalizer.denormalize(action_trajectory)
        if self.action_is_delta:
            action_dim = delta_action_trajectory.shape[-1]
            state_indices = normalize_action_delta_state_indices(
                self.action_delta_state_indices,
                action_dim=action_dim,
            )
            basis = torch.zeros(
                (q_current.shape[0], action_dim), device=q_current.device, dtype=q_current.dtype
            )
            for action_idx, state_idx in enumerate(state_indices):
                if state_idx is None:
                    continue
                if int(state_idx) >= q_current.shape[-1]:
                    raise ValueError(
                        f"Action delta state index {state_idx} for action dim {action_idx} exceeds "
                        f"q_current dim {q_current.shape[-1]}"
                    )
                basis[:, action_idx] = q_current[:, int(state_idx)]
            return delta_action_trajectory + basis[:, None, :]
        return delta_action_trajectory

    @torch.inference_mode()
    def sample_action_chunk(self, pixel_values, q_current):
        """Run the high-frequency vision path and return an absolute action chunk."""
        if self.vision_encoder is None:
            raise RuntimeError("sample_action_chunk() requires vision_encoder.")
        if self.latest_vlm_feature is None:
            raise RuntimeError("VLM cache is not initialized; call update_vlm_feature first.")

        policy_param = next(self.policy_head.parameters())
        policy_device = policy_param.device
        policy_dtype = policy_param.dtype

        f_vision = self.vision_encoder(pixel_values).to(device=policy_device, dtype=policy_dtype)
        c_sem = self.latest_vlm_feature.to(device=policy_device, dtype=policy_dtype)
        frame_delay = torch.full(
            (f_vision.shape[0],),
            self.current_step - self.last_vlm_update_step,
            dtype=torch.float32,
            device=policy_device,
        )
        action_trajectory = self.sample_action_chunk_from_features(
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=torch.ones(
                (f_vision.shape[0], c_sem.shape[1]),
                dtype=torch.bool,
                device=policy_device,
            ),
            frame_delay=frame_delay,
            q_current=q_current,
        )

        self.current_step += 1
        return action_trajectory

    @torch.inference_mode()
    def step(self, pixel_values, q_current):
        action_trajectory = self.sample_action_chunk(pixel_values=pixel_values, q_current=q_current)
        return action_trajectory[0, 0, :].detach().cpu().numpy()

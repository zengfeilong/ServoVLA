from __future__ import annotations

import torch


class FlowMatchingEulerSolver:
    """Simple Euler solver for flow-matching action sampling.

    The training path defines x_t = (1 - t) * x_0 + t * x_1, so t=0 is pure noise and
    t=1 is the clean action chunk. During inference we must therefore start from noise
    and integrate forward in time toward the data manifold.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        num_inference_steps: int = 10,
    ) -> None:
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.num_inference_steps = max(1, int(num_inference_steps))

    @torch.inference_mode()
    def sample(
        self,
        policy_head,
        f_vision: torch.Tensor,
        c_sem: torch.Tensor,
        c_sem_mask: torch.Tensor,
        frame_delay: torch.Tensor,
        q_current: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = f_vision.shape[0]
        device = f_vision.device
        dtype = f_vision.dtype

        if noise is None:
            x_t = torch.randn(
                batch_size,
                self.chunk_size,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
        else:
            x_t = noise.to(device=device, dtype=dtype)

        dt = 1.0 / self.num_inference_steps
        for step_idx in range(self.num_inference_steps):
            t_value = step_idx * dt
            t = torch.full((batch_size,), t_value, device=device, dtype=dtype)
            v_t = policy_head(
                x_t=x_t,
                t=t,
                f_vision=f_vision,
                c_sem=c_sem,
                c_sem_mask=c_sem_mask,
                frame_delay=frame_delay,
                q_current=q_current,
            )
            x_t = x_t + dt * v_t
        return x_t

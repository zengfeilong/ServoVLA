"""Flow Matching Diffusion Transformer (DiT) Policy Head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from servovla.config.compile_padding import ATTENTION_SEQUENCE_MULTIPLE, round_up_to_multiple

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception:  # pragma: no cover - depends on optional CUDA extension availability
    _flash_attn_func = None
    _flash_attn_varlen_func = None


class SinusoidalPosEmb2D(nn.Module):
    def __init__(self, hidden_dim: int, grid_size: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        pos_embed = self._build_2d_sincos_position_embedding()
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _build_2d_sincos_position_embedding(self) -> torch.Tensor:
        grid_h = torch.arange(self.grid_size, dtype=torch.float32)
        grid_w = torch.arange(self.grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0).reshape(2, 1, self.grid_size, self.grid_size)
        emb_h = self._get_1d_sincos_pos_embed_from_grid(self.hidden_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(self.hidden_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1)

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = 1.0 / 10000 ** (
            torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
        )
        out = torch.einsum("m,d->md", pos.reshape(-1), omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Account for a backbone that prepends a [CLS] token.
        seq_len = x.shape[1]
        pos_embed = self.pos_embed.to(device=x.device, dtype=x.dtype)
        if seq_len == pos_embed.shape[0] + 1:
            # Assign a zero positional embedding to the [CLS] token.
            cls_pos = torch.zeros((1, self.hidden_dim), device=x.device, dtype=x.dtype)
            pe = torch.cat([cls_pos, pos_embed], dim=0)
        elif seq_len == pos_embed.shape[0]:
            pe = pos_embed
        else:
            # Fall back to a generic interpolation for unexpected token counts.
            pe = (
                pos_embed[:seq_len]
                if seq_len < pos_embed.shape[0]
                else torch.cat(
                    [
                        pos_embed,
                        torch.zeros(
                            (seq_len - pos_embed.shape[0], self.hidden_dim),
                            device=x.device,
                            dtype=x.dtype,
                        ),
                    ],
                    dim=0,
                )
            )

        return x + pe.unsqueeze(0)


class ModulateAdaLNZero(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.linear(self.silu(c)).chunk(6, dim=-1)


class FlashCompatibleAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.flash_attention_dtype: torch.dtype | None = None

    def set_flash_attention_dtype(self, dtype: torch.dtype | None) -> None:
        if dtype in {None, torch.float32}:
            self.flash_attention_dtype = None
            return
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(f"flash attention dtype must be fp16/bf16/None, got {dtype}")
        self.flash_attention_dtype = dtype

    def _shape_projection(self, tensor: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return projection(tensor).view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _can_use_flash(self, q: torch.Tensor) -> bool:
        return (
            q.is_cuda
            and q.dtype in {torch.float16, torch.bfloat16}
            and _flash_attn_func is not None
            and _flash_attn_varlen_func is not None
        )

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if key_padding_mask is None:
            return _flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)

        valid_k = ~key_padding_mask
        key_lengths = valid_k.sum(dim=1, dtype=torch.int32)
        query_len = int(q.shape[1])
        cu_q = torch.arange(
            0,
            (int(q.shape[0]) + 1) * query_len,
            query_len,
            dtype=torch.int32,
            device=q.device,
        ).contiguous()
        cu_k = torch.cat(
            [
                key_lengths.new_zeros((1,)),
                torch.cumsum(key_lengths, dim=0, dtype=torch.int32),
            ],
            dim=0,
        ).contiguous()
        q_unpad = q.reshape(q.shape[0] * q.shape[1], self.num_heads, self.head_dim)
        k_unpad = k[valid_k]
        v_unpad = v[valid_k]
        out_unpad = _flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_q,
            cu_k,
            int(q.shape[1]),
            int(k.shape[1]),
            dropout_p=dropout_p,
            causal=False,
        )
        return out_unpad.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                (key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1]),
                dtype=q.dtype,
                device=q.device,
            )
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return out.transpose(1, 2)

    @staticmethod
    def _pad_sequence_to_multiple(
        tensor: torch.Tensor, multiple: int = ATTENTION_SEQUENCE_MULTIPLE
    ) -> torch.Tensor:
        target_len = round_up_to_multiple(int(tensor.shape[1]), multiple)
        pad_len = target_len - int(tensor.shape[1])
        if pad_len <= 0:
            return tensor
        return F.pad(tensor, (0, 0, 0, pad_len))

    @staticmethod
    def _pad_mask_to_length(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = int(target_len) - int(mask.shape[1])
        if pad_len <= 0:
            return mask.bool()
        return F.pad(mask.bool(), (0, pad_len), value=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._shape_projection(query, self.q_proj)
        k = self._shape_projection(key, self.k_proj)
        v = self._shape_projection(value, self.v_proj)
        output_dtype = q.dtype

        flash_q, flash_k, flash_v = q, k, v
        if self.flash_attention_dtype is not None:
            flash_q = q.to(dtype=self.flash_attention_dtype)
            flash_k = k.to(dtype=self.flash_attention_dtype)
            flash_v = v.to(dtype=self.flash_attention_dtype)

        if self._can_use_flash(flash_q):
            out = self._flash_attention(flash_q, flash_k, flash_v, key_padding_mask).to(
                dtype=output_dtype
            )
        else:
            out = self._sdpa_attention(q, k, v, key_padding_mask)
        return self.out_proj(out.reshape(query.shape[0], query.shape[1], self.hidden_dim))


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = FlashCompatibleAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = FlashCompatibleAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = ModulateAdaLNZero(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        x_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(x, c)

        x_res = x
        x = self.norm1(x)
        x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        self_key_padding_mask = ~x_mask if x_mask is not None else None
        attn_out = self.attn(x, x, x, key_padding_mask=self_key_padding_mask)
        x = x_res + gate_msa.unsqueeze(1) * attn_out

        x_res = x
        x = self.norm2(x)
        key_padding_mask = ~context_mask if context_mask is not None else None
        cross_out = self.cross_attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
        )
        x = x_res + cross_out

        x_res = x
        x = self.norm3(x)
        x = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x)
        return x_res + gate_mlp.unsqueeze(1) * mlp_out


class FlowMatchingDiT(nn.Module):
    def __init__(
        self,
        action_dim: int = 10,
        state_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        vision_feature_dim: int = 1024,
        semantic_feature_dim: int = 1024,
        dropout: float = 0.0,
        vision_grid_size: int = 16,
        num_cameras: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_cameras = max(1, int(num_cameras))
        self.x_embedder = nn.Linear(action_dim, hidden_dim)
        self.register_buffer(
            "horizon_pos_embed",
            self._get_1d_sincos_pos_embed(1000, hidden_dim),
            persistent=False,
        )

        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Frame-delay embedding network.
        self.delay_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.sem_proj = nn.Sequential(
            nn.Linear(semantic_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.view_embed = nn.Embedding(self.num_cameras, hidden_dim)

        # Use the configured grid size instead of a hard-coded token layout.
        self.vision_pos_embed = SinusoidalPosEmb2D(hidden_dim, grid_size=vision_grid_size)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6), nn.Linear(hidden_dim, action_dim)
        )

        nn.init.constant_(self.final_layer[1].weight, 0)
        nn.init.constant_(self.final_layer[1].bias, 0)

    def _get_1d_sincos_pos_embed(self, max_len: int, embed_dim: int) -> torch.Tensor:
        pos = torch.arange(max_len, dtype=torch.float32)
        omega = 1.0 / 10000 ** (
            torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
        )
        out = torch.einsum("m,d->md", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def _get_timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_vision: torch.Tensor,
        c_sem: torch.Tensor,
        c_sem_mask: torch.Tensor,
        frame_delay: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        B, Horizon, _ = x_t.shape
        x = self.x_embedder(x_t)
        horizon_pos = self.horizon_pos_embed[:Horizon].to(device=x.device, dtype=x.dtype)
        x = x + horizon_pos.unsqueeze(0)
        x_mask = torch.ones((B, Horizon), dtype=torch.bool, device=x.device)
        x = FlashCompatibleAttention._pad_sequence_to_multiple(x)
        x_mask = FlashCompatibleAttention._pad_mask_to_length(x_mask, int(x.shape[1]))

        t_emb = self._get_timestep_embedding(t * 1000, self.hidden_dim).to(
            device=x.device,
            dtype=x.dtype,
        )
        c = self.t_embedder(t_emb)

        # Inject sensor-delay context.
        delay_emb = self._get_timestep_embedding(frame_delay * 100.0, self.hidden_dim).to(
            device=x.device,
            dtype=x.dtype,
        )
        emb_delay = self.delay_embedder(delay_emb)
        c = c + emb_delay

        f_vis = self._encode_visual_tokens(f_vision)
        vis_mask = torch.ones((B, f_vis.shape[1]), dtype=torch.bool, device=x.device)
        c_s = self.sem_proj(c_sem)
        c_sem_mask = c_sem_mask.to(device=x.device, dtype=torch.bool)
        q_state = self.state_proj(q_current).unsqueeze(1)
        context = torch.cat([f_vis, c_s, q_state], dim=1)

        state_mask = torch.ones((B, 1), dtype=torch.bool, device=x.device)
        context_mask = torch.cat([vis_mask, c_sem_mask, state_mask], dim=1)
        context = FlashCompatibleAttention._pad_sequence_to_multiple(context)
        context_mask = FlashCompatibleAttention._pad_mask_to_length(
            context_mask, int(context.shape[1])
        )

        for block in self.blocks:
            x = block(x, c, context, context_mask, x_mask=x_mask)

        return self.final_layer(x[:, :Horizon])

    def _encode_visual_tokens(self, f_vision: torch.Tensor) -> torch.Tensor:
        f_vis = self.vision_proj(f_vision)
        if self.num_cameras <= 1:
            return self.vision_pos_embed(f_vis)

        if f_vis.shape[1] % self.num_cameras != 0:
            raise ValueError(
                f"Visual token sequence length {f_vis.shape[1]} is not divisible by num_cameras={self.num_cameras}"
            )

        batch_size, total_seq_len, _ = f_vis.shape
        seq_per_camera = total_seq_len // self.num_cameras
        f_vis = f_vis.view(batch_size, self.num_cameras, seq_per_camera, self.hidden_dim)

        encoded_per_view = []
        for camera_idx in range(self.num_cameras):
            view_tokens = self.vision_pos_embed(f_vis[:, camera_idx])
            view_tokens = view_tokens + self.view_embed.weight[camera_idx].view(1, 1, -1)
            encoded_per_view.append(view_tokens)
        return torch.cat(encoded_per_view, dim=1)


def build_policy_head_from_cfg(cfg) -> FlowMatchingDiT:
    # Derive the token grid from the configured vision image size.
    image_size = cfg.model.vision_encoder.get("image_size", 256)
    # DINOv3 ViT-16 backbones use a patch size of 16.
    patch_size = 16
    vision_grid_size = image_size // patch_size

    return FlowMatchingDiT(
        action_dim=cfg.model.policy_head.action_dim,
        state_dim=cfg.model.policy_head.state_dim,
        hidden_dim=cfg.model.policy_head.hidden_dim,
        num_layers=cfg.model.policy_head.num_layers,
        num_heads=cfg.model.policy_head.num_heads,
        vision_feature_dim=cfg.model.vision_encoder.feature_dim,
        semantic_feature_dim=cfg.model.vlm_encoder.feature_dim,
        dropout=cfg.model.policy_head.get("dropout", 0.0),
        vision_grid_size=vision_grid_size,
        num_cameras=cfg.model.policy_head.get("num_cameras", 2),
    )

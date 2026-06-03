"""
Concatenate text and image token sequences, and use separate QKV & output projections and MLP weights for text and image streams.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.partitioning import remat

from models.components import (
    TextEncoderAdapterMLP,
    TextEncoderAdapterTransformer,
    ImageConnector,
    DiTConfig,
    DiTFinalLayerNoAdaLN,
)
from models import components as base

Array = jnp.ndarray


@dataclasses.dataclass
class DualStreamDiTConfig(DiTConfig):
    rope_axes_dims: Optional[tuple[int, int, int]] = None
    rope_axes_lens: Optional[tuple[int, int, int]] = None
    rope_theta: float = 10000.0
    use_separate_norms: bool = False


class MultimodalRopeEmbedder(nn.Module):
    axes_dims: tuple[int, ...]
    axes_lens: tuple[int, ...]
    axes_scales: Optional[tuple[float, ...]] = None
    theta: float = 10000.0

    def setup(self) -> None:
        if len(self.axes_dims) != len(self.axes_lens):
            raise ValueError("axes_dims and axes_lens must have the same length")
        if self.axes_scales is None:
            axes_scales = (1.0,) * len(self.axes_dims)
        else:
            axes_scales = self.axes_scales
        if len(axes_scales) != len(self.axes_dims):
            raise ValueError("axes_scales must have the same length as axes_dims")
        freqs = []
        for dim, axis_len, axis_scale in zip(self.axes_dims, self.axes_lens, axes_scales):
            if dim % 2 != 0:
                raise ValueError("Each axis dimension must be even to form complex pairs")
            steps = jnp.arange(0, dim, 2, dtype=jnp.float32)
            base = 1.0 / (self.theta ** (steps / dim))
            positions = jnp.arange(axis_len, dtype=jnp.float32) * axis_scale
            angles = positions[:, None] * base[None, :]
            freq = jnp.exp(1j * angles).astype(jnp.complex64)
            freqs.append(freq)
        object.__setattr__(self, "freq_tables", tuple(freqs))

    def __call__(self, position_ids: Array) -> Array:
        gathered = []
        for axis_idx, table in enumerate(self.freq_tables):
            positions = jnp.clip(position_ids[:, :, axis_idx], 0, table.shape[0] - 1)
            gathered.append(jnp.take(table, positions, axis=0))
        return jnp.concatenate(gathered, axis=-1)


def _default_rope_axes_dims(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")
    time_dim = head_dim // 2
    if time_dim % 2 != 0:
        time_dim -= 1
    remaining = head_dim - time_dim
    if remaining <= 0:
        raise ValueError("Not enough dimensions left for spatial RoPE axes.")
    row_dim = remaining // 2
    col_dim = remaining - row_dim
    if row_dim % 2 != 0:
        row_dim -= 1
        col_dim += 1
    if col_dim % 2 != 0:
        col_dim -= 1
        row_dim += 1
    if min(time_dim, row_dim, col_dim) <= 0:
        raise ValueError("Each RoPE axis must receive at least two dimensions.")
    return time_dim, row_dim, col_dim


class MMDiTAttention(nn.Module):
    hidden_size: int
    num_heads: int
    qk_norm: bool
    use_rmsnorm: bool
    dtype: jnp.dtype = jnp.float32
    separate_qk_norms: bool = False

    def setup(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_image = nn.Dense(3 * self.hidden_size, use_bias=True, dtype=self.dtype, name="qkv_image")
        self.qkv_text = nn.Dense(3 * self.hidden_size, use_bias=True, dtype=self.dtype, name="qkv_text")
        norm_cls = base.RMSNorm if self.use_rmsnorm else base.LayerNorm
        if self.separate_qk_norms:
            self.q_norm_image = norm_cls(self.head_dim, dtype=self.dtype, name="q_norm_image") if self.qk_norm else None
            self.k_norm_image = norm_cls(self.head_dim, dtype=self.dtype, name="k_norm_image") if self.qk_norm else None
            self.q_norm_text = norm_cls(self.head_dim, dtype=self.dtype, name="q_norm_text") if self.qk_norm else None
            self.k_norm_text = norm_cls(self.head_dim, dtype=self.dtype, name="k_norm_text") if self.qk_norm else None
        else:
            self.q_norm = norm_cls(self.head_dim, dtype=self.dtype, name="q_norm") if self.qk_norm else None
            self.k_norm = norm_cls(self.head_dim, dtype=self.dtype, name="k_norm") if self.qk_norm else None
        self.out_image = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj_image")
        self.out_text = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj_text")

    def __call__(
        self,
        image_tokens: Array,
        text_tokens: Array,
        *,
        image_freqs_cis: Optional[Array],
        text_freqs_cis: Optional[Array],
        deterministic: bool,
        text_mask: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        del deterministic
        b, image_len, _ = image_tokens.shape
        _, text_len, _ = text_tokens.shape
        qkv_image = self.qkv_image(image_tokens)
        qkv_text = self.qkv_text(text_tokens)
        qkv_image = qkv_image.reshape(b, image_len, 3, self.num_heads, self.head_dim)
        qkv_text = qkv_text.reshape(b, text_len, 3, self.num_heads, self.head_dim)
        q_image, k_image, v_image = jnp.split(qkv_image, 3, axis=2)
        q_text, k_text, v_text = jnp.split(qkv_text, 3, axis=2)
        q_image = jnp.squeeze(q_image, axis=2).transpose(0, 2, 1, 3)
        k_image = jnp.squeeze(k_image, axis=2).transpose(0, 2, 1, 3)
        v_image = jnp.squeeze(v_image, axis=2).transpose(0, 2, 1, 3)
        q_text = jnp.squeeze(q_text, axis=2).transpose(0, 2, 1, 3)
        k_text = jnp.squeeze(k_text, axis=2).transpose(0, 2, 1, 3)
        v_text = jnp.squeeze(v_text, axis=2).transpose(0, 2, 1, 3)
        if self.separate_qk_norms:
            if self.q_norm_image is not None:
                q_image = self.q_norm_image(q_image)
                k_image = self.k_norm_image(k_image)
                q_text = self.q_norm_text(q_text)
                k_text = self.k_norm_text(k_text)
        else:
            if self.q_norm is not None:
                q_image = self.q_norm(q_image)
                k_image = self.k_norm(k_image)
                q_text = self.q_norm(q_text)
                k_text = self.k_norm(k_text)
        if image_freqs_cis is not None:
            q_image = self._apply_multimodal_rope(q_image, image_freqs_cis)
            k_image = self._apply_multimodal_rope(k_image, image_freqs_cis)
        if text_freqs_cis is not None:
            q_text = self._apply_multimodal_rope(q_text, text_freqs_cis)
            k_text = self._apply_multimodal_rope(k_text, text_freqs_cis)
        q = jnp.concatenate([q_image, q_text], axis=2)
        k = jnp.concatenate([k_image, k_text], axis=2)
        v = jnp.concatenate([v_image, v_text], axis=2)
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q * self.scale, k).astype(jnp.float32)
        key_mask = None
        if text_mask is not None:
            image_mask = jnp.ones((b, image_len), dtype=bool)
            key_mask = jnp.concatenate([image_mask, text_mask.astype(bool)], axis=1)
        if key_mask is not None:
            expanded = key_mask[:, None, None, :]
            neg_inf = jnp.finfo(attn_logits.dtype).min
            attn_logits = jnp.where(expanded, attn_logits, neg_inf)
        attn = nn.softmax(attn_logits, axis=-1).astype(q.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, image_len + text_len, self.hidden_size)
        if key_mask is not None:
            out = out * key_mask[:, :, None].astype(out.dtype)
        image_out = out[:, :image_len, :]
        text_out = out[:, image_len:, :]
        return self.out_image(image_out), self.out_text(text_out)

    @staticmethod
    def _apply_multimodal_rope(x: Array, freqs_cis: Array) -> Array:
        b, h, seq, dim = x.shape
        if dim % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE.")
        x_pair = x.reshape(b, h, seq, dim // 2, 2)
        x_complex = jax.lax.complex(x_pair[..., 0], x_pair[..., 1])
        freqs = freqs_cis[:, None, :, :]
        rotated = x_complex * freqs
        rotated = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-1)
        return rotated.reshape(b, h, seq, dim)


class DualStreamDiTBlock(nn.Module):
    """LightningDiT transformer block with joint attention over two streams."""

    hidden_size: int
    num_heads: int
    mlp_ratio: float
    use_qknorm: bool
    use_swiglu: bool
    use_rmsnorm: bool
    wo_shift: bool
    use_adaln: bool = True
    use_sandwich_norm: bool = False
    use_separate_norms: bool = False
    use_skip: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        image_tokens: Array,
        text_tokens: Array,
        cond: Array,
        image_freqs_cis: Optional[Array],
        text_freqs_cis: Optional[Array],
        deterministic: bool,
        text_mask: Optional[Array] = None,
        skip: Optional[tuple[Array, Array]] = None,
    ) -> tuple[Array, Array]:
        if self.use_skip:
            if skip is None:
                raise ValueError("Skip connection is required when use_skip is True.")
            skip_image, skip_text = skip
            skip_linear_image = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=self.dtype,
                name="skip_linear_image",
            )
            skip_linear_text = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=self.dtype,
                name="skip_linear_text",
            )
            image_tokens = skip_linear_image(jnp.concatenate([image_tokens, skip_image], axis=-1))
            text_tokens = skip_linear_text(jnp.concatenate([text_tokens, skip_text], axis=-1))
        norm_cls = base.RMSNorm if self.use_rmsnorm else base.LayerNorm
        if self.use_separate_norms:
            norm1_image = norm_cls(self.hidden_size, dtype=self.dtype, name="norm1_image")
            norm1_text = norm_cls(self.hidden_size, dtype=self.dtype, name="norm1_text")
            norm2_image = norm_cls(self.hidden_size, dtype=self.dtype, name="norm2_image")
            norm2_text = norm_cls(self.hidden_size, dtype=self.dtype, name="norm2_text")
        else:
            norm1 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm1")
            norm2 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm2")
            norm1_image = norm1_text = norm1
            norm2_image = norm2_text = norm2
        norm3 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm3") if self.use_sandwich_norm else None
        norm4 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm4") if self.use_sandwich_norm else None
        attn = MMDiTAttention(
            self.hidden_size,
            self.num_heads,
            self.use_qknorm,
            self.use_rmsnorm,
            separate_qk_norms=self.use_separate_norms,
            dtype=self.dtype,
            name="attn",
        )
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        if self.use_swiglu:
            mlp_image = base.SwiGLUFFN(self.hidden_size, int(2 / 3 * mlp_hidden), dtype=self.dtype, name="mlp_image")
            mlp_text = base.SwiGLUFFN(self.hidden_size, int(2 / 3 * mlp_hidden), dtype=self.dtype, name="mlp_text")
        else:
            mlp_image = base.MlpBlock(self.hidden_size, mlp_hidden, dtype=self.dtype, name="mlp_image")
            mlp_text = base.MlpBlock(self.hidden_size, mlp_hidden, dtype=self.dtype, name="mlp_text")
        if not self.use_adaln:
            image_attn, text_attn = attn(
                norm1_image(image_tokens),
                norm1_text(text_tokens),
                image_freqs_cis=image_freqs_cis,
                text_freqs_cis=text_freqs_cis,
                deterministic=deterministic,
                text_mask=text_mask,
            )
            image_tokens = image_tokens + (norm3(image_attn) if norm3 is not None else image_attn)
            text_tokens = text_tokens + (norm3(text_attn) if norm3 is not None else text_attn)
            image_mlp_out = mlp_image(norm2_image(image_tokens))
            text_mlp_out = mlp_text(norm2_text(text_tokens))
            image_tokens = image_tokens + (norm4(image_mlp_out) if norm4 is not None else image_mlp_out)
            text_tokens = text_tokens + (norm4(text_mlp_out) if norm4 is not None else text_mlp_out)
            if text_mask is not None:
                text_tokens = text_tokens * text_mask[:, :, None].astype(text_tokens.dtype)
            return image_tokens, text_tokens
        mod_dim = 4 * self.hidden_size if self.wo_shift else 6 * self.hidden_size
        ada = base.AdaLNModulation(self.hidden_size, mod_dim, dtype=self.dtype, name="adaLN_modulation")
        params = ada(cond)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(params, 4, axis=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(params, 6, axis=-1)
        image_attn, text_attn = attn(
            base.modulate(norm1_image(image_tokens), shift_msa, scale_msa),
            base.modulate(norm1_text(text_tokens), shift_msa, scale_msa),
            image_freqs_cis=image_freqs_cis,
            text_freqs_cis=text_freqs_cis,
            deterministic=deterministic,
            text_mask=text_mask,
        )
        if norm3 is not None:
            image_attn = norm3(image_attn)
            text_attn = norm3(text_attn)
        image_tokens = image_tokens + gate_msa[:, None, :] * image_attn
        text_tokens = text_tokens + gate_msa[:, None, :] * text_attn
        image_mlp_out = mlp_image(
            base.modulate(norm2_image(image_tokens), shift_mlp, scale_mlp)
        )
        text_mlp_out = mlp_text(
            base.modulate(norm2_text(text_tokens), shift_mlp, scale_mlp)
        )
        if norm4 is not None:
            image_mlp_out = norm4(image_mlp_out)
            text_mlp_out = norm4(text_mlp_out)
        image_tokens = image_tokens + gate_mlp[:, None, :] * image_mlp_out
        text_tokens = text_tokens + gate_mlp[:, None, :] * text_mlp_out
        if text_mask is not None:
            text_tokens = text_tokens * text_mask[:, :, None].astype(text_tokens.dtype)
        return image_tokens, text_tokens


class DualStreamDiT(nn.Module):
    config: DualStreamDiTConfig
    text_num_tokens: Optional[int] = None
    text_embed_dim: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        cfg = self.config
        if self.text_num_tokens is not None or self.text_embed_dim is not None:
            cfg = dataclasses.replace(
                cfg,
                text_num_tokens=self.text_num_tokens or cfg.text_num_tokens,
                text_embed_dim=self.text_embed_dim or cfg.text_embed_dim,
            )
            object.__setattr__(self, "config", cfg)
        self.input_size = cfg.input_size
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.in_channels
        self.x_embedder = base.PatchEmbed(cfg.patch_size, cfg.hidden_size, cfg.in_channels, dtype=self.dtype, name="x_embedder")
        if cfg.use_image_connector:
            self.image_connector = ImageConnector(
                cfg.hidden_size,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.use_qknorm,
                cfg.use_swiglu,
                cfg.use_rmsnorm,
                dtype=self.dtype,
                name="image_connector",
            )
        else:
            self.image_connector = None
        hw = cfg.input_size // cfg.patch_size
        grid = hw * hw
        self.hw = hw
        row_ids = jnp.repeat(jnp.arange(hw, dtype=jnp.int32), hw)
        col_ids = jnp.tile(jnp.arange(hw, dtype=jnp.int32), hw)
        self.image_row_ids = row_ids
        self.image_col_ids = col_ids
        """
        Keep exact 256 behavior for pretraining, but rescale index grid when
        fine-tuning at a different image resolution (e.g., 512 from 256 base).
        """
        if int(cfg.image_resolution) == 256:
            pos = base._get_pos_embed(cfg.hidden_size, hw)
        else:
            pos = base._get_interpolated_pos_embed(
                cfg.hidden_size,
                hw,
                image_resolution=int(cfg.image_resolution),
                base_image_resolution=256,
            )
        def _pos_init(rng, shape, dtype=None):
            del rng
            arr = pos.reshape(shape)
            if dtype is None:
                dtype = self.dtype
            return arr.astype(dtype)

        if cfg.position_embedding not in base.POSITION_EMBEDDING_OPTIONS:
            raise ValueError(f"Unknown position_embedding: {cfg.position_embedding}")
        if cfg.position_embedding in ("sinusoidal_and_rope", "sinusoidal_only"):
            self.pos_embed = self.param("pos_embed", _pos_init, (1, grid, cfg.hidden_size))
        else:
            self.pos_embed = None
        self.t_embedder = base.TimestepEmbedder(cfg.hidden_size, dtype=self.dtype, name="t_embedder")
        if cfg.text_encoder_adapter_type == "mlp":
            self.text_encoder_adapter = TextEncoderAdapterMLP(
                cfg.text_embed_dim,
                cfg.hidden_size,
                cfg.drop_text_prob,
                token_len=cfg.text_num_tokens,
                name="text_encoder_adapter",
            )
        elif cfg.text_encoder_adapter_type == "transformer":
            self.text_encoder_adapter = TextEncoderAdapterTransformer(
                cfg.text_embed_dim,
                cfg.hidden_size,
                cfg.drop_text_prob,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.use_qknorm,
                cfg.use_swiglu,
                cfg.use_rmsnorm,
                token_len=cfg.text_num_tokens,
                dtype=self.dtype,
                num_blocks=cfg.text_encoder_adapter_num_blocks,
                name="text_encoder_adapter",
            )
        else:
            raise ValueError(f"Unknown text_encoder_adapter_type: {cfg.text_encoder_adapter_type}")
        if cfg.position_embedding in ("sinusoidal_and_rope", "rope_only"):
            head_dim = cfg.hidden_size // cfg.num_heads
            axes_dims = cfg.rope_axes_dims or _default_rope_axes_dims(head_dim)
            if sum(axes_dims) != head_dim:
                raise ValueError(
                    f"Sum of rope_axes_dims ({axes_dims}) must equal head_dim={head_dim}"
                )
            text_rope_len = (
                sum(cfg.text_num_tokens)
                if isinstance(cfg.text_num_tokens, (list, tuple))
                else cfg.text_num_tokens
            )
            if cfg.repeat_text_emb:
                text_rope_len *= 2
            axes_lens = cfg.rope_axes_lens or (
                text_rope_len + 1,
                hw,
                hw,
            )
            if len(axes_lens) != len(axes_dims):
                raise ValueError("rope_axes_lens must have the same length as rope_axes_dims")
            image_scale = 256.0 / cfg.image_resolution
            axes_scales = (1.0,) + (image_scale,) * (len(axes_dims) - 1)
            self.rope_embedder = MultimodalRopeEmbedder(
                axes_dims=axes_dims,
                axes_lens=axes_lens,
                axes_scales=axes_scales,
                theta=cfg.rope_theta,
                name="rope_embedder",
            )
        else:
            self.rope_embedder = None
        block_cls = DualStreamDiTBlock
        if cfg.use_grad_ckpt:
            block_cls = remat(
                DualStreamDiTBlock,
                prevent_cse=True,
            )
        if cfg.use_long_skip:
            num_in_blocks = cfg.depth // 2
            self.in_blocks = [
                block_cls(
                    cfg.hidden_size,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    cfg.use_qknorm,
                    cfg.use_swiglu,
                    cfg.use_rmsnorm,
                    cfg.wo_shift,
                    use_adaln=cfg.use_adaln,
                    use_sandwich_norm=cfg.use_sandwich_norm,
                    use_separate_norms=cfg.use_separate_norms,
                    dtype=self.dtype,
                    name=f"blocks_{i}",
                )
                for i in range(num_in_blocks)
            ]
            self.mid_block = block_cls(
                cfg.hidden_size,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.use_qknorm,
                cfg.use_swiglu,
                cfg.use_rmsnorm,
                cfg.wo_shift,
                use_adaln=cfg.use_adaln,
                use_sandwich_norm=cfg.use_sandwich_norm,
                use_separate_norms=cfg.use_separate_norms,
                dtype=self.dtype,
                name=f"blocks_{num_in_blocks}",
            )
            self.out_blocks = [
                block_cls(
                    cfg.hidden_size,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    cfg.use_qknorm,
                    cfg.use_swiglu,
                    cfg.use_rmsnorm,
                    cfg.wo_shift,
                    use_adaln=cfg.use_adaln,
                    use_sandwich_norm=cfg.use_sandwich_norm,
                    use_separate_norms=cfg.use_separate_norms,
                    use_skip=True,
                    dtype=self.dtype,
                    name=f"blocks_{num_in_blocks + 1 + i}",
                )
                for i in range(num_in_blocks)
            ]
        else:
            self.blocks = [
                block_cls(
                    cfg.hidden_size,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    cfg.use_qknorm,
                    cfg.use_swiglu,
                    cfg.use_rmsnorm,
                    cfg.wo_shift,
                    use_adaln=cfg.use_adaln,
                    use_sandwich_norm=cfg.use_sandwich_norm,
                    use_separate_norms=cfg.use_separate_norms,
                    dtype=self.dtype,
                    name=f"blocks_{i}",
                )
                for i in range(cfg.depth)
            ]
        final_layer_cls = base.DiTFinalLayer if cfg.use_adaln else DiTFinalLayerNoAdaLN
        self.final_layer = final_layer_cls(
            cfg.hidden_size,
            cfg.patch_size,
            self.out_channels,
            cfg.use_rmsnorm,
            dtype=self.dtype,
            name="final_layer",
        )

    def __call__(
        self,
        x: Array,
        t: Array,
        caption: Array,
        *,
        mask: Optional[Array] = None,
        train: bool = False,
    ) -> Array:
        cfg = self.config
        x = jnp.transpose(x, (0, 2, 3, 1))
        tokens = self.x_embedder(x)
        if self.image_connector is not None:
            tokens = self.image_connector(tokens, train=train)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        t_emb = self.t_embedder(t)
        if isinstance(caption, (list, tuple)):
            if isinstance(cfg.text_embed_dim, (list, tuple)):
                assert len(caption) == len(cfg.text_embed_dim)
                for cap, dim in zip(caption, cfg.text_embed_dim):
                    if cap.shape[-1] != dim:
                        raise ValueError(
                            f"Caption embedding dim {cap.shape[-1]} does not match config.text_embed_dim={dim}"
                        )
        else:
            if caption.shape[-1] != cfg.text_embed_dim:
                raise ValueError(
                    f"Caption embedding dim {caption.shape[-1]} does not match config.text_embed_dim={cfg.text_embed_dim}"
                )
        caption_emb = self.text_encoder_adapter(caption, train=train)
        if isinstance(mask, (list, tuple)):
            mask = jnp.concatenate(list(mask), axis=1)
        if cfg.repeat_text_emb:
            caption_emb = jnp.concatenate([caption_emb, caption_emb], axis=1)
            if mask is not None:
                mask = jnp.concatenate([mask, mask], axis=1)
        text_mask_bool: Optional[Array]
        seq_text = caption_emb.shape[1]
        if mask is not None:
            if mask.shape[-1] != caption_emb.shape[1]:
                raise ValueError(
                    f"Mask length {mask.shape[-1]} does not match number of text tokens {caption_emb.shape[1]}"
                )
            text_mask_bool = mask.astype(bool)
            weights = text_mask_bool.astype(jnp.float32)
            denom = jnp.clip(jnp.sum(weights, axis=1, keepdims=True), a_min=1.0)
            pooled = jnp.einsum("bth,bt->bh", caption_emb, weights) / denom
        else:
            pooled = jnp.mean(caption_emb, axis=1)
            text_mask_bool = None
        cond = t_emb + pooled
        num_image_tokens = tokens.shape[1]
        if self.rope_embedder is not None:
            if text_mask_bool is None:
                text_mask_for_pos = jnp.ones((caption_emb.shape[0], seq_text), dtype=bool)
            else:
                text_mask_for_pos = text_mask_bool
            text_lengths = jnp.sum(text_mask_for_pos.astype(jnp.int32), axis=1)
            position_ids = self._build_position_ids(text_mask_for_pos, text_lengths, num_image_tokens)
            freqs_cis = self.rope_embedder(position_ids)
            text_freqs_cis = freqs_cis[:, :seq_text, :]
            image_freqs_cis = freqs_cis[:, seq_text:seq_text + num_image_tokens, :]
        else:
            text_freqs_cis = None
            image_freqs_cis = None
        image_tokens = tokens
        text_tokens = caption_emb
        if cfg.use_long_skip:
            skips = []
            for block in self.in_blocks:
                image_tokens, text_tokens = block(
                    image_tokens,
                    text_tokens,
                    cond,
                    image_freqs_cis,
                    text_freqs_cis,
                    not train,
                    text_mask_bool,
                    None,
                )
                skips.append((image_tokens, text_tokens))
            image_tokens, text_tokens = self.mid_block(
                image_tokens,
                text_tokens,
                cond,
                image_freqs_cis,
                text_freqs_cis,
                not train,
                text_mask_bool,
                None,
            )
            for block in self.out_blocks:
                image_tokens, text_tokens = block(
                    image_tokens,
                    text_tokens,
                    cond,
                    image_freqs_cis,
                    text_freqs_cis,
                    not train,
                    text_mask_bool,
                    skips.pop(),
                )
        else:
            for block in self.blocks:
                image_tokens, text_tokens = block(
                    image_tokens,
                    text_tokens,
                    cond,
                    image_freqs_cis,
                    text_freqs_cis,
                    not train,
                    text_mask_bool,
                    None,
                )
        tokens = image_tokens
        tokens = self.final_layer(tokens, cond)
        b = x.shape[0]
        h = w = cfg.input_size // cfg.patch_size
        tokens = tokens.reshape(b, h, w, cfg.patch_size, cfg.patch_size, self.out_channels)
        tokens = jnp.transpose(tokens, (0, 1, 3, 2, 4, 5))
        image = tokens.reshape(b, h * cfg.patch_size, w * cfg.patch_size, self.out_channels)
        image = jnp.transpose(image, (0, 3, 1, 2))
        return image

    def _build_position_ids(
        self,
        text_mask: Array,
        text_lengths: Array,
        num_image_tokens: int,
    ) -> Array:
        bsz, text_len = text_mask.shape
        caption_positions = jnp.broadcast_to(jnp.arange(text_len, dtype=jnp.int32), (bsz, text_len))
        caption_positions = jnp.where(text_mask, caption_positions, 0)
        zeros = jnp.zeros_like(caption_positions)
        caption_ids = jnp.stack((caption_positions, zeros, zeros), axis=-1)
        row_template = self.image_row_ids[:num_image_tokens]
        col_template = self.image_col_ids[:num_image_tokens]
        row_ids = jnp.broadcast_to(row_template[None, :], (bsz, num_image_tokens))
        col_ids = jnp.broadcast_to(col_template[None, :], (bsz, num_image_tokens))
        image_time = jnp.broadcast_to(text_lengths[:, None], (bsz, num_image_tokens))
        image_ids = jnp.stack((image_time, row_ids, col_ids), axis=-1)
        return jnp.concatenate([caption_ids, image_ids], axis=1).astype(jnp.int32)


DualStreamDiT_models = {
    "DiT-XL": dict(depth=29, hidden_size=1152, num_heads=16, mlp_ratio=4.0),
    "DiT-XL_1296": dict(depth=29, hidden_size=1296, num_heads=18, mlp_ratio=4.0),
    "DiT-XL_1440": dict(depth=29, hidden_size=1440, num_heads=20, mlp_ratio=4.0),
    "DiT-XL_1584": dict(depth=29, hidden_size=1584, num_heads=22, mlp_ratio=4.0),
    "DiT-XL_1728": dict(depth=29, hidden_size=1728, num_heads=24, mlp_ratio=4.0),
    "DiT-XL_1872": dict(depth=29, hidden_size=1872, num_heads=26, mlp_ratio=4.0),
    "DiT-XL_2016": dict(depth=29, hidden_size=2016, num_heads=28, mlp_ratio=4.0),
}

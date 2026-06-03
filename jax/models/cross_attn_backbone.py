"""
Inject text conditioning through cross-attention layers between self-attention and MLP layers.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

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
class CrossAttnDiTConfig(DiTConfig):
    use_pooled_text: bool = True
    use_timestep: bool = True


class CrossAttention(nn.Module):
    hidden_size: int
    num_heads: int
    qk_norm: bool
    use_rmsnorm: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_dim = self.hidden_size // self.num_heads
        norm_cls = base.RMSNorm if self.use_rmsnorm else base.LayerNorm
        self.q_proj = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="q")
        self.k_proj = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="k")
        self.v_proj = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="v")
        self.q_norm = norm_cls(self.head_dim, dtype=self.dtype, name="q_norm") if self.qk_norm else None
        self.k_norm = norm_cls(self.head_dim, dtype=self.dtype, name="k_norm") if self.qk_norm else None
        self.out = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj")
        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        x: Array,
        context: Array,
        *,
        deterministic: bool,
        mask: Optional[Array] = None,
    ) -> Array:
        del deterministic
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        b, q_len, _ = q.shape
        k_len = k.shape[1]
        q = q.reshape(b, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, k_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, k_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q * self.scale, k).astype(jnp.float32)
        if mask is not None:
            expanded = mask[:, None, None, :].astype(bool)
            neg_inf = jnp.finfo(attn_logits.dtype).min
            attn_logits = jnp.where(expanded, attn_logits, neg_inf)
        attn = nn.softmax(attn_logits, axis=-1).astype(q.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, q_len, self.hidden_size)
        return self.out(out)


class CrossAttnDiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float
    use_qknorm: bool
    use_swiglu: bool
    use_rmsnorm: bool
    wo_shift: bool
    use_adaln: bool = True
    use_sandwich_norm: bool = False
    use_skip: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        cond: Array,
        context: Array,
        rope: Optional[base.VisionRotaryEmbeddingFast],
        deterministic: bool,
        mask: Optional[Array] = None,
        skip: Optional[Array] = None,
    ) -> Array:
        if self.use_skip:
            if skip is None:
                raise ValueError("Skip connection is required when use_skip is True.")
            skip_linear = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=self.dtype,
                name="skip_linear",
            )
            x = skip_linear(jnp.concatenate([x, skip], axis=-1))
        norm_cls = base.RMSNorm if self.use_rmsnorm else base.LayerNorm
        norm1 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm1")
        norm2 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm2")
        norm3 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm3") if self.use_sandwich_norm else None
        norm4 = norm_cls(self.hidden_size, dtype=self.dtype, name="norm4") if self.use_sandwich_norm else None
        attn = base.Attention(
            self.hidden_size,
            self.num_heads,
            self.use_qknorm,
            self.use_rmsnorm,
            dtype=self.dtype,
            name="attn",
        )
        cross_attn = CrossAttention(
            self.hidden_size,
            self.num_heads,
            self.use_qknorm,
            self.use_rmsnorm,
            dtype=self.dtype,
            name="cross_attn",
        )
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        if self.use_swiglu:
            mlp = base.SwiGLUFFN(self.hidden_size, int(2 / 3 * mlp_hidden), dtype=self.dtype, name="mlp")
        else:
            mlp = base.MlpBlock(self.hidden_size, mlp_hidden, dtype=self.dtype, name="mlp")
        if not self.use_adaln:
            x = x + (norm3(attn(norm1(x), rope, deterministic)) if norm3 is not None else attn(norm1(x), rope, deterministic))
            x = x + cross_attn(x, context, deterministic=deterministic, mask=mask)
            mlp_out = mlp(norm2(x))
            x = x + (norm4(mlp_out) if norm4 is not None else mlp_out)
            return x
        mod_dim = 4 * self.hidden_size if self.wo_shift else 6 * self.hidden_size
        ada = base.AdaLNModulation(self.hidden_size, mod_dim, dtype=self.dtype, name="adaLN_modulation")
        params = ada(cond)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(params, 4, axis=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(params, 6, axis=-1)
        attn_out = attn(
            base.modulate(norm1(x), shift_msa, scale_msa),
            rope,
            deterministic,
        )
        if norm3 is not None:
            attn_out = norm3(attn_out)
        x = x + gate_msa[:, None, :] * attn_out
        x = x + cross_attn(x, context, deterministic=deterministic, mask=mask)
        mlp_out = mlp(base.modulate(norm2(x), shift_mlp, scale_mlp))
        if norm4 is not None:
            mlp_out = norm4(mlp_out)
        x = x + gate_mlp[:, None, :] * mlp_out
        return x


class CrossAttnDiT(nn.Module):
    config: CrossAttnDiTConfig
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
        self.t_embedder = (
            base.TimestepEmbedder(cfg.hidden_size, dtype=self.dtype, name="t_embedder")
            if cfg.use_timestep
            else None
        )
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
            half_head_dim = cfg.hidden_size // cfg.num_heads // 2
            hw_seq_len = cfg.input_size // cfg.patch_size
            self.feat_rope = base.VisionRotaryEmbeddingFast(
                half_head_dim,
                pt_seq_len=hw_seq_len,
                image_resolution=cfg.image_resolution,
                dtype=self.dtype,
                name="feat_rope",
            )
        else:
            self.feat_rope = None
        block_cls = CrossAttnDiTBlock
        if cfg.use_grad_ckpt:
            block_cls = remat(
                CrossAttnDiTBlock,
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
        t_emb = self.t_embedder(t) if self.t_embedder is not None else None
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
        if mask is not None:
            weights = mask.astype(jnp.float32)
            denom = jnp.clip(jnp.sum(weights, axis=1, keepdims=True), a_min=1.0)
            pooled = jnp.einsum("bth,bt->bh", caption_emb, weights) / denom
        else:
            pooled = jnp.mean(caption_emb, axis=1)
        if cfg.use_timestep and cfg.use_pooled_text:
            cond = t_emb + pooled
        elif cfg.use_timestep:
            cond = t_emb
        elif cfg.use_pooled_text:
            cond = pooled
        else:
            raise ValueError("At least one of use_timestep or use_pooled_text must be enabled.")
        rope = self.feat_rope
        if cfg.use_long_skip:
            skips = []
            for block in self.in_blocks:
                tokens = block(
                    tokens,
                    cond,
                    caption_emb,
                    rope,
                    not train,
                    mask,
                    None,
                )
                skips.append(tokens)
            tokens = self.mid_block(
                tokens,
                cond,
                caption_emb,
                rope,
                not train,
                mask,
                None,
            )
            for block in self.out_blocks:
                tokens = block(
                    tokens,
                    cond,
                    caption_emb,
                    rope,
                    not train,
                    mask,
                    skips.pop(),
                )
        else:
            for block in self.blocks:
                tokens = block(
                    tokens,
                    cond,
                    caption_emb,
                    rope,
                    not train,
                    mask,
                    None,
                )
        tokens = self.final_layer(tokens, cond)
        b = x.shape[0]
        h = w = cfg.input_size // cfg.patch_size
        tokens = tokens.reshape(b, h, w, cfg.patch_size, cfg.patch_size, self.out_channels)
        tokens = jnp.transpose(tokens, (0, 1, 3, 2, 4, 5))
        image = tokens.reshape(b, h * cfg.patch_size, w * cfg.patch_size, self.out_channels)
        image = jnp.transpose(image, (0, 3, 1, 2))
        return image


CrossAttnDiT_models = {
    "DiT-XL": dict(depth=29, hidden_size=1152, num_heads=16, mlp_ratio=4.0),
    "DiT-XL_1296": dict(depth=29, hidden_size=1296, num_heads=18, mlp_ratio=4.0),
    "DiT-XL_1440": dict(depth=29, hidden_size=1440, num_heads=20, mlp_ratio=4.0),
    "DiT-XL_1584": dict(depth=29, hidden_size=1584, num_heads=22, mlp_ratio=4.0),
    "DiT-XL_1728": dict(depth=29, hidden_size=1728, num_heads=24, mlp_ratio=4.0),
    "DiT-XL_1872": dict(depth=29, hidden_size=1872, num_heads=26, mlp_ratio=4.0),
    "DiT-XL_2016": dict(depth=29, hidden_size=2016, num_heads=28, mlp_ratio=4.0),
}

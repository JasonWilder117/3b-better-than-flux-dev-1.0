"""
Shared building blocks used by different backbone families.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

Array = jnp.ndarray
base = sys.modules[__name__]
POSITION_EMBEDDING_OPTIONS = frozenset(("sinusoidal_and_rope", "sinusoidal_only", "rope_only"))


@dataclass
class DiTConfig:
    input_size: int = 16
    patch_size: int = 1
    in_channels: int = 32
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    use_qknorm: bool = False
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    wo_shift: bool = False
    image_resolution: int = 256
    text_embed_dim: int = 1024
    text_num_tokens: int = 77
    drop_text_prob: float = 0.1
    use_grad_ckpt: bool = False
    use_long_skip: bool = True
    text_encoder_adapter_type: str = "mlp"
    text_encoder_adapter_num_blocks: int = 1
    use_image_connector: bool = False
    use_adaln: bool = True
    repeat_text_emb: bool = False
    position_embedding: str = "sinusoidal_and_rope"
    use_sandwich_norm: bool = False


class ImageConnector(nn.Module):
    """Small transformer connector applied to image tokens before positional embeddings."""

    hidden_size: int
    num_heads: int
    mlp_ratio: float
    use_qknorm: bool
    use_swiglu: bool
    use_rmsnorm: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, tokens: Array, train: bool) -> Array:
        norm_cls = base.RMSNorm if self.use_rmsnorm else base.LayerNorm
        norm1 = norm_cls(self.hidden_size, dtype=self.dtype, name="connector_norm1")
        norm2 = norm_cls(self.hidden_size, dtype=self.dtype, name="connector_norm2")
        attn = base.Attention(
            self.hidden_size,
            self.num_heads,
            self.use_qknorm,
            self.use_rmsnorm,
            dtype=self.dtype,
            name="connector_attn",
        )
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        if self.use_swiglu:
            mlp = base.SwiGLUFFN(
                self.hidden_size,
                int(2 / 3 * mlp_hidden),
                dtype=self.dtype,
                name="connector_mlp",
            )
        else:
            mlp = base.MlpBlock(
                self.hidden_size,
                mlp_hidden,
                dtype=self.dtype,
                name="connector_mlp",
            )
        tokens = tokens + attn(norm1(tokens), None, not train)
        return tokens + mlp(norm2(tokens))


class TextEncoderAdapterMLP(nn.Module):
    """
    Pass captions through an MLP. Also handles whole-caption level caption dropout for classifier-free guidance.
    """
    in_channels: int
    hidden_size: int
    drop_text_prob: float
    token_len: int = 77

    @nn.compact
    def __call__(self, caption, train):
        inits = dict(kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)

        def _apply_mlp(x, suffix):
            x = nn.Dense(self.hidden_size, **inits, name=f"mlp_dense_0{suffix}")(x)
            x = nn.gelu(x, approximate=False)
            x = nn.Dense(self.hidden_size, **inits, name=f"mlp_dense_1{suffix}")(x)
            return x

        if isinstance(caption, (list, tuple)):
            captions = list(caption)
            in_channels = list(self.in_channels) if isinstance(self.in_channels, (list, tuple)) else [cap.shape[-1] for cap in captions]
            token_lens = list(self.token_len) if isinstance(self.token_len, (list, tuple)) else [self.token_len] * len(captions)
            assert (len(in_channels) == len(captions)) and (len(token_lens) == len(captions))
            per_encoder_probs = self.drop_text_prob if isinstance(self.drop_text_prob, (list, tuple)) else None
            drop_ids_list = None
            if train:
                if per_encoder_probs is not None:
                    rng = self.make_rng("drop_text")
                    rngs = jax.random.split(rng, len(captions))
                    drop_ids_list = [
                        jax.random.bernoulli(rngs[i], per_encoder_probs[i], (captions[i].shape[0],))
                        if per_encoder_probs[i] > 0
                        else None
                        for i in range(len(captions))
                    ]
                elif self.drop_text_prob > 0:
                    rng = self.make_rng("drop_text")
                    drop_ids = jax.random.bernoulli(rng, self.drop_text_prob, (captions[0].shape[0],))
                    drop_ids_list = [drop_ids] * len(captions)

            outputs = []
            for i, cap in enumerate(captions):
                learnable_null_caption = self.param(
                    f"learnable_null_caption_{i}",
                    nn.initializers.normal(stddev=in_channels[i] ** -0.5),
                    (1, token_lens[i], in_channels[i]),
                )
                seq_len = cap.shape[1]
                if seq_len < token_lens[i]:
                    learnable_null_caption = learnable_null_caption[:, :seq_len, :]
                drop_ids = drop_ids_list[i] if drop_ids_list is not None else None
                if drop_ids is not None:
                    shape = (cap.shape[0],) + (1,) * (cap.ndim - 1)
                    cap = jnp.where(jnp.reshape(drop_ids, shape) == 1, learnable_null_caption, cap)
                outputs.append(_apply_mlp(cap, f"_{i}"))

            return jnp.concatenate(outputs, axis=1)

        learnable_null_caption = self.param('learnable_null_caption', nn.initializers.normal(stddev=self.in_channels ** -0.5), (1, self.token_len, self.in_channels))
        seq_len = caption.shape[1]
        if seq_len < self.token_len:
            learnable_null_caption = learnable_null_caption[:, :seq_len, :]

        if train and self.drop_text_prob > 0:
            rng = self.make_rng("drop_text")
            shape = (caption.shape[0],)
            drop_ids = jax.random.bernoulli(rng, self.drop_text_prob, shape)
        else:
            drop_ids = None
        if drop_ids is not None:
            shape = (caption.shape[0],) + (1,) * (caption.ndim - 1)
            caption = jnp.where(jnp.reshape(drop_ids, shape) == 1, learnable_null_caption, caption)

        return _apply_mlp(caption, "")


class TextEncoderAdapterTransformer(nn.Module):
    """
    Pass captions through a small transformer connector. Also handles whole-caption level caption dropout for classifier-free guidance.
    """
    in_channels: int
    hidden_size: int
    drop_text_prob: float
    num_heads: int
    mlp_ratio: float
    use_qknorm: bool
    use_swiglu: bool
    use_rmsnorm: bool
    token_len: int = 77
    dtype: jnp.dtype = jnp.float32
    num_blocks: int = 1

    @nn.compact
    def __call__(self, caption, train):
        inits = dict(kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)

        def _apply_connector(x, suffix):
            x = nn.Dense(self.hidden_size, **inits, name=f"connector_in{suffix}")(x)
            norm_cls = base.RMSNorm if self.use_rmsnorm else base.LayerNorm
            mlp_hidden = int(self.hidden_size * self.mlp_ratio)
            for block_idx in range(self.num_blocks):
                block_suffix = "" if block_idx == 0 else str(block_idx + 1)
                norm1 = norm_cls(self.hidden_size, dtype=self.dtype, name=f"connector_norm{2 * block_idx + 1}{suffix}")
                norm2 = norm_cls(self.hidden_size, dtype=self.dtype, name=f"connector_norm{2 * block_idx + 2}{suffix}")
                attn = base.Attention(
                    self.hidden_size,
                    self.num_heads,
                    self.use_qknorm,
                    self.use_rmsnorm,
                    dtype=self.dtype,
                    name=f"connector_attn{block_suffix}{suffix}",
                )
                if self.use_swiglu:
                    mlp = base.SwiGLUFFN(self.hidden_size, int(2 / 3 * mlp_hidden), dtype=self.dtype, name=f"connector_mlp{block_suffix}{suffix}")
                else:
                    mlp = base.MlpBlock(self.hidden_size, mlp_hidden, dtype=self.dtype, name=f"connector_mlp{block_suffix}{suffix}")
                attn_out = attn(norm1(x), None, not train)
                x = x + attn_out
                x = x + mlp(norm2(x))
            return x

        if isinstance(caption, (list, tuple)):
            captions = list(caption)
            in_channels = list(self.in_channels) if isinstance(self.in_channels, (list, tuple)) else [cap.shape[-1] for cap in captions]
            token_lens = list(self.token_len) if isinstance(self.token_len, (list, tuple)) else [self.token_len] * len(captions)
            assert (len(in_channels) == len(captions)) and (len(token_lens) == len(captions))
            per_encoder_probs = self.drop_text_prob if isinstance(self.drop_text_prob, (list, tuple)) else None
            drop_ids_list = None
            if train:
                if per_encoder_probs is not None:
                    rng = self.make_rng("drop_text")
                    rngs = jax.random.split(rng, len(captions))
                    drop_ids_list = [
                        jax.random.bernoulli(rngs[i], per_encoder_probs[i], (captions[i].shape[0],))
                        if per_encoder_probs[i] > 0
                        else None
                        for i in range(len(captions))
                    ]
                elif self.drop_text_prob > 0:
                    rng = self.make_rng("drop_text")
                    drop_ids = jax.random.bernoulli(rng, self.drop_text_prob, (captions[0].shape[0],))
                    drop_ids_list = [drop_ids] * len(captions)

            outputs = []
            for i, cap in enumerate(captions):
                learnable_null_caption = self.param(
                    f"learnable_null_caption_{i}",
                    nn.initializers.normal(stddev=in_channels[i] ** -0.5),
                    (1, token_lens[i], in_channels[i]),
                )
                seq_len = cap.shape[1]
                if seq_len < token_lens[i]:
                    learnable_null_caption = learnable_null_caption[:, :seq_len, :]
                drop_ids = drop_ids_list[i] if drop_ids_list is not None else None
                if drop_ids is not None:
                    shape = (cap.shape[0],) + (1,) * (cap.ndim - 1)
                    cap = jnp.where(jnp.reshape(drop_ids, shape) == 1, learnable_null_caption, cap)
                outputs.append(_apply_connector(cap, f"_{i}"))

            return jnp.concatenate(outputs, axis=1)

        learnable_null_caption = self.param('learnable_null_caption', nn.initializers.normal(stddev=self.in_channels ** -0.5), (1, self.token_len, self.in_channels))
        seq_len = caption.shape[1]
        if seq_len < self.token_len:
            learnable_null_caption = learnable_null_caption[:, :seq_len, :]

        if train and self.drop_text_prob > 0:
            rng = self.make_rng("drop_text")
            shape = (caption.shape[0],)
            drop_ids = jax.random.bernoulli(rng, self.drop_text_prob, shape)
        else:
            drop_ids = None
        if drop_ids is not None:
            shape = (caption.shape[0],) + (1,) * (caption.ndim - 1)
            caption = jnp.where(jnp.reshape(drop_ids, shape) == 1, learnable_null_caption, caption)

        return _apply_connector(caption, "")


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        def _init(rng, shape, _dtype=None):
            del rng
            return jnp.ones(shape, dtype=jnp.float32)

        self.scale = self.param("scale", _init, (self.dim,))

    def __call__(self, x: Array) -> Array:
        x_f32 = x.astype(jnp.float32)
        mean_square = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        normed = x_f32 * jax.lax.rsqrt(mean_square + self.eps)
        return (normed * self.scale).astype(x.dtype)


class LayerNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        def _scale_init(rng, shape, _dtype=None):
            del rng
            return jnp.ones(shape, dtype=jnp.float32)

        def _bias_init(rng, shape, _dtype=None):
            del rng
            return jnp.zeros(shape, dtype=jnp.float32)

        self.scale = self.param("scale", _scale_init, (self.dim,))
        self.bias = self.param("bias", _bias_init, (self.dim,))

    def __call__(self, x: Array) -> Array:
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
        normed = (x_f32 - mean) * jax.lax.rsqrt(variance + self.eps)
        return (normed * self.scale + self.bias).astype(x.dtype)


class PatchEmbed(nn.Module):
    patch_size: int
    hidden_size: int
    in_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        conv = nn.Conv(
            self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=True,
            dtype=self.dtype,
            name="proj",
        )
        x = conv(x)
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c)


class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256
    dtype: jnp.dtype = jnp.float32

    @staticmethod
    def timestep_embedding(t: Array, dim: int, max_period: int = 10000) -> Array:
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, t: Array) -> Array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_size, dtype=self.dtype, name="linear1")(t_freq)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, dtype=self.dtype, name="linear2")(x)
        return x


class SwiGLUFFN(nn.Module):
    hidden_size: int
    hidden_features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        w12 = nn.Dense(2 * self.hidden_features, dtype=self.dtype, name="w12")(x)
        x1, x2 = jnp.split(w12, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        out = nn.Dense(self.hidden_size, dtype=self.dtype, name="w3")(hidden)
        return out


class MlpBlock(nn.Module):
    hidden_size: int
    hidden_features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.hidden_features, dtype=self.dtype, name="fc1")(x)
        x = nn.gelu(x, approximate="tanh")
        x = nn.Dense(self.hidden_size, dtype=self.dtype, name="fc2")(x)
        return x


def _rotate_half(x: Array) -> Array:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def _broadcat(arrays: Sequence[Array], axis: int = -1) -> Array:
    arrays = list(arrays)
    rank = arrays[0].ndim
    axis = axis if axis >= 0 else axis + rank
    broadcast_shapes = []
    for i in range(rank):
        sizes = [arr.shape[i] for arr in arrays]
        if i == axis:
            broadcast_shapes.append(tuple(sizes))
        else:
            max_size = max(sizes)
            broadcast_shapes.append((max_size,) * len(arrays))
    target_shapes = list(zip(*broadcast_shapes))
    arrays = [jnp.broadcast_to(arr, shape) for arr, shape in zip(arrays, target_shapes)]
    return jnp.concatenate(arrays, axis=axis)


class VisionRotaryEmbeddingFast(nn.Module):
    dim: int
    pt_seq_len: int
    image_resolution: int
    ft_seq_len: Optional[int] = None
    base_resolution: int = 256
    theta: float = 10000.0
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        ft_seq_len = self.ft_seq_len or self.pt_seq_len
        base_freqs = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=self.dtype) / self.dim))
        scale = jnp.asarray(self.base_resolution, dtype=self.dtype) / jnp.asarray(self.image_resolution, dtype=self.dtype)
        t = jnp.arange(ft_seq_len, dtype=self.dtype) * scale
        freqs = jnp.einsum('..., f -> ... f', t, base_freqs)
        freqs = jnp.repeat(freqs, repeats=2, axis=-1)
        freqs = _broadcat((freqs[:, None, :], freqs[None, :, :]), axis=-1)
        self.freqs_cos = jnp.cos(freqs).reshape(-1, freqs.shape[-1]).astype(self.dtype)
        self.freqs_sin = jnp.sin(freqs).reshape(-1, freqs.shape[-1]).astype(self.dtype)

    def __call__(self, t: Array) -> Array:
        _, _, seq_len, _ = t.shape
        cos = self.freqs_cos
        sin = self.freqs_sin
        repeats = max(1, seq_len // cos.shape[0])
        if repeats > 1:
            cos = jnp.repeat(cos, repeats, axis=0)
            sin = jnp.repeat(sin, repeats, axis=0)
        cos = cos[:seq_len]
        sin = sin[:seq_len]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return t * cos + _rotate_half(t) * sin


class Attention(nn.Module):
    hidden_size: int
    num_heads: int
    qk_norm: bool
    use_rmsnorm: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Dense(3 * self.hidden_size, use_bias=True, dtype=self.dtype, name="qkv")
        norm_cls = RMSNorm if self.use_rmsnorm else LayerNorm
        self.q_norm = norm_cls(self.head_dim, dtype=self.dtype, name="q_norm") if self.qk_norm else None
        self.k_norm = norm_cls(self.head_dim, dtype=self.dtype, name="k_norm") if self.qk_norm else None
        self.out = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj")

    def __call__(self, x: Array, rope: Optional[VisionRotaryEmbeddingFast], deterministic: bool) -> Array:
        del deterministic
        b, n, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = jnp.squeeze(q, axis=2)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q * self.scale, k).astype(jnp.float32)
        attn = nn.softmax(attn_logits, axis=-1).astype(q.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, n, self.hidden_size)
        out = self.out(out)
        return out


class AdaLNModulation(nn.Module):
    hidden_size: int
    out_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, c: Array) -> Array:
        x = nn.silu(c)
        x = nn.Dense(self.out_dim, dtype=self.dtype, name="linear")(x)
        return x


def modulate(x: Array, shift: Optional[Array], scale: Array) -> Array:
    if shift is None:
        return x * (1 + scale[:, None, :])
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTFinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int
    use_rmsnorm: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, c: Array) -> Array:
        norm_cls = RMSNorm if self.use_rmsnorm else LayerNorm
        norm = norm_cls(self.hidden_size, dtype=self.dtype, name="norm_final")
        ada = AdaLNModulation(self.hidden_size, 2 * self.hidden_size, dtype=self.dtype, name="adaLN_modulation")
        shift, scale = jnp.split(ada(c), 2, axis=-1)
        x = modulate(norm(x), shift, scale)
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, dtype=self.dtype, name="linear")(x)
        return x


class DiTFinalLayerNoAdaLN(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int
    use_rmsnorm: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, cond: Array) -> Array:
        del cond
        norm_cls = base.RMSNorm if self.use_rmsnorm else base.LayerNorm
        norm = norm_cls(self.hidden_size, dtype=self.dtype, name="norm_final")
        x = norm(x)
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, dtype=self.dtype, name="linear")(x)
        return x


def _get_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_interpolated_pos_embed(
    embed_dim: int,
    grid_size: int,
    image_resolution: int,
    base_image_resolution: int = 256,
) -> np.ndarray:
    """
    Scale the index grid such that the range is the same as the 256 case
    """
    scale = float(base_image_resolution) / float(image_resolution)
    grid_h = np.arange(grid_size, dtype=np.float32) * scale
    grid_w = np.arange(grid_size, dtype=np.float32) * scale
    """
    Everything below is exactly the same as _get_pos_embed
    """
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = np.outer(pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb

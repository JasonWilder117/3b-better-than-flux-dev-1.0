import math
from typing import TYPE_CHECKING, Callable, List, Mapping, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.models.vae_flax import (
    FlaxAutoencoderKLOutput,
    FlaxDecoderOutput,
)
from diffusers.models.modeling_flax_utils import FlaxModelMixin
from flax.traverse_util import flatten_dict, unflatten_dict

if TYPE_CHECKING:
    import torch


def _get_activation(act_fn: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    act_fn = act_fn.lower()
    if act_fn in {"silu", "swish"}:
        return nn.swish
    if act_fn == "relu":
        return nn.relu
    if act_fn == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation '{act_fn}' for Flax Qwen VAE")


def _flatten_video_frames(x: jnp.ndarray) -> Tuple[jnp.ndarray, int, int, int, int]:
    b, c, t, h, w = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3, 4)).reshape(b * t, c, h, w)
    return x, b, t, h, w


def _restore_video_frames(x: jnp.ndarray, b: int, t: int) -> jnp.ndarray:
    n, c, h, w = x.shape
    assert n == b * t
    x = x.reshape(b, t, c, h, w)
    return jnp.transpose(x, (0, 2, 1, 3, 4))


class FlaxQwenImageRMSNorm(nn.Module):
    dim: int
    channel_first: bool = True
    images: bool = True
    bias: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        broadcast_dims = (1, 1, 1) if not self.images else (1, 1)
        param_shape = (self.dim, *broadcast_dims) if self.channel_first else (self.dim,)
        self.gamma = self.param("gamma", nn.initializers.ones, param_shape, self.dtype)
        if self.bias:
            self.beta = self.param("bias", nn.initializers.zeros, param_shape, self.dtype)
        else:
            self.beta = None
        self.scale = math.sqrt(self.dim)

    def __call__(self, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        axis = (1,) if self.channel_first else (-1,)
        norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
        norm = jnp.maximum(norm, eps)
        x = x / norm * self.scale
        x = x * self.gamma
        if self.beta is not None:
            x = x + self.beta
        return x


class FlaxQwenImageConv2d(nn.Module):
    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)  # left, right, top, bottom
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=jax.lax.Precision.HIGHEST,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if any(self.padding):
            pad_left, pad_right, pad_top, pad_bottom = self.padding
            x = jnp.pad(
                x,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
            )
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.conv(x)
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


class FlaxQwenImageCausalConv3d(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int, int]
    strides: Tuple[int, int, int] = (1, 1, 1)
    padding: Tuple[int, int, int] = (0, 0, 0)
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        pad_t, pad_h, pad_w = self.padding
        self._pad_config = (
            (0, 0),  # batch
            (0, 0),  # channels
            (pad_t * 2, 0),  # time (causal)
            (pad_h, pad_h),
            (pad_w, pad_w),
        )
        self.conv = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=jax.lax.Precision.HIGHEST,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if any(pad > 0 for cfg in self._pad_config[2:] for pad in cfg):
            x = jnp.pad(x, self._pad_config, mode="constant")
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.conv(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x


class FlaxQwenImageUpsample(nn.Module):
    scale_factor: Tuple[float, float] = (2.0, 2.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n, c, h, w = x.shape
        scale_h = int(self.scale_factor[0])
        scale_w = int(self.scale_factor[1])
        if scale_h != self.scale_factor[0] or scale_w != self.scale_factor[1]:
            raise ValueError("FlaxQwenImageUpsample only supports integer scale factors")
        x = jnp.repeat(x, scale_h, axis=2)
        x = jnp.repeat(x, scale_w, axis=3)
        return x


class FlaxQwenImageDiagonalGaussianDistribution:
    def __init__(self, parameters: jnp.ndarray, deterministic: bool = False):
        self.mean, self.logvar = jnp.split(parameters, 2, axis=1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            zeros = jnp.zeros_like(self.mean)
            self.std = zeros
            self.var = zeros

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)

    def kl(self, other=None):
        if self.deterministic:
            return jnp.array([0.0])

        axes = (1, 2, 3, 4)

        if other is None:
            return 0.5 * jnp.sum(self.mean**2 + self.var - 1.0 - self.logvar, axis=axes)

        return 0.5 * jnp.sum(
            jnp.square(self.mean - other.mean) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            axis=axes,
        )

    def nll(self, sample, axis=(1, 2, 3, 4)):
        if self.deterministic:
            return jnp.array([0.0])

        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis)

    def mode(self):
        return self.mean


class FlaxQwenImageResample(nn.Module):
    dim: int
    mode: str
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        mode = self.mode.lower()
        self._mode = mode

        if mode == "upsample2d":
            self.resample_layers = [
                FlaxQwenImageUpsample(),
                FlaxQwenImageConv2d(self.dim // 2, (3, 3), padding=(1, 1, 1, 1), dtype=self.dtype),
            ]
            self.time_conv = None
        elif mode == "upsample3d":
            self.resample_layers = [
                FlaxQwenImageUpsample(),
                FlaxQwenImageConv2d(self.dim // 2, (3, 3), padding=(1, 1, 1, 1), dtype=self.dtype),
            ]
            self.time_conv = FlaxQwenImageCausalConv3d(
                self.dim,
                self.dim * 2,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                dtype=self.dtype,
            )
        elif mode == "downsample2d":
            self.resample_layers = [
                FlaxQwenImageConv2d(
                    self.dim,
                    (3, 3),
                    strides=(2, 2),
                    padding=(0, 1, 0, 1),
                    dtype=self.dtype,
                ),
            ]
            self.time_conv = None
        elif mode == "downsample3d":
            self.resample_layers = [
                FlaxQwenImageConv2d(
                    self.dim,
                    (3, 3),
                    strides=(2, 2),
                    padding=(0, 1, 0, 1),
                    dtype=self.dtype,
                ),
            ]
            self.time_conv = FlaxQwenImageCausalConv3d(
                self.dim,
                self.dim,
                kernel_size=(3, 1, 1),
                strides=(2, 1, 1),
                padding=(0, 0, 0),
                dtype=self.dtype,
            )
        else:
            self.resample_layers = []
            self.time_conv = None

    def _apply_resample_layers(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.resample_layers:
            if isinstance(layer, FlaxQwenImageUpsample):
                x = layer(x)
            else:
                x = layer(x)
        return x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_flat, b_flat, t_flat, _, _ = _flatten_video_frames(x)
        x_flat = self._apply_resample_layers(x_flat)
        x = _restore_video_frames(x_flat, b_flat, t_flat)

        return x


class FlaxQwenImageResidualBlock(nn.Module):
    in_dim: int
    out_dim: int
    dropout: float = 0.0
    non_linearity: str = "silu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.nonlinearity = _get_activation(self.non_linearity)
        self.norm1 = FlaxQwenImageRMSNorm(self.in_dim, images=False, dtype=self.dtype)
        self.conv1 = FlaxQwenImageCausalConv3d(
            self.in_dim,
            self.out_dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            dtype=self.dtype,
        )
        self.norm2 = FlaxQwenImageRMSNorm(self.out_dim, images=False, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        self.conv2 = FlaxQwenImageCausalConv3d(
            self.out_dim,
            self.out_dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            dtype=self.dtype,
        )
        if self.in_dim != self.out_dim:
            self.conv_shortcut = FlaxQwenImageCausalConv3d(
                self.in_dim,
                self.out_dim,
                kernel_size=(1, 1, 1),
                dtype=self.dtype,
            )
        else:
            self.conv_shortcut = None

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout_layer(x, deterministic=deterministic)
        x = self.conv2(x)

        return x + residual


class FlaxQwenImageAttentionBlock(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.norm = FlaxQwenImageRMSNorm(self.dim, dtype=self.dtype)
        self.to_qkv = FlaxQwenImageConv2d(3 * self.dim, (1, 1), dtype=self.dtype)
        self.proj = FlaxQwenImageConv2d(self.dim, (1, 1), dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        identity = x
        n, c, t, h, w = x.shape

        x_btchw, b, t_frames, _, _ = _flatten_video_frames(x)
        x_btchw = self.norm(x_btchw)
        qkv = self.to_qkv(x_btchw)

        seq = h * w
        qkv = qkv.reshape(b * t_frames, 3, c, seq)
        qkv = jnp.transpose(qkv, (0, 1, 3, 2))
        q, k, v = qkv[:, 0:1], qkv[:, 1:2], qkv[:, 2:3]

        scale = 1.0 / math.sqrt(c)
        attn_scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.squeeze(1)
        attn_output = attn_output.transpose(0, 2, 1).reshape(b * t_frames, c, h, w)

        x_btchw = self.proj(attn_output)
        x = _restore_video_frames(x_btchw, b, t_frames)

        return x + identity


class FlaxQwenImageMidBlock(nn.Module):
    dim: int
    dropout: float = 0.0
    non_linearity: str = "silu"
    num_layers: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets: List[nn.Module] = []
        attentions: List[Optional[nn.Module]] = []

        resnets.append(FlaxQwenImageResidualBlock(self.dim, self.dim, self.dropout, self.non_linearity, self.dtype))
        for _ in range(self.num_layers):
            attentions.append(FlaxQwenImageAttentionBlock(self.dim, self.dtype))
            resnets.append(
                FlaxQwenImageResidualBlock(self.dim, self.dim, self.dropout, self.non_linearity, self.dtype)
            )

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.resnets[0](x, deterministic=deterministic)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, deterministic=deterministic)
        return x


class FlaxQwenImageEncoder3d(nn.Module):
    base_dim: int
    z_dim: int
    dim_mult: Sequence[int]
    num_res_blocks: int
    attn_scales: Sequence[float]
    temporal_downsample: Sequence[bool]
    dropout: float = 0.0
    non_linearity: str = "silu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        dims = [self.base_dim * u for u in [1] + list(self.dim_mult)]
        scale = 1.0

        self.conv_in = FlaxQwenImageCausalConv3d(3, dims[0], (3, 3, 3), padding=(1, 1, 1), dtype=self.dtype)

        down_layers: List[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            current_dim = in_dim
            for _ in range(self.num_res_blocks):
                down_layers.append(
                    FlaxQwenImageResidualBlock(current_dim, out_dim, self.dropout, self.non_linearity, self.dtype)
                )
                if scale in self.attn_scales:
                    down_layers.append(FlaxQwenImageAttentionBlock(out_dim, self.dtype))
                current_dim = out_dim

            if i != len(self.dim_mult) - 1:
                mode = "downsample3d" if self.temporal_downsample[i] else "downsample2d"
                down_layers.append(FlaxQwenImageResample(out_dim, mode=mode, dtype=self.dtype))
                scale /= 2.0

        self.down_blocks = down_layers

        self.mid_block = FlaxQwenImageMidBlock(out_dim, self.dropout, self.non_linearity, 1, self.dtype)
        self.norm_out = FlaxQwenImageRMSNorm(out_dim, images=False, dtype=self.dtype)
        self.nonlinearity = _get_activation(self.non_linearity)
        self.conv_out = FlaxQwenImageCausalConv3d(out_dim, self.z_dim, (3, 3, 3), padding=(1, 1, 1), dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.conv_in(x)
        for layer in self.down_blocks:
            if isinstance(layer, FlaxQwenImageResidualBlock):
                x = layer(x, deterministic=deterministic)
            elif isinstance(layer, FlaxQwenImageAttentionBlock):
                x = layer(x)
            else:
                x = layer(x)

        x = self.mid_block(x, deterministic=deterministic)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x


class FlaxQwenImageUpBlock(nn.Module):
    in_dim: int
    out_dim: int
    num_res_blocks: int
    dropout: float
    upsample_mode: Optional[str]
    non_linearity: str
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets: List[nn.Module] = []
        current_dim = self.in_dim
        for _ in range(self.num_res_blocks + 1):
            resnets.append(
                FlaxQwenImageResidualBlock(current_dim, self.out_dim, self.dropout, self.non_linearity, self.dtype)
            )
            current_dim = self.out_dim

        self.resnets = resnets

        if self.upsample_mode is not None:
            self.upsampler = FlaxQwenImageResample(self.out_dim, mode=self.upsample_mode, dtype=self.dtype)
        else:
            self.upsampler = None

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for resnet in self.resnets:
            x = resnet(x, deterministic=deterministic)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


class FlaxQwenImageDecoder3d(nn.Module):
    base_dim: int
    z_dim: int
    dim_mult: Sequence[int]
    num_res_blocks: int
    attn_scales: Sequence[float]
    temporal_upsample: Sequence[bool]
    dropout: float = 0.0
    non_linearity: str = "silu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        dims = [self.base_dim * self.dim_mult[-1]] + [self.base_dim * u for u in reversed(self.dim_mult)]
        scale = 1.0 / (2 ** max(len(self.dim_mult) - 2, 0))

        self.conv_in = FlaxQwenImageCausalConv3d(self.z_dim, dims[0], (3, 3, 3), padding=(1, 1, 1), dtype=self.dtype)
        self.mid_block = FlaxQwenImageMidBlock(dims[0], self.dropout, self.non_linearity, 1, self.dtype)

        up_layers: List[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block_in_dim = in_dim // 2 if i > 0 else in_dim
            upsample_mode = None
            if i != len(self.dim_mult) - 1:
                upsample_mode = "upsample3d" if self.temporal_upsample[i] else "upsample2d"
            up_layers.append(
                FlaxQwenImageUpBlock(
                    block_in_dim,
                    out_dim,
                    self.num_res_blocks,
                    self.dropout,
                    upsample_mode,
                    self.non_linearity,
                    self.dtype,
                )
            )
            if upsample_mode is not None:
                scale *= 2.0

        self.up_blocks = up_layers
        self.norm_out = FlaxQwenImageRMSNorm(out_dim, images=False, dtype=self.dtype)
        self.nonlinearity = _get_activation(self.non_linearity)
        self.conv_out = FlaxQwenImageCausalConv3d(out_dim, 3, (3, 3, 3), padding=(1, 1, 1), dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.conv_in(x)
        x = self.mid_block(x, deterministic=deterministic)
        for block in self.up_blocks:
            x = block(x, deterministic=deterministic)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x


# -----------------------------------------------------------------------------
# Top-level model
# -----------------------------------------------------------------------------


@flax_register_to_config
class FlaxAutoencoderKLQwenImage(nn.Module, FlaxModelMixin, ConfigMixin):
    base_dim: int = 96
    z_dim: int = 16
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_downsample: Sequence[bool] = (False, True, True)
    dropout: float = 0.0
    latents_mean: Sequence[float] = ()
    latents_std: Sequence[float] = ()
    in_channels: int = 3
    out_channels: int = 3
    act_fn: str = "silu"
    sample_size: int = 256
    sample_frames: int = 1
    dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        temporal_downsample = tuple(self.temperal_downsample)
        temporal_upsample = tuple(reversed(temporal_downsample))

        self.encoder = FlaxQwenImageEncoder3d(
            self.base_dim,
            self.z_dim * 2,
            self.dim_mult,
            self.num_res_blocks,
            self.attn_scales,
            temporal_downsample,
            self.dropout,
            self.act_fn,
            self.compute_dtype,
        )
        self.quant_conv = FlaxQwenImageCausalConv3d(
            self.z_dim * 2,
            self.z_dim * 2,
            kernel_size=(1, 1, 1),
            dtype=self.compute_dtype,
        )
        self.post_quant_conv = FlaxQwenImageCausalConv3d(
            self.z_dim,
            self.z_dim,
            kernel_size=(1, 1, 1),
            dtype=self.compute_dtype,
        )
        self.decoder = FlaxQwenImageDecoder3d(
            self.base_dim,
            self.z_dim,
            self.dim_mult,
            self.num_res_blocks,
            self.attn_scales,
            temporal_upsample,
            self.dropout,
            self.act_fn,
            self.compute_dtype,
        )

    def init_weights(self, rng: jax.Array) -> flax.core.FrozenDict:
        sample_shape = (
            1,
            self.in_channels,
            self.sample_frames,
            self.sample_size,
            self.sample_size,
        )
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        params_rng, dropout_rng, gaussian_rng = jax.random.split(rng, 3)
        rngs = {"params": params_rng, "dropout": dropout_rng, "gaussian": gaussian_rng}
        return self.init(rngs, sample, deterministic=True)["params"]

    def encode(
        self,
        sample: jnp.ndarray,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> FlaxAutoencoderKLOutput:
        hidden_states = self.encoder(sample, deterministic=deterministic)
        moments = self.quant_conv(hidden_states)
        posterior = FlaxQwenImageDiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return FlaxAutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self,
        latents: jnp.ndarray,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> FlaxDecoderOutput:
        hidden_states = self.post_quant_conv(latents)
        decoded = self.decoder(hidden_states, deterministic=deterministic)
        decoded = jnp.clip(decoded, -1.0, 1.0)
        if not return_dict:
            return (decoded,)
        return FlaxDecoderOutput(sample=decoded)

    def __call__(
        self,
        sample: jnp.ndarray,
        sample_posterior: bool = False,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> FlaxDecoderOutput:
        posterior = self.encode(sample, deterministic=deterministic, return_dict=True)
        if sample_posterior:
            rng = self.make_rng("gaussian")
            hidden_states = posterior.latent_dist.sample(rng)
        else:
            hidden_states = posterior.latent_dist.mode()
        decoded = self.decode(hidden_states, deterministic=deterministic, return_dict=return_dict)
        if not return_dict:
            return (decoded.sample,)
        return decoded


# -----------------------------------------------------------------------------
# PyTorch -> Flax weight conversion
# -----------------------------------------------------------------------------


def convert_pytorch_to_flax_qwenimage(
    pt_state_dict: Mapping[str, "torch.Tensor"],
    flax_model: FlaxAutoencoderKLQwenImage,
    init_key: int = 42,
) -> flax.core.FrozenDict:
    template = flax_model.init_weights(jax.random.PRNGKey(init_key))
    flat_template = flatten_dict(template)

    def to_torch_key(key_tuple: Tuple[str, ...]) -> str:
        parts: List[str] = []
        for segment in key_tuple:
            if segment == "conv":
                continue
            if segment == "kernel":
                parts.append("weight")
                continue
            if segment == "gamma":
                parts.append("gamma")
                continue
            if segment in {"beta", "bias"}:
                parts.append("bias")
                continue
            if segment == "upsampler":
                parts.append("upsamplers.0")
                continue
            if segment.startswith("resample_layers_"):
                parts.append("resample.1")
                continue
            if "_" in segment and segment.rsplit("_", 1)[-1].isdigit():
                base, idx = segment.rsplit("_", 1)
                parts.append(f"{base}.{idx}")
                continue
            parts.append(segment)
        return ".".join(parts)

    def convert_array(array: np.ndarray, expected_shape: Tuple[int, ...]) -> np.ndarray:
        if array.shape == expected_shape:
            return array
        if array.ndim == 5:  # Conv3d: (out, in, kT, kH, kW) -> (kT, kH, kW, in, out)
            array = np.transpose(array, (2, 3, 4, 1, 0))
        elif array.ndim == 4:  # Conv2d: (out, in, kH, kW) -> (kH, kW, in, out)
            array = np.transpose(array, (2, 3, 1, 0))
        elif array.ndim == 2:  # Linear layers if any
            array = np.transpose(array)
        if array.shape != expected_shape:
            raise ValueError(
                f"Converted tensor has shape {array.shape}, expected {expected_shape}"
            )
        return array

    flat_converted = {}
    missing_keys: List[str] = []

    for key_tuple, template_value in flat_template.items():
        torch_key = to_torch_key(key_tuple)
        if torch_key not in pt_state_dict:
            missing_keys.append(torch_key)
            continue
        tensor = pt_state_dict[torch_key]
        np_value = tensor.detach().cpu().numpy()
        converted = convert_array(np_value, template_value.shape)
        flat_converted[key_tuple] = jnp.asarray(converted, dtype=template_value.dtype)

    if missing_keys:
        raise KeyError(
            "Missing PyTorch parameters for conversion: "
            + ", ".join(missing_keys[:5])
            + (" ..." if len(missing_keys) > 5 else "")
        )

    recovered = unflatten_dict(flat_converted)
    return flax.core.freeze(recovered)

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class RectifiedFlowConfig:
    prediction: str = "velocity"
    use_lognorm: bool = True
    lognorm_mu: float = 0.0
    lognorm_sigma: float = 1.0
    train_timestep_shift: float = 0.0
    cfg_interval_start: float = 0
    inference_timestep_shift: float = 0.3
    sampling_method: str = "euler"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RectifiedFlowConfig":
        cfg = dict(config)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


def _broadcast_t(t: jnp.ndarray, target_shape: Sequence[int]) -> jnp.ndarray:
    dims = (1,) * (len(target_shape) - 1)
    return t.reshape((t.shape[0],) + dims)


def sample_times(
    rng: jax.Array,
    batch_size: int,
    cfg: RectifiedFlowConfig,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Sample interpolation times according to Rectified Flow settings."""

    if cfg.use_lognorm:
        # the same as https://github.com/hustvl/LightningDiT/blob/2725fed42a14898744433809949834e26957bcdd/transport/transport.py#L113
        rng, normal_key = jax.random.split(rng)
        normal_samples = cfg.lognorm_mu + cfg.lognorm_sigma * jax.random.normal(normal_key, (batch_size,), dtype=dtype)
        t = jax.nn.sigmoid(normal_samples)
    else:
        rng, uniform_key = jax.random.split(rng)
        t = jax.random.uniform(uniform_key, (batch_size,), dtype=dtype)

    shift = cfg.train_timestep_shift or 0.0
    if shift != 0.0 and shift != 1.0:
        t = (shift * t) / (1.0 + (shift - 1.0) * t)

    return t


def prepare_rectified_flow_inputs(
    latents: jnp.ndarray,
    noise_key: jax.Array,
    time_key: jax.Array,
    cfg: RectifiedFlowConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare (x_t, u_t, t) tuples for velocity training."""

    x1 = latents
    x0 = jax.random.normal(noise_key, latents.shape, dtype=latents.dtype)
    t = sample_times(time_key, latents.shape[0], cfg, dtype=latents.dtype)
    t_expanded = _broadcast_t(t, latents.shape)
    xt = (1.0 - t_expanded) * x0 + t_expanded * x1
    ut = x1 - x0
    return xt, ut, t.astype(latents.dtype)

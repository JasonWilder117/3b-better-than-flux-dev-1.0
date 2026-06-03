from dataclasses import dataclass
from typing import Callable, Optional

import jax
import numpy as np
import jax.numpy as jnp
from flax import linen

ACT2FN_FLAX = {"gelu": linen.gelu, 'silu': linen.silu}

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

def flax_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def flax_repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.expand_dims(hidden_states, axis=2)
    hidden_states = jnp.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def flax_eager_attention_forward(
    num_key_value_groups: int,
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    scaling: float,
    dropout: float = 0.0,
    deterministic: bool = False,
    rng: Optional[jax.Array] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = flax_repeat_kv(key, num_key_value_groups)
    value_states = flax_repeat_kv(value, num_key_value_groups)

    attn_weights = jnp.matmul(query, key_states.transpose(0, 1, 3, 2)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = linen.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(query.dtype)
    attn_weights = linen.Dropout(rate=dropout)(attn_weights, deterministic = deterministic, rng = rng)
    attn_output = jnp.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(0, 2, 1, 3)

    return attn_output, attn_weights

def flax_compute_default_rope_parameters(
    config: Optional[Qwen3Config] = None,
    seq_len: Optional[int] = None,
) -> tuple[jnp.ndarray, float]:
    rope_parameters = normalize_qwen3_rope_parameters(config)
    base = rope_parameters["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq, attention_factor

def normalize_qwen3_rope_parameters(config: Qwen3Config):
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": config.rope_theta,
        }
        config.rope_parameters = rope_parameters
    return rope_parameters

ROPE_INIT_FUNCTIONS_FLAX = {
    "default": flax_compute_default_rope_parameters,
}

class FlaxQwen3RotaryEmbedding(linen.Module):
    config: Qwen3Config

    @linen.compact
    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

        rope_type = normalize_qwen3_rope_parameters(self.config).get("rope_type", "default")

        rope_init_fn = ROPE_INIT_FUNCTIONS_FLAX[rope_type]

        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 1.0)
        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq_shape = (dim // 2,)

        inv_freq = self.param(
            "inv_freq",
            lambda key, shape, dtype=jnp.float32: rope_init_fn(self.config)[0].astype(dtype),
            inv_freq_shape,
        )
        attention_scaling = self.param(
            "attention_scaling",
            lambda key, shape, dtype=jnp.float32: jnp.asarray(rope_init_fn(self.config)[1], dtype=dtype),
            (),
        )

        if position_ids.ndim == 1:
            position_ids = jnp.expand_dims(position_ids, axis=0)

        inv_freq_expanded = jnp.expand_dims(inv_freq, axis=(0, 2)).astype(jnp.float32)
        inv_freq_expanded = jnp.broadcast_to(inv_freq_expanded, (position_ids.shape[0], inv_freq.shape[0], 1))
        position_ids_expanded = jnp.expand_dims(position_ids, axis=1).astype(jnp.float32)

        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb) * attention_scaling
        sin = jnp.sin(emb) * attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class FlaxQwen3RMSNorm(linen.Module):
    hidden_size: int
    eps: float = 1e-6

    @linen.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.power(hidden_states, 2), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        weight = self.param('weight', linen.initializers.ones, (self.hidden_size,))
        return (weight * hidden_states).astype(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.hidden_size)}, eps={self.eps}"


def flax_apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, position_ids=None, unsqueeze_dim: int = 1) -> tuple[jnp.ndarray, jnp.ndarray]:
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (flax_rotate_half(q) * sin)
    k_embed = (k * cos) + (flax_rotate_half(k) * sin)
    return q_embed, k_embed

class FlaxQwen3Attention(linen.Module):
    config: Qwen3Config
    layer_idx: int

    @linen.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        num_key_value_groups = num_attention_heads // num_key_value_heads
        scaling = head_dim ** -0.5

        batch_size, seq_length, _ = hidden_states.shape

        q_proj = linen.Dense(features=num_attention_heads * head_dim, use_bias=self.config.attention_bias, name="q_proj")
        k_proj = linen.Dense(features=num_key_value_heads * head_dim, use_bias=self.config.attention_bias, name="k_proj")
        v_proj = linen.Dense(features=num_key_value_heads * head_dim, use_bias=self.config.attention_bias, name="v_proj")

        q_norm = FlaxQwen3RMSNorm(head_dim, eps=self.config.rms_norm_eps, name="q_norm")
        k_norm = FlaxQwen3RMSNorm(head_dim, eps=self.config.rms_norm_eps, name="k_norm")

        query_states = q_norm(q_proj(hidden_states).reshape(batch_size, seq_length, num_attention_heads, head_dim))
        key_states = k_norm(k_proj(hidden_states).reshape(batch_size, seq_length, num_key_value_heads, head_dim))
        value_states = v_proj(hidden_states).reshape(batch_size, seq_length, num_key_value_heads, head_dim)

        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))

        cos, sin = position_embeddings
        query_states, key_states = flax_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            raise NotImplementedError('KV Cache is not implemented')

        attention_interface: Callable = flax_eager_attention_forward

        attn_output, attn_weights = attention_interface(
            num_key_value_groups,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling = scaling,
            dropout = 0.0 if deterministic else self.config.attention_dropout,
            deterministic = deterministic,
            rng = rng,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        o_proj = linen.Dense(features=self.config.hidden_size, use_bias=self.config.attention_bias, name="o_proj")
        attn_output = o_proj(attn_output)

        return attn_output, attn_weights


class FlaxQwen3MLP(linen.Module):
    config: Qwen3Config

    @linen.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        gate_proj = linen.Dense(features=self.config.intermediate_size, use_bias=False, name="gate_proj")
        up_proj = linen.Dense(features=self.config.intermediate_size, use_bias=False, name="up_proj")
        down_proj = linen.Dense(features=self.config.hidden_size, use_bias=False, name="down_proj")
        act_fn = ACT2FN_FLAX[self.config.hidden_act]

        down_output = down_proj(act_fn(gate_proj(hidden_states)) * up_proj(hidden_states))
        return down_output

class FlaxQwen3DecoderLayer(linen.Module):
    config: Qwen3Config
    layer_idx: int

    @linen.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rng : Optional[jnp.ndarray] = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> jnp.ndarray:
        return self._forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            deterministic=deterministic,
            rng = rng,
            **kwargs,
        )

    def _forward(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rng : Optional[jnp.ndarray] = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> jnp.ndarray:

        self_attn = FlaxQwen3Attention(config=self.config, layer_idx=self.layer_idx, name="self_attn")
        mlp = FlaxQwen3MLP(self.config, name="mlp")
        input_layernorm = FlaxQwen3RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, name="input_layernorm")
        post_attention_layernorm = FlaxQwen3RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, name="post_attention_layernorm")

        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states, _ = self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            deterministic = deterministic,
            rng = rng,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

@dataclass

@dataclass
class FlaxBaseModelOutputWithPast(ModelOutput):

    last_hidden_state: Optional[jnp.ndarray] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[jnp.ndarray, ...]] = None
    attentions: Optional[tuple[jnp.ndarray, ...]] = None

class FlaxQwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlaxQwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": FlaxQwen3DecoderLayer,
        "attentions": FlaxQwen3Attention,
    }


class FlaxQwen3Model(linen.Module):
    config: Qwen3Config
    dtype: jnp.dtype = jnp.float32

    @linen.compact
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[tuple[tuple[jnp.ndarray, jnp.ndarray]]] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        use_cache: bool = False,
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None
    ) -> FlaxBaseModelOutputWithPast:

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            embed_tokens = linen.Embed(num_embeddings=self.config.vocab_size,
                features=self.config.hidden_size, dtype=self.dtype, name="embed_tokens")

            hidden_states = embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_length, _ = hidden_states.shape

        if past_key_values is None:
            past_seen_tokens = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_seen_tokens = past_key_values[0][0].shape[2]
            raise NotImplementedError('KV Cache is not supported')

        cache_position = jnp.arange(past_seen_tokens, past_seen_tokens + seq_length, dtype=jnp.int32)

        if position_ids is None:
            position_ids = cache_position.reshape(1, -1)
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_length))
        elif position_ids.ndim == 1:
            position_ids = jnp.expand_dims(position_ids, axis=0)
            position_ids = jnp.broadcast_to(position_ids, (batch_size, position_ids.shape[1]))

        causal_mask = linen.make_causal_mask(jnp.ones((batch_size, seq_length), dtype=jnp.bool_), dtype=jnp.bool_)
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.bool_)
        padding_mask = attention_mask.astype(jnp.bool_)
        combined_mask = jnp.logical_and(causal_mask, padding_mask[:, jnp.newaxis, jnp.newaxis, :])
        flax_attention_mask = jnp.where(combined_mask, 0.0, jnp.finfo(self.dtype).min)
        rotary_emb = FlaxQwen3RotaryEmbedding(config=self.config, name="rotary_emb")
        position_embeddings = rotary_emb(hidden_states, position_ids)

        for i in range(self.config.num_hidden_layers):
            DecoderLayer = FlaxQwen3DecoderLayer(config=self.config, layer_idx=i, name=f"layer_{i}")

            layer_outputs = DecoderLayer(
                hidden_states,
                attention_mask=flax_attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=None,
                deterministic=deterministic,
                rng = rng
            )

            hidden_states = layer_outputs

        norm = FlaxQwen3RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, name="norm")
        hidden_states = norm(hidden_states)

        return FlaxBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None
        )

__all__ = [
    "FlaxQwen3PreTrainedModel",
    "FlaxQwen3Model",
]

def convert_pytorch_to_flax_params_qwen3(torch_model):
    flax_params = {'params': {}}

    flax_params['params']['embed_tokens'] = {
        'embedding': torch_model.embed_tokens.weight.detach().cpu().numpy()
    }
    flax_params['params']['rotary_emb'] = {
        'inv_freq': torch_model.rotary_emb.inv_freq.detach().cpu().numpy(),
        'attention_scaling': np.array(torch_model.rotary_emb.attention_scaling)
    }
    for i, layer in enumerate(torch_model.layers):
        layer_key = f'layer_{i}'
        flax_params['params'][layer_key] = {
            'input_layernorm': {
                'weight': layer.input_layernorm.weight.detach().cpu().numpy()
            },
            'post_attention_layernorm': {
                'weight': layer.post_attention_layernorm.weight.detach().cpu().numpy()
            },
            'self_attn': {
                'q_proj': {'kernel': layer.self_attn.q_proj.weight.T.detach().cpu().numpy()},
                'k_proj': {'kernel': layer.self_attn.k_proj.weight.T.detach().cpu().numpy()},
                'v_proj': {'kernel': layer.self_attn.v_proj.weight.T.detach().cpu().numpy()},
                'o_proj': {'kernel': layer.self_attn.o_proj.weight.T.detach().cpu().numpy()},
                'q_norm': {'weight': layer.self_attn.q_norm.weight.detach().cpu().numpy()},
                'k_norm': {'weight': layer.self_attn.k_norm.weight.detach().cpu().numpy()},
            },
            'mlp': {
                'gate_proj': {'kernel': layer.mlp.gate_proj.weight.T.detach().cpu().numpy()},
                'up_proj': {'kernel': layer.mlp.up_proj.weight.T.detach().cpu().numpy()},
                'down_proj': {'kernel': layer.mlp.down_proj.weight.T.detach().cpu().numpy()},
            }
        }

    flax_params['params']['norm'] = {
        'weight': torch_model.norm.weight.detach().cpu().numpy()
    }
    return flax_params

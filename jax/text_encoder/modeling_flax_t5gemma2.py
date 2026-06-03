from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Optional, Union

import torch

import jax
import numpy as np
import jax.numpy as jnp
from flax import linen

ACT2FN_FLAX = {'gelu': linen.gelu, 'silu': linen.silu, 'gelu_pytorch_tanh': lambda x: linen.gelu(x, approximate=True)}

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.models.siglip import SiglipVisionConfig

from transformers.utils import logging
logger = logging.get_logger(__name__)

class T5Gemma2TextConfig(PretrainedConfig):

    model_type = "t5gemma2_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __init__(
        self,
        vocab_size: Optional[int] = 262_208,
        hidden_size: Optional[int] = 2304,
        intermediate_size: Optional[int] = 9216,
        num_hidden_layers: Optional[int] = 26,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 4,
        head_dim: Optional[int] = 256,
        hidden_activation: Optional[str] = "gelu_pytorch_tanh",
        max_position_embeddings: Optional[int] = 131_072,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = True,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        query_pre_attn_scalar: Optional[int] = 256,
        sliding_window: Optional[int] = 4096,
        layer_types: Optional[list[str]] = None,
        final_logit_softcapping: Optional[float] = None,
        attn_logit_softcapping: Optional[float] = None,
        rope_parameters = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.layer_types = layer_types

        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs


class T5Gemma2EncoderConfig(PretrainedConfig):

    model_type = "t5gemma2_encoder"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    sub_configs = {
        "text_config": T5Gemma2TextConfig,
        "vision_config": SiglipVisionConfig,
    }

    def __init__(
        self,
        text_config: Optional[Union[T5Gemma2TextConfig, dict[str, Any]]] = None,
        vision_config: Optional[Union[SiglipVisionConfig, dict[str, Any]]] = None,
        mm_tokens_per_image: int = 256,
        boi_token_index: int = 255_999,
        eoi_token_index: int = 256_000,
        image_token_index: int = 262_144,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if text_config is None:
            text_config = T5Gemma2TextConfig()
            logger.info("text_config is None, using default T5Gemma2EncoderTextConfig text config.")
        elif isinstance(text_config, dict):
            text_config = T5Gemma2TextConfig(**text_config)

        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = SiglipVisionConfig()
            logger.info("vision_config is None, using default SiglipVisionConfig vision config.")

        self.text_config = text_config
        self.vision_config = vision_config
        self.mm_tokens_per_image = mm_tokens_per_image
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.initializer_range = initializer_range

        super().__init__(**kwargs)



class T5Gemma2DecoderConfig(PretrainedConfig):

    model_type = "t5gemma2_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __init__(
        self,
        vocab_size: Optional[int] = 262_208,
        hidden_size: Optional[int] = 2304,
        intermediate_size: Optional[int] = 9216,
        num_hidden_layers: Optional[int] = 26,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 4,
        head_dim: Optional[int] = 256,
        hidden_activation: Optional[str] = "gelu_pytorch_tanh",
        max_position_embeddings: Optional[int] = 131_072,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = True,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        query_pre_attn_scalar: Optional[int] = 256,
        sliding_window: Optional[int] = 4096,
        layer_types: Optional[list[str]] = None,
        final_logit_softcapping: Optional[float] = None,
        attn_logit_softcapping: Optional[float] = None,
        rope_parameters = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.layer_types = layer_types

        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs


class T5Gemma2Config(PretrainedConfig):

    model_type = "t5gemma2"
    keys_to_ignore_at_inference = ["past_key_values"]

    sub_configs = {
        "encoder": T5Gemma2EncoderConfig,
        "decoder": T5Gemma2DecoderConfig,
    }

    attribute_map = {
        "image_token_id": "image_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    def __init__(
        self,
        encoder: Optional[Union[T5Gemma2EncoderConfig, dict[str, Any]]] = None,
        decoder: Optional[Union[T5Gemma2DecoderConfig, dict[str, Any]]] = None,
        is_encoder_decoder: bool = True,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        classifier_dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        image_token_index: int = 256_001,
        **kwargs,
    ):
        if isinstance(encoder, dict):
            encoder = T5Gemma2EncoderConfig(**encoder)
        elif encoder is None:
            encoder = T5Gemma2EncoderConfig()
            logger.info("encoder is None, using default T5Gemma2EncoderConfig encoder config.")
        else:
            if not isinstance(encoder, T5Gemma2EncoderConfig):
                raise ValueError(f"{type(encoder)} is not supported.")

        if isinstance(decoder, dict):
            decoder = T5Gemma2DecoderConfig(**decoder)
        elif decoder is None:
            decoder = T5Gemma2DecoderConfig()
            logger.info("decoder is None, using default T5Gemma2DecoderConfig decoder config.")
        else:
            if not isinstance(decoder, T5Gemma2DecoderConfig):
                raise ValueError(f"{type(decoder)} is not supported.")

        if encoder.text_config.hidden_size != decoder.hidden_size:
            raise ValueError(
                "Imbalanced encoder-decoder is not supported in T5Gemma2: "
                f"encoder ({encoder.text_config.hidden_size}) vs decoder ({decoder.hidden_size})."
            )

        if not is_encoder_decoder:
            raise ValueError("T5Gemma2Model only support encoder-decoder modeling.")

        if encoder.text_config.vocab_size != decoder.vocab_size:
            raise ValueError(
                "Imbalanced encoder-decoder vocabulary size is not supported in T5Gemma2: "
                f"encoder ({encoder.text_config.vocab_size}) vs decoder ({decoder.vocab_size})."
            )

        encoder.text_config.dropout_rate = dropout_rate
        encoder.text_config.attention_dropout = attention_dropout
        encoder.vision_config.attention_dropout = attention_dropout
        encoder.image_token_index = image_token_index
        self.encoder = encoder

        decoder.dropout_rate = dropout_rate
        decoder.attention_dropout = attention_dropout
        self.decoder = decoder

        for special_token_key in ["bos_token_id", "pad_token_id", "eos_token_id", "vocab_size"]:
            if special_token_key not in kwargs:
                kwargs[special_token_key] = getattr(decoder, special_token_key)

        super().__init__(**kwargs)

        self.is_encoder_decoder = is_encoder_decoder
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.classifier_dropout_rate = classifier_dropout_rate
        self.initializer_range = initializer_range
        self.eoi_token_index = encoder.eoi_token_index
        self.image_token_index = image_token_index

    def __setattr__(self, key, value):
        shared_attr_with_submodules = [
            "output_hidden_states",
            "output_attentions",
            "_attn_implementation_internal",
            "dropout_rate",
            "attention_dropout",
            "vocab_size",
            "dtype",
        ]

        if key in shared_attr_with_submodules:
            setattr(self.encoder.text_config, key, value)
            setattr(self.encoder.vision_config, key, value)
            setattr(self.decoder, key, value)
            setattr(self.encoder, key, value)
        super().__setattr__(key, value)

class FlaxT5Gemma2RMSNorm(linen.Module):
    hidden_size: int
    eps: float = 1e-6

    @linen.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.power(hidden_states, 2), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        weight = self.param('weight', linen.initializers.zeros, (self.hidden_size,))
        return (hidden_states * (1.0 + weight)).astype(input_dtype)

    def extra_repr(self):
        return f"{tuple((self.hidden_size,))}, eps={self.eps}"

class FlaxT5Gemma2MLP(linen.Module):
    config: T5Gemma2TextConfig

    @linen.compact
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True,
                    rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        gate_proj = linen.Dense(features=self.config.intermediate_size, use_bias=False, name="gate_proj")
        up_proj = linen.Dense(features=self.config.intermediate_size, use_bias=False, name="up_proj")
        down_proj = linen.Dense(features=self.config.hidden_size, use_bias=False, name="down_proj")
        dropout = linen.Dropout(rate=self.config.dropout_rate)
        act_fn = ACT2FN_FLAX[self.config.hidden_activation]

        hidden_states = act_fn(gate_proj(hidden_states)) * up_proj(hidden_states)
        hidden_states = dropout(hidden_states, deterministic=deterministic, rng = rng)
        down_output = down_proj(hidden_states)
        return down_output

def flax_compute_default_rope_parameters(
    config: Optional[T5Gemma2TextConfig] = None,
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple[jnp.ndarray, float]:
    base = config.rope_parameters[layer_type]["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    attention_factor = 1.0

    inv_freq = 1.0 / (
        base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )
    return inv_freq, attention_factor

def flax_compute_linear_rope_parameters(
    config: Optional[T5Gemma2TextConfig] = None,
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple[jnp.ndarray, float]:
    rope_params = config.rope_parameters[layer_type]
    factor = rope_params["factor"]
    base = rope_params["rope_theta"]

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    inv_freq /= factor
    return inv_freq, attention_factor

ROPE_INIT_FUNCTIONS_FLAX = {
    "default": flax_compute_default_rope_parameters,
    "linear": flax_compute_linear_rope_parameters,
}

class FlaxT5Gemma2RotaryEmbedding(linen.Module):
    config: T5Gemma2TextConfig

    @linen.compact
    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray,
                    layer_type: Optional[str] = None) -> tuple[jnp.ndarray, jnp.ndarray]:

        rope_params = self.config.rope_parameters[layer_type]
        rope_type = rope_params["rope_type"] if rope_params is not None else "default"

        rope_init_fn = flax_compute_default_rope_parameters
        if rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS_FLAX[rope_type]

        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        if rope_type == "default":
            dim = head_dim
        elif rope_type == "linear":
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
            dim = int(head_dim * partial_rotary_factor)
        else:
            raise NotImplementedError
        inv_freq_shape = (dim // 2,)

        inv_freq = self.param(
            f"{layer_type}_inv_freq",
            lambda key, shape, dtype=jnp.float32: rope_init_fn(self.config, layer_type=layer_type)[0].astype(dtype),
            inv_freq_shape,
        )

        self.param(
            f"{layer_type}_original_inv_freq",
            lambda key, shape, dtype=jnp.float32: rope_init_fn(self.config, layer_type=layer_type)[0].astype(dtype),
            inv_freq_shape,
        )

        attention_scaling = self.param(
            f"{layer_type}_attention_scaling",
            lambda key, shape, dtype=jnp.float32: jnp.asarray(rope_init_fn(self.config, layer_type=layer_type)[1], dtype=dtype),
            (),
        )

        inv_freq_expanded = jnp.expand_dims(inv_freq, axis=(0, 2)).astype(jnp.float32)
        inv_freq_expanded = jnp.broadcast_to(inv_freq_expanded, (position_ids.shape[0], inv_freq.shape[0], 1))
        if position_ids.ndim == 2:
            position_ids_expanded = jnp.expand_dims(position_ids, axis=1).astype(jnp.float32)
        else:
            position_ids_expanded = position_ids.astype(jnp.float32)

        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb) * attention_scaling
        sin = jnp.sin(emb) * attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)

def flax_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def flax_apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, position_ids=None, unsqueeze_dim: int = 1) -> tuple[jnp.ndarray, jnp.ndarray]:
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (flax_rotate_half(q) * sin)
    k_embed = (k * cos) + (flax_rotate_half(k) * sin)
    return q_embed, k_embed


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
    softcap: Optional[float] = None,
    dropout: float = 0.0,
    deterministic: bool = False,
    rng: Optional[jax.Array] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = flax_repeat_kv(key, num_key_value_groups)
    value_states = flax_repeat_kv(value, num_key_value_groups)

    attn_weights = jnp.matmul(query, key_states.transpose(0, 1, 3, 2)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = jnp.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = linen.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(query.dtype)
    attn_weights = linen.Dropout(rate=dropout)(attn_weights, deterministic = deterministic, rng = rng)
    attn_output = jnp.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(0, 2, 1, 3)

    return attn_output, attn_weights


class FlaxT5Gemma2SelfAttention(linen.Module):
    config: T5Gemma2TextConfig
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
        scaling = self.config.query_pre_attn_scalar**-0.5

        layer_type = self.config.layer_types[self.layer_idx] if hasattr(self.config, "layer_types") else None
        sliding_window = self.config.sliding_window if layer_type == "sliding_attention" else None

        batch_size, seq_length, _ = hidden_states.shape

        q_proj = linen.Dense(features=num_attention_heads * head_dim, use_bias=self.config.attention_bias, name="q_proj")
        k_proj = linen.Dense(features=num_key_value_heads * head_dim, use_bias=self.config.attention_bias, name="k_proj")
        v_proj = linen.Dense(features=num_key_value_heads * head_dim, use_bias=self.config.attention_bias, name="v_proj")

        q_norm = FlaxT5Gemma2RMSNorm(head_dim, eps=self.config.rms_norm_eps, name="q_norm")
        k_norm = FlaxT5Gemma2RMSNorm(head_dim, eps=self.config.rms_norm_eps, name="k_norm")

        query_states = q_proj(hidden_states).reshape(batch_size, seq_length, num_attention_heads, head_dim)
        key_states = k_proj(hidden_states).reshape(batch_size, seq_length, num_key_value_heads, head_dim)
        value_states = v_proj(hidden_states).reshape(batch_size, seq_length, num_key_value_heads, head_dim)

        query_states = q_norm(query_states)
        key_states = k_norm(key_states)

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
            scaling=scaling,
            dropout=0.0 if deterministic else self.config.attention_dropout,
            deterministic=deterministic,
            rng=rng,
            softcap=self.config.attn_logit_softcapping,
            sliding_window=sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        o_proj = linen.Dense(features=self.config.hidden_size, use_bias=self.config.attention_bias, name="o_proj")
        attn_output = o_proj(attn_output)

        return attn_output, attn_weights

class FlaxT5Gemma2TextScaledWordEmbedding(linen.Module):
    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    embed_scale: float = 1.0
    eoi_token_index: int = 256_000

    @linen.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        input_embeddings = linen.Embed(self.num_embeddings, self.embedding_dim, name="weight")(input_ids) * self.embed_scale
        eoi_embedding = self.param('eoi_embedding', linen.initializers.zeros, (self.embedding_dim,))
        return jnp.where((input_ids == self.eoi_token_index)[..., None], eoi_embedding, input_embeddings)

class FlaxT5Gemma2EncoderLayer(linen.Module):
    config: T5Gemma2TextConfig
    layer_idx: int

    @linen.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray],
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> jnp.ndarray:

        rng_attn, rng_resid_attn, rng_mlp, rng_resid_mlp = (None, None, None, None)
        if not deterministic and rng is not None:
            rng_attn, rng_resid_attn, rng_mlp, rng_resid_mlp = jax.random.split(rng, 4)

        self_attn = FlaxT5Gemma2SelfAttention(self.config, layer_idx=self.layer_idx, name="self_attn")
        pre_self_attn_layernorm = FlaxT5Gemma2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, name="pre_self_attn_layernorm")
        post_self_attn_layernorm = FlaxT5Gemma2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, name="post_self_attn_layernorm")

        mlp = FlaxT5Gemma2MLP(self.config, name="mlp")
        pre_feedforward_layernorm = FlaxT5Gemma2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, name="pre_feedforward_layernorm")
        post_feedforward_layernorm = FlaxT5Gemma2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, name="post_feedforward_layernorm")

        dropout = linen.Dropout(rate=self.config.dropout_rate)

        residual = hidden_states
        hidden_states = pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            deterministic=deterministic,
            rng=rng_attn,
            **kwargs
        )
        hidden_states = post_self_attn_layernorm(hidden_states)
        hidden_states = residual + dropout(hidden_states, deterministic=deterministic, rng=rng_resid_attn)

        residual = hidden_states
        hidden_states = pre_feedforward_layernorm(hidden_states)
        hidden_states = mlp(hidden_states, deterministic=deterministic, rng=rng_mlp)
        hidden_states = post_feedforward_layernorm(hidden_states)
        hidden_states = residual + dropout(hidden_states, deterministic=deterministic, rng=rng_resid_mlp)

        return hidden_states

@dataclass
class FlaxBaseModelOutput(ModelOutput):

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None

class FlaxT5Gemma2Encoder(linen.Module):
    config: T5Gemma2EncoderConfig
    eoi_token_index: int = 256_000

    @linen.compact
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rng: Optional[jax.Array] = None,
    ) -> FlaxBaseModelOutput:

        text_config = self.config.text_config

        embed_tokens = FlaxT5Gemma2TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            self.config.pad_token_id,
            embed_scale=text_config.hidden_size**0.5,
            eoi_token_index=self.eoi_token_index,
            name="embed_tokens"
        )
        norm = FlaxT5Gemma2RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps, name="norm")
        dropout = linen.Dropout(text_config.dropout_rate)
        rotary_emb = FlaxT5Gemma2RotaryEmbedding(text_config, name="rotary_emb")

        rng_drop_in, rng_drop_out, rng_stream = None, None, None
        if not deterministic and rng is not None:
            rng_drop_in, rng_drop_out, rng_stream = jax.random.split(rng, 3)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids)

        if position_ids is None:
            position_ids = jnp.arange(0, inputs_embeds.shape[1], dtype=jnp.int32).reshape(1, -1)

        if attention_mask is None:
            attention_mask = jnp.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]), dtype=jnp.bool_)

        def make_4d_mask(mask_2d, sliding_window=None):
            batch, seq_len = mask_2d.shape
            mask_4d = mask_2d[:, None, None, :]
            mask_4d = jnp.where(mask_4d > 0, 0, jnp.finfo(jnp.float32).min)
            mask_4d = jnp.broadcast_to(mask_4d, (batch, 1, seq_len, seq_len))

            if sliding_window is not None:

                left_window = (sliding_window + 1) // 2
                right_window = (sliding_window) // 2 + 1

                idx = jnp.arange(seq_len)
                q_idx = idx[:, None]
                k_idx = idx[None, :]
                dist = q_idx - k_idx

                left_mask = (dist >= 0) & (dist < left_window)
                right_mask = (dist < 0) & (-dist < right_window)
                window_mask = left_mask | right_mask
                window_mask = window_mask[None, None, :, :]
                mask_4d = jnp.where(window_mask, mask_4d, jnp.finfo(jnp.float32).min)

            return mask_4d

        self_attn_mask_mapping = {
            "full_attention": make_4d_mask(attention_mask),
            "sliding_attention": make_4d_mask(attention_mask, sliding_window=text_config.sliding_window)
        }

        hidden_states = inputs_embeds

        position_embeddings = {}
        for layer_type in text_config.layer_types:
            position_embeddings[layer_type] = rotary_emb(hidden_states, position_ids, layer_type)

        hidden_states = dropout(hidden_states, deterministic=deterministic, rng=rng_drop_in)

        for i in range(text_config.num_hidden_layers):
            layer_module = FlaxT5Gemma2EncoderLayer(text_config, layer_idx=i, name=f"layers_{i}")

            layer_type = text_config.layer_types[i] if hasattr(text_config, "layer_types") else "full_attention"
            rng_layer = None
            if not deterministic and rng_stream is not None:
                rng_stream, rng_layer = jax.random.split(rng_stream)

            hidden_states = layer_module(
                hidden_states,
                position_embeddings[layer_type],
                self_attn_mask_mapping[layer_type],
                position_ids=position_ids,
                deterministic=deterministic,
                rng=rng_layer
            )

        hidden_states = norm(hidden_states)
        hidden_states = dropout(hidden_states, deterministic=deterministic, rng=rng_drop_out)

        return FlaxBaseModelOutput(last_hidden_state = hidden_states)


__all__ = [
    "FlaxT5Gemma2Encoder",
]

def convert_pytorch_to_flax_params_t5gemma2(torch_model):
    flax_params = {'params': {}}

    flax_params['params']['embed_tokens'] = {
        'eoi_embedding': torch_model.embed_tokens.eoi_embedding.detach().numpy(),
        'weight':{'embedding':torch_model.embed_tokens.weight.detach().numpy()}
    }
    flax_params['params']['rotary_emb'] = {
        'full_attention_inv_freq': torch_model.rotary_emb.full_attention_inv_freq.detach().numpy(),
        'full_attention_original_inv_freq': torch_model.rotary_emb.full_attention_original_inv_freq.detach().numpy(),
        'full_attention_attention_scaling': np.array(torch_model.rotary_emb.full_attention_attention_scaling),
        'sliding_attention_inv_freq': torch_model.rotary_emb.sliding_attention_inv_freq.detach().numpy(),
        'sliding_attention_original_inv_freq': torch_model.rotary_emb.sliding_attention_original_inv_freq.detach().numpy(),
        'sliding_attention_attention_scaling': np.array(torch_model.rotary_emb.sliding_attention_attention_scaling)
    }
    for i, layer in enumerate(torch_model.layers):
        layer_key = f'layers_{i}'
        flax_params['params'][layer_key] = {
            'pre_self_attn_layernorm': {
                'weight': layer.pre_self_attn_layernorm.weight.detach().numpy()
            },
            'post_self_attn_layernorm': {
                'weight': layer.post_self_attn_layernorm.weight.detach().numpy()
            },
            'self_attn': {
                'q_proj': {'kernel': layer.self_attn.q_proj.weight.T.detach().numpy()},
                'k_proj': {'kernel': layer.self_attn.k_proj.weight.T.detach().numpy()},
                'v_proj': {'kernel': layer.self_attn.v_proj.weight.T.detach().numpy()},
                'o_proj': {'kernel': layer.self_attn.o_proj.weight.T.detach().numpy()},
                'q_norm': {'weight': layer.self_attn.q_norm.weight.detach().numpy()},
                'k_norm': {'weight': layer.self_attn.k_norm.weight.detach().numpy()},
            },
            'pre_feedforward_layernorm': {
                'weight': layer.pre_feedforward_layernorm.weight.detach().numpy()
            },
            'post_feedforward_layernorm': {
                'weight': layer.post_feedforward_layernorm.weight.detach().numpy()
            },
            'mlp': {
                'gate_proj': {'kernel': layer.mlp.gate_proj.weight.T.detach().numpy()},
                'up_proj': {'kernel': layer.mlp.up_proj.weight.T.detach().numpy()},
                'down_proj': {'kernel': layer.mlp.down_proj.weight.T.detach().numpy()},
            }
        }

    flax_params['params']['norm'] = {
        'weight': torch_model.norm.weight.detach().numpy()
    }
    return flax_params

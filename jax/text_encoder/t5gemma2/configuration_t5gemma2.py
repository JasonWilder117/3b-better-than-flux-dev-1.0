from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.utils import logging
from transformers.models.siglip import SiglipVisionConfig


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


__all__ = ["T5Gemma2Config", "T5Gemma2TextConfig", "T5Gemma2EncoderConfig", "T5Gemma2DecoderConfig"]

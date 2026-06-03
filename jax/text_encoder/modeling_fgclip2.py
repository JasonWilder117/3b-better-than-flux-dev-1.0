import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from torchvision.ops import roi_align

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, can_return_tuple, filter_out_non_signature_kwargs
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Fgclip2TextConfig(PretrainedConfig):

    model_type = "fgclip2_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        projection_size=None,
        keep_len=20,
        longtext_len=196,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.projection_size = projection_size if projection_size is not None else hidden_size
        self.keep_len = keep_len
        self.longtext_len = longtext_len


class Fgclip2VisionConfig(PretrainedConfig):

    model_type = "fgclip2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_patches = num_patches


class Fgclip2Config(PretrainedConfig):

    model_type = "fgclip2"
    sub_configs = {"text_config": Fgclip2TextConfig, "vision_config": Fgclip2VisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `Fgclip2TextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `Fgclip2VisionConfig` with default values.")

        self.text_config = Fgclip2TextConfig(**text_config)
        self.vision_config = Fgclip2VisionConfig(**vision_config)

        self.initializer_factor = 1.0

@dataclass
class Fgclip2VisionOutput(ModelOutput):

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class Fgclip2TextOutput(ModelOutput):

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class Fgclip2Output(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class Fgclip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Fgclip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Fgclip2Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Fgclip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Fgclip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Union[Fgclip2VisionConfig, Fgclip2TextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Fgclip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Fgclip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Fgclip2Encoder(nn.Module):

    def __init__(self, config: Fgclip2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Fgclip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


class Fgclip2VisionTransformer(nn.Module):
    def __init__(self, config: Fgclip2VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Fgclip2VisionEmbeddings(config)
        self.encoder = Fgclip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = Fgclip2MultiheadAttentionPoolingHead(config)

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        if attention_mask is not None and self.config._attn_implementation != "flash_attention_2":
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)

    tensor.erfinv_()

    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


class Fgclip2PreTrainedModel(PreTrainedModel):
    config: Fgclip2Config
    base_model_prefix = "fgclip2"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Fgclip2TextEmbeddings",
        "Fgclip2VisionEmbeddings",
        "Fgclip2EncoderLayer",
        "Fgclip2MultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": Fgclip2EncoderLayer,
        "attentions": Fgclip2Attention,
    }

    def _init_weights(self, module):
        if isinstance(module, Fgclip2VisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, Fgclip2Config)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, Fgclip2Attention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, Fgclip2MLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, Fgclip2MultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, Fgclip2Model):
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Fgclip2TextEmbeddings(nn.Module):
    def __init__(self, config: Fgclip2TextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        keep_len = config.keep_len
        longtext_len = config.longtext_len

        self.position_embedding_res = nn.Embedding(longtext_len, embed_dim)
        self.position_embedding_ori = nn.Embedding(longtext_len, embed_dim)

        self.mask1 = torch.zeros([longtext_len, 1])
        self.mask1[:keep_len, :] = 1
        self.mask2 = torch.zeros([longtext_len, 1])
        self.mask2[keep_len:, :] = 1

        self.register_buffer("position_ids", torch.arange(longtext_len).expand((1, -1)), persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_short_position_ids: Optional[bool] = True,
    ) -> torch.Tensor:

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        if use_short_position_ids:
            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings
        else:
            position_embeddings_res = self.position_embedding_res(position_ids)
            position_embeddings_ori = self.position_embedding_ori(position_ids)
            embeddings = (
                inputs_embeds
                + (position_embeddings_ori * self.mask1.to(inputs_embeds.device))
                .type(inputs_embeds.dtype)
                .to(inputs_embeds.device)
                + (position_embeddings_res * self.mask2.to(inputs_embeds.device))
                .type(inputs_embeds.dtype)
                .to(inputs_embeds.device)
            )

        return embeddings


class Fgclip2TextTransformer(nn.Module):
    def __init__(self, config: Fgclip2TextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = Fgclip2TextEmbeddings(config)
        self.encoder = Fgclip2Encoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, config.projection_size)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        walk_type: str = "short",
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        walk_type = walk_type.lower()
        if walk_type not in ["short", "box", "long"]:
            raise ValueError(f"Invalid `walk_type`: {walk_type}. Must be one of 'short', 'box', 'long'.")

        walk_short = walk_type == "short"
        walk_box = walk_type == "box"
        walk_long = walk_type == "long"

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, use_short_position_ids=(not walk_long)
        )
        # FG-CLIP2 uses bidirectional text attention; do not create CLIP-style causal masks.
        uses_flash_attention = "flash" in self.config._attn_implementation
        if uses_flash_attention:
            attention_mask = None
        elif attention_mask is not None and not uses_flash_attention:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        # Match FG-CLIP2 pooling: use the final token hidden state even when it is padding.
        pooled_output = last_hidden_state[:, -1, :]
        if walk_short == True:
            assert walk_box == False
            assert walk_long == False
            temp_pool_out = []
            for i in range(pooled_output.shape[0]):
                temp_pool_out.append(self.head(pooled_output[i : i + 1]))
            pooled_output = torch.cat(temp_pool_out, dim=0)
        if walk_box == True:
            assert walk_short == False
            assert walk_long == False
            pooled_output = pooled_output
        if walk_long == True:
            assert walk_short == False
            assert walk_box == False
            pooled_output = pooled_output
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class Fgclip2TextModel(Fgclip2PreTrainedModel):
    config: Fgclip2TextConfig

    def __init__(self, config: Fgclip2TextConfig):
        super().__init__(config)
        self.text_model = Fgclip2TextTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        walk_type: str = "short",
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            walk_type=walk_type,
            **kwargs,
        )


class Fgclip2MultiheadAttentionPoolingHead(nn.Module):

    def __init__(self, config: Fgclip2VisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Fgclip2MLP(config)
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        group_size = self.num_heads
        outputs = []
        for i in range(batch_size):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            out_i = self.attention(
                probe[i : i + 1],
                hidden_state[i : i + 1],
                hidden_state[i : i + 1],
                attn_mask=attention_mask[start_idx:end_idx] if attention_mask is not None else None,
            )[0]
            outputs.append(out_i)

        hidden_state = torch.cat(outputs, dim=0)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)

        temp_outs = []
        for k in range(batch_size):
            out_k = self.mlp(hidden_state[k : k + 1])
            temp_outs.append(out_k)
        hidden_state = residual + torch.cat(temp_outs, dim=0)

        return hidden_state[:, 0]


class Fgclip2VisionModel(Fgclip2PreTrainedModel):
    config: Fgclip2VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Fgclip2VisionConfig):
        super().__init__(config)

        self.vision_model = Fgclip2VisionTransformer(config)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class Fgclip2Model(Fgclip2PreTrainedModel):
    config: Fgclip2Config

    def __init__(self, config: Fgclip2Config):
        super().__init__(config)

        if not isinstance(config.text_config, Fgclip2TextConfig):
            raise TypeError(
                "config.text_config is expected to be of type Fgclip2TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, Fgclip2VisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type Fgclip2VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        text_model = Fgclip2TextModel._from_config(text_config)
        vision_model = Fgclip2VisionModel._from_config(vision_config)

        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self.dense_feature_head = Fgclip2MultiheadAttentionPoolingHead(vision_config)
        self.embed_dim = text_config.hidden_size
        self.longtext_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.boxtext_head = nn.Linear(self.embed_dim, self.embed_dim)

        self.post_init()

    @filter_out_non_signature_kwargs()
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        walk_type: str = "short",
    ) -> torch.FloatTensor:

        walk_type = walk_type.lower()

        if walk_type not in ["short", "box", "long"]:
            raise ValueError(f"Invalid `walk_type`: {walk_type}. Must be one of 'short', 'box', 'long'.")

        walk_short = walk_type == "short"
        walk_box = walk_type == "box"
        walk_long = walk_type == "long"

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            walk_type=walk_type,
        )

        if walk_short:
            pooled_output = text_outputs.pooler_output

        if walk_box:
            pooled_output = self.boxtext_head(text_outputs.pooler_output)

        if walk_long:
            pooled_output = self.longtext_head(text_outputs.pooler_output)

        return pooled_output

    @filter_out_non_signature_kwargs()
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        pooled_output = vision_outputs.pooler_output

        return pooled_output

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        walk_type: str = "short",
    ) -> Fgclip2Output:
        walk_type = walk_type.lower()

        if walk_type not in ["short", "box", "long"]:
            raise ValueError(f"Invalid `walk_type`: {walk_type}. Must be one of 'short', 'box', 'long'.")

        walk_short = walk_type == "short"
        walk_box = walk_type == "box"
        walk_long = walk_type == "long"

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            walk_type=walk_type,
        )

        image_embeds = vision_outputs.pooler_output

        if walk_short:
            text_embeds = text_outputs.pooler_output

        if walk_box:
            text_embeds = self.boxtext_head(text_outputs.pooler_output)

        if walk_long:
            text_embeds = self.longtext_head(text_outputs.pooler_output)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        return Fgclip2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    @filter_out_non_signature_kwargs()
    def get_image_dense_feature(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

        probe = vision_outputs.last_hidden_state
        hidden_state = vision_outputs.last_hidden_state
        attention_mask = pixel_attention_mask

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.dense_feature_head.num_heads, 1, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.dense_feature_head.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[
            0
        ]
        residual = hidden_state
        hidden_state = self.dense_feature_head.layernorm(hidden_state)
        hidden_state = residual + self.dense_feature_head.mlp(hidden_state)
        feature_map = hidden_state

        return feature_map

    @filter_out_non_signature_kwargs()
    def get_image_region_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        image_sizes: Optional[list[tuple]] = None,
        region_infos: Optional[list[list[list[float]]]] = None,
    ) -> list[torch.FloatTensor]:
        if region_infos is None or len(region_infos) == 0:
            return []

        dense_feature_map = self.get_image_dense_feature(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        bs, _, hidden_dim = dense_feature_map.shape

        all_region_features = []

        for i in range(bs):
            h, w = spatial_shapes[i].tolist()
            img_h, img_w = image_sizes[i]
            bboxes = region_infos[i]

            if not bboxes:
                all_region_features.append(torch.empty(0, hidden_dim, device=dense_feature_map.device))
                continue

            num_valid = h * w
            feat_seq = dense_feature_map[i, :num_valid]
            feat_map = feat_seq.view(h, w, hidden_dim).permute(2, 0, 1).unsqueeze(0)

            rois = []
            for x1, y1, x2, y2 in bboxes:
                nx1 = (x1 / img_w) * w
                ny1 = (y1 / img_h) * h
                nx2 = (x2 / img_w) * w
                ny2 = (y2 / img_h) * h
                rois.append([0, nx1, ny1, nx2, ny2])
            rois_tensor = torch.tensor(rois, dtype=torch.float32, device=feat_map.device)

            pooled = roi_align(
                input=feat_map,
                boxes=rois_tensor,
                output_size=(1, 1),
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )
            region_feats = pooled.squeeze(-1).squeeze(-1)

            all_region_features.append(region_feats)

        return all_region_features


__all__ = ["Fgclip2Model", "Fgclip2PreTrainedModel", "Fgclip2TextModel", "Fgclip2VisionModel"]

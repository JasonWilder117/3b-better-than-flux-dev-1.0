from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen

import torch

from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs

ACT2FN_FLAX = {'gelu': linen.gelu, 'silu': linen.silu, 'gelu_pytorch_tanh': lambda x: linen.gelu(x, approximate=True)}
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

@dataclass
class FlaxBaseModelOutput(ModelOutput):

    last_hidden_state: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray, ...]] = None
    attentions: Optional[tuple[jnp.ndarray, ...]] = None

@dataclass
class FlaxFgclip2TextOutput(ModelOutput):

    text_embeds: Optional[jnp.ndarray] = None
    last_hidden_state: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray, ...]] = None
    attentions: Optional[tuple[jnp.ndarray, ...]] = None

@dataclass
class FlaxBaseModelOutputWithPooling(ModelOutput):

    last_hidden_state: Optional[jnp.ndarray] = None
    pooler_output: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray, ...]] = None
    attentions: Optional[tuple[jnp.ndarray, ...]] = None


def flax_eager_attention_forward(
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

    attn_weights = jnp.matmul(query, key.transpose(0, 1, 3, 2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = linen.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(query.dtype)
    attn_weights = linen.Dropout(rate=dropout)(attn_weights, deterministic = deterministic, rng = rng)
    attn_output = jnp.matmul(attn_weights, value)
    attn_output = attn_output.transpose(0, 2, 1, 3)

    return attn_output, attn_weights

class FlaxFgclip2Attention(linen.Module):
    config: Fgclip2TextConfig

    @linen.compact
    def __call__(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None,
        **kwargs) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:

        embed_dim = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = embed_dim // num_heads

        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:"
                f" {num_heads})."
            )

        scale = head_dim ** -0.5

        k_proj = linen.Dense(features = embed_dim, name = 'k_proj')
        v_proj = linen.Dense(features = embed_dim, name = 'v_proj')
        q_proj = linen.Dense(features = embed_dim, name = 'q_proj')
        out_proj = linen.Dense(features = embed_dim, name = 'out_proj')

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = q_proj(hidden_states)
        keys = k_proj(hidden_states)
        values = v_proj(hidden_states)

        queries = queries.reshape(batch_size, seq_length, num_heads, head_dim)
        queries = jnp.transpose(queries, (0, 2, 1, 3))
        keys = keys.reshape(batch_size, seq_length, num_heads, head_dim)
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = values.reshape(batch_size, seq_length, num_heads, head_dim)
        values = jnp.transpose(values, (0, 2, 1, 3))

        attention_interface: Callable = flax_eager_attention_forward

        attn_output, attn_weights = attention_interface(
            queries,
            keys,
            values,
            attention_mask,
            scaling = scale,
            dropout = 0.0 if deterministic else self.config.attention_dropout,
            deterministic = deterministic,
            rng = rng
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        attn_output = out_proj(attn_output)

        return attn_output, attn_weights

class FlaxFgclip2MLP(linen.Module):
    config: Fgclip2TextConfig

    @linen.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:

        activation_fn = ACT2FN_FLAX[self.config.hidden_act]
        fc1 = linen.Dense(features = self.config.intermediate_size, name = 'fc1')
        fc2 = linen.Dense(features = self.config.hidden_size, name = 'fc2')

        hidden_states = fc1(hidden_states)
        hidden_states = activation_fn(hidden_states)
        hidden_states = fc2(hidden_states)

        return hidden_states


class FlaxFgclip2EncoderLayer(linen.Module):
    config: Fgclip2TextConfig

    @linen.compact
    def __call__(self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> jnp.ndarray:

        layer_norm1 = linen.LayerNorm(epsilon = self.config.layer_norm_eps, name = 'layer_norm1')
        self_attn = FlaxFgclip2Attention(self.config, name = 'self_attn')
        layer_norm2 = linen.LayerNorm(epsilon = self.config.layer_norm_eps, name = 'layer_norm2')
        mlp = FlaxFgclip2MLP(self.config, name = 'mlp')
        residual = hidden_states

        hidden_states = layer_norm1(hidden_states)
        hidden_states, _ = self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            deterministic = deterministic,
            rng = rng,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer_norm2(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



class FlaxFgclip2Encoder(linen.Module):
    config: Fgclip2TextConfig

    @linen.compact
    def __call__(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> FlaxBaseModelOutput:

        hidden_states = inputs_embeds
        for i in range(self.config.num_hidden_layers):
            encoder_layer = FlaxFgclip2EncoderLayer(self.config, name = f'layer_{i}')
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                deterministic = deterministic,
                rng = rng,
                **kwargs,
            )

        return FlaxBaseModelOutput(last_hidden_state=hidden_states)


class FlaxFgclip2PreTrainedModel(PreTrainedModel):
    config: Fgclip2Config
    base_model_prefix = "fgclip2"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Fgclip2TextEmbeddings",
        "Fgclip2EncoderLayer",
    ]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": FlaxFgclip2EncoderLayer,
        "attentions": FlaxFgclip2Attention,
    }


class FlaxFgclip2TextEmbeddings(linen.Module):
    config: Fgclip2TextConfig

    @linen.compact
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        use_short_position_ids: Optional[bool] = True,
    ) -> jnp.ndarray:
        embed_dim = self.config.hidden_size

        token_embedding = linen.Embed(num_embeddings = self.config.vocab_size,
                features = embed_dim, name = "token_embedding")

        position_embedding = linen.Embed(num_embeddings = self.config.max_position_embeddings,
                features = embed_dim, name = "position_embedding")

        keep_len = self.config.keep_len
        longtext_len = self.config.longtext_len

        position_embedding_res = linen.Embed(num_embeddings = longtext_len,
                features = embed_dim, name = 'position_embedding_res')
        position_embedding_ori = linen.Embed(num_embeddings = longtext_len,
                features = embed_dim, name = 'position_embedding_ori')

        mask1 = jnp.zeros((longtext_len, 1)).at[:keep_len, :].set(1)
        mask2 = jnp.zeros((longtext_len, 1)).at[keep_len:, :].set(1)

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = jnp.expand_dims(jnp.arange(longtext_len,),axis=0)[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = token_embedding(input_ids)

        if use_short_position_ids:
            position_embeddings = position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings
        else:
            position_embeddings_res = position_embedding_res(position_ids)
            position_embeddings_ori = position_embedding_ori(position_ids)
            embeddings = (
                inputs_embeds
                + (position_embeddings_ori * mask1[:seq_length,:].astype(inputs_embeds.dtype))
                + (position_embeddings_res * mask2[:seq_length,:].astype(inputs_embeds.dtype))
            )

        return embeddings


class FlaxFgclip2TextTransformer(linen.Module):
    config: Fgclip2TextConfig
    dtype: jnp.dtype = jnp.float32
    @linen.compact
    def __call__(self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        walk_type: str = "short",
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> FlaxBaseModelOutputWithPooling:


        embed_dim = self.config.hidden_size
        embeddings = FlaxFgclip2TextEmbeddings(self.config, name = 'embeddings')
        encoder = FlaxFgclip2Encoder(self.config, name = 'encoder')
        final_layer_norm = linen.LayerNorm(epsilon = self.config.layer_norm_eps, name = 'final_layer_norm')

        head = linen.Dense(embed_dim, self.config.projection_size, name = 'head')

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        walk_type = walk_type.lower()
        if walk_type not in ["short", "box", "long"]:
            raise ValueError(f"Invalid `walk_type`: {walk_type}. Must be one of 'short', 'box', 'long'.")

        walk_short = walk_type == "short"
        walk_box = walk_type == "box"
        walk_long = walk_type == "long"

        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        hidden_states = embeddings(
            input_ids=input_ids, position_ids=position_ids, use_short_position_ids=(not walk_long)
        )
        # FG-CLIP2 uses bidirectional text attention; do not create CLIP-style causal masks.
        batch_size, seq_len = input_shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        attention_mask = attention_mask.astype(jnp.bool_)
        attention_mask = jnp.broadcast_to(attention_mask[:,jnp.newaxis, jnp.newaxis,:],
                                            (batch_size, 1, seq_len, seq_len))
        attention_mask = jnp.where(attention_mask, 0.0, jnp.finfo(self.dtype).min)

        encoder_outputs: FlaxBaseModelOutput = encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            deterministic = deterministic,
            rng = rng,
            **kwargs,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = final_layer_norm(last_hidden_state)
        # The model uses the last token's hidden state, which may be padding.
        pooled_output = last_hidden_state[:, -1, :]
        if walk_short == True:
            assert walk_box == False
            assert walk_long == False
            temp_pool_out = []
            for i in range(pooled_output.shape[0]):
                temp_pool_out.append(head(pooled_output[i : i + 1]))
            pooled_output = jnp.concatenate(temp_pool_out, axis=0)
        if walk_box == True:
            assert walk_short == False
            assert walk_long == False
            pooled_output = pooled_output
        if walk_long == True:
            assert walk_short == False
            assert walk_box == False
            pooled_output = pooled_output
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class FlaxFgclip2TextModel(linen.Module):
    config: Fgclip2TextConfig

    @linen.compact
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        walk_type: str = "short",
        deterministic: bool = True,
        rng: Optional[jnp.ndarray] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> FlaxBaseModelOutputWithPooling:

        text_model = FlaxFgclip2TextTransformer(self.config, name = 'text_model')

        return text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            walk_type=walk_type,
            deterministic = deterministic,
            rng = rng,
            **kwargs,
        )

__all__ = ["FlaxFgclip2PreTrainedModel", "FlaxFgclip2TextModel"]

def convert_pytorch_to_flax_params_fgclip2(torch_model):
    flax_params = {'params': {'text_model':{'encoder':{}}}}
    for i, layer in enumerate(torch_model.text_model.encoder.layers):
        layer_key = f'layer_{i}'
        flax_params['params']['text_model']['embeddings'] = {
            "token_embedding": {'embedding': torch_model.text_model.embeddings.token_embedding.weight.detach().numpy()},
            "position_embedding": {'embedding': torch_model.text_model.embeddings.position_embedding.weight.detach().numpy()},
            "position_embedding_res": {'embedding': torch_model.text_model.embeddings.position_embedding_res.weight.detach().numpy()},
            "position_embedding_ori": {'embedding': torch_model.text_model.embeddings.position_embedding_ori.weight.detach().numpy()}
        }
        flax_params['params']['text_model']['final_layer_norm'] = {
            'scale': torch_model.text_model.final_layer_norm.weight.detach().numpy(),
            'bias': torch_model.text_model.final_layer_norm.bias.detach().numpy(),
        }
        flax_params['params']['text_model']['head'] = {
            'kernel': torch_model.text_model.head.weight.T.detach().numpy(),
            'bias': torch_model.text_model.head.bias.detach().numpy()
        }
        flax_params['params']['text_model']['encoder'][layer_key] = {
            'layer_norm1': {
                'scale': layer.layer_norm1.weight.detach().numpy(),
                'bias': layer.layer_norm1.bias.detach().numpy()
            },
            'self_attn': {
                'k_proj': {'kernel': layer.self_attn.k_proj.weight.T.detach().numpy(),
                        'bias': layer.self_attn.k_proj.bias.detach().numpy()},
                'v_proj': {'kernel': layer.self_attn.v_proj.weight.T.detach().numpy(),
                        'bias': layer.self_attn.v_proj.bias.detach().numpy()},
                'q_proj': {'kernel': layer.self_attn.q_proj.weight.T.detach().numpy(),
                        'bias': layer.self_attn.q_proj.bias.detach().numpy()},
                'out_proj': {'kernel': layer.self_attn.out_proj.weight.T.detach().numpy(),
                        'bias': layer.self_attn.out_proj.bias.detach().numpy()},
            },
            'layer_norm2': {
                'scale': layer.layer_norm2.weight.detach().numpy(),
                'bias': layer.layer_norm2.bias.detach().numpy()
            },
            'mlp': {
                'fc1' : {'kernel': layer.mlp.fc1.weight.T.detach().numpy(),
                    'bias': layer.mlp.fc1.bias.detach().numpy()},
                'fc2' : {'kernel': layer.mlp.fc2.weight.T.detach().numpy(),
                        'bias': layer.mlp.fc2.bias.detach().numpy()}
            }
        }
    return flax_params

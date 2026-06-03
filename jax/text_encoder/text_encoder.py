import jax
import jax.numpy as jnp
import numpy as np
from typing import Literal
import transformers
from transformers import AutoTokenizer
from transformers import FlaxCLIPTextModel
from transformers import FlaxT5EncoderModel

from text_encoder.modeling_flax_qwen3 import FlaxQwen3Model
from text_encoder.modeling_flax_qwen3_vl import FlaxQwen3VLTextModel, convert_pytorch_to_flax_params_qwen3
from text_encoder.modeling_flax_fgclip2 import FlaxFgclip2TextModel, convert_pytorch_to_flax_params_fgclip2
from text_encoder.modeling_fgclip2 import Fgclip2TextModel
from text_encoder.modeling_flax_t5gemma2 import FlaxT5Gemma2Encoder, convert_pytorch_to_flax_params_t5gemma2
from text_encoder.t5gemma2.modeling_t5gemma2 import T5Gemma2ForConditionalGeneration, T5Gemma2Encoder
from gemma.gm import ckpts
from gemma.research import t5gemma
import functools

QWEN_IMAGE_T2I_SYSTEM_PROMPT = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
QWEN_IMAGE_SYSTEM_PROMPT_TEXT_ENCODER_TYPES = {
    "qwen3-vl-2b-instruct_system_prompt",
    "qwen3-vl-4b-instruct_system_prompt",
}


def is_qwen_image_system_prompt_text_encoder_type(text_encoder_type: str) -> bool:
    return text_encoder_type in QWEN_IMAGE_SYSTEM_PROMPT_TEXT_ENCODER_TYPES


class QwenImageChatTemplateTokenizer:
    """Tokenizer wrapper matching Qwen-Image-style text conditioning."""

    def __init__(self, tokenizer, max_input_length: int, system_prompt: str, tokenizer_kwargs: dict):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.system_prompt = system_prompt
        self.tokenizer_kwargs = dict(tokenizer_kwargs)
        self.drop_prefix_token_len = self._compute_drop_prefix_token_len()

    def _messages(self, prompt: str):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _render_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            self._messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )

    def _compute_drop_prefix_token_len(self) -> int:
        marker = "<|qwen_image_user_prompt|>"
        rendered = self._render_prompt(marker)
        marker_start = rendered.index(marker)
        prefix = rendered[:marker_start]
        return len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])

    def __call__(self, text_prompts, **kwargs):
        if isinstance(text_prompts, str):
            prompts = [text_prompts]
        else:
            prompts = list(text_prompts)

        call_kwargs = dict(self.tokenizer_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs["max_length"] = self.max_input_length + self.drop_prefix_token_len
        call_kwargs.setdefault("add_special_tokens", False)
        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        return self.tokenizer(rendered_prompts, **call_kwargs)


def _drop_prefix(hidden_states, attention_mask=None, drop_prefix_token_len: int = 0):
    if not drop_prefix_token_len:
        return hidden_states, attention_mask
    hidden_states = hidden_states[:, drop_prefix_token_len:, :]
    if attention_mask is not None:
        attention_mask = attention_mask[:, drop_prefix_token_len:]
    return hidden_states, attention_mask


try:
    jax.default_backend()
except RuntimeError:
    print("No GPU/TPU, use CPU.")
    
def encode_text_encoder(
    text_encoder,
    text_encoder_params,
    text_encoder_type: str,
    input_ids,
    attention_mask,
    drop_prefix_token_len: int = 0,
):
    if text_encoder_type in {"T5Gemma", "T5Gemma_2B_no_IT", "T5Gemma_9B_2B", "T5Gemma_9B_2B_no_IT"}:
        outputs = text_encoder.apply(
            {"params": text_encoder_params},
            input_ids,
            attention_mask,
            method=text_encoder.compute_encoder_activations,
        )
        hidden_states = outputs.activations[-1]
    elif ("qwen3" in text_encoder_type) or ("T5Gemma2" in text_encoder_type):
        encoder_outputs = text_encoder.apply(
            text_encoder_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs[0]
    elif "fg-clip2" in text_encoder_type:
        encoder_outputs = text_encoder.apply(
            text_encoder_params,
            input_ids=input_ids,
            attention_mask=None,
            walk_type="long" if "long" in text_encoder_type else "short",
        )
        hidden_states = encoder_outputs[0]
    else:
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=text_encoder_params,
            train=False,
        )
        hidden_states = outputs[0]
    hidden_states, _ = _drop_prefix(hidden_states, attention_mask, drop_prefix_token_len)
    return jax.lax.stop_gradient(hidden_states)

class TextEncoder:
    def __init__(self, config = None, text_encoder_type: Literal["T5", "T5-small", "T5-base", "T5-large", "T5-xl", "T5-xxl",
                                                    "clip-large", "fg-clip2-short-so400m", "fg-clip2-long-so400m",
                                                    "qwen3-1.7b", "qwen3-4b-instruct-2507",
                                                    "qwen3-vl-2b-instruct", "qwen3-vl-2b-instruct_system_prompt",
                                                    "qwen3-vl-4b-instruct", "qwen3-vl-4b-instruct_system_prompt",
                                                    "T5Gemma", "T5Gemma_2B_no_IT", "T5Gemma_9B_2B", "T5Gemma_9B_2B_no_IT",
                                                    "T5Gemma2-1b", "T5Gemma2-4b"] = "clip-large",
                    text_token_len = None, weight_dtype=jnp.float32, dit_checkpoint_path=None):

        self.config = config
        self.dit_checkpoint_path = dit_checkpoint_path

        if isinstance(text_encoder_type, (list, tuple)):
            self.text_encoder_type = list(text_encoder_type)
            if text_token_len is None or isinstance(text_token_len, (int, np.integer)):
                token_lens = [text_token_len] * len(self.text_encoder_type)
            elif isinstance(text_token_len, (list, tuple)):
                assert len(text_token_len) == len(self.text_encoder_type)
                token_lens = list(text_token_len)
            else:
                raise ValueError("text_token_len must be an int, a sequence, or None.")
            bundles = [
                TextEncoder(
                    config=config,
                    text_encoder_type=enc_type,
                    text_token_len=token_lens[idx],
                    weight_dtype=weight_dtype,
                    dit_checkpoint_path=dit_checkpoint_path,
                )
                for idx, enc_type in enumerate(self.text_encoder_type)
            ]
            self.tokenizer = [bundle.tokenizer for bundle in bundles]
            self.text_encoder = [bundle.text_encoder for bundle in bundles]
            self.text_encoder_params = [bundle.text_encoder_params for bundle in bundles]
            self.hidden_dim = [bundle.hidden_dim for bundle in bundles]
            self.text_token_len = [bundle.text_token_len for bundle in bundles]
            self.drop_prefix_token_len = [bundle.drop_prefix_token_len for bundle in bundles]
            self.weight_dtype = weight_dtype
            return

        self.text_encoder_type = text_encoder_type
        self.drop_prefix_token_len = 0
        
        self.text_encoder_dict = {
            "T5": "DeepFloyd/t5-v1_1-xxl", "T5-small": "google/t5-v1_1-small",
            "T5-base": "google/t5-v1_1-base", "T5-large": "google/t5-v1_1-large",
            "T5-xl": "google/t5-v1_1-xl", "T5-xxl": "google/t5-v1_1-xxl",
            "clip-large": "openai/clip-vit-large-patch14", 
            "fg-clip2-long-so400m": "qihoo360/fg-clip2-so400m",
            "fg-clip2-short-so400m": "qihoo360/fg-clip2-so400m",
            "qwen3-1.7b": "Qwen/Qwen3-1.7B",
            "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
            "qwen3-vl-2b-instruct": "Qwen/Qwen3-VL-2B-Instruct",
            "qwen3-vl-2b-instruct_system_prompt": "Qwen/Qwen3-VL-2B-Instruct",
            "qwen3-vl-4b-instruct": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-vl-4b-instruct_system_prompt": "Qwen/Qwen3-VL-4B-Instruct",
            "T5Gemma": "T5Gemma", "T5Gemma_2B_no_IT": "T5Gemma_2B_no_IT", "T5Gemma_9B_2B": "T5Gemma_9B_2B", "T5Gemma_9B_2B_no_IT": "T5Gemma_9B_2B_no_IT",
            "T5Gemma2-1b": "google/t5gemma-2-1b-1b",
            "T5Gemma2-4b": "google/t5gemma-2-4b-4b"
        }

        self.text_encoder_hidden_dim_dict = {
            "T5": 4096, "T5-small": 512, "T5-base": 768, "T5-large": 1024, 
            "T5-xl": 2048, "T5-xxl": 4096, "clip-large": 768, "fg-clip2-short-so400m": 1152,
            "T5Gemma": 2304, "T5Gemma_2B_no_IT": 2304, "T5Gemma_9B_2B": 3584, "T5Gemma_9B_2B_no_IT": 3584,
            "fg-clip2-long-so400m": 1152,
            "qwen3-1.7b": 2048, "qwen3-4b-instruct-2507": 2560,
            "qwen3-vl-2b-instruct": 2048, "qwen3-vl-2b-instruct_system_prompt": 2048,
            "qwen3-vl-4b-instruct": 2560, "qwen3-vl-4b-instruct_system_prompt": 2560,
            "T5Gemma2-1b": 1152, "T5Gemma2-4b": 2560
        }
        
        self.text_token_len_dict = {
            "T5": 120, "T5-small": 120, "T5-base": 120, "T5-large": 120, 
            "T5-xl": 120, "T5-xxl": 120, "clip-large": 77, "fg-clip2-short-so400m": 64,
            "T5Gemma": 256, "T5Gemma_2B_no_IT": 256, "T5Gemma_9B_2B": 256, "T5Gemma_9B_2B_no_IT": 256,
            "fg-clip2-long-so400m": 196,
            "qwen3-1.7b": 256, "qwen3-4b-instruct-2507": 256,
            "qwen3-vl-2b-instruct": 256, "qwen3-vl-2b-instruct_system_prompt": 256,
            "qwen3-vl-4b-instruct": 256, "qwen3-vl-4b-instruct_system_prompt": 256,
            "T5Gemma2-1b": 256, "T5Gemma2-4b": 256
        }
        
        self.text_token_len = self.text_token_len_dict[self.text_encoder_type
                                ] if text_token_len is None else text_token_len
        
        if text_encoder_type not in self.text_encoder_dict:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model_name = self.text_encoder_dict[self.text_encoder_type]
        self.hidden_dim = self.text_encoder_hidden_dim_dict[self.text_encoder_type]
        self.weight_dtype = weight_dtype

        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):

        print(f"Loading text encoder {self.text_encoder_type} and tokenizer.")
        if self.text_encoder_type == "T5Gemma":
            preset = t5gemma.T5GemmaPreset.GEMMA2_2B_2B
            self.tokenizer = t5gemma.Sampler(model=None, params=None, tokenizer=preset.tokenizer, max_input_length=self.text_token_len)
            self.text_encoder = preset.config.make("transformer")
            """
            Normally, this should be:
            t5gemma_ckpt = preset.get_checkpoint_from_kaggle(
                t5gemma.CKPTType.IT,
                t5gemma.PretrainType.UL2,
            )
            However, the request is replicated on each TPU worker, and Kaggle wouldn't allow it.
            Thus, I saved the checkpoint on local machine and uploaded it to Google Cloud bucket.
            Note that the folder should contain an empty file "commit_success.txt" for the checkpoint to be considered valid by gemma.gm.ckpts.
            """
            t5gemma_ckpt = "gs://path/to/checkpoint"
            # Load text encoder parameters and ensure they are fully addressable on CPU
            self.text_encoder_params = ckpts.load_params(t5gemma_ckpt)
            # Force parameters to CPU to make them fully addressable for replication
            self.text_encoder_params = jax.device_get(self.text_encoder_params)
            del self.text_encoder_params["decoder"]
        elif self.text_encoder_type == "T5Gemma_2B_no_IT":
            preset = t5gemma.T5GemmaPreset.GEMMA2_2B_2B
            self.tokenizer = t5gemma.Sampler(model=None, params=None, tokenizer=preset.tokenizer, max_input_length=self.text_token_len)
            self.text_encoder = preset.config.make("transformer")
            t5gemma_ckpt = "gs://path/to/checkpoint"
            # Load text encoder parameters and ensure they are fully addressable on CPU
            self.text_encoder_params = ckpts.load_params(t5gemma_ckpt)
            # Force parameters to CPU to make them fully addressable for replication
            self.text_encoder_params = jax.device_get(self.text_encoder_params)
            del self.text_encoder_params["decoder"]
        elif self.text_encoder_type == "T5Gemma_9B_2B":
            preset = t5gemma.T5GemmaPreset.GEMMA2_9B_2B
            self.tokenizer = t5gemma.Sampler(model=None, params=None, tokenizer=preset.tokenizer, max_input_length=self.text_token_len)
            self.text_encoder = preset.config.make("transformer")
            t5gemma_ckpt = "gs://path/to/checkpoint"
            # Load text encoder parameters and ensure they are fully addressable on CPU
            self.text_encoder_params = ckpts.load_params(t5gemma_ckpt)
            # Force parameters to CPU to make them fully addressable for replication
            self.text_encoder_params = jax.device_get(self.text_encoder_params)
            del self.text_encoder_params["decoder"]
        elif self.text_encoder_type == "T5Gemma_9B_2B_no_IT":
            preset = t5gemma.T5GemmaPreset.GEMMA2_9B_2B
            self.tokenizer = t5gemma.Sampler(model=None, params=None, tokenizer=preset.tokenizer, max_input_length=self.text_token_len)
            self.text_encoder = preset.config.make("transformer")
            t5gemma_ckpt = "gs://path/to/checkpoint"
            # Load text encoder parameters and ensure they are fully addressable on CPU
            self.text_encoder_params = ckpts.load_params(t5gemma_ckpt)
            # Force parameters to CPU to make them fully addressable for replication
            self.text_encoder_params = jax.device_get(self.text_encoder_params)
            del self.text_encoder_params["decoder"]
        elif "T5Gemma2" in self.text_encoder_type:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer_kwargs = dict(max_length=self.text_token_len,padding="max_length",
                truncation=True, return_attention_mask=True, return_tensors='np')
            self.tokenizer = functools.partial(self.tokenizer, **tokenizer_kwargs)
            pytorch_text_encoder: T5Gemma2Encoder = (T5Gemma2ForConditionalGeneration.from_pretrained(
                                                self.model_name)).model.encoder
            self.text_encoder = FlaxT5Gemma2Encoder(pytorch_text_encoder.config)
            self.text_encoder.params = convert_pytorch_to_flax_params_t5gemma2(pytorch_text_encoder)
            self.text_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(self.weight_dtype), self.text_encoder.params)
            self.text_encoder_params = self.text_encoder.params
            del pytorch_text_encoder
        elif "T5" in self.text_encoder_type:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer_kwargs = dict(max_length=self.text_token_len, padding='max_length',
                truncation=True, return_tensors='np',return_attention_mask=True, add_special_tokens=True)
            self.tokenizer = functools.partial(self.tokenizer, **tokenizer_kwargs)
            self.text_encoder = FlaxT5EncoderModel.from_pretrained(self.model_name, from_pt=True, dtype=self.weight_dtype)
            self.text_encoder_params = self.text_encoder.params
        elif "clip-large" in self.text_encoder_type:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer_kwargs = dict(max_length=self.text_token_len, padding="max_length",
                truncation="longest_first", return_tensors='np')
            self.tokenizer = functools.partial(self.tokenizer, **tokenizer_kwargs)
            self.text_encoder = FlaxCLIPTextModel.from_pretrained(self.model_name, from_pt=True, dtype=self.weight_dtype)
            self.text_encoder_params = self.text_encoder.params
        elif "qwen3-vl" in self.text_encoder_type:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer_kwargs = dict(max_length=self.text_token_len,padding="max_length",
                truncation=True, return_attention_mask=True, return_tensors='np')
            if is_qwen_image_system_prompt_text_encoder_type(self.text_encoder_type):
                self.tokenizer = QwenImageChatTemplateTokenizer(
                    self.tokenizer,
                    max_input_length=self.text_token_len,
                    system_prompt=QWEN_IMAGE_T2I_SYSTEM_PROMPT,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                self.drop_prefix_token_len = self.tokenizer.drop_prefix_token_len
            else:
                self.tokenizer = functools.partial(self.tokenizer, **tokenizer_kwargs)
            pytorch_text_encoder = (
                transformers.Qwen3VLForConditionalGeneration.from_pretrained(self.model_name)
            ).model.language_model
            self.text_encoder = FlaxQwen3VLTextModel(pytorch_text_encoder.config, dtype=self.weight_dtype)
            self.text_encoder.params = convert_pytorch_to_flax_params_qwen3(pytorch_text_encoder)
            self.text_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(self.weight_dtype), self.text_encoder.params)
            self.text_encoder_params = self.text_encoder.params
            del pytorch_text_encoder
        elif "qwen3" in self.text_encoder_type:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer_kwargs = dict(max_length=self.text_token_len,padding="max_length",
                truncation=True, return_attention_mask=True, return_tensors='np')
            self.tokenizer = functools.partial(self.tokenizer, **tokenizer_kwargs)
            pytorch_text_encoder = (transformers.AutoModelForCausalLM.from_pretrained(
                                                    self.model_name)).model
            self.text_encoder = FlaxQwen3Model(pytorch_text_encoder.config, dtype=self.weight_dtype)
            self.text_encoder.params = convert_pytorch_to_flax_params_qwen3(pytorch_text_encoder)
            self.text_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(self.weight_dtype), self.text_encoder.params)
            self.text_encoder_params = self.text_encoder.params
            del pytorch_text_encoder
        elif "fg-clip2" in self.text_encoder_type:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer_kwargs = dict(max_length=self.text_token_len, padding="max_length",
                truncation=True, return_tensors="np")
            self.tokenizer = functools.partial(self.tokenizer, **tokenizer_kwargs)
            pytorch_text_encoder = Fgclip2TextModel.from_pretrained(self.model_name,trust_remote_code=True)
            self.text_encoder = FlaxFgclip2TextModel(pytorch_text_encoder.config)
            self.text_encoder.params = convert_pytorch_to_flax_params_fgclip2(pytorch_text_encoder)
            self.text_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(self.weight_dtype), self.text_encoder.params)
            self.text_encoder_params = self.text_encoder.params
            del pytorch_text_encoder
        else:
            raise ValueError(f"Loading {self.text_encoder_type} model is not implemented.")
                
        print(f"Successfully loaded text encoder {self.text_encoder_type} and tokenizer.")

    def prepare_inputs(self, text_prompts: list[str]):
        tokenized_output = self.tokenizer(text_prompts)
        # print(tokenized_output['input_ids'], tokenized_output.keys())
        return {"input_ids": jnp.array(tokenized_output.input_ids),
                "attention_mask": jnp.array(tokenized_output.attention_mask) if \
                                    tokenized_output.get("attention_mask") is not None else jnp.array(np.full_like(tokenized_output.input_ids, True, dtype=bool))}
            
    def encode(self, text_prompts: list[str]):

        if not isinstance(text_prompts, list):
            raise TypeError("text_prompts should be a list of string")

        inputs = self.prepare_inputs(text_prompts)
        if "fg-clip2-long" in self.text_encoder_type:
            inputs['walk_type'] = 'long'
        elif "fg-clip2-short" in self.text_encoder_type:
            inputs['walk_type'] = 'short'
            
        encoder_outputs = self.text_encoder.apply(
            self.text_encoder.params, **inputs
        )

        hidden_states, attention_mask = _drop_prefix(
            encoder_outputs.last_hidden_state,
            inputs["attention_mask"],
            self.drop_prefix_token_len,
        )
        return hidden_states, attention_mask

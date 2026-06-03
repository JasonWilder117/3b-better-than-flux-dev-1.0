import diffrax as dfx
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from diffusers.models import FlaxAutoencoderKL
from transformers import FlaxCLIPTextModel, FlaxT5EncoderModel
from diffusers.pipelines.pipeline_flax_utils import FlaxDiffusionPipeline, FlaxImagePipelineOutput
from typing import Dict, Optional, Tuple, Union
import numpy as np

from . import rectified_flow
from text_encoder.text_encoder import encode_text_encoder, is_qwen_image_system_prompt_text_encoder_type
from vae.vae import VAE_CONFIGS, reverse_scale_latents

class FlaxInferencePipeline(FlaxDiffusionPipeline):
    def __init__(
        self,
        transformer,
        vae: FlaxAutoencoderKL,
        text_encoder: Union[FlaxCLIPTextModel, FlaxT5EncoderModel, None],
        *,
        config: Dict,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        super().__init__()
        self.register_modules(vae=vae)
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.dtype = dtype
        self.model_config = config
        self.transport_config = rectified_flow.RectifiedFlowConfig.from_config(config.transport)
        prediction_mode = getattr(self.transport_config, "prediction", "velocity")
        self._use_x_prediction = str(prediction_mode).lower() in ("x", "sample", "data")
        self._t_eps = 0.05

    def _decode_latents(
        self,
        params: Union[Dict, FrozenDict],
        latents: jax.Array,
    ) -> jax.Array:
        pretrained_vae_name_or_path = VAE_CONFIGS[self.model_config.vae_type]["pretrained_vae_name_or_path"]
        latents = reverse_scale_latents(latents, self.model_config.vae_type)
        vae_params = params["vae"]
        vae_dtype = getattr(self.vae, "dtype", self.dtype)
        if pretrained_vae_name_or_path == "Qwen/Qwen-Image" and latents.ndim == 4:
            latents = latents[:, :, None, :, :]
        latents = latents.astype(vae_dtype)
        decoded = jax.lax.stop_gradient(
            self.vae.apply({"params": vae_params}, latents, method=self.vae.decode)
        )
        if isinstance(decoded, dict):
            decoded = decoded.get("sample", decoded)
        elif hasattr(decoded, "sample"):
            decoded = decoded.sample
        if pretrained_vae_name_or_path == "Qwen/Qwen-Image" and decoded.ndim == 5:
            decoded = decoded[:, :, 0, :, :]
        return decoded

    def _encode_text(
        self,
        params: Union[Dict, FrozenDict],
        text: jax.Array,
        masks: jax.Array,
    ) -> jax.Array:
        if self.text_encoder is None:
            return text
        encoder_params = params["text_encoder"]
        if isinstance(self.text_encoder, (list, tuple)):
            assert isinstance(text, (list, tuple)) and isinstance(masks, (list, tuple)) and isinstance(self.model_config.text_encoder_type, (list, tuple))
            assert (len(self.text_encoder) == len(encoder_params)) and (len(self.text_encoder) == len(self.model_config.text_encoder_type))
            drop_prefix_lens = self._get_text_encoder_drop_prefix_lens(text)
            hidden_states = [
                encode_text_encoder(enc, enc_params, enc_type, enc_text, enc_mask, drop_prefix_len)
                for enc, enc_params, enc_type, enc_text, enc_mask, drop_prefix_len in zip(
                    self.text_encoder,
                    encoder_params,
                    self.model_config.text_encoder_type,
                    text,
                    masks,
                    drop_prefix_lens,
                )
            ]
            return hidden_states
        return encode_text_encoder(
            self.text_encoder,
            encoder_params,
            self.model_config.text_encoder_type,
            text,
            masks,
            self._get_text_encoder_drop_prefix_lens(text),
        )

    def _get_batch_size(self, text):
        if isinstance(text, (list, tuple)):
            return text[0].shape[0]
        return text.shape[0]

    def _get_configured_text_token_lens(self):
        text_encoder_type = getattr(self.model_config, "text_encoder_type", None)
        token_len = getattr(self.model_config, "token_len", None)
        if isinstance(text_encoder_type, (list, tuple)):
            if isinstance(token_len, (list, tuple)):
                return list(token_len)
            return [token_len] * len(text_encoder_type)
        return token_len

    def _infer_text_encoder_drop_prefix_len(self, text_encoder_type, text, token_len):
        if not is_qwen_image_system_prompt_text_encoder_type(text_encoder_type):
            return 0
        expected_token_len = 256 if token_len is None else token_len
        return max(int(text.shape[1]) - int(expected_token_len), 0)

    def _get_text_encoder_drop_prefix_lens(self, text=None):
        drop_prefix_lens = getattr(self.model_config, "text_encoder_drop_prefix_token_len", None)
        if isinstance(self.text_encoder, (list, tuple)):
            if isinstance(drop_prefix_lens, (list, tuple)):
                return list(drop_prefix_lens)
            if drop_prefix_lens is not None:
                return [drop_prefix_lens] * len(self.text_encoder)
            text_encoder_types = self.model_config.text_encoder_type
            token_lens = self._get_configured_text_token_lens()
            return [
                self._infer_text_encoder_drop_prefix_len(enc_type, enc_text, token_len)
                for enc_type, enc_text, token_len in zip(text_encoder_types, text, token_lens)
            ]
        if drop_prefix_lens is not None:
            return drop_prefix_lens
        return self._infer_text_encoder_drop_prefix_len(
            self.model_config.text_encoder_type,
            text,
            self._get_configured_text_token_lens(),
        )

    def _drop_text_encoder_prefix_from_masks(self, masks, text):
        drop_prefix_lens = self._get_text_encoder_drop_prefix_lens(text)
        if isinstance(masks, (list, tuple)):
            return [
                mask[:, drop_prefix_len:] if drop_prefix_len else mask
                for mask, drop_prefix_len in zip(masks, drop_prefix_lens)
            ]
        return masks[:, drop_prefix_lens:] if drop_prefix_lens else masks

    def _concat_attention_mask(self, masks):
        if masks is None:
            return None
        if isinstance(masks, (list, tuple)):
            return jnp.concatenate(list(masks), axis=1)
        return masks

    def _get_unconditional_tokens(
        self,
        params: Union[Dict, FrozenDict],
        batch_size: int,
        dtype: jnp.dtype,
    ) -> Optional[jnp.ndarray]:
        transformer_params = params["transformer"]
        module_params = transformer_params["text_encoder_adapter"]

        def _collect_prefixed(module_params, prefix: str):
            matches = []
            for key in module_params:
                if not isinstance(key, str):
                    continue
                if key.startswith(prefix + "_"):
                    suffix = key[len(prefix) + 1:]
                    if suffix.isdigit():
                        matches.append((int(suffix), key))
            if matches:
                return [module_params[key] for _, key in sorted(matches)]
            return [module_params[prefix]]

        def _broadcast_one_set_of_unconditional_tokens(tokens):
            tokens = jnp.asarray(tokens, dtype=dtype)
            if tokens.shape[0] == 1 and batch_size > 1:
                tokens = jnp.repeat(tokens, batch_size, axis=0)
            elif tokens.shape[0] != batch_size:
                tokens = jnp.broadcast_to(tokens, (batch_size,) + tokens.shape[1:])
            return tokens

        tokens = _collect_prefixed(module_params, "learnable_null_caption")
        tokens = [_broadcast_one_set_of_unconditional_tokens(t) for t in tokens]
        if len(tokens) == 1:
            return tokens[0]
        return tokens

    def _prepare_text_conditioning(
        self,
        params: Union[Dict, FrozenDict],
        encoder_hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        guidance_scale: Union[float, jnp.ndarray],
        batch_size: int,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
        dtype_ref = encoder_hidden_states[0].dtype if isinstance(encoder_hidden_states, (list, tuple)) else encoder_hidden_states.dtype
        unconditional_tokens = self._get_unconditional_tokens(
            params,
            batch_size,
            dtype_ref,
        )

        guidance = jnp.asarray(guidance_scale, dtype=self.dtype)
        if guidance.ndim == 0:
            guidance = jnp.broadcast_to(guidance, (batch_size,))
        elif guidance.ndim == 1:
            if guidance.shape[0] != batch_size:
                raise ValueError(
                    f"guidance_scale has length {guidance.shape[0]} but batch size is {batch_size}"
                )
        else:
            guidance = guidance.reshape((batch_size,))

        if isinstance(encoder_hidden_states, (list, tuple)):
            assert isinstance(unconditional_tokens, (list, tuple))
            assert len(encoder_hidden_states) == len(unconditional_tokens)
            encoder_hidden_states = [
                jnp.concatenate([cond, uncond], axis=0)
                for cond, uncond in zip(encoder_hidden_states, unconditional_tokens)
            ]
            if attention_mask is not None:
                attention_mask = jnp.concatenate([attention_mask, attention_mask], axis=0)
        else:
            assert not isinstance(unconditional_tokens, (list, tuple))
            encoder_hidden_states = jnp.concatenate([encoder_hidden_states, unconditional_tokens], axis=0)
            if attention_mask is not None:
                attention_mask = jnp.concatenate([attention_mask, attention_mask], axis=0)

        return encoder_hidden_states, attention_mask, guidance

    def _prediction_to_velocity(
        self,
        model_output: jnp.ndarray,
        latents: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        if not self._use_x_prediction:
            return model_output
        if timesteps.ndim == 0:
            timesteps = jnp.full((latents.shape[0],), timesteps, dtype=latents.dtype)
        timesteps = timesteps.astype(model_output.dtype)
        t_shape = (timesteps.shape[0],) + (1,) * (model_output.ndim - 1)
        eps = jnp.asarray(self._t_eps, dtype=model_output.dtype)
        denom = jnp.maximum(1.0 - timesteps.reshape(t_shape), eps)
        return (model_output - latents) / denom

    def _time_grid(self, num_steps: int, dtype: jnp.dtype) -> jnp.ndarray:
        times = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=dtype)
        shift = self.transport_config.inference_timestep_shift or 0.0
        if shift != 0.0:
            """
            As "shift" gets smaller, there are larger proportion of "times" that are closer to 0.
            I.e., spend more time on noisier timesteps.
            """
            times = (shift * times) / (1.0 + (shift - 1.0) * times)
        return times

    def _diffrax_solver_and_controller(self, method: str) -> tuple[dfx.AbstractSolver, dfx.AbstractStepSizeController]:
        method = method.lower()
        if method == "euler":
            return dfx.Euler(), dfx.ConstantStepSize()
        raise ValueError(f"Unsupported sampling method: {method}")

    def _rectified_velocity(
        self,
        params: Union[Dict, FrozenDict],
        latents: jnp.ndarray,
        t_scalar: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        masks: Optional[jnp.ndarray],
        guidance_scale: Optional[jnp.ndarray],
        cfg_rescale: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        batch = latents.shape[0]
        latents_dtype = latents.dtype
        t = jnp.full((batch,), t_scalar, dtype=latents_dtype)
        transformer_params = {"params": params["transformer"]}
        attention_mask = masks

        if guidance_scale is not None:
            latent_input = jnp.concatenate([latents, latents], axis=0)
            t_input = jnp.concatenate([t, t], axis=0)
            model_output = self.transformer.apply(
                transformer_params,
                latent_input,
                t_input,
                encoder_hidden_states,
                mask=attention_mask,
            )
            velocity = self._prediction_to_velocity(model_output, latent_input, t_input)
            cond, uncond = jnp.split(velocity, 2, axis=0)
            if cond.shape[1] != latents.shape[1]:
                cond = cond[:, : latents.shape[1]]
                uncond = uncond[:, : latents.shape[1]]
            scale = guidance_scale.reshape((batch,) + (1,) * (cond.ndim - 1))
            cfg_mask = jnp.ones_like(scale)
            if self.transport_config.cfg_interval_start is not None:
                cfg_mask = jnp.where(t >= self.transport_config.cfg_interval_start, 1.0, 0.0)
                cfg_mask = cfg_mask.reshape(scale.shape)
                
            guided = cond + cfg_mask * (scale - 1) * (cond - uncond)
            if cfg_rescale is not None:
                phi = jnp.asarray(cfg_rescale, dtype=guided.dtype)
                phi = phi.reshape((batch,) + (1,) * (guided.ndim - 1)) if phi.ndim else phi
                axes = tuple(range(1, guided.ndim))
                std_c = jnp.std(cond, axis=axes, keepdims=True)
                std_g = jnp.std(guided, axis=axes, keepdims=True)
                factor = std_c / (std_g + 1e-8)
                guided = guided * (1.0 - phi + phi * factor)
            return guided

        model_output = self.transformer.apply(
            transformer_params,
            latents,
            t,
            encoder_hidden_states,
            mask=attention_mask,
        )
        velocity = self._prediction_to_velocity(model_output, latents, t)
        if velocity.shape[1] != latents.shape[1]:
            velocity = velocity[:, : latents.shape[1]]
        return velocity

    def _integrate_rectified_velocity(
        self,
        params: Union[Dict, FrozenDict],
        latents: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        masks: Optional[jnp.ndarray],
        guidance_scale: Optional[jnp.ndarray],
        cfg_rescale: Optional[jnp.ndarray],
        num_inference_steps: int,
    ) -> jnp.ndarray:
        if num_inference_steps <= 0:
            return latents

        method = (self.transport_config.sampling_method or "euler").lower()
        solver, controller = self._diffrax_solver_and_controller(method)

        dtype = latents.dtype
        times = self._time_grid(num_inference_steps, dtype)
        t0 = times[0]
        t1 = times[-1]

        steps = max(num_inference_steps, 1)
        dt0 = (t1 - t0) / steps
        dt0 = jnp.maximum(dt0, jnp.asarray(1e-6, dtype=dtype))

        guidance = None
        if guidance_scale is not None:
            guidance = jnp.asarray(guidance_scale, dtype=dtype)
        rescale = None
        if cfg_rescale is not None:
            rescale = jnp.asarray(cfg_rescale, dtype=dtype)

        def velocity_fn(t, y, _args):
            t_scalar = jnp.asarray(t, dtype=dtype)
            return self._rectified_velocity(
                params,
                y,
                t_scalar,
                encoder_hidden_states,
                masks,
                guidance,
                rescale,
            )

        term = dfx.ODETerm(velocity_fn)
        solution = dfx.diffeqsolve(
            term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=latents,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=controller,
            max_steps=10000,
        )
        latents_final = solution.ys
        if hasattr(latents_final, "ndim") and latents_final.ndim == latents.ndim + 1:
            latents_final = jnp.squeeze(latents_final, axis=0)
        return latents_final

    def _generate_rectified(
        self,
        params: Union[Dict, FrozenDict],
        text: jax.Array,
        masks: jax.Array,
        key: jax.Array,
        guidance_scale: Union[float, jnp.ndarray],
        cfg_rescale: Union[float, jnp.ndarray],
        num_inference_steps: int,
    ) -> jnp.ndarray:
        batch_size = self._get_batch_size(text)
        latent_size = self.transformer.config.input_size
        latent_channels = self.transformer.config.in_channels
        latents_shape = (batch_size, latent_channels, latent_size, latent_size)
        latents = jax.random.normal(key, latents_shape, dtype=self.dtype)

        encoder_hidden_states = self._encode_text(params, text, masks)
        dit_masks = self._drop_text_encoder_prefix_from_masks(masks, text)
        attention_mask = self._concat_attention_mask(dit_masks)

        (
            encoder_hidden_states,
            attention_mask,
            guidance_scale_tensor,
        ) = self._prepare_text_conditioning(
            params,
            encoder_hidden_states,
            attention_mask,
            guidance_scale,
            batch_size,
        )

        latents = self._integrate_rectified_velocity(
            params,
            latents,
            encoder_hidden_states,
            attention_mask,
            guidance_scale_tensor,
            cfg_rescale,
            num_inference_steps,
        )

        image = self._decode_latents(params, latents)
        image = jnp.clip(image / 2 + 0.5, 0, 1).transpose(0, 2, 3, 1)
        return image

    def __call__(
        self,
        params: Union[Dict, FrozenDict],
        text: jax.Array,
        masks: jax.Array,
        key: jax.Array,
        guidance_scale: Union[float, jnp.ndarray] = 4.0,
        num_inference_steps: int = 250,
        cfg_rescale: Union[float, jnp.ndarray] = 0.0,
        return_dict: bool = True,
    ):
        images = _p_generate_text(
            self,
            params,
            text,
            masks,
            key,
            guidance_scale,
            cfg_rescale,
            num_inference_steps,
        )
        images = np.asarray(images)
        if return_dict:
            return FlaxImagePipelineOutput(images=images)
        return images


from functools import partial


@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, 0, 0, 0, None),
    static_broadcasted_argnums=(0, 7),
)
def _p_generate_text(
    pipe: FlaxInferencePipeline,
    params: Union[Dict, FrozenDict],
    text: jax.Array,
    masks: jax.Array,
    key: jax.Array,
    guidance_scale: Union[float, jnp.ndarray],
    cfg_rescale: Union[float, jnp.ndarray],
    num_inference_steps: int,
):
    return pipe._generate_rectified(
        params,
        text,
        masks,
        key,
        guidance_scale,
        cfg_rescale,
        num_inference_steps,
    )

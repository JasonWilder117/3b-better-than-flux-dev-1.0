import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import gc
from pathlib import Path
import os
import sys

# Add the repo to path so we can import from generate.py
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from torch_inference.generate import (
    i1DiT3B,
    encode_prompt,
    denoise_latents,
    decode_vae,
    rewrite_prompts,
    time_grid,
    prepare_cfg_conditioning,
    reverse_scale_flux2_latents,
)

from transformers import AutoTokenizer, T5GemmaModel
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download


class i1SamplerNode:
    """ComfyUI node for i1 text-to-image generation"""
    
    def __init__(self):
        self.model = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self.device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A detailed portrait of a person with intricate details"
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.0,
                    "max": 25.0,
                    "step": 0.5
                }),
                "cfg_rescale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 1024,
                    "step": 256
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 1024,
                    "step": 256
                }),
                "rewrite_prompt": ("BOOLEAN", {
                    "default": False
                }),
                "rewriter_model": (["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-4B-Instruct-2507"], {
                    "default": "Qwen/Qwen3-4B-Instruct-2507"
                }),
                "inference_timestep_shift": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "sampling"
    
    def generate(self, prompt, steps, cfg_scale, cfg_rescale, seed, width, height, 
                 rewrite_prompt, rewriter_model, inference_timestep_shift, checkpoint_path=""):
        
        # Initialize device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model if not already loaded
        if self.model is None:
            print("[i1 Sampler] Loading models...")
            
            # Load checkpoint
            if not checkpoint_path:
                checkpoint_path = hf_hub_download(
                    repo_id="zlab-princeton/i1-3B",
                    filename="1024_resolution_checkpoint_torch.pt",
                    repo_type="model",
                )
            
            self.model = i1DiT3B().to(device=self.device, dtype=torch.bfloat16).eval()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            
            # Load text encoder
            self.tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-ul2-it")
            self.text_encoder = T5GemmaModel.from_pretrained(
                "google/t5gemma-2b-2b-ul2-it",
                dtype=torch.bfloat16,
            ).encoder.to(self.device).eval()
            
            # Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                subfolder="vae"
            ).to(device=self.device, dtype=torch.bfloat16).eval()
        
        # Rewrite prompt if requested
        if rewrite_prompt:
            print(f"[i1 Sampler] Rewriting prompt with {rewriter_model}...")
            prompts = rewrite_prompts([prompt], self.device, rewriter_model, batch_size=1)
            prompt = prompts[0]
            print(f"[i1 Sampler] Rewritten prompt: {prompt}")
        
        # Encode prompt
        print("[i1 Sampler] Encoding prompt...")
        with torch.inference_mode():
            text, mask = encode_prompt(self.tokenizer, self.text_encoder, [prompt], self.device)
        
        # Create args-like object for denoise_latents
        class Args:
            def __init__(self):
                self.num_steps = steps
                self.cfg_scale = cfg_scale
                self.cfg_rescale = cfg_rescale
                self.inference_timestep_shift = inference_timestep_shift
        
        args = Args()
        
        # Generate latents
        print("[i1 Sampler] Generating image...")
        with torch.inference_mode():
            latents = denoise_latents(self.model, text, mask, args, self.device)
        
        # Decode VAE
        print("[i1 Sampler] Decoding VAE...")
        images = decode_vae(self.vae, latents, batch_size=1)
        
        # Convert to tensor format expected by ComfyUI
        image_tensor = torch.from_numpy(images[0]).float() / 255.0
        
        # Clean up
        del text, mask, latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[i1 Sampler] Complete!")
        return (image_tensor.unsqueeze(0),)


class i1ModelLoader:
    """Node to preload i1 models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("I1_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    
    def load_model(self, checkpoint_path=""):
        if not checkpoint_path:
            checkpoint_path = hf_hub_download(
                repo_id="zlab-princeton/i1-3B",
                filename="1024_resolution_checkpoint_torch.pt",
                repo_type="model",
            )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = i1DiT3B().to(device=device, dtype=torch.bfloat16).eval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        
        return ({"model": model, "device": device, "checkpoint_path": checkpoint_path},)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "i1Sampler": i1SamplerNode,
    "i1ModelLoader": i1ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "i1Sampler": "i1 Text-to-Image Sampler",
    "i1ModelLoader": "Load i1 Model",
}

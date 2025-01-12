from typing import Union
from PIL import Image

import torch

from .utils import tools


class ControlNeXtSDXL:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
            unet_model_name_or_path: str = None,
            controlnet_model_name_or_path: str = None,
            vae_model_name_or_path: str = None,
            lora_path: str = None,
            load_weight_increasement: bool = True,
            enable_xformers_memory_efficient_attention: bool = False,
            revision: str = None,
            variant: str = None,
            hf_cache_dir: str = None,
            use_safetensors: bool = True,
            device: str = "cuda",
    ):
        self.pipe = tools.get_pipeline(
            pretrained_model_name_or_path,
            unet_model_name_or_path,
            controlnet_model_name_or_path,
            vae_model_name_or_path=vae_model_name_or_path,
            lora_path=lora_path,
            load_weight_increasement=load_weight_increasement,
            enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
            revision=revision,
            variant=variant,
            hf_cache_dir=hf_cache_dir,
            use_safetensors=use_safetensors,
            device=device,
        )
        self.device = device

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            negative_prompt: str = None,
            seed: int = None,
            width: int = 1024,
            height: int = 1024,
            controlnet_scale: float = 1.0,
            num_inference_steps: int = 20,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if isinstance(control_image, str):
            control_image = Image.open(control_image).convert("RGB")
        control_image = control_image.resize((width, height))

        with torch.autocast(self.device):
            image = self.pipe(
                prompt=prompt,
                controlnet_image=control_image,
                controlnet_scale=controlnet_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
            ).images[0]
        return image

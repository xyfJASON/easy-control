from typing import Union
from PIL import Image

import torch
from diffusers import StableDiffusionXLAdapterPipeline, AutoencoderKL, T2IAdapter as T2IAdapter_diffusers
from diffusers.utils import load_image


class T2IAdapter:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
            adapter_model_name_or_path: str = "TencentARC/t2i-adapter-lineart-sdxl-1.0",
            vae_model_name_or_path: str = "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype: torch.dtype = torch.float16,
            variant: str = None,
            device: str = "cuda",
    ):
        self.device = device
        self.adapter = T2IAdapter_diffusers.from_pretrained(
            pretrained_model_name_or_path=adapter_model_name_or_path,
            torch_dtype=torch_dtype,
            varient=variant,
        ).to(device)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=vae_model_name_or_path,
            torch_dtype=torch_dtype,
        )
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            vae=self.vae,
            adapter=self.adapter,
            torch_dtype=torch_dtype,
            variant=variant,
        ).to(device)

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            negative_prompt: str = None,
            width: int = 1024,
            height: int = 1024,
            num_inference_steps: int = 20,
            adapter_conditioning_scale: float = 1.0,
            guidance_scale: float = 7.5,
    ):
        control_image = load_image(control_image)
        control_image = control_image.resize((width, height))

        image = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=adapter_conditioning_scale,
            guidance_scale=guidance_scale,
        ).images[0]
        return image

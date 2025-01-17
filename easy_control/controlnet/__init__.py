from typing import Union
from PIL import Image

import torch
from diffusers import AutoencoderKL, ControlNetModel
from diffusers import StableDiffusionControlNetPipeline
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image


class ControlNet:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
            control_model_name_or_path: str = "lllyasviel/control_v11p_sd15_canny",
            torch_dtype: torch.dtype = torch.float32,
            device: str = "cuda",
    ):
        self.device = device

        self.controlnet = ControlNetModel.from_pretrained(
            pretrained_model_name_or_path=control_model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
        ).to(device)

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            negative_prompt: str = None,
            seed: int = None,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 20,
            guidance_scale: float = 7.5,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        control_image = load_image(control_image)
        control_image = control_image.resize((width, height))

        image = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image


class ControlNetSDXL:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
            control_model_name_or_path: str = "diffusers/controlnet-canny-sdxl-1.0",
            vae_model_name_or_path: str = "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype: torch.dtype = torch.float16,
            device: str = "cuda",
    ):
        self.device = device

        self.controlnet = ControlNetModel.from_pretrained(
            pretrained_model_name_or_path=control_model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(device)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=vae_model_name_or_path,
            torch_dtype=torch_dtype,
        )
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            vae=self.vae,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
        ).to(device)

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            negative_prompt: str = None,
            seed: int = None,
            width: int = 1024,
            height: int = 1024,
            num_inference_steps: int = 20,
            guidance_scale: float = 5.0,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        control_image = load_image(control_image)
        control_image = control_image.resize((width, height))

        image = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image

from typing import Union
from PIL import Image

import torch
from diffusers import StableDiffusionAdapterPipeline, StableDiffusionXLAdapterPipeline
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, T2IAdapter as T2IAdapterModel
from diffusers.utils import load_image


class T2IAdapter:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
            adapter_model_name_or_path: str = "TencentARC/t2iadapter_canny_sd15v2",
            torch_dtype: torch.dtype = torch.float32,
            device: str = "cuda",
    ):
        self.device = device

        self.adapter = T2IAdapterModel.from_pretrained(
            pretrained_model_name_or_path=adapter_model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(device)
        self.pipe = StableDiffusionAdapterPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            adapter=self.adapter,
            torch_dtype=torch_dtype,
        ).to(device)

        self.convert_method = None
        if "canny" in adapter_model_name_or_path:
            self.convert_method = lambda x: x.convert("L")

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            negative_prompt: str = None,
            seed: int = None,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 20,
            adapter_conditioning_scale: float = 1.0,
            guidance_scale: float = 7.5,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        control_image = load_image(control_image, convert_method=self.convert_method)
        control_image = control_image.resize((width, height))

        image = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=adapter_conditioning_scale,
            guidance_scale=guidance_scale,
        ).images[0]
        return image


class T2IAdapterSDXL:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
            adapter_model_name_or_path: str = "TencentARC/t2i-adapter-lineart-sdxl-1.0",
            vae_model_name_or_path: str = "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype: torch.dtype = torch.float16,
            variant: str = "fp16",
            device: str = "cuda",
    ):
        self.device = device

        self.adapter = T2IAdapterModel.from_pretrained(
            pretrained_model_name_or_path=adapter_model_name_or_path,
            torch_dtype=torch_dtype,
            varient=variant,
        ).to(device)
        self.euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=vae_model_name_or_path,
            torch_dtype=torch_dtype,
        )
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            vae=self.vae,
            adapter=self.adapter,
            scheduler=self.euler_a,
            torch_dtype=torch_dtype,
            variant=variant,
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
            adapter_conditioning_scale: float = 1.0,
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
            adapter_conditioning_scale=adapter_conditioning_scale,
            guidance_scale=guidance_scale,
        ).images[0]
        return image

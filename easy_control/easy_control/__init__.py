from typing import Union
from PIL import Image

import torch
from diffusers.utils import load_image

from .src.pipeline import FluxPipeline
from .src.transformer_flux import FluxTransformer2DModel
from .src.lora_helper import set_single_lora, set_multi_lora


class EasyControl:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
            lora_path: str = "./ckpts/easy_control/models",
            condition_type: str = "canny",
            device: str = "cuda",
    ):
        self.device = device

        self.pipe = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device=device,
        )
        self.pipe.transformer = transformer
        self.pipe.to(device)

        path = f"{lora_path}/{condition_type}.safetensors"
        set_single_lora(self.pipe.transformer, path, lora_weights=[1], cond_size=512)

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            seed: int = None,
            width: int = 1024,
            height: int = 1024,
            num_inference_steps: int = 25,
            guidance_scale: float = 3.5,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        control_image = load_image(control_image)
        control_image = control_image.resize((width, height))

        result = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator,
            spatial_images=[control_image],
            subject_images=[],
            cond_size=512,
        ).images[0]
        return result

from typing import Union
from PIL import Image

import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

from .models.controlnet_union import ControlNetModel_Union
from .pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline


TASKNAME2ID = {
    "openpose": 0,
    "depth": 1,
    "hed": 2,
    "pidi": 2,
    "scribble": 2,
    "ted": 2,
    "canny": 3,
    "lineart": 3,
    "anime_lineart": 3,
    "mlsd": 3,
    "normal": 4,
    "segment": 5,
}


class ControlNetUnion:
    def __init__(
            self,
            task_name: str,
            pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet_union_model_name_or_path: str = "xinsir/controlnet-union-sdxl-1.0",
            vae_model_name_or_path: str = None,
            torch_dtype: torch.dtype = torch.float16,
            use_safetensors: bool = True,
            device: str = "cuda",
    ):
        assert task_name in TASKNAME2ID.keys(), f"Task name {task_name} not supported."
        self.task_id = TASKNAME2ID[task_name]

        self.device = device

        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=vae_model_name_or_path,
            torch_dtype=torch_dtype,
        )
        controlnet_model = ControlNetModel_Union.from_pretrained(
            pretrained_model_name_or_path=controlnet_union_model_name_or_path,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
        )
        self.pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            controlnet=controlnet_model,
            vae=vae,
            torch_dtype=torch_dtype,
            scheduler=eulera_scheduler,
        ).to(self.device)

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            negative_prompt: str = None,
            seed: int = None,
            width: int = 1024,
            height: int = 1024,
            num_inference_steps: int = 20,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        control_image = load_image(control_image).resize((width, height))

        image_list = [0, 0, 0, 0, 0, 0]
        image_list[self.task_id] = control_image

        union_control_type = torch.Tensor([0, 0, 0, 0, 0, 0])
        union_control_type[self.task_id] = 1

        image = self.pipe(
            prompt=[prompt] * 1,
            image_list=image_list,
            negative_prompt=[negative_prompt] * 1,
            generator=generator,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            union_control=True,
            union_control_type=union_control_type,
        ).images[0]
        return image

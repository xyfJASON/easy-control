import yaml
from typing import Union
from PIL import Image

import torch
from diffusers import DDIMScheduler
from diffusers.utils import load_image

from .utils.sdxl import get_control_config, register_control
from .utils.utils import get_self_recurrence_schedule
from .pipelines.pipeline_sdxl import CtrlXStableDiffusionXLPipeline


class CtrlX:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype: torch.dtype = torch.float16,
            variant: str = "fp16",
            use_safetensors: bool = True,
            device: str = "cuda",

            num_inference_steps: int = 20,
            structure_schedule: float = 0.6,
            appearance_schedule: float = 0.6,
    ):
        self.num_inference_steps = num_inference_steps

        # build pipeline
        self.scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        self.pipe = CtrlXStableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            scheduler=self.scheduler,
            torch_dtype=torch_dtype,
            variant=variant,
            use_safetensors=use_safetensors,
        ).to(device)
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.pipe.set_progress_bar_config(desc="Ctrl-X inference")

        # build config
        self.config = yaml.safe_load(get_control_config(structure_schedule, appearance_schedule))
        register_control(
            model=self.pipe,
            timesteps=self.pipe.scheduler.timesteps,
            control_schedule=self.config["control_schedule"],
            control_target=self.config["control_target"],
        )
        self.self_recurrence_schedule = get_self_recurrence_schedule(
            schedule=self.config["self_recurrence_schedule"], num_inference_steps=num_inference_steps,
        )

    def sample(
            self,
            prompt: str,
            structure_prompt: str = "",
            appearance_prompt: str = "",
            structure_image: Union[str, Image.Image] = None,
            appearance_image: Union[str, Image.Image] = None,
            positive_prompt: str = None,
            negative_prompt: str = None,
            eta: float = 0.0,
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 5.0,
            structure_guidance_scale: float = 5.0,
            appearance_guidance_scale: float = 5.0,
    ):
        if structure_image is not None:
            structure_image = load_image(structure_image)
        if appearance_image is not None:
            appearance_image = load_image(appearance_image)

        result, structure, appearance = self.pipe(
            prompt=prompt,
            structure_prompt=structure_prompt,
            appearance_prompt=appearance_prompt,
            structure_image=structure_image,
            appearance_image=appearance_image,
            num_inference_steps=self.num_inference_steps,
            negative_prompt=negative_prompt,
            positive_prompt=positive_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            structure_guidance_scale=structure_guidance_scale,
            appearance_guidance_scale=appearance_guidance_scale,
            eta=eta,
            output_type="pil",
            return_dict=False,
            control_schedule=self.config["control_schedule"],
            self_recurrence_schedule=self.self_recurrence_schedule,
        )
        return result[0]

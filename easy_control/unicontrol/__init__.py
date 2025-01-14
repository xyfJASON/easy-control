import os
import numpy as np
from typing import Union
from PIL import Image

import torch
import torchvision.transforms as T
from diffusers.utils import load_image
from transformers import logging

from .cldm.model import create_model, load_state_dict
from .cldm.ddim_unicontrol_hacked import DDIMSampler


logging.set_verbosity_error()


CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
TASKNAMES = [
    "hed", "canny", "seg", "depth", "normal", "img", "openpose",
    "hedsketch", "bbox", "outpainting", "grayscale", "blur", "inpainting",
]
TASK2INSTRUCTION = {
    "control_hed": "hed edge to image",
    "control_canny": "canny edge to image",
    "control_seg": "segmentation map to image",
    "control_depth": "depth map to image",
    "control_normal": "normal surface map to image",
    "control_img": "image editing",
    "control_openpose": "human pose skeleton to image",
    "control_hedsketch": "sketch to image",
    "control_bbox": "bounding box to image",
    "control_outpainting": "image outpainting",
    "control_grayscale": "gray image to color image",
    "control_blur": "deblur image to clean image",
    "control_inpainting": "image inpainting",
}


class UniControl:
    def __init__(
            self,
            pretrained_model_path: str,
            task_name: str,
            version: str = "v1.1",
            device: str = "cuda",
    ):
        assert task_name in TASKNAMES, f"Task name {task_name} not supported."

        if version == "v1":
            self.config = os.path.join(CONFIG, "cldm_v15_unicontrol.yaml")
        elif version == "v1.1":
            self.config = os.path.join(CONFIG, "cldm_v15_unicontrol_v11.yaml")
        else:
            raise ValueError(f"Version {version} not supported.")

        self.device = device

        self.model = create_model(self.config).to(device)
        self.model.load_state_dict(load_state_dict(pretrained_model_path, location="cpu"), strict=False)

        self.ddim_sampler = DDIMSampler(self.model)

        self.task_dic = {"name": f"control_{task_name}"}
        task_instruction = TASK2INSTRUCTION[self.task_dic["name"]]
        self.task_dic["feature"] = self.model.get_learned_conditioning(task_instruction)[:, :1, :]

    @torch.no_grad()
    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            positive_prompt: str = None,
            negative_prompt: str = None,
            width: int = 512,
            height: int = 512,
            guidance_scale: float = 7.5,
            num_inference_steps: int = 20,
    ):
        if positive_prompt is not None:
            prompt = prompt.strip() + ", " + positive_prompt.strip()
        if negative_prompt is None:
            negative_prompt = ""

        control = load_image(control_image).resize((width, height))
        control = T.ToTensor()(control).unsqueeze(0).to(self.device)

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt])], "task": self.task_dic}
        un_cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([negative_prompt])]}

        samples, intermediates = self.ddim_sampler.sample(
            num_inference_steps,
            batch_size=1,
            shape=(4, height // 8, width // 8),
            conditioning=cond,
            verbose=False,
            eta=0,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond,
        )

        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        x_samples = (x_samples * 255).astype(np.uint8)
        return Image.fromarray(x_samples[0])

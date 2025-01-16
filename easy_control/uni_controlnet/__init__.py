import os
import numpy as np
from typing import Union
from PIL import Image

import torch
from diffusers.utils import load_image
from transformers import logging, AutoProcessor, CLIPModel

from .models.util import create_model, load_state_dict
from .models.ddim_hacked import DDIMSampler


logging.set_verbosity_error()


CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
TASKNAME2ID = {
    "canny": 0,
    "mlsd": 1,
    "hed": 2,
    "sketch": 3,
    "openpose": 4,
    "midas": 5,
    "seg": 6,
}


class UniControlNet:
    def __init__(
            self,
            pretrained_model_path: str,
            task_name: str,
            device: str = "cuda",
    ):
        assert task_name in TASKNAME2ID.keys(), f"Task name {task_name} not supported."

        self.task_name = task_name
        self.config = os.path.join(CONFIG, "uni_v15.yaml")
        self.device = device

        self.model = create_model(self.config).to(device)
        self.model.load_state_dict(load_state_dict(pretrained_model_path, location="cpu"), strict=False)
        self.ddim_sampler = DDIMSampler(self.model)

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def detect_content(self, content_image: Image.Image):
        inputs = self.processor(images=content_image, return_tensors="pt").to(self.device)
        image_features = self.clip.get_image_features(**inputs)
        image_feature = image_features[0].detach()
        return image_feature

    @torch.no_grad()
    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            content_image: Union[str, Image.Image] = None,
            positive_prompt: str = None,
            negative_prompt: str = None,
            width: int = 512,
            height: int = 512,
            control_strength: float = 1.0,
            global_strength: float = 1.0,
            guidance_scale: float = 7.5,
            num_inference_steps: int = 20,
    ):
        if positive_prompt is not None:
            prompt = prompt.strip() + ", " + positive_prompt.strip()
        if negative_prompt is None:
            negative_prompt = ""

        control = load_image(control_image).resize((width, height))
        detected_maps = np.zeros((height, width, 3*7)).astype(np.uint8)
        i = TASKNAME2ID[self.task_name]
        detected_maps[:, :, i*3:(i+1)*3] = np.array(control).astype(np.uint8)

        local_control = torch.from_numpy(detected_maps.copy()).float().to(self.device) / 255.0
        local_control = local_control.permute(2, 0, 1).unsqueeze(0)

        global_control = torch.zeros((768, ), device=self.device)
        if content_image is not None:
            content_image = load_image(content_image)
            global_control = self.detect_content(content_image)
        global_control = global_control.unsqueeze(0)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {"local_control": [local_control], "c_crossattn": [self.model.get_learned_conditioning([prompt])], "global_control": [global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [self.model.get_learned_conditioning([negative_prompt])], "global_control": [uc_global_control]}

        self.model.control_scales = [control_strength] * 13
        samples, _ = self.ddim_sampler.sample(
            num_inference_steps,
            batch_size=1,
            shape=(4, height // 8, width // 8),
            conditioning=cond,
            verbose=False,
            eta=0,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond,
            global_strength=global_strength,
        )
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        x_samples = (x_samples * 255).astype(np.uint8)
        return Image.fromarray(x_samples[0])

from typing import Union
from PIL import Image

import torch
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel

from .flux.condition import Condition
from .flux.generate import generate, seed_everything


def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


class OminiControl:
    def __init__(
            self,
            pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-schnell",
            lora_model_name_or_path: str = "Yuanshi/OminiControl",
            condition_type: str = "canny",
            device: str = "cuda",
            use_int8: bool = False,
    ):
        assert condition_type in ["canny", "depth", "coloring", "deblurring"]
        self.condition_type = condition_type

        if use_int8 or get_gpu_memory() < 33:
            assert pretrained_model_name_or_path == "black-forest-labs/FLUX.1-schnell"
            transformer_model = FluxTransformer2DModel.from_pretrained(
                "sayakpaul/flux.1-schell-int8wo-improved",
                torch_dtype=torch.bfloat16,
                use_safetensors=False,
            )
            self.pipe = FluxPipeline.from_pretrained(
                pretrained_model_name_or_path,
                transformer=transformer_model,
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            self.pipe = FluxPipeline.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
            ).to(device)

        self.pipe.load_lora_weights(
            lora_model_name_or_path,
            weight_name=f"experimental/{condition_type}.safetensors",
            adapter_name=condition_type,
        )

    def sample(
            self,
            prompt: str,
            control_image: Union[str, Image.Image],
            seed: int = None,
            width: int = 512,
            height: int = 512,
    ):
        if seed is not None:
            seed_everything(seed)

        if isinstance(control_image, str):
            control_image = Image.open(control_image).convert("RGB")
        control_image = control_image.resize((width, height))
        condition = Condition(self.condition_type, condition=control_image)

        result = generate(
            self.pipe,
            prompt=prompt,  # type: ignore
            conditions=[condition],
        ).images[0]
        return result

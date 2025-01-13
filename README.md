# easy-control-diffusion

We provide simple python wrappers for various diffusion-based controllable generation models so that you can easily establish the baselines for your paper.
For full functionalities of these models, please refer to their original repositories.



## Installation

> The code is tested with python 3.12.8, torch 2.5.1 and cuda 12.4.

Clone the repository:

```shell
git clone https://github.com/xyfJASON/easy-control-diffusion.git
cd easy-control-diffusion
```

Create and activate a conda environment:

```shell
conda create -n easy-control python=3.12
conda activate easy-control
```

Install the dependencies:

```shell
pip install torch torchvision
pip install -r requirements.txt
```



## Usage

### T2I-Adapter ([arXiv](https://arxiv.org/abs/2302.08453) | [GitHub](https://github.com/TencentARC/T2I-Adapter))

```python
from easy_control import T2IAdapter

t2iadapter = T2IAdapter(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    adapter_model_name_or_path="TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
    vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
    variant="fp16",
    device="cuda",
)
result = t2iadapter.sample(
    control_image="./test_images/city-depth.png",
    prompt="a photo of a city at night, stars in the sky",
    negative_prompt="worst quality",
)
result.show()
```

### Ctrl-X ([arXiv](https://arxiv.org/abs/2406.07540) | [GitHub](https://github.com/genforce/ctrl-x/tree/main) | [website](https://genforce.github.io/ctrl-x))

```python
from easy_control import CtrlX

ctrl_x = CtrlX(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    device="cuda",
)
result = ctrl_x.sample(
    appearance_image="./test_images/cat.jpeg",
    structure_image="./test_images/rabbit-canny.png",
    prompt="a rabbit",
    positive_prompt="high quality",
    negative_prompt="ugly, blurry, dark, low res, unrealistic",
)
result.show()
```

### ControlNeXt ([arXiv](https://arxiv.org/abs/2408.06070) | [GitHub](https://github.com/dvlab-research/ControlNeXt) | [website](https://pbihao.github.io/projects/controlnext/index.html))

```python
from easy_control import ControlNeXtSDXL

controlnext_sdxl = ControlNeXtSDXL(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    unet_model_name_or_path="Eugeoter/controlnext-sdxl-vidit-depth",
    controlnet_model_name_or_path="Eugeoter/controlnext-sdxl-vidit-depth",
    vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
    variant="fp16",
)
result = controlnext_sdxl.sample(
    control_image="./test_images/tower-depth-vidit.png",
    prompt="a diamond tower in the middle of a lava lake",
)
result.show()
```



## References

```
@inproceedings{mou2024t2i,
  title={T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models},
  author={Mou, Chong and Wang, Xintao and Xie, Liangbin and Wu, Yanze and Zhang, Jian and Qi, Zhongang and Shan, Ying},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4296--4304},
  year={2024}
}
```

```
@inproceedings{lin2024ctrlx,
  title={Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance},
  author={Kuan Heng Lin and Sicheng Mo and Ben Klingher and Fangzhou Mu and Bolei Zhou},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=ZulWEWQOp9}
}
```

```
@article{peng2024controlnext,
  title={Controlnext: Powerful and efficient control for image and video generation},
  author={Peng, Bohao and Wang, Jian and Zhang, Yuechen and Li, Wenbo and Yang, Ming-Chang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2408.06070},
  year={2024}
}
```

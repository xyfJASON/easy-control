# easy-control-diffusion

We provide simple python wrappers for various controllable generation models so that you can easily establish the baselines for your paper.
For full functionalities, please refer to their original repositories.



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

### SDXL-based Models

**Ctrl-X** ([arXiv](https://arxiv.org/abs/2406.07540) | [GitHub](https://github.com/genforce/ctrl-x/tree/main) | [website](https://genforce.github.io/ctrl-x)):

```python
from easy_control import CtrlX

ctrl_x = CtrlX()
result = ctrl_x.sample(
    appearance_image="./test_images/cat.jpeg",
    structure_image="./test_images/rabbit-canny.png",
    prompt="a rabbit",
)
result[0].show()
```



## References

```
@inproceedings{lin2024ctrlx,
  title={Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance},
  author={Kuan Heng Lin and Sicheng Mo and Ben Klingher and Fangzhou Mu and Bolei Zhou},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=ZulWEWQOp9}
}
```

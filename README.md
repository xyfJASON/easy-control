# easy-control-diffusion

Quickly try out diffusion-based controllable generation models with a few lines of code.



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

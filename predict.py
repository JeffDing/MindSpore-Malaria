# The MIT license:

# Copyright <2023> <lisenlingood、热带鱼>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#  and associated documentation files (the “Software”), to deal in the Software without restric
# tion, including without limitation the rights to use, copy, modify, merge, publish, distribute, sub
# license, and/or sell copies of the Software, and to permit persons to whom the Software is furni
# shed to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantia
# l portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED
# , INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PART
# ICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT H
# OLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF C
# ONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWA
# RE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



import os

from mindspore.dataset import ImageFolderDataset

from Vit import *


import os
import pathlib
import cv2
import numpy as np
from PIL import Image
from enum import Enum
from scipy import io


class Color(Enum):
    """dedine enum color."""
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def check_file_exist(file_name: str):
    """check_file_exist."""
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File `{file_name}` does not exist.")


def color_val(color):
    """color_val."""
    if isinstance(color, str):
        return Color[color].value
    if isinstance(color, Color):
        return color.value
    if isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    if isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    if isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    raise TypeError(f'Invalid type for color: {type(color)}')


def imread(image, mode=None):
    """imread."""
    if isinstance(image, pathlib.Path):
        image = str(image)

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        check_file_exist(image)
        image = Image.open(image)
        if mode:
            image = np.array(image.convert(mode))
    else:
        raise TypeError("Image must be a `ndarray`, `str` or Path object.")

    return image


def imwrite(image, image_path, auto_mkdir=True):
    """imwrite."""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(image_path))
        if dir_name != '':
            dir_name = os.path.expanduser(dir_name)
            os.makedirs(dir_name, mode=777, exist_ok=True)

    image = Image.fromarray(image)
    image.save(image_path)


def imshow(img, win_name='', wait_time=0):
    """imshow"""
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def show_result(img: str,
                result: Dict[int, float],
                text_color: str = 'green',
                font_scale: float = 0.5,
                row_width: int = 20,
                show: bool = False,
                win_name: str = '',
                wait_time: int = 0,
                thickness:int = 1,
                out_file: Optional[str] = None) -> None:
    """Mark the prediction results on the picture."""
    img = imread(img, mode="RGB")
    img = img.copy()
    x, y = 0, row_width
    text_color = color_val(text_color)
    for k, v in result.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    font_scale, text_color,thickness = thickness )
        y += row_width
    if out_file:
        show = False
        imwrite(img, out_file)

    if show:
        imshow(img, win_name, wait_time)


def index2label():
    return {0:"falciparum",1:"uninfected",2:"vivax"}

network = ViT(image_size=IMAGESIZE)

ascend_target = (ms.get_context("device_target") == "Ascend")
if ascend_target:
    model = train.Model(network)
else:
    model = train.Model(network)

vit_path = './ViT/vit_672_4832_0.80897.ckpt'
param_dict = ms.load_checkpoint(vit_path)

ms.load_param_into_net(network, param_dict)



if __name__ == '__main__':

    # image is in './out/input/input.jpg'
    data_path = './'
    dataset_infer = ImageFolderDataset(os.path.join(data_path, "out"), shuffle=True)

    dataset_infer = datapipe(dataset_infer,1)
    # Read data for inference
    for i, image in enumerate(dataset_infer.create_dict_iterator(output_numpy=True)):
        image = image["image"]
        image = ms.Tensor(image)
        prob = model.predict(image)
        label = np.argmax(prob.asnumpy(), axis=1)
        mapping = index2label()
        output = {int(label): mapping[int(label)]}
        print(output)
        show_result(img="./out/input/input.jpg",
                    result=output,
                    out_file="./out/out.jpg",
                    font_scale= 5,
                    row_width=200,
                    thickness = 10)

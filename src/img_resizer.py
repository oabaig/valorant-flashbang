import os
from sys import argv
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

in_dir = argv[1]
out_dir = argv[2]
xDim = argv[3]
yDim = argv[4]

images = []
for file in os.listdir(in_dir):
    if file.endswith('.jpg'):
        images.append(file)

new_images = []
for image in images:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_images.append(resize_with_pad(np.array(img), (xDim, yDim)))

count = 0
for img in new_images:
    i = Image.fromarray(img)
    i.save(os.path.join(out_dir, f'{count}.jpg'))
    count += 1
    
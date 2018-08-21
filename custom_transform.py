#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Custom transform for the PyTorch framework.
"""

from PIL import Image
import numpy as np
import numbers
from torchvision.transforms.functional import _is_pil_image

def substract(tensor, value):
    """Substract a tensor image with a value."""

    for t, v in zip(tensor, value):
        t.sub_(v)
    
    return tensor

class Substract(object):
    """Substract a tensor image with a value."""

    def __init__(self, value):
        self.value = value

    def __call__(self, tensor):
        return substract(tensor, self.value)


def translate(img, horizontal=0, vertical=0):
    """Translate the img by horizontal and vertical pixels.

    Args:
        img (PIL Image): PIL Image to be rotated.
        horizontal (int): Number of horizontal pixels to translate.
            If horizontal > 0, img will be translated LEFT.
            If horizontal < 0, img will be translated RIGHT.
        vertical (int): Number of vertical pixels to translate.
            If vertical > 0, img will be translated UP.
            If vertical < 0, img will be translated DOWN.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transform(img.size, Image.AFFINE, (1, 0, horizontal, 0, 1, vertical))

class RandomTranslation(object):
    """Tanslate the image horizontally and vertically.
    Args:
        horizontal (sequence or int): Range of horizontal pixels to select from.
            If horizontal is a number instead of sequence like (min, max), the range of pixels
            will be (-horizontal, +horizontal).
            If horizontal > 0, img will be translated LEFT.
            If horizontal < 0, img will be translated RIGHT.
        vertical (sequence or int): Range of vertical pixels to select from.
            If vertical is a number instead of sequence like (min, max), the range of pixels
            will be (-vertical, +vertical).
            If vertical > 0, img will be translated UP.
            If vertical < 0, img will be translated DOWN.
    """

    def __init__(self, horizontal=0, vertical=0):
        if isinstance(horizontal, numbers.Number):
            if horizontal < 0:
                raise ValueError("If horizontal is a single number, it must be positive.")
            self.horizontal = (-horizontal, horizontal)
        else:
            if len(horizontal) != 2:
                raise ValueError("If horizontal is a sequence, it must be of len 2.")
            self.horizontal = horizontal

        if isinstance(vertical, numbers.Number):
            if vertical < 0:
                raise ValueError("If vertical is a single number, it must be positive.")
            self.vertical = (-vertical, vertical)
        else:
            if len(vertical) != 2:
                raise ValueError("If vertical is a sequence, it must be of len 2.")
            self.vertical = vertical

    @staticmethod
    def get_params(horizontal, vertical):
        """Get parameters for ``translate`` for a random translation.
        Returns:
            h, v: params to be passed to ``translate`` for random translation.
        """
        h = np.random.uniform(horizontal[0], horizontal[1])
        v = np.random.uniform(vertical[0], vertical[1])

        return h, v

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        h, v = self.get_params(self.horizontal, self.vertical)

        return translate(img, h, v)
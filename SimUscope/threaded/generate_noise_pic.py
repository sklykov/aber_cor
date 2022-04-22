# -*- coding: utf-8 -*-
"""
Simulation of some picture with noise.

@author: ssklykov
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt


# %% Noise generation
def generate_noise_picture(height: int, width: int, pixel_type: str = 'uint8') -> np.ndarray:
    """
    Generate of a noise image with even distribution of noise (pixel values) on that.

    Parameters
    ----------
    height : int
        Height of a generated image.
    width : int
        Width of a generated image.
    pixel_type : str, optional
        Type of pixels in an image. The default is 'uint8'.

    Raises
    ------
    Exception
        When the specified height or width are less than 2.

    Returns
    -------
    img : np.ndarray
        Generate image with even noise.

    """
    img = np.zeros((1, 1), dtype='uint8')
    if (height >= 2) and (width >= 2):
        if pixel_type == 'uint8':
            img = np.random.randint(0, high=255, size=(height, width), dtype='uint8')
        if pixel_type == 'float':
            img = np.random.rand(height, width)
    else:
        raise Exception("Specified height or width are less than 2")

    return img


# %% Tests
if __name__ == "__main__":
    plt.close('all')
    img = generate_noise_picture(200, 200)
    plt.imshow(img, cmap='gray'); plt.axis('off'); plt.tight_layout()

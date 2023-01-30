# -*- coding: utf-8 -*-
"""
Sample of simple single image acquisition using uManager.

@author: sklykov

@license: GPLv3, general terms on: https://www.gnu.org/licenses/gpl-3.0.en.html

"""
# %% Imports
import matplotlib.pyplot as plt
import numpy as np

# %% Basic initialization
# !!! If the ZMQ server versions in Java and Python are different,
# I faced the problem with exception that Python str not transferred to java.lang.String
# Initialization of pycromanager (installed by the pip)
from pycromanager import Core
core = Core(); print(core)

# %% Acquisition from Denoising applications example
# Straight-forward way of acquiring of a single image (snap) and use it for further processing
core.snap_image()
tagged_image = core.get_tagged_image()
pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"],
                                                tagged_image.tags["Width"]])
plt.imshow(pixels, cmap="magma")

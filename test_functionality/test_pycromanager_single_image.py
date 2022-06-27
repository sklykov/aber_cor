# -*- coding: utf-8 -*-
"""
Sample of simple single image acquisition using uManager.

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3
"""
# %% Imports
import os
import matplotlib.pyplot as plt
import numpy as np

# %% Basic initialization
# !!! If the ZMQ server versions in Java and Python are different,
# I faced the problem with exception that Python str not transferred to java.lang.String
# Initialization of pycromanager (installed by the pip)
from pycromanager import Core, Acquisition
core = Core(); print(core)

# %% Attempt to use section 'Acquisition' for single snap image (image not read)
# current_active_path = str(os.path.dirname(__file__))
# standard_image_name = 'snap'
# # Specifiyng the single image acquisition event
# event = {'axes': {'z': 0}}
# with Acquisition(directory=current_active_path, name=standard_image_name) as acq:
#     acq.acquire(event)
#     image = acq.get_dataset().read_image()

# %% Acquisition from Denoising applications example
# Straight-forward way of acquiring of a single image (snap) and use it for further processing
core.snap_image()
tagged_image = core.get_tagged_image()
pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"],
                                                tagged_image.tags["Width"]])
plt.imshow(pixels, cmap="magma")
# Make some calculations on the image (FFT, metric counting - should take not so much)

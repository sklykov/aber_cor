# -*- coding: utf-8 -*-
"""
Concatenate different calculated integral matricies.

@author: ssklykov
"""
# %% Imports
import os
import numpy as np
calibration_folder = os.path.dirname(__file__)

# %% Compose already calculated values
Z1_name = "IntegralMatrix14Zernike_RecordedAberrations.npy"; Z1_path = os.path.join(calibration_folder, Z1_name)
Z2_name = "IntegralMatrix5OrderZernike_RecordedAberrations.npy"; Z2_path = os.path.join(calibration_folder, Z2_name)
Z_overall_name = "IntegralMatrix20Zernike_RecordedAberrations.npy"; Z_overall_path = os.path.join(calibration_folder, Z_overall_name)
if not(os.path.exists(Z_overall_path)) and os.path.exists(Z1_path) and os.path.exists(Z2_path):
    Z1 = np.load(Z1_path); Z2 = np.load(Z2_path)
    Z_overall = np.concatenate((Z1, Z2), axis=1)
    # np.save(Z_overall_path, Z_overall)

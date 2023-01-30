# -*- coding: utf-8 -*-
"""
Test IDS camera initialization.

Reference
---------
    https://en.ids-imaging.com/programming-examples-details/simple-live-image-acquisition-with-the-python-interface-pyueye.html

"""
import time
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

try:
    from pyueye import ueye
    n_cameras = ueye.INT()
    ueye.is_GetNumberOfCameras(n_cameras)
    print(n_cameras)
    camera_reference = ueye.HIDS(0)  # 0 = first available camera
    # From IDS example:
    sInfo = ueye.SENSORINFO()
    cInfo = ueye.CAMINFO()
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    rectAOI = ueye.IS_RECT()
    pitch = ueye.INT()
    nBitsPerPixel = ueye.INT(8)    # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
    channels = 1  # 3: channels for color mode(RGB); take 1 channel for monochrome
    m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
    bytes_per_pixel = int(nBitsPerPixel / 8)
    camera_status = ueye.is_InitCamera(camera_reference, None)  # see the example from IDS
    # print("Success status number:", ueye.IS_SUCCESS)
    if camera_status == ueye.IS_SUCCESS:
        print("camera status is OK ")
        # Below - calls of function for getting info afterwards
        ueye.is_ResetToDefault(camera_reference)  # without resetting to default, it fails to get images below
        ueye.is_GetCameraInfo(camera_reference, cInfo)
        ueye.is_GetSensorInfo(camera_reference, sInfo)
        ueye.is_AOI(camera_reference, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
        # ueye.is_SetDisplayMode(camera_reference, ueye.IS_SET_DM_DIB)
        ueye.is_SetColorMode(camera_reference, ueye.IS_CM_MONO8)

        # Set the right color mode
        if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        else:
            # for monochrome camera models use Y8 mode
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("else")

        # Some info
        print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
        width = rectAOI.s32Width
        height = rectAOI.s32Height
        print("Maximum image width:\t", width)
        print("Maximum image height:\t", height)

        # Captrure (memory preallocation and starting of capturing) settings
        ueye.is_AllocImageMem(camera_reference, width, height, nBitsPerPixel, pcImageMemory, MemID)
        ueye.is_SetImageMem(camera_reference, pcImageMemory, MemID)
        ueye.is_CaptureVideo(camera_reference, ueye.IS_DONT_WAIT)
        ueye.is_InquireImageMem(camera_reference, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)

        # Exposure time settings - check local documentation after installing IDS
        var = ueye.IS_EXPOSURE_CMD_SET_EXPOSURE.real
        set_exp_t = ueye.DOUBLE(0.1)
        rc = ueye.is_Exposure(camera_reference, var, set_exp_t, 8)

        exposure_time = ueye.DOUBLE()
        var = ueye.IS_EXPOSURE_CMD_GET_EXPOSURE.real
        rc = ueye.is_Exposure(camera_reference, var, exposure_time, 8)
        print(f'exposure time = {exposure_time}')
        var = ueye.IS_EXPOSURE_CMD_SET_EXPOSURE.real

        # Acquisition
        i = 0
        while i < 10:
            # Extract and reshape the image
            array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=True)
            frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
            print(np.max(frame))
            time.sleep(0.02)
            # Display the images after #5
            if i > 0:
                set_exp_t = ueye.DOUBLE(500*(i-4)/1000)
                rc = ueye.is_Exposure(camera_reference, var, set_exp_t, 8)
                plt.figure()
                plt.axis('off')
                plt.imshow(frame, cmap='gray')
                plt.tight_layout()
            i += 1

        time.sleep(1)
    else:
        print("Could't initialize a camera")


except ValueError or ImportError or NotImplementedError as e:
    print(e)

finally:
    ueye.is_FreeImageMem(camera_reference, pcImageMemory, MemID)
    ueye.is_ExitCamera(camera_reference)

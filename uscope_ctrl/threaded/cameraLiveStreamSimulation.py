# -*- coding: utf-8 -*-
"""
Compose classes for camera images streaming simulation.

@author: ssklykov
"""
# %% Imports
from generate_noise_pic import generate_noise_picture
from threading import Thread
import numpy as np
import time
from queue import Empty


# %% Classes definitions - for further imports inside the GUI interfaces or other top-level programs
# %% Class wrapper for threaded noisy single picture generation
class SingleImageThreadedGenerator(Thread):
    """Threaded class for generation of single noisy image."""

    height = 100; width = 100; image = np.zeros((height, width), dtype='uint8')

    def __init__(self, height: int = 100, width: int = 100):
        Thread.__init__(self); self.height = height; self.width = width

    def run(self):
        """
        Generate of single image and storing it inside the class variable.

        Returns
        -------
        None.

        """
        self.image = generate_noise_picture(self.width, self.height)  # Width and height order re-defined in the subfunction as well


# %% Class wrapper for threaded noisy continuous pictures generation for checking the performance of generation in Python
class ContinuousImageThreadedGenerator(Thread):
    """Threaded class for generation of continuous stream of noisy images and updating the ImageView widget."""

    meanGenTimes = np.zeros(200, dtype="uint"); meanGenerationT = 0

    def __init__(self, imageWidget, messages_queue, refresh_delay_ms: int = 100,
                 height: int = 100, width: int = 100, testPerformance: bool = False, autoRange: bool = True):
        """
        Initialize of threaded class for continuous generation of noisy pictures.

        Parameters
        ----------
        imageWidget : pyqtgraph.ImageView
            ImageView widget from pyqtgraph for showing the stream of generated images.
        messages_queue : queue.Queue
            The communication queue through which the message to stop generation is transferred.
        refresh_delay_ms : int, optional
            Delay between each new generation of noisy picture. The default is 100.
        height : int, optional
            Of the generated image. The default is 100.
        width : int, optional
            Of the generated image. The default is 100.
        testPerformance : bool, optional
            Flag for storing and printing mean value of passed for each generation time (refresh delay + overhead). The default is False.
        autoRange: bool, optional
            Flag for enabling / disabling of auto range (scale and place) of new generated images. The default is True.

        Returns
        -------
        None.

        """
        Thread.__init__(self); self.height = height; self.width = width; self.testPerformance = testPerformance
        self.messages_queue = messages_queue; self.autoRange = autoRange
        self.imageWidget = imageWidget; self.refresh_delay_ms = refresh_delay_ms

    def run(self):
        """
        Make continuous generation of noisy pictures and updating the ImageView widget from pyqtgraph for their showing.

        Returns
        -------
        None.

        """
        flag_generation = True
        i = 0  # for adding the elements into preinitilized array for further mean generation time calculation
        while(flag_generation):
            if self.testPerformance:
                t1 = time.time()
                # Below - the workaround for preventing kernel diying during continuous generation without any delays
                if self.refresh_delay_ms == 0:  # if the delay between frames is 0, than the generation is unstable
                    self.refresh_delay_ms += 1  # make the delay at least 1 ms
            image = generate_noise_picture(self.width, self.height)  # Get the noisy picture, width and height swapped
            # Set the image for representation by passed ImageView pyqtgraph widget - below
            self.imageWidget.setImage(image, autoRange=self.autoRange)
            time.sleep(self.refresh_delay_ms/1000)  # Applying artificial delays between each image generation
            # If testing of Performance requested, then accumulating of passed times in the array performed
            if self.testPerformance:
                t2 = time.time()
                if i < np.size(self.meanGenTimes):
                    self.meanGenTimes[i] = np.uint(np.round((t2-t1)*1000, 0)); i += 1  # Add the passed time to the array
            # Checking the messages for getting each time the notification about stopping the generation and thread
            if (self.messages_queue.qsize() > 0) and not(self.messages_queue.empty()):
                try:
                    message = self.messages_queue.get_nowait()  # trying to get the stored message
                except Empty:  # Embedded queue.Empty exception raised when the Queue is empty but quired by get() method
                    pass
                if isinstance(message, str):  # the obtained message has the String type
                    if message == "Stop Generation" or message == "Stop":
                        flag_generation = False   # Stop the while loop above
                elif isinstance(message, Exception):
                    flag_generation = False  # if the obtained message has the Exception type, then stop generation
        # If generation stopped and the test of performance was asked, then print out the mean generation time
        if self.testPerformance:
            # Calculation of the final element that in the array is zero (passed time not saved)
            for j in range(np.size(self.meanGenTimes)):
                if self.meanGenTimes[j] == 0:
                    break
            self.meanGenTimes = self.meanGenTimes[0:j]  # Truckate array till the non-zerp element for mean value calculation
            self.meanGenerationT = np.uint(np.round(np.mean(self.meanGenTimes), 0))
            print("Mean generation time is:", self.meanGenerationT, "ms")


# %% Attempt to create implementation of Process class is failed because of complications with its import to other module
# !!! Seems that implementation of Process from multiprocessing and further importing it as submodule to the main python
# script can't be achieved

# %% Tests of functionality
if __name__ == "__main__":
    pass

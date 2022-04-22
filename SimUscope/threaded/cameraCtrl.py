# -*- coding: utf-8 -*-
"""
Class wrapper for controlling PCO camera using separate thread and implementation of its native functions.

@author: ssklykov
"""
# %% Imports
from threading import Thread
from queue import Queue, Empty
import time
import pco


# %% Class wrapper
class PCOcamera(Thread):
    """Class for wrapping controls of the PCO camera and provided API features."""

    initialized = False  # Start the mail inifinite loop if the class initialized
    mainLoopTimeDelay = 25  # Internal constant - delaying for the main loop for receiving and process the commands
    liveStream: bool  # force type checking
    exposure_time_ms: int

    def __init__(self, messagesQueue: Queue, exceptionsQueue: Queue, imagesQueue: Queue, exposure_time_ms: int):
        self.messagesQueue = messagesQueue  # For receiving the commands to stop / start live stream
        self.exceptionsQueue = exceptionsQueue  # For adding the exceptions that should stop the main program
        Thread.__init__(self)  # Initialize this class in the other thread
        self.initialized = True  # Additional flag for the start the loop in the run method
        self.imagesQueue = imagesQueue  # For storing the acquired images
        self.liveStream = False  # Set default live stream state to false
        self.exposure_time_ms = exposure_time_ms  # Initializing with the default exposure time
        # Initialization code for the camera
        try:
            self.cameraReference = pco.Camera()
            print("The PCO camera initialized")
            self.cameraReference.set_exposure_time(self.exposure_time_ms/1000)
            print("Default exposure ime is set for the camera")
            print("The camera status report:", self.cameraReference.sdk.get_camera_health_status())
        except ImportError:
            print("The PCO library is unavailable, check the installation. The camera not initialized!")
            self.cameraReference = None
        except ValueError:
            print("CAMERA NOT INITIALIZED! THE HANDLE TO IT - NONE")
            self.cameraReference = None
        self.max_width = 1024; self.max_height = 1024

    def run(self):
        """
        Keep the camera initializedand waiting the commands to start Live Stream / Single snap imaging.

        Returns
        -------
        None.

        """
        while self.initialized:
            # Checking for commands created by clicking buttons or by any events
            if not(self.messagesQueue.empty()) and (self.messagesQueue.qsize() > 0):
                try:
                    message = self.messagesQueue.get_nowait()  # get the message from the main controlling GUI
                    if isinstance(message, str):
                        if message == "Close the camera" or message == "Stop" or message == "Stop Program":
                            try:
                                print("Received by the camera process:", message)
                                self.close()  # closing the connection to the camera and release all resources
                            except Exception as error:
                                print("Raised exception during closing the camera:", error)
                                self.exceptionsQueue.put_nowait(error)  # re-throw to the main program the error
                            finally:
                                self.initialized = False  # In any case stop the loop waiting the commands from the GUI
                        if message == "Start Live Stream":
                            print("Camera start live streaming")
                            self.live_imaging()  # call the function
                        if message == "Snap single image":
                            try:
                                # The single acquired image is sent back to the calling controlling program via Queue
                                self.snap_single_image()
                            except Exception as e:
                                # Any encountered exceptions should be reported to the main controlling program
                                self.close()  # An attempt to close the camera
                                self.initialized = False  # Stop this running loop
                                self.exceptionsQueue.put_nowait(e)  # Send to the main controlling program the caught Exception e
                        if message == "Restore Full Frame":
                            print("Full frame image will be restored with sizes: ", self.max_width, self.max_height)  # TODO
                    if isinstance(message, tuple):
                        (command, parameters) = message
                        if command == "Crop Image":
                            print("Received sizes for cropping:", parameters)  # TODO
                        if command == "Set exposure time":
                            self.setExposureTime(parameters)
                    if isinstance(message, Exception):
                        print("Camera will be stopped because of throw from the main GUI exception")
                        self.close()
                except Empty:
                    pass

            time.sleep(self.mainLoopTimeDelay/1000)  # Delays of each step of processing of commands
        print("The threaded process for the camera is closed")

    def snap_single_image(self):
        """
        Get the single image from the camera.

        Returns
        -------
        None.

        """
        if self.cameraReference is not None:
            self.cameraReference.record(number_of_images=1, mode='sequence')  # setup the camera to acquire a single image
            self.cameraReference.wait_for_first_image()
            image, metadata = self.cameraReference.image()  # get the single image
            self.imagesQueue.put_nowait(image)  # put the image to the queue for getting it in the main thread
        else:
            self.imagesQueue.put_nowait("String replacer for an image")

    def live_imaging(self):
        """
        Make of Live Imaging stream for the PCO camera.

        Returns
        -------
        None.

        """
        self.liveStream = True
        if self.cameraReference is not None:  # then it's supposed that the camera has been properly initialized
            self.cameraReference.record(number_of_images=25, mode='ring buffer')  # configure the live stream acquisition
            # make the loop below for infinite live stream, that could be stopped only by receiving the command or exception
            print("Live streaming started by the PCO camera")
            while (self.liveStream):
                self.cameraReference.wait_for_first_image()  # wait that the image acquired actually
                image, metadata = self.cameraReference.image()  # get the acquired image
                if not(self.imagesQueue.full()):
                    self.imagesQueue.put_nowait(image)  # put the image to the queue for getting it in the main thread
                if not(self.messagesQueue.empty()) and (self.messagesQueue.qsize() > 0):
                    try:
                        message = self.messagesQueue.get_nowait()  # get the message from the main controlling GUI
                        if isinstance(message, str):
                            if message == "Stop Live Stream":
                                print("Camera stop live streaming")
                                self.cameraReference.stop()  # stop the live stream from the camera
                                self.liveStream = False; break
                        elif isinstance(message, Exception):
                            print("Camera stop live streaming because of reported error")
                            self.cameraReference.stop()  # stop the live stream from the camera
                            self.messagesQueue.put_nowait(message)  # setting for run() method again the error report for stopping the camera
                            self.liveStream = False; break
                    except Empty:
                        pass
                time.sleep(0.015)  # trying to induce the delay between acquiring and send images, for letting the GUI refresh image
        else:
            # Substituion of actual image generation
            while (self.liveStream):
                self.exposure_time_ms += 10  # some overhead delay
                # Below time sleep - the artificial application of delays resembling the acquisition of images from the camera
                time.sleep(self.exposure_time_ms/1000)
                self.imagesQueue.put_nowait("Live Image substituted by this string")
                if not(self.messagesQueue.empty()) and (self.messagesQueue.qsize() > 0):
                    try:
                        message = self.messagesQueue.get_nowait()  # get the message from the main controlling GUI
                        if isinstance(message, str):
                            if message == "Stop Live Stream":
                                print("Camera stop live streaming")
                                self.liveStream = False; break
                        elif isinstance(message, Exception):
                            print("Camera stop live streaming because of reported error")
                            self.messagesQueue.put_nowait(message)  # setting for run() method again the error report for stopping the camera
                            self.liveStream = False; break
                    except Empty:
                        pass

    def setExposureTime(self, exposure_time_ms: int):
        """
        Set exposure time for the camera.

        Parameters
        ----------
        exposure_time_ms : int
            Provided value for exposure time from GUI.

        Returns
        -------
        None.

        """
        if isinstance(exposure_time_ms, int):
            if exposure_time_ms == 0:
                exposure_time_ms = 1
            self.exposure_time_ms = exposure_time_ms
            if self.cameraReference is not None:  # if the camera is really activated, then call the function
                self.cameraReference.set_exposure_time(self.exposure_time_ms/1000)
            print("The set exposure time:", self.exposure_time_ms)

    def close(self):
        """
        Deinitialize the camera and close the connection to it.

        Returns
        -------
        None.

        """
        if self.cameraReference is not None:  # If the camera was initialized, then the cameraReference will be not None
            self.cameraReference.close()
        time.sleep(self.mainLoopTimeDelay/1000)
        print("The PCO camera closed")

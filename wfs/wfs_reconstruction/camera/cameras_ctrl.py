# -*- coding: utf-8 -*-
"""
Class wrapper for controlling the IDS or simulated camera using separate process.

It implements API functions provided in the IDS controlling library https://pypi.org/project/pyueye/

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3 (check https://www.gnu.org/licenses/ )
"""
# %% Imports - global dependecies (from standard library and installed by conda / pip)
from multiprocessing import Process, Queue
from queue import Empty, Full
import time
import numpy as np


# %% Class wrapper
class CameraWrapper(Process):
    """Class for wrapping controls of the IDS camera and provided API features."""

    initialized = False  # Start the mail infinite loop if the class initialized
    main_loop_time_delay = 25  # Internal constant - delaying for the main loop for receiving and process the commands
    live_stream_flag: bool  # force type checking
    exposure_time_ms: int
    camera_type: str
    max_width: int; max_height: int
    image_width: int; image_height: int
    camera_reference = None

    def __init__(self, messages_queue: Queue, exceptions_queue: Queue, images_queue: Queue, messages2caller: Queue,
                 exposure_t_ms: int, image_width: int, image_height: int, camera_type: str = "Simulated"):
        Process.__init__(self)  # Initialize this class on the separate process with its own memory and core
        self.messages_queue = messages_queue  # For receiving the commands to stop / start live stream
        self.exceptions_queue = exceptions_queue  # For adding the exceptions that should stop the main program
        self.messages2caller = messages2caller  # For sending internal messages from this class for debugging
        self.images_queue = images_queue  # For storing the acquired images
        self.live_stream_flag = False  # Set default live stream state to false
        self.exposure_time_ms = exposure_t_ms  # Initializing with the default exposure time
        self.camera_type = camera_type  # Type of initialized camera
        # Initialization code for the IDS camera -> MOVED to the run() because it's impossible to pickle handle to the camera
        if self.camera_type == "IDS":
            # All the camera initialization code moved to the run() method!
            # It's because the camera handle returned by the DLL library is not pickleable by the pickle method !!!
            try:
                # from pyueye import ueye
                print("The controlling library for IDS camera installed: ")
                self.initialized = False  # Additional flag for the start the loop in the run method (run Process!)
                self.max_width = 2048; self.max_height = 2048
            except ImportError:
                print("The IDS library for controlling of the camera isn't available."
                      + " Install it from: ")
                self.initialized = False
        # Initialization code for the Simulated camera
        elif self.camera_type == "Simulated":
            self.camera_reference = None  # no associated library call to any API functions
            self.initialized = True  # Additional flag for the start the loop in the run() method
            self.max_width = image_width; self.max_height = image_height
            self.image_height = image_height; self.image_width = image_width
            try:
                self.generate_noise_picture()
                self.messages2caller.put_nowait("The Simulated camera initialized")
            except Exception as e:
                # Only arises if image width or height are too small (less than 2 pixels)
                self.messages_queue.put_nowait(str(e))
                self.exceptions_queue.put_nowait(e)
                self.initialized = False
        # Stop initialization because the camera type couldn't be recognized
        else:
            self.camera_reference = None
            self.messages2caller.put_nowait("The specified type of the camera hasn't been implemented")
            self.initialized = False  # Additional flag for the start the loop in the run method
            self.exceptions_queue.put_nowait(Exception("The specified type of the camera not implemented"))

    def run(self):
        """
        Keep the camera initialized waiting the commands to start Live Stream / Single snap imaging.

        Returns
        -------
        None.

        """
        print("**** Camera Process() print stream WILL BE PRINTED BELOW if the camera changed ****")
        if self.camera_type == "IDS":
            # Since the import of the IDS library already has been tested above, again import for availability:
            # from pyueye import ueye
            try:
                raise ValueError
                # self.camera_reference = ... # open connection to the camera in the default mode
                self.messages2caller.put_nowait("The IDS camera initialized")
                # self.messages2caller.put_nowait("Default exposure time is set ms: " )
                self.initialized = True  # Additional flag for the start the loop in the run metho
                self.images_queue.put_nowait("The IDS camera initialized")
            except ValueError:
                self.messages2caller.put_nowait("CAMERA NOT INITIALIZED! THE HANDLE TO IT - 'NONE'")  # Only for debugging
                self.images_queue.put_nowait("The Simulated IDS camera initialized")  # Notify the main GUI about initialization
                self.camera_reference = None
        else:
            print("**** Simulated Camera Print Stream ****")
        self.messages2caller.put_nowait(self.camera_type + " camera Process has been launched")  # Send the command for debugging
        # The main loop - the handler in the Process loop
        n_check_camera_status = 0  # Check and report the status of the IDS camera each dozen of seconds (specification below)
        # Below - the loop that receives the commands from GUI and initialize function to handle them
        while self.initialized:
            # Checking for commands created by clicking buttons or by any events
            if not(self.messages_queue.empty()) and (self.messages_queue.qsize() > 0) and self.initialized:
                try:
                    message = self.messages_queue.get_nowait()  # get the message from the main controlling GUI
                    if isinstance(message, str):
                        # Close the Camera
                        if ((message == "Close the camera" or message == "Stop" or message == "Stop Program")
                           or (message == "Close Camera")):
                            try:
                                self.messages2caller.put_nowait("Received by the Camera: " + message)
                                self.close()  # closing the connection to the camera and release all resources
                                if self.camera_reference is not None:
                                    self.camera_reference = None
                                self.images_queue.close()  # close this queue for further usage
                            except Exception as error:
                                self.messages2caller.put_nowait("Raised exception during closing the camera:" + str(error))
                                self.exceptions_queue.put_nowait(error)  # re-throw to the main program the error
                            finally:
                                self.initialized = False; break  # In any case stop the loop waiting the commands from the GUI

                        # Live stream mode
                        if message == "Start Live Stream":
                            self.messages2caller.put_nowait("Camera start live streaming")
                            try:
                                self.live_imaging()  # call the function
                            except Exception as error:
                                self.messages2caller.put_nowait("Error string: " + str(error))
                                self.messages2caller.put_nowait(str(error).split(sep=" "))
                                self.exceptions_queue.put_nowait(error)

                        # Acquiring single image
                        if message == "Snap single image":
                            try:
                                # The single acquired image is sent back to the calling controlling program via Queue
                                self.snap_single_image()
                                if not self.messages2caller.full():
                                    self.messages2caller.put_nowait("Single image snap performed")
                            except Exception as e:
                                # Any encountered exceptions should be reported to the main controlling program
                                self.close()  # An attempt to close the camera
                                self.initialized = False  # Stop this running loop
                                self.exceptions_queue.put_nowait(e)  # Send to the main controlling program the caught Exception e

                        # Check and return the actual camera status
                        if message == "Get the IDS camera status":
                            self.return_camera_status()

                        # Restore full frame
                        if message == "Restore Full Frame":
                            self.messages2caller.put_nowait(("Full frame restored: " + str((self.max_width, self.max_height))))
                            self.restore_full_frame()  # TODO
                    if isinstance(message, tuple):
                        (command, parameters) = message
                        if command == "Crop Image":
                            # Send back for debugging crop parameters - below
                            self.messages2caller.put_nowait("Crop coordinates: " + str(parameters))
                            (yLeftUpper, xLeftUpper, height, width) = parameters
                            self.crop_image(yLeftUpper, xLeftUpper, width, height)  # TODO
                        # Set exposure time
                        if command == "Set exposure time":
                            self.set_exposure_time(parameters)
                        # Set the new image sizes for Simulated camera
                        if command == "Change simulate picture sizes to:":
                            (command, parameters) = message
                            (width, height) = parameters
                            self.update_simulated_sizes(width, height)
                    # Exceptions handling => close the camera if it receives from other parts of the program the exception
                    if isinstance(message, Exception):
                        print("Camera will be stopped because of throw from the main GUI exception")
                        self.close()
                except Empty:
                    pass
            time.sleep(self.main_loop_time_delay/1000)  # Delays of each step of processing of commands
            # Below: check each 20s the camera status and send it to the main GUI
            if self.camera_type == "IDS" and self.camera_reference is not None:
                if n_check_camera_status < (300000/self.main_loop_time_delay):
                    n_check_camera_status += 1
                else:
                    n_check_camera_status = 0
                    self.messages2caller.put_nowait("Camera status after 5min: "
                                                    + str(self.camera_reference.sdk.get_camera_health_status()))
        self.messages2caller.put_nowait("run() of Process finished for " + self.camera_type + " camera")  # DEBUG

    def snap_single_image(self):
        """
        Get the single image from the camera.

        Returns
        -------
        None.

        """
        if (self.camera_reference is not None) and (self.camera_type == "IDS"):
            print("specify code for IDS")
            # self.messages2caller.put_nowait("Image timing: " + str(self.camera_reference.sdk.get_image_timing()))
            # self.messages2caller.put_nowait("Delay exp time infor: " + str(self.camera_reference.sdk.get_delay_exposure_time()))
            # self.images_queue.put_nowait(image)  # put the image to the queue for getting it in the main thread
        elif (self.camera_reference is None) and (self.camera_type == "IDS"):
            self.images_queue.put_nowait("String replacer of an image")
        elif self.camera_type == "Simulated":
            image = self.generate_noise_picture()  # No need to evoke try, because the width and height conformity already checked
            self.images_queue.put_nowait(image)

    def live_imaging(self):
        """
        Make of Live Imaging stream for the IDS camera.

        Returns
        -------
        None.

        """
        self.live_stream_flag = True  # flag for the infinite loop for the streaming images continuously
        # self.messages2caller.put_nowait("AT = Time of Acquisition, TT - time for putting image to queue")  # DEBUG
        if (self.camera_type == "IDS") and (self.camera_reference is not None):  # check that physical camera initialized
            # self.messages2caller.put_nowait("Settings: " + str(self.camera_reference.rec.get_settings()))  # DEBUG
            # self.messages2caller.put_nowait("Status of recorder: " + str(self.camera_reference.rec.get_status()))  # DEBUG
            print("specify code for IDS")
        # General handle for live-streaming - starting with acquiring of the first image
        while self.live_stream_flag:
            # then it's supposed that the camera has been properly initialized - below
            if (self.camera_type == "IDS") and (self.camera_reference is not None):
                # make the loop below for infinite live stream, that could be stopped only by receiving the command or exception
                # try:
                #     if not (self.images_queue.full()):
                #         try:
                #             self.images_queue.put_nowait(image)  # put the image to the queue for getting it in the main thread
                #         except Full:
                #             pass  # do nothing for now about the overloaded queue
                # except Exception as error:
                #     self.messages2caller.put_nowait("The Live Mode finished by IDS camera because of thrown Exception")
                #     self.messages2caller.put_nowait("Thrown error: " + str(error))
                #     self.live_stream_flag = False  # stop the loop
                print("specify code for IDS")
            elif (self.camera_type == "IDS") and (self.camera_reference is None):
                # Substitution of actual image generation by the IDS camera
                time.sleep(self.exposure_time_ms/1000)
                self.images_queue.put_nowait("Live Image substituted by this string")
            elif self.camera_type == "Simulated":
                image = self.generate_noise_picture()  # simulate some noise image
                if not(self.images_queue.full()):
                    self.images_queue.put_nowait(image)
                time.sleep(self.exposure_time_ms/1000)  # Delay due to the simulated exposure
            # Below - checking for the command "Stop Live stream
            if not(self.messages_queue.empty()) and (self.messages_queue.qsize() > 0):
                try:
                    message = self.messages_queue.get_nowait()  # get the message from the main controlling GUI
                    if isinstance(message, str):
                        if message == "Stop Live Stream":
                            self.messages2caller.put_nowait("Camera stop live streaming")
                            if (self.camera_type == "IDS") and (self.camera_reference is not None):
                                self.camera_reference.stop()  # stop the live stream from the IDS  camera
                            self.live_stream_flag = False; break
                    elif isinstance(message, Exception):
                        self.messages2caller.put_nowait("Camera stop live streaming because of the reported error")
                        if (self.camera_type == "IDS") and (self.camera_reference is not None):
                            self.camera_reference.stop()  # stop the live stream from the IDS camera
                        self.messages_queue.put_nowait(message)  # send for run() method again the error report
                        self.live_stream_flag = False; break
                except Empty:
                    pass

    def set_exposure_time(self, exposure_t_ms: int):
        """
        Set exposure time for the camera.

        Parameters
        ----------
        exposure_t_ms : int
            Provided value for exposure time from GUI.

        Returns
        -------
        None.

        """
        if isinstance(exposure_t_ms, int):
            if exposure_t_ms <= 0:  # exposure time cannot be 0
                exposure_t_ms = 1
            self.exposure_time_ms = exposure_t_ms
            # if the camera is really activated, then call the function
            if self.camera_reference is not None and self.camera_type == "IDS":
                print("specify code for IDS")
                # Report back the set exposure time for the actual camera
                self.messages2caller.put_nowait("The set exposure time ms: "
                                                + str(int(np.round(1000*(self.camera_reference.get_exposure_time()), 0))))

    def return_camera_status(self):
        """
        Put to the printing stream the camera status.

        Returns
        -------
        None.

        """
        if (self.camera_type == "IDS") and (self.camera_reference is not None):
            print("specify code for IDS")

    def crop_image(self, yLeftUpper: int, xLeftUpper: int, width: int, height: int):
        """
        Crop selected ROI from the image.

        Parameters
        ----------
        yLeftUpper : int
            y coordinate of left upper corner of ROI region.
        xLeftUpper : int
            x coordinate of left upper corner of ROI region.
        width : int
            ROI width.
        height : int
            ROI height.

        Returns
        -------
        None.

        """
        # For simulated camera only reassign the image height and width
        if self.camera_type == "Simulated":
            self.image_width = width; self.image_height = height
        # For actual IDS camera implementation
        if (self.camera_type == "IDS") and (self.camera_reference is not None):  # check that physical camera initialized
            print("specify code for IDS")

    def restore_full_frame(self):
        """
        Restore full size image.

        Returns
        -------
        None.

        """
        # For simulated camera only reassign the image height and width
        if self.camera_type == "Simulated":
            self.image_width = self.max_width; self.image_height = self.max_height
        # For actual IDS camera implementation
        if (self.camera_type == "IDS") and (self.camera_reference is not None):  # check that physical camera initialized
            print("specify code for IDS")

    def update_simulated_sizes(self, width: int, height: int):
        """
        Update of sizes of a simulated image.

        Parameters
        ----------
        width : int
            Updated image width.
        height : int
            Updated image height.

        Returns
        -------
        None.

        """
        self.image_width = width; self.max_width = width; self.image_height = height; self.max_height = height

    def close(self):
        """
        De-initialize the camera and close the connection to it.

        Returns
        -------
        None.

        """
        #  If the camera - IDS, then call close() function from the IDS module
        if (self.camera_type == "IDS") and (self.camera_reference is not None):
            print("specify code for IDS")
        time.sleep(self.main_loop_time_delay/1000)  #
        self.messages2caller.put_nowait("The " + self.camera_type + " camera close() performed")

    def generate_noise_picture(self, pixel_type: str = 'uint8') -> np.ndarray:
        """
        Generate of a noise image with even distribution of noise (pixel values) on that.

        Parameters
        ----------
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
        height = self.image_height; width = self.image_width
        if (height >= 2) and (width >= 2):
            if pixel_type == 'uint8':
                img = np.random.randint(0, high=255, size=(height, width), dtype='uint8')
            if pixel_type == 'float':
                img = np.random.rand(height, width)
        else:
            raise Exception("Specified height or width are less than 2")

        return img


# %% It's testing and unnecessary code, only valuable to check if the real camera is initialized before running the GUI
if __name__ == "__main__":
    messages_queue = Queue(maxsize=5); exceptions_queue = Queue(maxsize=2); images_queue = Queue(maxsize=2)
    messages2caller = Queue(maxsize=5); exposure_time_ms = 100; img_width = 100; img_height = 100
    camera = CameraWrapper(messages_queue, exceptions_queue, images_queue, messages2caller,
                           exposure_time_ms, img_width, img_height, camera_type="IDS")  # "Simulated", "IDS"
    camera.start(); time.sleep(7)
    if not(messages2caller.empty()):
        while not(messages2caller.empty()):
            try:
                print(messages2caller.get_nowait())
            except Empty:
                pass
    messages_queue.put_nowait("Stop"); time.sleep(4)
    if not(messages2caller.empty()):
        while not(messages2caller.empty()):
            try:
                print(messages2caller.get_nowait())
            except Empty:
                pass
    camera.join()

# -*- coding: utf-8 -*-
"""
Class wrapper for controlling PCO or simulated camera using separate process.

It implements API functions provided in the pco controlling library https://pypi.org/project/pco/
For dependency list - see the imports (the simplest form).

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3 (check https://www.gnu.org/licenses/ )
"""
# %% Imports - global dependecies (from standard library and installed by conda / pip)
from multiprocessing import Process, Queue
from queue import Empty, Full
import time
import numpy as np
import skimage.io as io
import os


# %% Class wrapper
class CameraWrapper(Process):
    """Class for wrapping controls of the PCO camera and provided API features."""

    initialized = False  # Start the mail infinite loop if the class initialized
    main_loop_time_delay = 25  # Internal constant - delaying for the main loop for receiving and process the commands
    live_stream_flag: bool  # force type checking
    exposure_time_ms: int
    camera_type: str
    max_width: int; max_height: int
    image_width: int; image_height: int
    camera_reference = None
    replace_img_camera = None

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
        # Initialization code for the PCO camera -> MOVED to the run() because it's impossible to pickle handle to the camera
        if self.camera_type == "PCO":
            # All the camera initialization code moved to the run() method!
            # It's because the camera handle returned by the DLL library is not pickleable by the pickle method !!!
            # Additionally, the import statement of controlling library moved here
            try:
                import pco
                print("The controlling library for PCO camera installed: ", pco.__package__)
                self.initialized = True  # Additional flag for the start the loop in the run method (run Process!)
                self.max_width = 2048; self.max_height = 2048
            except ImportError:
                print("The pco library for controlling of the camera isn't available."
                      + " Install it from: https://pypi.org/project/pco/")
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
        # !!! Below - initialization code for the PCO camera, because in the __init__() method it's impossible to make,
        # because the handle returned by the call pco.Camera() is the not pickleable object
        print(f"**** {self.camera_type} Process() print stream start ****")
        if self.camera_type == "PCO":
            # Since the import of the pco library already has been tested above, again import for availability:
            import pco
            try:
                self.camera_reference = pco.Camera(debuglevel='none')  # open connection to the camera with some default mode
                # printing statement won't be redirected to stdout, thus sending this message to the queue
                self.messages2caller.put_nowait("The PCO camera initialized")
                self.camera_reference.set_exposure_time(self.exposure_time_ms/1000)
                self.messages2caller.put_nowait("Default exposure time is set ms: "
                                                + str(int(np.round(1000*(self.camera_reference.get_exposure_time()), 0))))
                self.messages2caller.put_nowait("The camera status report: "
                                                + str(self.camera_reference.sdk.get_camera_health_status()))
                self.messages2caller.put_nowait("The camera acquire mode: " + str(self.camera_reference.sdk.get_acquire_mode()))
                self.messages2caller.put_nowait("The camera trigger mode: " + str(self.camera_reference.sdk.get_trigger_mode()))
                # self.messages2caller.put_nowait("Sizes of full image " + str(self.camera_reference.sdk.get_sizes()))  # DEBUG
                self.max_width = self.camera_reference.sdk.get_sizes()['y max']
                self.max_height = self.camera_reference.sdk.get_sizes()['x max']
                self.messages2caller.put_nowait("Camera max width / height: " + str((self.max_width, self.max_height)))  # DEBUG
                # self.messages2caller.put_nowait("Camera ROI " + str(self.camera_reference.sdk.get_camera_description()))
                self.initialized = True  # Additional flag for the start the loop in the run method
                self.images_queue.put_nowait("The PCO camera initialized")
            except ValueError:
                self.messages2caller.put_nowait("CAMERA NOT INITIALIZED! THE HANDLE TO IT - 'NONE'")  # Only for debugging
                self.images_queue.put_nowait("The Simulated PCO camera initialized")  # Notify the main GUI about initialization
                self.camera_reference = None
        self.messages2caller.put_nowait(self.camera_type + " camera Process has been launched")  # Send the command for debugging
        # The main loop - the handler in the Process loop
        n_check_camera_status = 0  # Check and report the status of the PCO camera each dozen of seconds (specification below)
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
                                self.messages2caller.put_nowait("Single image snap performed")
                            except Exception as e:
                                # Any encountered exceptions should be reported to the main controlling program
                                self.close()  # An attempt to close the camera
                                self.initialized = False  # Stop this running loop
                                self.exceptions_queue.put_nowait(e)  # Send to the main controlling program the caught Exception e
                        # Check and return the actual camera status
                        if message == "Get the PCO camera status":
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
            if self.camera_type == "PCO" and self.camera_reference is not None:
                if n_check_camera_status < (300000/self.main_loop_time_delay):
                    n_check_camera_status += 1
                else:
                    n_check_camera_status = 0
                    self.messages2caller.put_nowait("Camera status after 5min: "
                                                    + str(self.camera_reference.sdk.get_camera_health_status()))
        self.messages2caller.put_nowait("run() of Process finished for " + self.camera_type + " camera")  # DEBUG
        print(f"**** {self.camera_type} Process() print stream finish ****")

    def snap_single_image(self):
        """
        Get the single image from the camera.

        Returns
        -------
        None.

        """
        if (self.camera_reference is not None) and (self.camera_type == "PCO"):
            self.camera_reference.record(number_of_images=1, mode='sequence')  # set up the camera to acquire a single image
            (image, metadata) = self.camera_reference.image()  # get the single image
            # self.messages2caller.put_nowait("Image timing: " + str(self.camera_reference.sdk.get_image_timing()))
            # self.messages2caller.put_nowait("Delay exp time infor: " + str(self.camera_reference.sdk.get_delay_exposure_time()))
            self.images_queue.put_nowait(image)  # put the image to the queue for getting it in the main thread
        elif (self.camera_reference is None) and (self.camera_type == "PCO"):
            # self.images_queue.put_nowait("String replacer of an image")  # exchanged to read and put an image below:
            # Read the image placed in the same folder for convenience
            if self.replace_img_camera is None:
                self.beads_image_path = os.path.join(os.path.curdir, "beads.png")
                self.replace_img_camera = io.imread(self.beads_image_path)
                self.images_queue.put_nowait(self.replace_img_camera)
            else:
                self.images_queue.put_nowait(self.replace_img_camera)
        # Snap simulated picture with white noise
        elif self.camera_type == "Simulated":
            image = self.generate_noise_picture()  # No need to evoke try, because the width and height conformity already checked
            self.images_queue.put_nowait(image)

    def live_imaging(self):
        """
        Make of Live Imaging stream for the PCO camera.

        Returns
        -------
        None.

        """
        self.live_stream_flag = True  # flag for the infinite loop for the streaming images continuously
        # self.messages2caller.put_nowait("AT = Time of Acquisition, TT - time for putting image to queue")  # DEBUG
        if (self.camera_type == "PCO") and (self.camera_reference is not None):  # check that physical camera initialized
            self.camera_reference.record(number_of_images=100, mode='fifo')  # configure the live stream acquisition
            # self.messages2caller.put_nowait("Settings: " + str(self.camera_reference.rec.get_settings()))  # DEBUG
            # self.messages2caller.put_nowait("Status of recorder: " + str(self.camera_reference.rec.get_status()))  # DEBUG
        # General handle for live-streaming - starting with acquiring of the first image
        while self.live_stream_flag:
            # then it's supposed that the camera has been properly initialized - below
            if (self.camera_type == "PCO") and (self.camera_reference is not None):
                # t1 = time.time()  # DEBUG
                # make the loop below for infinite live stream, that could be stopped only by receiving the command or exception
                try:
                    self.camera_reference.wait_for_first_image()  # wait that the image acquired actually
                    (image, metadata) = self.camera_reference.image()  # get the acquired image
                    # t3 = time.time()  # DEBUG
                    if not (self.images_queue.full()):
                        try:
                            self.images_queue.put_nowait(image)  # put the image to the queue for getting it in the main thread
                        except Full:
                            pass  # do nothing for now about the overloaded queue
                        # t2 = time.time(); self.messages2caller.put_nowait("AT: " + str(int(np.round(((t3-t1)*1000), 0))) + " | "
                        #                                                    + "TT: " + str(int(np.round(((t2-t1)*1000), 0))))
                except Exception as error:
                    self.messages2caller.put_nowait("The Live Mode finished by PCO camera because of thrown Exception")
                    self.messages2caller.put_nowait("Thrown error: " + str(error))
                    self.live_stream_flag = False  # stop the loop
            elif (self.camera_type == "PCO") and (self.camera_reference is None):
                # Substitution of actual image generation by the PCO camera
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
                            if (self.camera_type == "PCO") and (self.camera_reference is not None):
                                self.camera_reference.stop()  # stop the live stream from the PCO  camera
                            self.live_stream_flag = False; break
                    elif isinstance(message, Exception):
                        self.messages2caller.put_nowait("Camera stop live streaming because of the reported error")
                        if (self.camera_type == "PCO") and (self.camera_reference is not None):
                            self.camera_reference.stop()  # stop the live stream from the PCO camera
                        self.messages_queue.put_nowait(message)  # setting for run() again the error report for stopping the camera
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
            if self.camera_reference is not None and self.camera_type == "PCO":
                self.camera_reference.set_exposure_time(self.exposure_time_ms/1000)
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
        if (self.camera_type == "PCO") and (self.camera_reference is not None):
            self.messages2caller.put_nowait("Camera status: " + str(self.camera_reference.sdk.get_camera_health_status()))
            self.messages2caller.put_nowait(str(self.camera_reference.sdk.get_acquire_mode()))
            self.messages2caller.put_nowait(str(self.camera_reference.sdk.get_frame_rate()))
            self.messages2caller.put_nowait(str(self.camera_reference.sdk.get_acquire_mode_ex()))

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
        # For actual PCO camera implementation
        if (self.camera_type == "PCO") and (self.camera_reference is not None):  # check that physical camera initialized
            self.camera_reference.sdk.set_roi(yLeftUpper + 1, xLeftUpper + 1, (yLeftUpper + width), (xLeftUpper + height))

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
        # For actual PCO camera implementation
        if (self.camera_type == "PCO") and (self.camera_reference is not None):  # check that physical camera initialized
            self.camera_reference.sdk.set_roi(1, 1, self.max_width, self.max_height)

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
        #  If the camera - PCO, then call close() function from the pco module
        if (self.camera_type == "PCO") and (self.camera_reference is not None):
            self.camera_reference.close()
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
                img = np.zeros(shape=(height, width), dtype='uint16')  # blank image
                # Generate a few stripes over noisy image - deprecated now
                # pixel_step = 20; j = pixel_step
                # while j < height:
                #     img[:, j] = 255
                #     j += pixel_step
                img += np.random.randint(0, high=255, size=(height, width), dtype='uint16')
                # maxI = np.max(img); img = (255*(img/maxI)).astype(dtype='uint8')
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
                           exposure_time_ms, img_width, img_height, camera_type="Simulated")  # "Simulated", "PCO"
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

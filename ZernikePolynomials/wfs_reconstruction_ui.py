# -*- coding: utf-8 -*-
"""
GUI for perform wavefront reconstruction.

It's can be used for test of implemented algorithm (refernces in the corresponding scripts) of
many recorded images, stored locally.
@author: ssklykov
"""
# %% Imports
import tkinter as tk
from tkinter.ttk import Progressbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # import canvas container from matplotlib for tkinter
import matplotlib.figure as plot_figure
import time
import numpy as np
import os
from skimage import io
from skimage.util import img_as_ubyte
from reconstruction_wfs_functions import (get_integral_limits_nonaberrated_centers, IntegralMatrixThreaded,
                                          get_localCoM_matrix, get_coms_shifts, get_zernike_coefficients_list,
                                          get_zernike_order_from_coefficients_number)
import re
from queue import Queue, Empty
from calc_zernikes_sh_wfs import get_polynomials_coefficients
from zernike_pol_calc import get_plot_zps_polar


# %% Reconstructor GUI
class ReconstructionUI(tk.Frame):  # The way of making the ui as the child of Frame class - from official tkinter docs
    """Class composing GUI for reconstruction of waveforms using modal reconstruction algorithm."""

    def __init__(self, master):
        # Basic initialization of class variables
        super().__init__(master)  # initialize the toplevel widget
        self.master.title("UI for reconstruction of recorded wavefronts")
        self.master.geometry("+50+70")  # opens the main window with some offset
        self.config(takefocus=True)   # make created window in focus
        self.calibrate_window = None  # holder for checking if the window created
        self.calibrate_axes = None  # the class for plotting in figure loaded pictures
        self.loaded_image = None  # holder for the loaded image for calibration / reconstruction
        self.calibration = False  # flag for switching for a calibration window
        self.calibrate_plots = None  # flag for plots on an image - CoMs, etc.
        self.default_threshold = 55; self.default_radius = 14.0; self.coms_spots = None
        self.order = 4  # default selected Zernike order
        self.messages_queue = Queue(maxsize=10)
        self.integral_matrix = []
        self.calculation_thread = None  # holder for calculation thread of integral matrix
        self.integration_running = False  # flag for tracing the running integration
        self.activate_load_aber_pic_count = 0  # if it is equal to 2, then both files for reconstruction can be loaded
        self.loaded_axes = None; self.loaded_figure = None  # holders for opening and loading figures
        self.reconstruction_window = None  # holder for top-level window represent the loaded picture
        self.reconstruction_axes = None; self.reconstruction_plots = None

        # Buttons and labels specification
        self.load_button = tk.Button(master=self, text="Load Aber.Picture", padx=2, pady=2, command=self.load_aberrated_picture)
        self.load_button.config(state="disabled")
        self.calibrate_button = tk.Button(master=self, text="Calibrate", padx=2, pady=2, command=self.calibrate)
        self.path_int_matrix = tk.StringVar()
        self.label_integral_matrix = tk.Label(master=self, textvariable=self.path_int_matrix, anchor=tk.CENTER)
        self.default_path_label = tk.Label(master=self, text="Default path to calibration files:", anchor=tk.CENTER)
        self.current_path = os.path.dirname(__file__); self.calibration_path = os.path.join(self.current_path, "calibrations")
        self.load_spots_button = tk.Button(master=self, text="Load Focal Spots", padx=1, pady=1, command=self.load_found_spots)
        self.load_integral_matrix_button = tk.Button(master=self, text="Load Integral Matrix", padx=1,
                                                     pady=1, command=self.load_integral_matrix)
        self.spots_text = tk.StringVar(); self.integralM_text = tk.StringVar()  # text variables
        self.spots_label = tk.Label(master=self, textvariable=self.spots_text, anchor=tk.CENTER)
        self.integralM_label = tk.Label(master=self, textvariable=self.integralM_text, anchor=tk.CENTER)
        self.set_default_path()  # calling the method for resolving the standard path for calibrations

        # Grid layout for placing buttons, labels, etc.
        pad = 4  # specification of additional space between widgets in the grid layout
        self.load_button.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)  # Load Button
        self.calibrate_button.grid(row=0, rowspan=1, column=5, columnspan=1, padx=pad, pady=pad)  # Calibrate Button
        self.label_integral_matrix.grid(row=1, rowspan=1, column=2, columnspan=3, padx=pad, pady=pad)  # Text about default path
        self.default_path_label.grid(row=0, rowspan=1, column=2, columnspan=3, padx=pad, pady=pad)  # Represent the default path
        # Load calibration file with focal spots:
        self.load_spots_button.grid(row=2, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        # Load calibration file with integral matrix:
        self.load_integral_matrix_button.grid(row=3, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.spots_label.grid(row=2, rowspan=1, column=2, columnspan=3, padx=pad, pady=pad)
        self.integralM_label.grid(row=3, rowspan=1, column=2, columnspan=3, padx=pad, pady=pad)
        self.grid()  # pack all buttons and labels

    def set_default_path(self):
        """
        Check the existing of "calibrations" folder and set the default path representing to the user.

        Returns
        -------
        None.

        """
        if os.path.exists(self.calibration_path) and os.path.isdir(self.calibration_path):
            self.path_int_matrix.set(self.calibration_path)
            self.spots_path = os.path.join(self.calibration_path, "detected_focal_spots.npy")
            self.integralM_path = os.path.join(self.calibration_path, "integral_calibration_matrix.npy")
            if os.path.exists(self.spots_path):
                self.spots_text.set("Calibration file with focal spots found")
                self.coms_spots = np.load(self.spots_path)
                rows, cols = self.coms_spots.shape
                if rows > 0 and cols > 0:
                    self.activate_load_aber_pic_count += 1
            else:
                self.spots_text.set("No default calibration file with spots found")
            if os.path.isfile(self.integralM_path):
                self.integralM_text.set("Calibration file with integral matrix found")
                self.integral_matrix = np.load(self.integralM_path)
                rows, cols = self.integral_matrix.shape
                if rows > 0 and cols > 0:
                    self.activate_load_aber_pic_count += 1
            else:
                self.integralM_text.set("No default calibration file with integral matrix found")
            if self.activate_load_aber_pic_count == 2:
                self.load_button.config(state="normal")
        else:
            self.path_int_matrix.set(self.current_path)
            self.spots_text.set("No default calibration file with spots found")
            self.integralM_text.set("No default calibration file with integral matrix found")

    # %% Calibration
    def calibrate(self):
        """
        Perform calibration on the recorded non-aberrated image, stored on the local drive.

        Returns
        -------
        None.

        """
        if self.calibrate_window is None:  # create the toplevel widget for holding all ctrls for calibration
            # Toplevel window configuration
            self.calibrate_window = tk.Toplevel(master=self); self.calibrate_window.geometry("+720+70")
            self.calibrate_window.protocol("WM_DELETE_WINDOW", self.calibration_exit)  # associate quit with the function
            self.calibrate_button.config(state="disabled")  # disable the Calibrate button
            self.calibrate_window.title("Calibration")
            self.load_button.config(state="disabled")

            # Buttons specification for calibration
            pad = 4  # universal additional distance between buttons on grid layout
            self.calibrate_load_button = tk.Button(master=self.calibrate_window, text="Load recorded spots",
                                                   command=self.load_picture)
            self.calibrate_localize_button = tk.Button(master=self.calibrate_window, text="Localize focal spots",
                                                       command=self.localize_spots)
            self.calibrate_localize_button.config(state="disabled")
            self.save_coms_button = tk.Button(master=self.calibrate_window, text="Save found spots",
                                              command=self.save_localized_coms)
            self.save_coms_button.config(state="disabled")
            self.calc_integral_matrix_button = tk.Button(master=self.calibrate_window, text="Calculate integrals",
                                                         command=self.calculate_integral_matrix)
            self.calc_integral_matrix_button.config(state="disabled")
            self.abort_integral_matrix_button = tk.Button(master=self.calibrate_window, text="Abort integrals",
                                                          command=self.abort_integration, fg="red")
            self.abort_integral_matrix_button.config(state="disabled")
            self.save_integral_matrix_button = tk.Button(master=self.calibrate_window, text="Save int. matrix",
                                                         command=self.save_integral_matrix)
            self.save_integral_matrix_button.config(state="disabled")

            # Packing progress bar
            self.progress_frame = tk.Frame(master=self.calibrate_window)
            self.progress_bar_label = tk.Label(master=self.progress_frame, text="Integration progress: ")
            self.integration_progress_bar = Progressbar(master=self.progress_frame, length=150)
            self.progress_bar_label.pack(side='left', padx=1, pady=1)
            self.integration_progress_bar.pack(side='left', padx=1, pady=1)

            # Threshold value ctrls and packing
            self.threshold_frame = tk.Frame(master=self.calibrate_window)  # for holding label and spinbox
            self.threshold_label = tk.Label(master=self.threshold_frame, text="Threshold (1...254): ")
            self.threshold_label.pack(side='left', padx=1, pady=1)
            self.threshold_value = tk.IntVar(); self.threshold_value.set(self.default_threshold)  # default threshold value
            self.threshold_value.trace_add(mode="write", callback=self.validate_threshold)
            self.threshold_ctrl_box = tk.Spinbox(master=self.threshold_frame, from_=1, to=254,
                                                 increment=1, textvariable=self.threshold_value,
                                                 wrap=True, width=4)   # adapt Spinbox to 4 digits in int value
            self.threshold_ctrl_box.pack(side='left', padx=1, pady=1)
            self.threshold_ctrl_box.config(state="disabled")

            # Radius of sub-apertures ctrls and packing
            self.radius_frame = tk.Frame(master=self.calibrate_window)  # for holding label and spinbox
            self.radius_label = tk.Label(master=self.radius_frame, text="Sub-aperture radius: ")
            self.radius_label.pack(side='left', padx=1, pady=1)
            self.radius_value = tk.DoubleVar(); self.radius_value.set(self.default_radius)  # default threshold value
            self.radius_value.trace_add(mode="write", callback=self.validate_radius)
            self.radius_ctrl_box = tk.Spinbox(master=self.radius_frame, from_=1.0, to=100.0,
                                              increment=0.2, textvariable=self.radius_value,
                                              wrap=True, width=4)   # adapt Spinbox to 4 digits in double value
            self.radius_ctrl_box.pack(side='left', padx=1, pady=1)
            self.radius_ctrl_box.config(state="disabled")

            # Number of orders control
            self.zernike_orders = ["1st", "2nd", "3rd", "4th", "5th"]; self.selected_order = tk.StringVar()
            self.order_list = ["Use up to " + item + " order" for item in self.zernike_orders]
            self.selected_order.set(self.order_list[3])
            self.zernike_order_selector = tk.OptionMenu(self.calibrate_window, self.selected_order, *self.order_list,
                                                        command=self.order_selected)
            self.zernike_order_selector.config(state="disabled")

            # Construction of figure holder for its representation
            self.calibrate_figure = plot_figure.Figure(figsize=(6.8, 5.7))
            self.calibrate_canvas = FigureCanvasTkAgg(self.calibrate_figure, master=self.calibrate_window)
            self.calibrate_fig_widget = self.calibrate_canvas.get_tk_widget()

            # Layout of widgets on the calibration window
            self.calibrate_load_button.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.calibrate_localize_button.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.threshold_frame.grid(row=0, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.radius_frame.grid(row=0, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.save_coms_button.grid(row=0, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
            self.calibrate_fig_widget.grid(row=1, rowspan=4, column=0, columnspan=5, padx=pad, pady=pad)
            self.calc_integral_matrix_button.grid(row=5, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.zernike_order_selector.grid(row=5, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.progress_frame.grid(row=5, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.abort_integral_matrix_button.grid(row=5, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.save_integral_matrix_button.grid(row=5, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
            self.calibrate_window.grid()
            self.calibrate_window.config(takefocus=True)  # put the created windows in focus
            self.calibration = True  # flag for association of images with the calibration window

    def calibration_exit(self):
        """
        Handle closing of the 'Calibration' window.

        Returns
        -------
        None.

        """
        if self.integration_running:  # if the integration running, then abort integration first
            self.abort_integration()
            self.after(520, self.calibration_exit)  # call this function again after some ms for closing window
        else:
            self.calibrate_button.config(state="normal")  # activate the Calibrate button
            self.config(takefocus=True)   # make main window in focus
            # Below - restore empty holders for recreation of figure, axes, etc.
            self.calibration = False; self.loaded_image = None
            self.calibrate_axes = None; self.calibrate_plots = None
            if not self.messages_queue.empty():
                self.messages_queue.queue.clear()  # clear all messages from the messages queue
            self.calibrate_window.destroy(); self.calibrate_window = None
        self.load_button.config(state="normal")  # activate the load picture button

    def load_picture(self):
        """
        Ask a user for selecting an image for loading and represent it in calibration / reconstruction window.

        Returns
        -------
        None.

        """
        self.calibrate_localize_button.config(state="normal")  # enable localization button after loading image
        self.pics_path = os.path.join(self.current_path, "pics")  # default folder with the pictures
        if self.calibrate_axes is not None:
            self.calibrate_figure.delaxes(self.calibrate_figure.get_axes()[0])
            self.calibrate_axes = None
        # construct absolute path to the folder with recorded pictures
        if os.path.exists(self.pics_path) and os.path.isdir(self.pics_path):
            initialdir = self.pics_path
        else:
            initialdir = self.current_path
        file_types = [("PNG image", "*.png"), ("JPG file", "*.jpg, *.jpeg"), ("Tiff file", "*.tiff, *.tif")]
        open_image_dialog = tk.filedialog.askopenfile(initialdir=initialdir, filetypes=file_types)
        if open_image_dialog is not None:
            self.path_loaded_picture = open_image_dialog.name  # record absolute path to the opened image
            self.loaded_image = io.imread(self.path_loaded_picture, as_gray=True)
            self.loaded_image = img_as_ubyte(self.loaded_image)  # convert to the ubyte U8 image
            rows, cols = self.loaded_image.shape
            if self.calibration:  # draw the loaded image in the opened calibration window (Toplevel)
                if self.calibrate_axes is None:
                    self.calibrate_axes = self.calibrate_figure.add_subplot()  # add axes without dimension
                if self.calibrate_axes is not None and self.loaded_image is not None:
                    self.calibrate_axes.imshow(self.loaded_image, cmap='gray')
                    self.calibrate_axes.axis('off'); self.calibrate_figure.tight_layout()
                    self.calibrate_canvas.draw()  # redraw image in the widget (stored in canvas)
                    self.threshold_ctrl_box.config(state="normal")  # enable the threshold button
                    self.radius_ctrl_box.config(state="normal")  # enable the radius button
                    self.calibrate_plots = None

    def validate_threshold(self, *args):
        """
        Call checking function after some time of the first changing of threshold value.

        Parameters
        ----------
        *args : list
            List with name of IntVar and operation (write).

        Returns
        -------
        None.

        """
        self.after(920, self.check_threshold_value)  # call once the checking procedure of input value

    def check_threshold_value(self):
        """
        Check input manually value in the text variable of the threshold controlling Spinbox.

        Returns
        -------
        None.

        """
        try:
            input_value = self.threshold_value.get()
            if input_value < 1 or input_value > 255:  # bounds for an ubyte (U8) image
                self.threshold_value.set(self.default_threshold)
        except Exception:
            self.threshold_value.set(self.default_threshold)

    def validate_radius(self, *args):
        """
        Validate input radius as a double value.

        Parameters
        ----------
        *args : several values provided by DoubleVar tkinter.
            There are a few parameters associated with DoubleVar.

        Returns
        -------
        None.

        """
        self.after(920, self.check_radius_value)

    def check_radius_value(self):
        """
        Check input manually value in the text variable of the radius controlling Spinbox.

        Returns
        -------
        None.

        """
        try:
            input_value = self.radius_value.get()
            if input_value < 1.0 or input_value > 100.0:  # bounds for an ubyte (U8) image
                self.radius_value.set(self.default_radius)
        except Exception:
            self.radius_value.set(self.default_radius)

    def redraw_loaded_image(self):
        """
        Recreate the Axes instance and redraw originally loaded picture (without any additional plots on it).

        Returns
        -------
        None.

        """
        if self.calibrate_plots is None:  # upon the creation
            self.calibrate_plots = True
        else:
            # Below - code for refreshing Axes class for again re-draw found CoMs and etc
            self.calibrate_figure.delaxes(self.calibrate_figure.get_axes()[0])
            self.calibrate_axes = self.calibrate_figure.add_subplot()
            self.calibrate_axes.imshow(self.loaded_image, cmap='gray')
            self.calibrate_axes.axis('off'); self.calibrate_figure.tight_layout()
            self.calibrate_canvas.draw()  # redraw image in the widget (stored in canvas)

    def localize_spots(self):
        """
        Call the function for localization of center of masses of focal spots on the loaded (non-aberrated) image.

        Returns
        -------
        None.

        """
        self.redraw_loaded_image()  # call for refreshing image without plotted CoMs and subapertures
        self.activate_load_aber_pic_count = 0  # disactivate the possibility to load aberrated picture before calibration complete
        self.load_button.config(state="disabled")
        (self.coms_spots, self.theta0, self.rho0,
         self.integration_limits) = get_integral_limits_nonaberrated_centers(self.calibrate_axes,
                                                                             self.loaded_image,
                                                                             self.threshold_value.get(),
                                                                             self.radius_value.get())
        self.calibrate_canvas.draw()  # call for redraw the image in the calibration frame
        if not isinstance(self.coms_spots, int):
            rows, cols = self.coms_spots.shape  # getting the size of found coordinates (X, Y) of CoMs
            if rows > 0 and cols > 0:
                self.save_coms_button.config(state="normal"); self.calc_integral_matrix_button.config(state="normal")
                self.zernike_order_selector.config(state="normal"); self.abort_integral_matrix_button.config(state="normal")
        else:
            self.save_coms_button.config(state="disabled"); self.calc_integral_matrix_button.config(state="disabled")
            self.zernike_order_selector.config(state="disabled"); self.abort_integral_matrix_button.config(state="disabled")

    def save_localized_coms(self):
        """
        Open dialog for saving of found center of masses of focal spots.

        Returns
        -------
        None.

        """
        if self.coms_spots is not None:
            coms_file = tk.filedialog.asksaveasfile(title="Save coordinates of localized spots",
                                                    initialdir=self.calibration_path, filetypes=[("numpy binary file", "*.npy")],
                                                    defaultextension=".npy", initialfile="detected_focal_spots.npy")
            if coms_file is not None:
                self.calibrated_spots_path = coms_file.name
                np.save(self.calibrated_spots_path, self.coms_spots)
                self.set_default_path()   # update the indicator of default detected spots path

    def order_selected(self, *args):
        """
        Extract the order number from StringVar list of controls.

        Parameters
        ----------
        *args : list
            Provided by tkinter as the StringVar change.

        Returns
        -------
        None.

        """
        reg_digit = re.compile('\d')  # match any digit in a string
        self.order = int(re.search(reg_digit, self.selected_order.get()).group(0))  # scan for number of Zernike order

    def calculate_integral_matrix(self):
        """
        Call the calculation of integral matrix.

        Returns
        -------
        None.

        """
        self.calculation_thread = IntegralMatrixThreaded(self.messages_queue, self.order, self.theta0, self.rho0,
                                                         self.integration_limits, self.radius_value.get(),
                                                         self.integration_progress_bar, self.integral_matrix)
        self.calculation_thread.start()
        self.integration_running = True
        self.save_integral_matrix_button.config(state="disabled")
        self.after(500, self.check_finish_integration)

    def abort_integration(self):
        """
        Send the command for stopping integration.

        Returns
        -------
        None.

        """
        if self.integration_running:
            self.messages_queue.put_nowait("Stop integration")  # send the command for stopping calculation thread
            self.save_integral_matrix_button.config(state="disabled")
            if self.calculation_thread is not None:  # checks that thread object still exists
                if self.calculation_thread.is_alive():
                    self.calculation_thread.join(1)  # wait 1 sec for active thread stops
            self.integration_running = False

    def check_finish_integration(self):
        """
        Check periodically whatever the calculation of integral matrix is finished or not.

        Returns
        -------
        None.

        """
        if not self.messages_queue.empty():
            try:
                message = self.messages_queue.get_nowait()
                if message == "Integration finished":
                    self.integration_running = False
                    print("Integration matrix acquired and can be saved")
                    self.integral_matrix = self.calculation_thread.integral_matrix
                    if not isinstance(self.integral_matrix, list):
                        rows, cols = self.integral_matrix.shape
                        if rows > 0 and cols > 0:
                            self.save_integral_matrix_button.config(state="normal")
                if message == "Integration aborted":
                    self.integration_running = False
                    print(message); self.integral_matrix = []
                elif message == "Stop integration":
                    self.messages_queue.put_nowait(message); time.sleep(0.15)  # resend abort calculation
            except Empty:
                self.after(400, self.check_finish_integration)
        else:
            self.after(400, self.check_finish_integration)

    def save_integral_matrix(self):
        """
        Save the calculated integral matrix on the specified path.

        Returns
        -------
        None.

        """
        integralM_file = tk.filedialog.asksaveasfile(title="Save integral matrix",
                                                     initialdir=self.calibration_path,
                                                     filetypes=[("numpy binary file", "*.npy")],
                                                     defaultextension=".npy",
                                                     initialfile="integral_calibration_matrix.npy")
        if integralM_file is not None:
            self.integral_matrix_path = integralM_file.name
            np.save(self.integral_matrix_path, self.integral_matrix)
            self.set_default_path()   # update the indicator of default detected spots path

    # %% Reconstruction
    def load_found_spots(self):
        """
        Load saved in the *npy file detected coordinates of focal spots.

        Returns
        -------
        None.

        """
        if os.path.exists(self.calibration_path) and os.path.isdir(self.calibration_path):
            initialdir = self.calibration_path
        else:
            initialdir = self.current_path
        coms_file = tk.filedialog.askopenfile(title="Load coordinates of localized spots",
                                              initialdir=initialdir, filetypes=[("numpy binary file", "*.npy")],
                                              defaultextension=".npy", initialfile="detected_focal_spots.npy")
        if coms_file is not None:
            self.calibrated_spots_path = coms_file.name
            self.coms_spots = np.load(self.spots_path)
            rows, cols = self.coms_spots.shape
            if rows > 0 and cols > 0:
                # below - force the user to load the integral matrix again, if the file with spots reloaded
                self.load_button.config(state="disabled")
                if self.activate_load_aber_pic_count == 0 or self.activate_load_aber_pic_count == 1:
                    self.activate_load_aber_pic_count += 1
                elif self.activate_load_aber_pic_count == 2:
                    self.activate_load_aber_pic_count -= 1
            if self.activate_load_aber_pic_count == 2:
                self.load_button.config(state="normal")

    def load_integral_matrix(self):
        """
        Load the precalculated integral matrix.

        Returns
        -------
        None.

        """
        if os.path.exists(self.calibration_path) and os.path.isdir(self.calibration_path):
            initialdir = self.calibration_path
        else:
            initialdir = self.current_path
        integralM_file = tk.filedialog.askopenfile(title="Load calculated integral matrix",
                                                   initialdir=initialdir, filetypes=[("numpy binary file", "*.npy")],
                                                   defaultextension=".npy", initialfile="integral_calibration_matrix.npy")
        if integralM_file is not None:
            self.integralM_path = integralM_file.name
            self.integral_matrix = np.load(self.integralM_path)
            rows, cols = self.integral_matrix.shape
            if rows > 0 and cols > 0:
                # below - force the user to load the integral matrix again, if the file with spots reloaded
                self.load_button.config(state="disabled")
                if self.activate_load_aber_pic_count == 0 or self.activate_load_aber_pic_count == 1:
                    self.activate_load_aber_pic_count += 1
                elif self.activate_load_aber_pic_count == 2:
                    self.activate_load_aber_pic_count -= 1
            if self.activate_load_aber_pic_count == 2:
                self.load_button.config(state="normal")

    def load_aberrated_picture(self):
        """
        Load the aberrated picture for calculation of aberrations profile.

        Returns
        -------
        None.

        """
        self.pics_path = os.path.join(self.current_path, "pics")  # default folder with the pictures
        # Below - delete the plots (e.g., the localized focal spots)
        if self.reconstruction_axes is not None:
            self.reconstruction_figure.delaxes(self.reconstruction_figure.get_axes()[0])
            self.reconstruction_axes = None
        # construct absolute path to the folder with recorded pictures
        if os.path.exists(self.pics_path) and os.path.isdir(self.pics_path):
            initialdir = self.pics_path
        else:
            initialdir = self.current_path
        file_types = [("PNG image", "*.png"), ("JPG file", "*.jpg, *.jpeg"), ("Tiff file", "*.tiff, *.tif")]
        open_image_dialog = tk.filedialog.askopenfile(initialdir=initialdir, filetypes=file_types)
        if open_image_dialog is not None:
            self.path_loaded_picture = open_image_dialog.name  # record absolute path to the opened image
            self.loaded_image = io.imread(self.path_loaded_picture, as_gray=True)
            self.loaded_image = img_as_ubyte(self.loaded_image)  # convert to the ubyte U8 image
            rows, cols = self.loaded_image.shape
            if rows > 0 and cols > 0:
                # construct the toplevel window
                if self.reconstruction_window is None:  # create the toplevel widget for holding all ctrls for calibration
                    # Toplevel window configuration
                    self.reconstruction_window = tk.Toplevel(master=self); self.reconstruction_window.geometry("+720+70")
                    self.reconstruction_window.protocol("WM_DELETE_WINDOW", self.reconstruction_exit)
                    self.calibrate_button.config(state="disabled")  # disable the Calibrate button
                    self.reconstruction_window.title("Reconstruction"); pad = 4

                    # Buttons specification
                    self.reconstruct_localize_button = tk.Button(master=self.reconstruction_window, text="Localize focal spots",
                                                                 command=self.localize_aberrated_spots)
                    self.reconstruct_get_shifts_button = tk.Button(master=self.reconstruction_window, text="Get shifts",
                                                                   command=self.calculate_coms_shifts)
                    self.reconstruct_get_shifts_button.config(state="disabled")
                    self.reconstruct_get_zernikes_button = tk.Button(master=self.reconstruction_window, text="Get Aberrations",
                                                                     command=self.calculate_zernikes_coefficients)
                    self.reconstruct_get_zernikes_button.config(state="disabled")

                    # Construction of figure holder for its representation
                    self.reconstruction_figure = plot_figure.Figure(figsize=(6.8, 5.7))
                    self.reconstruction_canvas = FigureCanvasTkAgg(self.reconstruction_figure, master=self.reconstruction_window)
                    self.reconstruction_fig_widget = self.reconstruction_canvas.get_tk_widget()

                    # Threshold value ctrls and packing
                    self.threshold_frame = tk.Frame(master=self.reconstruction_window)  # for holding label and spinbox
                    self.threshold_label = tk.Label(master=self.threshold_frame, text="Threshold (1...254): ")
                    self.threshold_label.pack(side='left', padx=1, pady=1)
                    self.threshold_value = tk.IntVar(); self.threshold_value.set(self.default_threshold)  # default threshold value
                    self.threshold_value.trace_add(mode="write", callback=self.validate_threshold)
                    self.threshold_ctrl_box = tk.Spinbox(master=self.threshold_frame, from_=1, to=254,
                                                         increment=1, textvariable=self.threshold_value,
                                                         wrap=True, width=4)   # adapt Spinbox to 4 digits in int value
                    self.threshold_ctrl_box.pack(side='left', padx=1, pady=1)

                    # Radius of sub-apertures ctrls and packing
                    self.radius_frame = tk.Frame(master=self.reconstruction_window)  # for holding label and spinbox
                    self.radius_label = tk.Label(master=self.radius_frame, text="Sub-aperture radius: ")
                    self.radius_label.pack(side='left', padx=1, pady=1)
                    self.radius_value = tk.DoubleVar(); self.radius_value.set(self.default_radius)  # default threshold value
                    self.radius_value.trace_add(mode="write", callback=self.validate_radius)
                    self.radius_ctrl_box = tk.Spinbox(master=self.radius_frame, from_=1.0, to=100.0,
                                                      increment=0.2, textvariable=self.radius_value,
                                                      wrap=True, width=4)   # adapt Spinbox to 4 digits in double value
                    self.radius_ctrl_box.pack(side='left', padx=1, pady=1)

                    # Place widgets on the Toplevel window
                    self.reconstruct_localize_button.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
                    self.threshold_frame.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
                    self.radius_frame.grid(row=0, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
                    self.reconstruct_get_shifts_button.grid(row=0, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
                    self.reconstruct_get_zernikes_button.grid(row=0, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
                    self.reconstruction_fig_widget.grid(row=1, rowspan=4, column=0, columnspan=5, padx=pad, pady=pad)

                    # Draw the loaded image
                    if self.reconstruction_axes is None:
                        self.reconstruction_axes = self.reconstruction_figure.add_subplot()  # add axes without dimension
                    if self.reconstruction_axes is not None and self.loaded_image is not None:
                        self.reconstruction_axes.imshow(self.loaded_image, cmap='gray')
                        self.reconstruction_axes.axis('off'); self.reconstruction_figure.tight_layout()
                        self.reconstruction_canvas.draw()  # redraw image in the widget (stored in canvas)
                        # self.threshold_ctrl_box.config(state="normal")  # enable the threshold button
                        # self.radius_ctrl_box.config(state="normal")  # enable the radius button
                        self.reconstruction_plots = None

                # Redraw reloaded image on the reconstruction window
                else:  # the reconstruction window has been already opened
                    if self.reconstruction_axes is None:
                        self.reconstruction_axes = self.reconstruction_figure.add_subplot()  # add axes without dimension
                    if self.reconstruction_axes is not None and self.loaded_image is not None:
                        self.reconstruction_axes.imshow(self.loaded_image, cmap='gray')
                        self.reconstruction_axes.axis('off'); self.reconstruction_figure.tight_layout()
                        self.reconstruction_canvas.draw()  # redraw image in the widget (stored in canvas)
                        # self.threshold_ctrl_box.config(state="normal")  # enable the threshold button
                        # self.radius_ctrl_box.config(state="normal")  # enable the radius button
                        self.reconstruction_plots = None
        else:  # the new image not loaded, but the window remains with the previous image not active actually
            if self.reconstruction_window is not None:
                self.reconstruction_exit()  # close previously opened toplevel window

    def redraw_aberrated_image(self):
        """
        Redraw loaded image without any plots on it.

        Returns
        -------
        None.

        """
        # Redraw the image on the reconstruction window without any additional plots on it
        if self.reconstruction_plots is None:
            self.reconstruction_plots = True
        else:
            # Below - code for refreshing Axes class for again re-draw found CoMs and etc
            self.reconstruction_figure.delaxes(self.reconstruction_figure.get_axes()[0])
            self.reconstruction_axes = self.reconstruction_figure.add_subplot()
            self.reconstruction_axes.imshow(self.loaded_image, cmap='gray')
            self.reconstruction_axes.axis('off'); self.reconstruction_figure.tight_layout()
            self.reconstruction_canvas.draw()  # redraw image in the widget (stored in canvas)

    def localize_aberrated_spots(self):
        """
        Localize the focal spots on aberrated images.

        Returns
        -------
        None.

        """
        min_dist_peaks = int(np.round(1.5*self.radius_value.get(), 0))  # also used in imported reconstruction_wfs_functions
        region_size = int(np.round(1.4*self.radius_value.get(), 0))  # also used in imported reconstruction_wfs_functions
        self.coms_aberrated = get_localCoM_matrix(image=self.loaded_image, axes_fig=self.reconstruction_axes,
                                                  threshold_abs=self.threshold_value.get(),
                                                  min_dist_peaks=min_dist_peaks, region_size=region_size)
        self.redraw_aberrated_image()  # redraw the originally loaded image without any plots
        # Below - plotting found spots
        rows, cols = self.coms_aberrated.shape
        if rows > 0 and cols > 0:
            self.reconstruction_axes.plot(self.coms_aberrated[:, 1], self.coms_aberrated[:, 0], '.', color="red")
            self.reconstruction_canvas.draw()  # redraw image in the widget (stored in canvas)
            self.reconstruct_get_shifts_button.config(state="normal")

    def calculate_coms_shifts(self):
        """
        Calculate and visualize shifts in center of masses between non-aberrated and aberrated pictures.

        Returns
        -------
        None.

        """
        self.redraw_aberrated_image()  # redraw the originally loaded image without any plots
        (self.coms_shifts, self.integral_matrix_aberrated,
         self.coms_aberrated) = get_coms_shifts(self.coms_spots, self.integral_matrix,
                                                self.coms_aberrated, self.radius_value.get())
        # Plot shifts
        rows, cols = self.coms_aberrated.shape
        if rows > 0 and cols > 0:
            for i in range(rows):
                # Plotting the arrows for visual representation of shifts direction. Changed signs - because of swapped Y axis
                self.reconstruction_axes.arrow(self.coms_aberrated[i, 1] - self.coms_shifts[i, 1],
                                               self.coms_aberrated[i, 0] + self.coms_shifts[i, 0],
                                               self.coms_shifts[i, 1], -self.coms_shifts[i, 0],
                                               color='red', linewidth=4)
            self.reconstruction_canvas.draw()  # redraw image in the widget (stored in canvas)
            self.reconstruct_get_zernikes_button.config(state="normal")

    def calculate_zernikes_coefficients(self):
        """
        Calculate amplitudes (alpha coefficients) of Zernike polynomials, used for integral matrix calculation.

        Returns
        -------
        None.

        """
        self.alpha_coefficients = get_polynomials_coefficients(self.integral_matrix_aberrated, self.coms_shifts)
        self.alpha_coefficients *= np.pi  # for adjusting to radians ???
        self.alpha_coefficients = list(self.alpha_coefficients)
        if len(self.alpha_coefficients) > 0:
            # Define below used orders of Zernikes and providing them for amplitudes calculation
            self.order = get_zernike_order_from_coefficients_number(len(self.alpha_coefficients))
            self.zernike_list_orders = get_zernike_coefficients_list(self.order)
            # Draw the profile with Zernike polynomials multiplied by coefficients (amplitudes)
            self.reconstruction_figure = get_plot_zps_polar(self.reconstruction_figure, orders=self.zernike_list_orders,
                                                            step_r=0.005, step_theta=0.9,
                                                            alpha_coefficients=self.alpha_coefficients, show_amplitudes=False)
            self.reconstruction_canvas.draw()  # redraw the figure

    def reconstruction_exit(self):
        """
        Handle close of the toplevel window for reconstruction controls.

        Returns
        -------
        None.

        """
        self.calibrate_button.config(state="normal")
        self.reconstruction_window.destroy(); self.reconstruction_window = None

    def destroy(self):
        """
        Rewrite the behaviour of the main window then it's closed.

        Returns
        -------
        None.

        """
        self.messages_queue.put_nowait("Stop integration")  # send the command for stopping calculation thread
        if self.calculation_thread is not None:
            if self.calculation_thread.is_alive():
                self.calculation_thread.join(1)  # wait 1 sec for active thread stops
        if not self.messages_queue.empty():
            self.messages_queue.queue.clear()  # clear all messages from the messages queue


# %% Main launch
if __name__ == "__main__":
    rootTk = tk.Tk()  # toplevel widget of Tk there the class - child of tk.Frame is embedded
    reconstructor_gui = ReconstructionUI(rootTk)
    reconstructor_gui.mainloop()

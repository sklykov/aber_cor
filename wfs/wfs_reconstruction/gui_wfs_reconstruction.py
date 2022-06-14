# -*- coding: utf-8 -*-
"""
GUI for perform wavefront reconstruction.

This script is used for testing of the wavefront reconstruction algorithm (see reference) on the images from
Shack-Hartmann sensor.

@author: Sergei Klykov (GitHub: @ssklykov).
@license: GPLv3 (check https://www.gnu.org/licenses/ ).

Reference
---------
    Antonello, J. (2014): https://doi.org/10.4233/uuid:f98b3b8f-bdb8-41bb-8766-d0a15dae0e27

"""

# %% Imports - global dependecies (from standard library and installed by conda / pip)
import tkinter as tk
from tkinter import font
from tkinter.ttk import Progressbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.figure as plot_figure
import time
import numpy as np
import os
from skimage import io
from skimage.util import img_as_ubyte
import re
from queue import Queue, Empty
from pathlib import Path
import platform
import ctypes
from multiprocessing import Queue as mpQueue
from threading import Thread

# %% Imports - local dependecies (modules / packages in the containing it folder / subfolders)
print("Calling signature:", __name__)  # inspection of called signature
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    # ???: Last condition arised because of attempt to reload modules then additional Process launched, seems
    # Actual call as the standalone module or from other module from this package (as a dependecy)
    # Note that in this case this module is aware only about modules / packages in the same folder
    from reconstruction_wfs_functions import (get_integral_limits_nonaberrated_centers, IntegralMatrixThreaded,
                                              get_localCoM_matrix, get_coms_shifts, get_zernike_coefficients_list,
                                              get_zernike_order_from_coefficients_number, get_coms_fast,
                                              get_coms_shifts_fast)
    from calc_zernikes_sh_wfs import get_polynomials_coefficients
    from zernike_pol_calc import get_plot_zps_polar, zernike_polynomials_sum_tuned
    import camera as cam  # for accessing controlling wrapper for the cameras (simulated and IDS)
else:  # relative imports for resolving these dependencies in the case of import as module from a package
    from .reconstruction_wfs_functions import (get_integral_limits_nonaberrated_centers, IntegralMatrixThreaded,
                                               get_localCoM_matrix, get_coms_shifts,
                                               get_zernike_coefficients_list,
                                               get_zernike_order_from_coefficients_number, get_coms_fast,
                                               get_coms_shifts_fast)
    from .calc_zernikes_sh_wfs import get_polynomials_coefficients
    from .zernike_pol_calc import get_plot_zps_polar, zernike_polynomials_sum_tuned
    from . import camera as cam   # for accessing controlling wrapper for the cameras (simulated and IDS)
    print("Implicit usage of camera module, list of importable modules: ", cam.__all__)


# %% Reconstructor GUI
class ReconstructionUI(tk.Frame):  # The way of making the ui as the child of Frame class - from official tkinter docs
    """Class composing GUI for reconstruction of waveforms using modal reconstruction algorithm."""

    def __init__(self, master):
        # Basic initialization of class variables
        super().__init__(master)  # initialize the toplevel widget
        self.master.title("UI for reconstruction of recorded wavefronts")
        self.master.geometry("+40+40")  # opens the main window with some offset from top screen coordinates
        self.config(takefocus=True)   # make created window in focus
        self.calibrate_window = None  # holder for checking if the window created
        self.calibrate_axes = None  # the class for plotting in figure loaded pictures
        self.loaded_image = None  # holder for the loaded image for calibration / reconstruction
        self.calibration = False  # flag for switching for a calibration window
        self.calibrate_plots = None  # flag for plots on an image - CoMs, etc.
        self.default_threshold = 55; self.default_radius = 14.0; self.coms_spots = None
        self.order = 4  # default selected Zernike order
        self.messages_queue = Queue(maxsize=10); self.integral_matrix = np.ndarray
        self.calculation_thread = None  # holder for calculation thread of integral matrix
        self.integration_running = False  # flag for tracing the running integration
        self.activate_load_aber_pic_count = 0  # if it == 2, then both files for reconstruction can be loaded
        self.loaded_axes = None; self.loaded_figure = None  # holders for opening and loading figures
        self.reconstruction_window = None  # holder for the top-level window representing the loaded picture
        self.reconstruction_axes = None; self.reconstruction_plots = None
        self.camera_ctrl_window = None  # holder for the top-level window controlling a camera
        self.default_font = font.nametofont("TkDefaultFont")
        self.coms_aberrated = None; self.coms_shifts = None

        # Buttons and labels specification
        self.load_aber_pic_button = tk.Button(master=self, text="Load Aber.Picture",
                                              command=self.load_aberrated_picture)
        self.load_aber_pic_button.config(state="disabled")
        self.calibrate_button = tk.Button(master=self, text="Calibrate", command=self.calibrate)
        # Display the default searching path
        self.default_path_display = tk.Text(master=self, height=3, width=46, wrap=tk.CHAR,
                                            font=(self.default_font.actual()['family'],
                                                  self.default_font.actual()['size']))
        self.current_path = os.path.dirname(__file__); self.calibration_path = os.path.join(self.current_path,
                                                                                            "calibrations")
        # Buttons and labels
        self.load_spots_button = tk.Button(master=self, text="Load Focal Spots", command=self.load_found_spots)
        self.load_integral_matrix_button = tk.Button(master=self, text="Load Integral Matrix",
                                                     command=self.load_integral_matrix)
        self.spots_text = tk.StringVar(); self.integralM_text = tk.StringVar()  # text variables
        self.spots_label = tk.Label(master=self, textvariable=self.spots_text, anchor=tk.CENTER)
        self.integralM_label = tk.Label(master=self, textvariable=self.integralM_text, anchor=tk.CENTER)
        self.set_default_path()  # calling the method for resolving the standard path for calibrations
        self.open_camera_button = tk.Button(master=self, text="Live Camera", command=self.open_camera)

        # Grid layout for placing buttons, labels, etc.
        self.pad = 8  # specification of additional space between widgets in the grid layout
        self.load_aber_pic_button.grid(row=0, rowspan=1, column=0, columnspan=1, padx=self.pad, pady=self.pad)  # Load Button
        self.calibrate_button.grid(row=0, rowspan=1, column=5, columnspan=1, padx=self.pad, pady=self.pad)  # Calibrate Button
        self.default_path_display.grid(row=0, rowspan=2, column=2, columnspan=3, padx=self.pad, pady=self.pad)
        # Load calibration file with focal spots:
        self.load_spots_button.grid(row=2, rowspan=1, column=0, columnspan=1, padx=self.pad, pady=self.pad)
        # Load calibration file with integral matrix:
        self.load_integral_matrix_button.grid(row=3, rowspan=1, column=0, columnspan=1, padx=self.pad, pady=self.pad)
        self.spots_label.grid(row=2, rowspan=1, column=2, columnspan=3, padx=self.pad, pady=self.pad)
        self.integralM_label.grid(row=3, rowspan=1, column=2, columnspan=3, padx=self.pad, pady=self.pad)
        self.open_camera_button.grid(row=3, rowspan=1, column=5, columnspan=1, padx=self.pad, pady=self.pad)
        self.grid(); self.master.update()  # pack all buttons and labels
        self.master_geometry = self.master.winfo_geometry()  # saves the main window geometry

    def set_default_path(self):
        """
        Check the existing of "calibrations" folder and set the default path representing to the user.

        Returns
        -------
        None.

        """
        if os.path.exists(self.calibration_path) and os.path.isdir(self.calibration_path):
            self.default_path_display.delete(0.0, 'end')  # clean previously assigned path
            self.default_path_display.insert('end', "Default path to calibration files: \n", ('header path',))
            self.default_path_display.tag_config('header path', justify=tk.CENTER)
            self.default_path_display.insert('end', (self.calibration_path))
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
                self.load_aber_pic_button.config(state="normal")
        else:
            self.default_path_display.insert('end', (self.current_path + "\n"))
            self.spots_text.set("No default calibration file with spots found")
            self.integralM_text.set("No default calibration file with integral matrix found")

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
        if self.camera_ctrl_window is not None:  # perform all required for closing camera ctrl operations
            self.camera_ctrl_exit()

    # %% Calibration
    def calibrate(self, camera_ctrl_call: bool = False):
        """
        Perform calibration on the recorded non-aberrated image, stored on the local drive.

        Parameters
        ----------
        camera_ctrl_call : bool, optional
            If this function is called by button from camera_ctrl window. The default is False.

        Returns
        -------
        None.

        """
        if self.calibrate_window is None:  # create the toplevel widget for holding all ctrls for calibration
            # Toplevel window configuration for calibration - relative to the main  (actual position!)
            y_shift = self.master.winfo_geometry().split("+")[2]  # vertical shift
            x_shift = self.master.winfo_x() + self.master.winfo_width() + self.pad  # horizontal shift
            self.calibrate_window = tk.Toplevel(master=self); self.calibrate_window.geometry(f'+{x_shift}+{y_shift}')
            self.calibrate_window.protocol("WM_DELETE_WINDOW", self.calibration_exit)  # associate quit with the function
            self.calibrate_button.config(state="disabled")  # disable the Calibrate button
            self.calibrate_window.title("Calibration"); self.load_aber_pic_button.config(state="disabled")
            self.camera_ctrl_call = camera_ctrl_call  # retain the call flag

            # Buttons specification for calibration
            pad = 4  # universal additional distance between buttons on grid layout
            self.calibrate_load_spots_button = tk.Button(master=self.calibrate_window, text="Load recorded spots",
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
            if not self.camera_ctrl_call:
                self.calibrate_figure = plot_figure.Figure(figsize=(6.8, 5.7))  # For provided sample image
            else:
                # Adjusted for the image coming from the camera
                self.calibrate_figure = plot_figure.Figure(figsize=(self.default_frame_figure,
                                                                    self.default_frame_figure))
            self.calibrate_canvas = FigureCanvasTkAgg(self.calibrate_figure, master=self.calibrate_window)
            self.calibrate_fig_widget = self.calibrate_canvas.get_tk_widget()
            # Toolbar - with tools for image manipulation
            self.calibrate_fig_toolbar = NavigationToolbar2Tk(self.calibrate_canvas, self.calibrate_window,
                                                              pack_toolbar=False)
            self.calibrate_fig_toolbar.update()

            # Draw content of transferred image from the camera ctrl window
            if self.camera_ctrl_call:
                self.calibrate_axes = self.calibrate_figure.add_subplot()
                self.calibrate_axes.axis('off'); self.calibrate_figure.tight_layout()
                self.calibrate_axes.imshow(self.current_image, cmap='gray', interpolation='none',
                                           vmin=0, vmax=255)
                self.calibrate_figure.subplots_adjust(left=0, bottom=0, right=1, top=1)  # remove white borders
                self.calibrate_canvas.draw()
                if len(self.current_image.shape) > 2:
                    self.loaded_image = np.squeeze(self.current_image, axis=2)  # Remove 3rd dimension from camera image
                else:
                    self.loaded_image = self.current_image
                # Disable the load feature of calibration and proceed to calibration procedure
                self.calibrate_load_spots_button.config(state="disabled")
                self.threshold_ctrl_box.config(state="normal")  # enable the threshold button
                self.radius_ctrl_box.config(state="normal")  # enable the radius button
                self.calibrate_localize_button.config(state="normal")  # enable localization button
                # Change default parameters for better adjustment to IDS camera pictures
                self.default_threshold = 127; self.threshold_value.set(self.default_threshold)
                self.default_radius = 10.0; self.radius_value.set(self.default_radius)

            # Layout of widgets on the calibration window
            self.calibrate_load_spots_button.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.calibrate_localize_button.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.threshold_frame.grid(row=0, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.radius_frame.grid(row=0, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.save_coms_button.grid(row=0, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
            self.calibrate_fig_widget.grid(row=1, rowspan=4, column=0, columnspan=5, padx=pad, pady=pad)
            self.calc_integral_matrix_button.grid(row=7, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.zernike_order_selector.grid(row=7, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.progress_frame.grid(row=7, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.abort_integral_matrix_button.grid(row=7, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.save_integral_matrix_button.grid(row=7, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
            self.calibrate_fig_toolbar.grid(row=5, rowspan=2, column=1, columnspan=4, padx=pad, pady=pad)
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
        self.load_aber_pic_button.config(state="normal")  # activate the load picture button
        self.camera_ctrl_call = False  # for possible re-open of calibration window from camera ctrl one

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
        except tk.TclError:
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
        except tk.TclError:
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
        self.redraw_loaded_image()  # call for refreshing image without plotted CoMs and sub-apertures
        self.activate_load_aber_pic_count = 0  # dis-activate the possibility to load aberrated picture before calibration complete
        self.load_aber_pic_button.config(state="disabled")
        # print("image min and max: ", np.min(self.loaded_image), np.max(self.loaded_image))
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
                self.calibration_path, tail = os.path.split(self.calibrated_spots_path)
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
            self.calibration_path, tail = os.path.split(self.integral_matrix_path)
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
            self.coms_spots = np.load(self.calibrated_spots_path)
            rows, cols = self.coms_spots.shape
            if rows > 0 and cols > 0:
                # below - force the user to load the integral matrix again, if the file with spots reloaded
                self.load_aber_pic_button.config(state="disabled")
                if self.activate_load_aber_pic_count == 0 or self.activate_load_aber_pic_count == 1:
                    self.activate_load_aber_pic_count += 1
                elif self.activate_load_aber_pic_count == 2:
                    self.activate_load_aber_pic_count -= 1
            if self.activate_load_aber_pic_count == 2:
                self.load_aber_pic_button.config(state="normal")

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
                self.load_aber_pic_button.config(state="disabled")
                if self.activate_load_aber_pic_count == 0 or self.activate_load_aber_pic_count == 1:
                    self.activate_load_aber_pic_count += 1
                elif self.activate_load_aber_pic_count == 2:
                    self.activate_load_aber_pic_count -= 1
            if self.activate_load_aber_pic_count == 2:
                self.load_aber_pic_button.config(state="normal")

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
                    # Toplevel window configuration for reconstruction - relative to the main window actual position!
                    y_shift = self.master.winfo_geometry().split("+")[2]  # vertical shift
                    x_shift = self.master.winfo_x() + self.master.winfo_width() + self.pad  # horizontal shift
                    self.reconstruction_window = tk.Toplevel(master=self)
                    self.reconstruction_window.geometry(f'+{x_shift}+{y_shift}')
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
                    self.reconstruct_save_zernikes_plot = tk.Button(master=self.reconstruction_window, text="Save sum plot",
                                                                    command=self.save_sum_reconstructed_zernikes)
                    self.reconstruct_save_zernikes_plot.config(state="disabled")

                    # Selector to show / hide a colorbar for polynomials plot
                    self.colorbar_show_options = ["No colorbar", "Show colorbar"]
                    self.colorbar_show_var = tk.StringVar()
                    self.colorbar_show_var.set(self.colorbar_show_options[0])
                    self.amplitude_show_selector = tk.OptionMenu(self.reconstruction_window,
                                                                 self.colorbar_show_var,
                                                                 *self.colorbar_show_options,
                                                                 command=self.colobar_option_selected)
                    self.amplitude_show_selector.config(state="disabled")

                    # Construction of figure holder for its representation
                    self.reconstruction_figure = plot_figure.Figure(figsize=(6.8, 5.7))
                    self.reconstruction_canvas = FigureCanvasTkAgg(self.reconstruction_figure,
                                                                   master=self.reconstruction_window)
                    self.reconstruction_fig_widget = self.reconstruction_canvas.get_tk_widget()

                    # Threshold value ctrls and packing
                    self.threshold_frame = tk.Frame(master=self.reconstruction_window)  # for holding ctrls below
                    self.threshold_label = tk.Label(master=self.threshold_frame, text="Threshold (1...254): ")
                    self.threshold_label.pack(side='left', padx=1, pady=1)
                    self.threshold_value = tk.IntVar(); self.threshold_value.set(self.default_threshold)
                    self.threshold_value.trace_add(mode="write", callback=self.validate_threshold)
                    self.threshold_ctrl_box = tk.Spinbox(master=self.threshold_frame, from_=1, to=254,
                                                         increment=1, textvariable=self.threshold_value,
                                                         wrap=True, width=4)   # adapt Spinbox to 4 digits in int
                    self.threshold_ctrl_box.pack(side='left', padx=1, pady=1)

                    # Radius of sub-apertures ctrls and packing
                    self.radius_frame = tk.Frame(master=self.reconstruction_window)  # for holding ctrls below
                    self.radius_label = tk.Label(master=self.radius_frame, text="Sub-aperture radius: ")
                    self.radius_label.pack(side='left', padx=1, pady=1)
                    self.radius_value = tk.DoubleVar(); self.radius_value.set(self.default_radius)
                    self.radius_value.trace_add(mode="write", callback=self.validate_radius)
                    self.radius_ctrl_box = tk.Spinbox(master=self.radius_frame, from_=1.0, to=100.0,
                                                      increment=0.2, textvariable=self.radius_value,
                                                      wrap=True, width=4)   # adapt Spinbox to 4 digits in double
                    self.radius_ctrl_box.pack(side='left', padx=1, pady=1)

                    # Place widgets on the Toplevel window
                    self.reconstruct_localize_button.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
                    self.threshold_frame.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
                    self.radius_frame.grid(row=0, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
                    self.reconstruct_get_shifts_button.grid(row=0, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
                    self.reconstruct_get_zernikes_button.grid(row=0, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
                    self.reconstruction_fig_widget.grid(row=1, rowspan=4, column=0, columnspan=5, padx=pad, pady=pad)
                    self.amplitude_show_selector.grid(row=5, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
                    self.reconstruct_save_zernikes_plot.grid(row=5, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)

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
        # Disable some buttons for preventing of usage of previous calculation results
        if self.amplitude_show_selector['state'] == 'normal':
            self.amplitude_show_selector.config(state='disabled')
        if self.reconstruct_get_zernikes_button['state'] == 'normal':
            self.reconstruct_get_zernikes_button.config(state='disabled')
            self.reconstruct_save_zernikes_plot.config(state='disabled')

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
        # Disable some buttons for preventing of usage of previous calculation results
        if self.amplitude_show_selector['state'] == 'normal':
            self.amplitude_show_selector.config(state='disabled')

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
            self.amplitude_show_selector.config(state="normal")
            self.reconstruct_save_zernikes_plot.config(state="normal")

    def colobar_option_selected(self, *args):
        """
        Draw or remove the colorbar on the profile with Zernike polynomials sum for representation of their amplitudes.

        Parameters
        ----------
        *args : list
            Provided by tkinter upon the changing tkinter.OptionMenu.

        Returns
        -------
        None.

        """
        if self.colorbar_show_var.get() == "No colorbar":
            show_amplitudes = False
        else:
            show_amplitudes = True
        if len(self.alpha_coefficients) > 0:
            # Draw the profile with Zernike polynomials multiplied by coefficients (amplitudes)
            self.reconstruction_figure = get_plot_zps_polar(self.reconstruction_figure, orders=self.zernike_list_orders,
                                                            step_r=0.005, step_theta=0.9,
                                                            alpha_coefficients=self.alpha_coefficients,
                                                            show_amplitudes=show_amplitudes)
            self.reconstruction_canvas.draw()  # redraw the figure

    def save_sum_reconstructed_zernikes(self):
        """
        Save plotted reconstructed sum of Zernike polynomials.

        Returns
        -------
        None.

        """
        file_types = [("PNG image", "*.png"), ("JPG file", "*.jpg")]
        if self.reconstruction_canvas is not None and self.reconstruction_figure is not None:
            sum_plot_file = tk.filedialog.asksaveasfile(title="Save integral matrix",
                                                        initialdir=self.calibration_path,
                                                        filetypes=file_types,
                                                        defaultextension=".png",
                                                        initialfile="Reconstructed sum of polynomials")
            if sum_plot_file is not None:
                self.sum_plot_file_path = sum_plot_file.name  # get the string path
                self.reconstruction_figure.savefig(self.sum_plot_file_path)  # save the drawn figure on the GUI

    def reconstruction_exit(self):
        """
        Handle close of the toplevel window for reconstruction controls.

        Returns
        -------
        None.

        """
        self.calibrate_button.config(state="normal")
        self.reconstruction_window.destroy(); self.reconstruction_window = None

    # %% Wavefront sensor Camera ctrl
    # recorded wavefront profiles from a Shack-Hartmann sensor
    def open_camera(self):
        """
        Open the additional window for controlling a camera.

        Returns
        -------
        None.

        """
        if self.camera_ctrl_window is None:
            # Toplevel window configuration for calibration - relative to the main window actual position!
            y_shift = self.master.winfo_geometry().split("+")[2]  # vertical shift
            x_shift = self.master.winfo_x() + self.master.winfo_width() + self.pad  # horizontal shift
            self.camera_ctrl_window = tk.Toplevel(master=self)
            self.camera_ctrl_window.geometry(f'+{x_shift}+{y_shift}')
            self.camera_ctrl_window.protocol("WM_DELETE_WINDOW", self.camera_ctrl_exit)
            self.global_timeout = 0.25  # global timeout in seconds
            self.frame_figure_axes = None; self.__flag_live_stream = False
            self.exposure_t_ms = 50; self.exposure_t_ms_min = 1; self.exposure_t_ms_max = 100
            self.gui_refresh_rate_ms = 10  # The constant time pause between each attempt to retrieve the image
            self.calibration_activation = False  # for activating the button for calibration window open
            self.coms_shifts = None; self.coms_aberrated = None
            self.__flag_live_localization = False; self.reconstruction_updater = None
            self.__flag_live_image_updater = True  # default - for displaying live streamed images
            self.__flag_update_amplitudes = False  # additional flag to stop updating the amplitudes
            self.amplitudes_figure = None; self.amplitudes_figure_axes = None
            self.amplitudes_showing = None; self.__flag_bar_plot = False
            self.frame_figure_pcolormesh = None

            # Buttons creation
            self.single_snap_button = tk.Button(master=self.camera_ctrl_window, text="Snap single image",
                                                command=self.snap_single_image)
            self.single_snap_button.config(state="disabled")
            self.live_stream_button = tk.Button(master=self.camera_ctrl_window, text="Start Live",
                                                command=self.live_stream, fg='green')
            self.live_stream_button.config(state="disabled")
            self.cameras = ["Simulated", "IDS"]; self.selected_camera = tk.StringVar()
            self.selected_camera.set(self.cameras[0])
            self.camera_selector = tk.OptionMenu(self.camera_ctrl_window, self.selected_camera, *self.cameras,
                                                 command=self.switch_active_camera)
            self.camera_selector.config(state="disabled")
            self.calibration_activate_button = tk.Button(master=self.camera_ctrl_window, text="Calibrate",
                                                         command=self.open_calibration, fg='blue')
            self.calibration_activate_button.config(state="disabled")
            self.live_reconstruction_button = tk.Button(master=self.camera_ctrl_window, text="Start Reconstruction",
                                                        command=self.live_reconstruction, fg='magenta')
            self.live_reconstruction_button.config(state="disabled")
            self.get_focal_spots_button = tk.Button(master=self.camera_ctrl_window, text="Start Localization",
                                                    command=self.live_localize_spots, fg='green')

            # Threshold value ctrls and packing
            self.threshold_frame_lvRec = tk.Frame(master=self.camera_ctrl_window)  # for holding ctrls below
            self.threshold_label_lvRec = tk.Label(master=self.threshold_frame_lvRec, text="Threshold(1..254): ")
            self.threshold_label_lvRec.pack(side='left', padx=1, pady=1)
            self.default_threshold = 110
            self.threshold_value_lvRec = tk.IntVar(); self.threshold_value_lvRec.set(self.default_threshold)
            # self.threshold_value.trace_add(mode="write", callback=self.validate_threshold)
            self.threshold_ctrl_box_lvRec = tk.Spinbox(master=self.threshold_frame_lvRec, from_=1, to=254,
                                                       increment=1, textvariable=self.threshold_value_lvRec,
                                                       wrap=True, width=4)   # adapt Spinbox to 4 digits in int
            self.threshold_ctrl_box_lvRec.pack(side='left', padx=1, pady=1)

            # Radius of sub-apertures ctrls and packing
            self.radius_frame_lvRec = tk.Frame(master=self.camera_ctrl_window)  # for holding ctrls below
            self.radius_label_lvRec = tk.Label(master=self.radius_frame_lvRec, text="Sub-ap. radius(px): ")
            self.radius_label_lvRec.pack(side='left', padx=1, pady=1)
            self.default_radius = 10.0
            self.radius_value_lvRec = tk.DoubleVar(); self.radius_value_lvRec.set(self.default_radius)
            # self.radius_value_lvRec.trace_add(mode="write", callback=self.validate_radius)
            self.radius_ctrl_box_lvRec = tk.Spinbox(master=self.radius_frame_lvRec, from_=1.0, to=50.0,
                                                    increment=0.2, textvariable=self.radius_value_lvRec,
                                                    wrap=True, width=4)   # adapt Spinbox to 4 digits in double
            self.radius_ctrl_box_lvRec.pack(side='left', padx=1, pady=1)

            # Provide selection of reconstruction views - spots or reconstructed profile
            self.views = ["Detected Spots", "Coefficients"]; self.selected_view = tk.StringVar()
            self.selected_view.set(self.views[0])
            self.view_selector = tk.OptionMenu(self.camera_ctrl_window, self.selected_view, *self.views,
                                               command=self.switch_reconstructed_view)

            # Exposure time control
            self.exposure_t_ms_box = tk.Frame(master=self.camera_ctrl_window)
            self.exposure_t_ms_label = tk.Label(master=self.exposure_t_ms_box, text="Exposure time[ms]: ")
            self.exposure_t_ms_ctrl = tk.DoubleVar(); self.exposure_t_ms_ctrl.set(self.exposure_t_ms)
            self.exposure_t_ms_selector = tk.Spinbox(master=self.exposure_t_ms_box, from_=self.exposure_t_ms_min,
                                                     to=self.exposure_t_ms_max,
                                                     increment=1.0, textvariable=self.exposure_t_ms_ctrl,
                                                     wrap=True, width=4, command=self.set_exposure_t_ms)
            self.exposure_t_ms_selector.bind('<Return>', self.validate_exposure_t_input)  # validate input
            self.exposure_t_ms_label.pack(side=tk.LEFT); self.exposure_t_ms_selector.pack(side=tk.LEFT)
            self.exposure_t_ms_selector.config(state="disabled")

            # Figure associated with live frame
            self.default_frame_figure = 6.0  # default figure size (in inches)
            self.frame_figure = plot_figure.Figure(figsize=(self.default_frame_figure, self.default_frame_figure))
            self.canvas = FigureCanvasTkAgg(self.frame_figure, master=self.camera_ctrl_window); self.canvas.draw()
            self.frame_widget = self.canvas.get_tk_widget()
            self.plot_toolbar = NavigationToolbar2Tk(self.canvas, self.camera_ctrl_window, pack_toolbar=False)
            self.plot_toolbar.update()
            # Assign subplot to the created figure
            if self.frame_figure_axes is None:
                self.frame_figure_axes = self.frame_figure.add_subplot()
                self.frame_figure_axes.axis('off'); self.frame_figure.tight_layout()
                self.frame_figure.subplots_adjust(left=0, bottom=0, right=1, top=1)  # remove white borders
            self.imshowing = None  # AxesImage instance

            # Grid layout of all widgets
            self.camera_ctrl_pad = 4
            self.plot_toolbar.grid(row=6, rowspan=1, column=2, columnspan=3,
                                   padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.single_snap_button.grid(row=0, rowspan=1, column=1, columnspan=1,
                                         padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.live_stream_button.grid(row=0, rowspan=1, column=2, columnspan=1,
                                         padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.camera_selector.grid(row=0, rowspan=1, column=0, columnspan=1,
                                      padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.exposure_t_ms_box.grid(row=0, rowspan=1, column=3, columnspan=1,
                                        padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.calibration_activate_button.grid(row=0, rowspan=1, column=4, columnspan=1,
                                                  padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.frame_widget.grid(row=1, rowspan=5, column=0, columnspan=5,
                                   padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.live_reconstruction_button.grid(row=6, rowspan=1, column=0, columnspan=1,
                                                 padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.camera_ctrl_window.update()

            # Camera Initialization
            self.messages2Camera = mpQueue(maxsize=10)  # create message queue for communication with the camera
            self.camera_messages = mpQueue(maxsize=10)  # create message queue for listening from the camera
            self.exceptions_queue = mpQueue(maxsize=5)  # Initialize separate queue for handling Exceptions
            self.images_queue = mpQueue(maxsize=40)  # Initialize the queue for holding acquired images
            self.image_height = 1000; self.image_width = 1000
            self.camera_handle = cam.cameras_ctrl.CameraWrapper(self.messages2Camera, self.exceptions_queue,
                                                                self.images_queue, self.camera_messages,
                                                                self.exposure_t_ms, self.image_width,
                                                                self.image_height,
                                                                self.selected_camera.get())
            self.camera_handle.start()  # start associated with the camera Process()
            # Wait the confirmation that camera initialized and Process launched
            camera_initialized_flag = False; time.sleep(self.gui_refresh_rate_ms/1000)
            while(not camera_initialized_flag):
                if not self.camera_messages.empty():
                    try:
                        message = self.camera_messages.get_nowait(); print(message)
                        if message == "Simulated camera Process has been launched":
                            self.single_snap_button.config(state="normal")
                            self.live_stream_button.config(state="normal")
                            self.camera_selector.config(state="normal")
                            self.exposure_t_ms_selector.config(state="normal")
                            camera_initialized_flag = True; break
                    except Empty:
                        pass
                else:
                    time.sleep(self.gui_refresh_rate_ms/1000)
            # Exceptions and messeges handling - start associated Threads
            self.exceptions_checker = cam.check_exception_streams.ExceptionsChecker(self.exceptions_queue,
                                                                                    self)
            self.messages_printer = cam.check_exception_streams.MessagesPrinter(self.camera_messages)
            self.exceptions_checker.start(); self.messages_printer.start()

    def snap_single_image(self):
        """
        Handle acquiring and representing of a single image.

        Returns
        -------
        None.

        """
        if self.camera_handle is not None:
            if not(self.messages2Camera.full()):
                self.messages2Camera.put_nowait("Snap single image")  # the command for acquiring single image
                # timeout to wait the image on the imagesQueue
                if self.exposure_t_ms > 5:
                    timeout_wait = 2*self.exposure_t_ms
                else:
                    timeout_wait = 10
                try:
                    # Waiting then image will be available
                    image = self.images_queue.get(block=True, timeout=(timeout_wait/1000))
                except Empty:
                    image = None
                    print("The snap image not acquired, timeout reached")
                # Represent image on the figure (associated widget)
                if not(isinstance(image, str)) and (image is not None) and isinstance(image, np.ndarray):
                    # Remove 3rd dimension from camera image for correct working of the cursor data update
                    if len(image.shape) > 2:
                        image = np.squeeze(image, axis=2)
                    self.current_image = image  # save it as single element available for other Threads
                    # For example, it will be available for Calibration and live spot detection
                    self.show_image(image)
                    # Activate below the button for calibration starting
                    if not self.calibration_activation:
                        self.calibration_activation = True
                        self.calibration_activate_button.config(state="normal")
                if isinstance(image, str):
                    print("Image: ", image)  # replacer for IDS simulation

    def live_stream(self):
        """
        Start and show live image stream.

        Returns
        -------
        None.

        """
        self.__flag_live_stream = not self.__flag_live_stream   # changing the state of the flag
        # Start Live Stream
        if self.__flag_live_stream:
            self.live_stream_button.config(text="Stop Live", fg='red')
            self.single_snap_button.config(state="disabled")  # disabling the Single Frame acquisition
            self.camera_selector.config(state="disabled")  # disabling selection of the active camera
            self.exposure_t_ms_selector.config(state="disabled")
            if self.selected_camera.get() == "IDS":
                self.live_reconstruction_button.config(state="disabled")
            self.plot_toolbar.grid_remove()  # remove toolbar from the widget
            self.frame_figure_axes.mouseover = False  # disable tracing mouse
            # self.imshowing.set_animated(True)  # tests say that it's unnecessary in this application
            self.messages2Camera.put_nowait("Start Live Stream")  # Send this command to the wrapper class
            # refresh of displayed images process => evoked Thread
            self.image_updater = Thread(target=self.update_image, args=())
            self.image_updater.start()  # start the Thread and assigned to it task
        # Stop Live Stream
        else:
            if self.__flag_live_localization:
                self.live_localize_spots()
                time.sleep(5*self.gui_refresh_rate_ms/1000)  # additional delay for stopping reconstructions
            if not(self.messages2Camera.full()):
                self.messages2Camera.put_nowait("Stop Live Stream")  # Send the message to stop live stream
            self.live_stream_button.config(text="Start Live", fg='green')
            self.single_snap_button.config(state="normal"); self.camera_selector.config(state="normal")
            self.exposure_t_ms_selector.config(state="normal")
            self.frame_figure_axes.format_coord = self.format_coord  # !!! Fix bug (next string)
            # Bug: after live streaming, hovering mouse pointer over the image not showing pixel value
            self.frame_figure_axes.mouseover = True  # enable tracing mouse
            # Restore toolbar on the same place as before
            self.plot_toolbar.grid(row=6, rowspan=1, column=2, columnspan=3,
                                   padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            if self.selected_camera.get() == "IDS":
                self.live_reconstruction_button.config(state="normal")

    def update_image(self):
        """
        Target for running on separate Thread updating of images coming from a camera.

        Returns
        -------
        None.

        """
        time.sleep(1.25*self.exposure_t_ms/1000)  # initial delay before querying for a new image from a queue
        delay = 1.04*self.exposure_t_ms  # delay specification for stream of querying for a new image
        while(self.__flag_live_stream):
            # t1 = time.perf_counter()
            try:
                image = self.images_queue.get_nowait()  # get new image from a queue
                if not(isinstance(image, str)) and (image is not None) and isinstance(image, np.ndarray):
                    # Remove 3rd dimension from camera image for correct working of the cursor data update
                    if len(image.shape) > 2:
                        image = np.squeeze(image, axis=2)
                    self.current_image = image  # save it as single element available for other Threads
                    # For example, it will be available for Calibration and live spot detection
                    # Below flag - for switching off the live stream displaying
                    if self.__flag_live_image_updater:
                        self.show_image(image)  # call the function for image refreshing
                    # Activate below the button for calibration starting
                    if not self.calibration_activation:
                        self.calibration_activation = True
                        self.calibration_activate_button.config(state="normal")
            except Empty:
                pass
            time.sleep(delay/1000)
            # t2 = time.perf_counter(); print("Image showing takes:", round((t2-t1)*1000))

    def show_image(self, image: np.ndarray):
        """
        Show image on the figure (widget).

        Parameters
        ----------
        image : np.ndarray
            Acquired / simulated image.

        Returns
        -------
        None.

        """
        # Initialize the AxesImage if this function called 1st time
        if image is not None and isinstance(image, np.ndarray) and self.imshowing is None:
            # Function below returns matplotlib.image.AxesImage class - need for further reference
            if self.selected_camera.get() == "Simulated":
                color_map = 'gray'
            else:
                color_map = 'plasma'
            self.imshowing = self.frame_figure_axes.imshow(image, cmap=color_map, interpolation='none',
                                                           vmin=0, vmax=255)
        # Redraw image in the figure
        if not self.__flag_live_stream:
            self.imshowing.set_data(image)  # set data for AxesImage for updating image content
            self.canvas.draw()
        else:
            # !!! Calling backend canvas for re-drawing of updated image in the idle state (draw_idle())
            # that is more effective then call self.canvas.draw()
            self.imshowing.set_data(image)  # set data for AxesImage for updating image content
            self.canvas.draw_idle()

    def format_coord(self, x, y):
        """
        Re-implementation of default matplotlib function that failed after live streaming launch.

        Reference
        ---------
            See https://matplotlib.org/stable/gallery/images_contours_and_fields/image_zcoord.html

        Parameters
        ----------
        x : float
            x coordinate provided by matplotlib events during hovering the mouse on an image.
        y : float
            y coordinate provided by matplotlib events.

        Returns
        -------
        str
            Formatted string that is shown by Navigation toolbar on the GUI.

        """
        numrows, numcols = self.current_image.shape  # image sizes
        x = int(round(x, 0)); y = int(round(y, 0))  # according the Reference above
        if 0 <= x < numcols and 0 <= y < numrows:
            z = self.current_image[y, x]
            return 'x=%1.0f, y=%1.0f \n [%1.0f]' % (x, y, z)
        else:
            return 'x=%1.0f, y=%1.0f' % (x, y)

    def close_current_camera(self):
        """
        Exit the associated with the active camera Process and close it (de-initialize).

        Returns
        -------
        None.

        """
        if self.camera_handle is not None:
            # If there is live streaming going on, just first call the function to stop it
            if self.__flag_live_stream:
                self.live_stream()
            # Disable all controlling buttons
            self.single_snap_button.config(state="disabled"); self.live_stream_button.config(state="disabled")
            self.camera_selector.config(state="disabled"); time.sleep(self.gui_refresh_rate_ms/1000)
            # Send the message to stop the imaging and deinitialize the camera:
            self.messages2Camera.put_nowait("Close the camera"); time.sleep(self.gui_refresh_rate_ms/1000)
            if self.camera_handle.is_alive():  # if the associated with the camera Process hasn't been finished
                self.camera_handle.join(timeout=self.global_timeout)  # wait the camera closing / deinitializing
                print("Camera process released")
            self.camera_handle = None  # for preventing again checking if it's alive
            # Print out all collected messages
            while not self.camera_messages.empty():
                try:
                    print(self.camera_messages.get_nowait())
                except Empty:
                    break

    def switch_active_camera(self, selected_camera: str):
        """
        Switch current active camera (IDS or Simulated).

        Parameters
        ----------
        selected_camera : str
            Returned tkinter OptionMenu button selected camera.

        Returns
        -------
        None.

        """
        # Clear the buffer with images
        if not self.images_queue.empty():
            for i in range(self.images_queue.qsize()):
                try:
                    self.images_queue.get_nowait()
                except Empty:
                    break
        self.close_current_camera()  # close the previously active camera
        print("Selected camera:", self.selected_camera.get())
        # Changing default exposure time for usability of the IDS camera
        if selected_camera == "IDS":
            self.exposure_t_ms = 2; self.exposure_t_ms_ctrl.set(2)
        else:
            self.exposure_t_ms = 50; self.exposure_t_ms_ctrl.set(50)
        # Initialize again the camera and associated Process
        self.camera_handle = cam.cameras_ctrl.CameraWrapper(self.messages2Camera, self.exceptions_queue,
                                                            self.images_queue, self.camera_messages,
                                                            self.exposure_t_ms, self.image_width, self.image_height,
                                                            self.selected_camera.get())
        if self.selected_camera.get() == "Simulated":
            self.camera_handle.start()  # start associated with the camera Process()
            self.live_reconstruction_button.config(state="disabled")  # disable reconstruction for simulations
        else:
            time.sleep(self.gui_refresh_rate_ms/1000)
        # Wait the confirmation that camera initialized
        camera_initialized_flag = False; time.sleep(self.gui_refresh_rate_ms/1000)
        if self.gui_refresh_rate_ms > 0 and self.gui_refresh_rate_ms < 1000:
            attempts = 6000//self.gui_refresh_rate_ms  # number of attempts to receive initialized message ~ 6 sec.
        else:
            attempts = 600
        i = 0  # counting attempts
        wait_camera = False  # for separate 2 events: import controlling library and confiramtion from a camera
        while(not camera_initialized_flag and i <= attempts):
            if not self.camera_messages.empty():
                try:
                    message = self.camera_messages.get_nowait()
                    if self.selected_camera.get() == "Simulated":
                        print(message)
                        if message == "Simulated camera Process has been launched":
                            camera_initialized_flag = True; break
                        else:
                            time.sleep(self.gui_refresh_rate_ms/1000); i += 1
                    else:
                        # print("Message from IDS camera:", message)
                        # This case - for looking for initial import success of controlling IDS library
                        if not wait_camera:
                            if message == "IDS controlling library and camera are both available":
                                wait_camera = True; self.camera_handle.start()  # start associated Process()
                                time.sleep(5*self.gui_refresh_rate_ms/1000)
                            else:
                                print("IDS camera not initialized, it reports:", message)
                                camera_initialized_flag = False; self.camera_handle = None
                                time.sleep(2*self.gui_refresh_rate_ms/1000); break
                        # This case - for waiting the confirmation, that camera initialized
                        else:
                            # print("Waiting confirmation from IDS camera and received: ", message)
                            if message == "The IDS camera initialized":
                                camera_initialized_flag = True; break
                            else:
                                time.sleep(self.gui_refresh_rate_ms/1000); i += 1
                except Empty:
                    time.sleep(self.gui_refresh_rate_ms/1000); i += 1
            else:
                time.sleep(self.gui_refresh_rate_ms/1000); i += 1
        # Below - handle if the IDS camera cannot be initialized
        if i > attempts and not camera_initialized_flag:
            print("Camera not initialized, timeout for initialization passed")
            self.camera_handle = None  # prevent to send to simulated camera close command
        # Below case - IDS camera library not installed or camera not connected or something else wrong
        if self.selected_camera.get() == "IDS" and not camera_initialized_flag:
            self.selected_camera.set("Simulated"); self.switch_active_camera("Simulated")
            self.exposure_t_ms = 50; self.exposure_t_ms_ctrl.set(50)
        # Change the accepted range for the Exposure time for IDS camera
        if self.selected_camera.get() == "IDS" and camera_initialized_flag:
            self.exposure_t_ms_min = 0.01; self.exposure_t_ms_max = 100
            self.exposure_t_ms_selector.config(from_=self.exposure_t_ms_min, to=self.exposure_t_ms_max,
                                               increment=self.exposure_t_ms_min, width=5)
            self.live_reconstruction_button.config(state="normal")
        # Below case - everything is ok, activate back buttons
        if camera_initialized_flag:
            self.single_snap_button.config(state="normal"); self.live_stream_button.config(state="normal")
            self.camera_selector.config(state="normal")

        # Configuration of color map for image show (if the switching between cameras happen)
        if self.imshowing is not None:
            # It's necessary to delete previous axes (subplot) for fully refresh the image and toolbar!
            self.frame_figure.clear()  # clear all axes
            self.frame_figure_axes = self.frame_figure.add_subplot()  # create new axis!
            self.frame_figure_axes.axis('off'); self.frame_figure.tight_layout()
            self.frame_figure.subplots_adjust(left=0, bottom=0, right=1, top=1)  # remove white borders
            # Set the zero image after the switching of camera
            if self.selected_camera.get() == "IDS":
                # Default for IDS camera sizes = 2048x2048, hard-coded now, but it reports this in the init. message
                self.image_width = 2048; self.image_height = 2048; color_map = 'plasma'
            else:
                self.image_width = 1000; self.image_height = 1000; color_map = 'gray'

            # Update blank image for picking up by toolbar right dimensions sizes
            self.current_image = np.zeros((self.image_width, self.image_height), dtype='uint8')
            self.imshowing = self.frame_figure_axes.imshow(self.current_image, cmap=color_map,
                                                           interpolation='none', vmin=0, vmax=255)
            self.canvas.draw(); self.frame_widget.update(); self.plot_toolbar.update()

    def validate_exposure_t_input(self, *args):
        """
        Validate user input into exposure time field.

        Parameters
        ----------
        *args : string arguments.
            Provided by tkinter signature call.

        Returns
        -------
        None.

        """
        try:
            exp_t = self.exposure_t_ms_ctrl.get()
            if exp_t < self.exposure_t_ms_min and self.exposure_t_ms_max > 100.0:
                self.exposure_t_ms_ctrl.set(self.exposure_t_ms)  # put previous assigned value
            else:
                self.exposure_t_ms = exp_t
        except tk.TclError:
            self.exposure_t_ms_ctrl.set(self.exposure_t_ms)  # put previous assigned value
        self.camera_ctrl_window.focus_set()  # removing focus from input text variable
        self.set_exposure_t_ms()   # call the set function of exposure time

    def set_exposure_t_ms(self):
        """
        Set exposure time for the active camera.

        Returns
        -------
        None.

        """
        self.exposure_t_ms = self.exposure_t_ms_ctrl.get()
        # below - send the tuple with string command and exposure time value
        if not self.messages2Camera.full():
            self.messages2Camera.put_nowait(("Set exposure time", self.exposure_t_ms))
        if self.selected_camera.get() == "Simulated":
            time.sleep(self.gui_refresh_rate_ms/2000)
        else:
            # Get actual set exposure time
            time.sleep((2*self.gui_refresh_rate_ms)/1000)
            if not self.camera_messages.empty():
                try:
                    # Checking the answeer about actually set exposure time from the camera
                    message = self.camera_messages.get_nowait()
                    exp_t = message.split(":")[1]; exp_t = round(float(exp_t), 2)
                    self.exposure_t_ms_ctrl.set(exp_t); print(message)
                except Empty:
                    pass

    def open_calibration(self):
        """
        Open calibration window and trasnfer last image to it.

        Returns
        -------
        None.

        """
        if self.__flag_live_stream:
            self.live_stream()  # stop live streaming, if it runs
        self.calibrate(True)  # open calibration window and transfer the current displayed image there
        self.camera_ctrl_exit()  # closes the camera controlling window, stop live stream

    def live_reconstruction(self):
        """
        Make live reconstruction of wavefronts.

        Returns
        -------
        None.

        """
        self.live_stream()  # call to the live stream initiliazation or stop
        self.live_reconstruction_button.config(state="normal")  # because it's disabled by self.live_stream()
        # Disable / enable calinbration feature and Stop Live / Start Live + put additional controls
        if self.__flag_live_stream:
            self.live_reconstruction_button.config(text="Stop Reconstruction", fg='red')
            self.get_focal_spots_button.config(text="Start Localization", fg='green')
            self.live_stream_button.config(state="disabled")
            self.calibration_activate_button.config(state="disabled")
            self.get_focal_spots_button.grid(row=6, rowspan=1, column=1, columnspan=1,
                                             padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.threshold_frame_lvRec.grid(row=6, rowspan=1, column=2, columnspan=1,
                                            padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.radius_frame_lvRec.grid(row=6, rowspan=1, column=3, columnspan=1,
                                         padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.plot_points = None
        else:
            # Buttons show / hide
            self.live_reconstruction_button.config(text="Start Reconstruction", fg='magenta')
            self.live_stream_button.config(state="normal"); self.radius_frame_lvRec.grid_remove()
            self.calibration_activate_button.config(state="normal"); self.view_selector.grid_remove()
            self.get_focal_spots_button.grid_remove(); self.threshold_frame_lvRec.grid_remove()

    def live_localize_spots(self):
        """
        Launch the localization of focal spots on the separate thread.

        Returns
        -------
        None.

        """
        self.__flag_live_localization = not self.__flag_live_localization
        time.sleep(10*self.gui_refresh_rate_ms/1000)
        if self.__flag_live_stream and self.__flag_live_localization:
            # refresh of displayed images process => evoked Thread
            self.reconstruction_updater = Thread(target=self.localize_spots_on_thread, args=())
            self.reconstruction_updater.start()  # start the Thread and assigned to it task
            self.get_focal_spots_button.config(text="Stop Localization", fg='red')
            self.radius_frame_lvRec.grid_remove(); self.threshold_frame_lvRec.grid_remove()
            self.view_selector.grid(row=6, rowspan=1, column=2, columnspan=1,
                                    padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
        elif not self.__flag_live_localization:
            self.get_focal_spots_button.config(text="Start Localization", fg='green')
            self.radius_frame_lvRec.grid(row=6, rowspan=1, column=3, columnspan=1,
                                         padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.threshold_frame_lvRec.grid(row=6, rowspan=1, column=2, columnspan=1,
                                            padx=self.camera_ctrl_pad, pady=self.camera_ctrl_pad)
            self.view_selector.grid_remove()

    def switch_reconstructed_view(self, view: str):
        """
        Turn on / off live displaying of images from a camera.

        Parameters
        ----------
        view : str
            Provided by tkinter call with the selected value.

        Returns
        -------
        None.

        """
        # Switch on / off the live displaying of coming from camera images
        if view == "Coefficients":
            self.__flag_live_image_updater = False; self.imshowing = None
            time.sleep(self.gui_refresh_rate_ms/1000)  # for stopping the live updater
            if self.plot_points is not None:
                self.plot_points.pop(0).remove(); self.plot_points = None
            time.sleep(6*self.gui_refresh_rate_ms/1000)
            # See the set up example in get_plot_zps_polar()
            self.frame_figure.clear()  # clear the axes from figure
            self.frame_figure_axes = self.frame_figure.add_subplot(projection='polar')
            self.frame_figure_axes.grid(False); self.frame_figure_axes.axis('off')
            self.frame_figure_axes.set_theta_direction(-1)
            self.frame_figure.subplots_adjust(left=0, bottom=0, right=1, top=1)
            self.frame_figure.tight_layout()
            self.__flag_update_amplitudes = True  # allowing to make amplitudes graph
            # Make the external window for the representation of the calculated coefficients
            y_shift = (int(1.3*self.master.winfo_height())
                       + int(self.master.winfo_geometry().split("+")[2]))  # vertical shift
            x_shift = self.master.winfo_x()  # horizontal shift
            height = self.master.winfo_height(); width = self.master.winfo_width()
            self.show_coefficients_win = tk.Toplevel(master=self.camera_ctrl_window)
            self.show_coefficients_win.title("Zernike coefficients")
            self.show_coefficients_win.geometry(f'{width}x{height}+{x_shift}+{y_shift}')
            self.show_coefficients_win.protocol("WM_DELETE_WINDOW", self.show_coefficients_win_close)
            # Create the holder for plots - figure and associated axes
            if self.amplitudes_figure is None:
                self.amplitudes_figure = plot_figure.Figure()
            self.amplitudes_canvas = FigureCanvasTkAgg(self.amplitudes_figure,
                                                       master=self.show_coefficients_win)
            self.amplitudes_canvas.draw()
            self.amplitudes_widget = self.amplitudes_canvas.get_tk_widget()
            self.amplitudes_widget.pack(side='top', padx=2, pady=2)
            self.__flag_bar_plot = True  # for enabling plotting bars
        else:
            self.__flag_update_amplitudes = False
            self.show_coefficients_win_close()
            time.sleep(12*self.gui_refresh_rate_ms/1000)  # put some delay for prevent draw on refresh image
            # Clear the 2D profile with sum of Zernike coefficients
            self.frame_figure.clear()  # clear all axes
            time.sleep(self.gui_refresh_rate_ms/1000)  # put some delay for cleaning
            self.frame_figure_pcolormesh = None
            self.frame_figure_axes = self.frame_figure.add_subplot()  # create new axis!
            self.frame_figure_axes.axis('off'); self.frame_figure.tight_layout()
            self.frame_figure.subplots_adjust(left=0, bottom=0, right=1, top=1)  # remove white borders
            self.imshowing = self.frame_figure_axes.imshow(self.current_image, cmap='plasma',
                                                           interpolation='none', vmin=0, vmax=255)
            self.canvas.draw(); time.sleep(self.gui_refresh_rate_ms/1000)
            self.plot_toolbar = NavigationToolbar2Tk(self.canvas, self.camera_ctrl_window, pack_toolbar=False)
            self.plot_toolbar.update(); self.__flag_live_image_updater = True
        print("Selected representation of reconstructed coefficients:", view)

    def localize_spots_on_thread(self):
        """
        Localize current focal spots, calculate shifts between them and calibrated spots and Zernike coefficients.

        Returns
        -------
        None.

        """
        while(self.__flag_live_stream and self.__flag_live_localization):
            t1 = time.perf_counter()
            region_size = int(np.round(1.6*self.radius_value_lvRec.get(), 0))
            self.coms_aberrated = get_coms_fast(image=self.current_image, nonaberrated_coms=self.coms_spots,
                                                threshold_abs=self.threshold_value_lvRec.get(),
                                                region_size=region_size)
            # Below - plotting found (locali spots
            rows, cols = self.coms_aberrated.shape
            if rows > 0 and cols > 0:
                (self.coms_shifts, self.integral_matrix_aberrated,
                 self.coms_aberrated) = get_coms_shifts_fast(self.coms_spots, self.integral_matrix,
                                                             self.coms_aberrated)
                # Show the detected focal spots, if the live stream shown in the GUI
                if self.__flag_live_image_updater:
                    # Use only one sample of plt.lines.Line2D for drawing found spots
                    if self.plot_points is None:
                        self.plot_points = self.frame_figure_axes.plot(self.coms_aberrated[:, 1],
                                                                       self.coms_aberrated[:, 0],
                                                                       '.', color="red")
                    else:
                        self.plot_points[0].set_data(self.coms_aberrated[:, 1], self.coms_aberrated[:, 0])
                    self.canvas.draw_idle()  # redraw image in the widget (stored in canvas)
                # t2 = time.perf_counter(); print("Shifts coms calc. takes sec.:", round((t2-t1), 1))
                rows, cols = self.coms_shifts.shape
                if rows > 0 and cols > 0:
                    self.alpha_coefficients = get_polynomials_coefficients(self.integral_matrix_aberrated,
                                                                           self.coms_shifts)
                    if np.size(self.alpha_coefficients) > 0:
                        self.alpha_coefficients = np.round(self.alpha_coefficients, 4)
                    # self.alpha_coefficients *= np.pi  # for adjusting to radians ???
                    self.alpha_coefficients = list(self.alpha_coefficients)
                    if len(self.alpha_coefficients) > 0:
                        # Define below used orders of Zernikes and providing them for amplitudes calculation
                        self.order = get_zernike_order_from_coefficients_number(len(self.alpha_coefficients))
                        self.zernike_list_orders = get_zernike_coefficients_list(self.order)
                        # print(self.alpha_coefficients)  # for debugging
                        if not self.__flag_live_image_updater and self.__flag_update_amplitudes:
                            # below - function for drawing 2D Zernike polynomials coefficients sum
                            # Explicit call for making the pcolormesh plot
                            R, Theta, S = zernike_polynomials_sum_tuned(self.zernike_list_orders,
                                                                        self.alpha_coefficients,
                                                                        step_r=0.01, step_theta=1.0)
                            # Get the sample of
                            if self.frame_figure_pcolormesh is None:
                                self.frame_figure_pcolormesh = self.frame_figure_axes.pcolormesh(Theta, R, S,
                                                                                                 cmap='coolwarm',
                                                                                                 shading='nearest')
                                # shows the colour bar on the figure
                                self.frame_figure_colorbar = self.frame_figure.colorbar(self.frame_figure_pcolormesh,
                                                                                        ax=self.frame_figure_axes)
                                self.canvas.draw_idle()  # redraw the figure
                            else:
                                # Updating the colormesh figure using the method of a QuadMesh class
                                # self.frame_figure_pcolormesh.set_array(S)
                                # Unfortunately, simple updating of array values doesn't provide the automatic
                                # update of colorbar. Therefore, below pcolormesh and colorbar deleted and re-created
                                self.frame_figure_colorbar.remove(); self.frame_figure_pcolormesh.remove()
                                self.frame_figure_pcolormesh = self.frame_figure_axes.pcolormesh(Theta, R, S,
                                                                                                 cmap='coolwarm',
                                                                                                 shading='nearest')
                                self.frame_figure_colorbar = self.frame_figure.colorbar(self.frame_figure_pcolormesh,
                                                                                        ax=self.frame_figure_axes)

                                self.canvas.draw_idle()
                            # Plot coefficients (amplitudes) as bars on the external window
                            if self.__flag_bar_plot:
                                if self.amplitudes_figure_axes is None:
                                    self.make_amplitudes_subplots()  # see the wrapper function
                                else:
                                    if self.amplitudes_showing is not None:
                                        # Bars updating using the method of a BarContainer class
                                        # Unfortunately, the command below doesn't send automatic command
                                        # to update the axes parameters
                                        # i = 0
                                        # for bar in self.amplitudes_showing:
                                        #     bar.set_height(self.alpha_coefficients[i]); i += 1
                                        # Because of the reason above, the dirty hack now is to remove /
                                        # recreate bar plots
                                        self.amplitudes_showing.remove()
                                        self.amplitudes_showing = self.amplitudes_figure_axes.bar(self.names,
                                                                                                  self.alpha_coefficients,
                                                                                                  color='blue')
                                self.amplitudes_canvas.draw_idle()
                                # FIXME: Made drawing on the main thread! Instead of process now
                                # or made it wrapped to another function as update_image above
                        t3 = time.perf_counter(); print("Coeff-s calc./showing takes sec.:", round((t3-t1), 1))
        # Below - removing drawn localized spots from the image
        # Ref.: https://www.adamsmith.haus/python/answers/how-to-remove-a-line-from-a-plot-in-python
        if self.plot_points is not None:
            self.plot_points.pop(0).remove(); self.plot_points = None
        # Return to normal representation if this function stop running (reconstruction stopped)
        if not self.__flag_live_image_updater:
            self.selected_view.set(self.views[0]); self.switch_reconstructed_view(self.views[0])

    def make_amplitudes_subplots(self):
        """
        Make the subplots for representation of calculated Zernike polynomial coefficients.

        Returns
        -------
        None.

        """
        if self.amplitudes_figure_axes is None:
            self.amplitudes_figure_axes = self.amplitudes_figure.add_subplot()
            self.amplitudes_figure.subplots_adjust(left=0.08, bottom=0.16, right=0.98, top=0.98)
        if self.amplitudes_showing is None:
            self.names = []
            for mode in self.zernike_list_orders:
                mode = str(mode).replace(" ", ""); self.names.append("Z"+mode)
            self.amplitudes_showing = self.amplitudes_figure_axes.bar(self.names,
                                                                      self.alpha_coefficients,
                                                                      color='blue')
            self.amplitudes_figure.tight_layout()

    def show_coefficients_win_close(self):
        """
        Handle close window with plotting bars with the Zernike polynomials coefficients.

        Returns
        -------
        None.

        """
        self.__flag_bar_plot = False
        time.sleep(10*self.gui_refresh_rate_ms/1000)  # for ensuring that the drawing of bars stopped
        self.amplitudes_figure = None
        self.show_coefficients_win.destroy()

    def camera_ctrl_exit(self):
        """
        Handle the closing event for camera control window (the additional Top-level window).

        Returns
        -------
        None.

        """
        if self.__flag_bar_plot:
            self.show_coefficients_win_close()
        self.close_current_camera()
        time.sleep(2*self.gui_refresh_rate_ms/1000)
        if self.exceptions_checker.is_alive():
            self.exceptions_queue.put_nowait("Stop Exception Checker")
            # The problem is here, that Thread below somehow waits for exit action, so
            # this Thread cannot be joined in the main thread, therefore
            # self.exceptions_checker.join() deleted from here
        if self.messages_printer.is_alive():
            self.camera_messages.put_nowait("Stop Messages Printer")
            self.messages_printer.join()
            print("Messages Printer stopped")
        self.frame_figure_axes = None
        self.camera_ctrl_window.destroy(); self.camera_ctrl_window = None


# %% Launch settings
def correct_blur_on_windows():
    """
    Fix the issue with blurring if the script is launched on Windows.

    Returns
    -------
    None.

    """
    if platform.system() == "Windows":
        ctypes.windll.shcore.SetProcessDpiAwareness(2)


def print_short_license_note():
    """
    Print out the GPLv3 short notification.

    Returns
    -------
    None.

    """
    print("'Wavefront reconstruction GUI' Copyright (C) 2022  Sergei Klykov (according to GPLv3)")


def launch():
    """
    Launch the UI application upon the call.

    Returns
    -------
    None.

    """
    correct_blur_on_windows(); print_short_license_note()  # initial functions
    rootTk = tk.Tk(); gui = ReconstructionUI(rootTk); gui.mainloop()


# %% Main launch
if __name__ == "__main__":
    correct_blur_on_windows(); print_short_license_note()  # initial functions
    rootTk = tk.Tk()  # toplevel widget of Tk there the class - child of tk.Frame is embedded
    reconstructor_gui = ReconstructionUI(rootTk)
    reconstructor_gui.mainloop()
    if reconstructor_gui.coms_shifts is not None and reconstructor_gui.coms_aberrated is not None:
        coms_shifts = reconstructor_gui.coms_shifts
        coms_aberrated = reconstructor_gui.coms_aberrated
        coms_nonaberrated = reconstructor_gui.coms_spots

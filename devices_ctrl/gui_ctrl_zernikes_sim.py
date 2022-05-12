# -*- coding: utf-8 -*-
"""
GUI for representing and controlling Zernike polynomials.

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3 (check https://www.gnu.org/licenses/ )

"""
# %% Imports
import tkinter as tk
# Below: themed buttons for tkinter, rewriting standard one from tk
from tkinter.ttk import Button, Frame, Label, OptionMenu, Checkbutton, Spinbox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # import canvas container from matplotlib for tkinter
from zernike_pol_calc import get_plot_zps_polar, get_classical_polynomial_name, get_osa_standard_index
import matplotlib.figure as plot_figure
import time
import numpy as np
from scipy.io import loadmat
import ctypes  # for fixing of blurred text if launched by pure Python
import platform
import os
import skimage.io


# %% GUI class
class ZernikeCtrlUI(Frame):  # all widgets master class - top level window
    """Class specified the GUI controlling instance."""

    master: tk.Tk; figure: plt.figure
    plotColorbar: bool; flagFlattened: bool; amplitudes: list; orders: list; changedSliders: int

    def __init__(self, master):
        """
        Initialize the main controlling window, that is extending tkinter.Frame.

        Parameters
        ----------
        master : tkinter.Tk()
            Root (main) window for starting application, according the sample from the Python reference().

        Returns
        -------
        None.

        """
        # Values initialization
        super().__init__(master)  # initialize the main window (frame) for all widgets
        self.plotColorbar = False; self.master.title("Zernike polynomials controls and representation")
        self.master.geometry("+3+40")  # put the main window on the (+x, +y) coordinate away from the top left display coord.
        self.amplitudes = [0.0, 0.0]  # default amplitudes for the 1st order
        self.orders = [(-1, 1), (1, 1)]; self.flagFlattened = False; self.changedSliders = 1
        self.amplitudes_sliders_dict = {}; self.minV = 200; self.maxV = 400
        self.deviceHandle = None  # holder for the opened serial communication handle
        self.serial_comm_ctrl = None  # empty holder for serial communication ctrl
        self.voltages_bits = None  # empty holder because of asking in the end of the script to return this value
        self.librariesImported = False  # libraries for import - PySerial and local device controlling library
        self.baudrate = 115200  # default rate for serial communication
        self.ampcomImported = False; self.converted_voltages = []; self.volts_written = False
        self.parse_indices = []; self.offset_zernikes = np.zeros(shape=1); self.fitted_offsets = np.zeros(shape=1)
        self.corrections_loaded = False; self.flatten_field_coefficients = np.zeros(shape=1)
        self.corrected_voltages = np.zeros(shape=1)
        self.sliders_shown = True  # flag for the controlling of controls representation
        # tk.ttk.Style().theme_use('default')  # also can be: classic, alt, clam
        self.amplitudes_doubleVars_dict = {}; self.amplitudes_inputs_dict = {}
        self.icons_dict = {}; self.canvas_dict = {}; self.plotWidgets_dict = {}
        self.slider_length = 168  # default length of sliders controllers
        self.default_font = tk.font.nametofont("TkDefaultFont"); self.default_entry_font = tk.font.nametofont("TkTextFont")
        self.default_menu_font = tk.font.nametofont("TkMenuFont"); self.tooltip_font = tk.font.nametofont("TkTooltipFont")
        self.main_figure_size = 5.0; self.icon_figure_size = 0.55

        # Below - matrices placeholders for possible returning some placeholders instead of exception
        self.voltages = np.empty(1); self.check_solution = np.empty(1); self.zernike_amplitudes = np.empty(1)
        self.diff_amplitudes = np.empty(1); self.influence_matrix = np.empty(1)

        # Widgets creation and specification
        self.refresh_plot_button = Button(self, text="Refresh Plot", command=self.plot_zernikes)
        self.zernikesLabel = Label(self, text=" Polynomials ctrls up to:")
        self.figure = plot_figure.Figure(figsize=(self.main_figure_size, self.main_figure_size))  # Empty figure for phase profile
        self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.plotWidget = self.canvas.get_tk_widget()

        # Below - the way of how associate tkinter buttons with the variables and their states
        self.color_bar_state = tk.BooleanVar(); self.color_bar_state.set(False)
        self.plot_colorbar_button = Checkbutton(self, text="Colorbar", command=self.color_bar_plotting,
                                                onvalue=True, offvalue=False, variable=self.color_bar_state)
        self.load_infl_matrix_button = Button(self, text="Load Infl. Matrix", command=self.load_influence_matrix)
        self.load_infl_matrix_button.state(['!disabled', 'disabled'])  # disable of ttk button
        self.flatten_button = Button(self, text="Flatten all", command=self.flatten_zernike_profile)
        self.getVoltsButton = Button(self, text="Get Volts", command=self.getVolts)
        self.getVoltsButton.state(['disabled'])
        self.adjust_gui_sizes_button = Button(self, text="Adjust Font Sizes", command=self.adjust_gui_sizes)

        # Below - specification of OptionMenu from ttk for polynomials order selection, fixed thanks to StackOverflow
        self.order_n = ["1st ", "2nd ", "3rd ", "4th ", "5th ", "6th ", "7th "]
        self.order_list = [item + "order" for item in self.order_n]
        self.selected_order = tk.StringVar(); self.selected_order.set(self.order_list[0])
        self.max_order_selector = OptionMenu(self, self.selected_order, self.order_list[0], *self.order_list,
                                             command=self.number_orders_changed)

        # Specification of two case selectors: Simulation / Controlling DPP
        self.operation_modes = ["Pure Simulator", "DPP + Simulator"]; self.operation_mode_selector = tk.StringVar()
        self.operation_mode_selector.set(self.operation_modes[0])
        self.operation_mode_sel_button = OptionMenu(self, self.operation_mode_selector, self.operation_modes[0],
                                                    *self.operation_modes, command=self.operation_mode_selection)

        # Max voltage control with the named label and Combobox for controlling voltage
        self.max_V_box = Frame(self); textVMaxLabel = Label(self.max_V_box, text="Max Volts: ")
        self.maxV_selector_value = tk.IntVar(); self.maxV_selector_value.set(200)  # initial voltage
        # Below - add the association of updating of integer values of Spinbox input value:
        self.maxV_selector_value.trace_add("write", self.maxV_changed)
        self.maxV_selector = Spinbox(self.max_V_box, from_=self.minV, to=self.maxV,
                                     increment=10, state=tk.DISABLED, width=4,
                                     exportselection=True, textvariable=self.maxV_selector_value)
        textVMaxLabel.pack(side=tk.LEFT); self.maxV_selector.pack(side=tk.LEFT)

        # Below - additional window for holding the sliders with the amplitudes
        self.ampl_ctrls = tk.Toplevel(master=self)  # additional window, master - the main window

        # Placing all created widgets in the grid layout on the main window
        padx = 6; pady = 4  # overall additional border distances for all widgets
        self.pady = pady; self.padx = padx
        self.zernikesLabel.grid(row=0, rowspan=1, column=0, columnspan=1, padx=padx, pady=pady)
        self.max_order_selector.grid(row=0, rowspan=1, column=1, columnspan=1, padx=padx, pady=pady)
        self.refresh_plot_button.grid(row=0, rowspan=1, column=2, columnspan=1, padx=padx, pady=pady)
        self.plot_colorbar_button.grid(row=0, rowspan=1, column=3, columnspan=1, padx=padx, pady=pady)
        self.flatten_button.grid(row=0, rowspan=1, column=4, columnspan=1, padx=padx, pady=pady)
        self.operation_mode_sel_button.grid(row=7, rowspan=1, column=0, columnspan=1, padx=padx, pady=pady)
        self.load_infl_matrix_button.grid(row=7, rowspan=1, column=1, columnspan=1, padx=padx, pady=pady)
        self.max_V_box.grid(row=7, rowspan=1, column=2, columnspan=1, padx=padx, pady=pady)
        self.getVoltsButton.grid(row=7, rowspan=1, column=3, columnspan=1, padx=padx, pady=pady)
        self.plotWidget.grid(row=1, rowspan=6, column=0, columnspan=5, padx=padx, pady=pady)
        self.adjust_gui_sizes_button.grid(row=7, rowspan=1, column=4, columnspan=1, padx=padx, pady=pady)
        self.grid(); self.master.update()  # for updating associate with master properties (geometry)

        # set default numbers of Zernike polynomials sliders controls
        self.selected_order.set(self.order_list[3]); self.after(0, self.number_orders_changed(self.order_list[3]))
        self.master_geometry = self.master.winfo_geometry()  # saves the main window geometry
        self.after_id = self.after(1000, self.always_on_top)  # launch the function bringing the main window always on top

        # Below - blurred text fixing for launching this script by Python console (not from ID, there it isn't observed)
        print("Script launched on:", platform.system())
        if platform.system() == "Windows":
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            dpi = master.winfo_fpixels('1i')
            # print(dpi)  #  check the usage!

    # %% Simulation
    def plot_zernikes(self):
        """
        Plot the sum of specified Zernike's polynomials amplitudes.

        Returns
        -------
        None.

        """
        # t1 = time.time()
        # below: update the plot
        self.figure = get_plot_zps_polar(self.figure, orders=self.orders, step_r=0.01, step_theta=0.8,
                                         alpha_coefficients=self.amplitudes, show_amplitudes=self.plotColorbar)
        self.canvas.draw()  # redraw the figure
        # t2 = time.time(); print("redraw time(ms):", int(np.round((t2-t1)*1000, 0)))  # for debugging

    def color_bar_plotting(self):
        """
        Redraw of colormap with the Zernike's polynomials sum on the unit radius aperture.

        Returns
        -------
        None.

        """
        self.plotColorbar = self.color_bar_state.get(); self.plot_zernikes()

    def sliderValueChanged(self, new_pos):
        """
        Any slider value has been changed and this function handles it.

        Parameters
        ----------
        new_pos : double
            It is sent by the evoking of this event button.

        Returns
        -------
        None.

        """
        # new_pos sent by the associated button
        i = 0
        for key in self.amplitudes_sliders_dict.keys():  # storing the amplitudes in the list
            self.amplitudes[i] = self.amplitudes_sliders_dict[key].get(); i += 1
        if self.changedSliders > 1:  # flatten operation (counting down sliders, redraw only once)
            self.changedSliders -= 1
        if not self.flagFlattened:  # if no flatten flag, redraw the plot (1 slider only changed)
            self.plot_zernikes()
        else:
            if self.changedSliders == 1:  # if all sliders finally zeroed, redraw the plot
                self.flagFlattened = False; self.plot_zernikes()

    def flatten_zernike_profile(self):
        """
        Make all amplitude sliders controls equal to 0.0 value.

        Returns
        -------
        None.

        """
        self.flagFlattened = True  # flag for preventing the redrawing
        if self.sliders_shown:
            for key in self.amplitudes_sliders_dict.keys():
                if abs(self.amplitudes_sliders_dict[key].get()) > 0.0001:  # if not actually equal to zero
                    self.amplitudes_sliders_dict[key].set(0.0)
                    self.changedSliders += 1  # counting number of zeroed sliders for preventing multiple redrawing
        else:
            for key in self.amplitudes_doubleVars_dict.keys():
                if abs(self.amplitudes_doubleVars_dict[key].get()) > 0.0001:  # if not actually equal to zero
                    self.amplitudes_doubleVars_dict[key].set(0.0)

    def number_orders_changed(self, selected_order: str):
        """
        Handle the event of order specification.

        Parameters
        ----------
        selected_order : str
            Reported string with the selected order.

        Returns
        -------
        None.

        """
        n_orders = int(selected_order[0]); pad = 5
        # Refresh the TopLevel window and the associated dictionary with buttons
        y_shift = self.master.winfo_geometry().split("+")[2]  # shift of Toplevel window vertically
        # shift of Toplevel window horizontally
        x_shift = self.master.winfo_x() + self.master.winfo_width() + (self.padx + self.padx//2)
        self.ampl_ctrls.destroy(); self.ampl_ctrls = tk.Toplevel(master=self)
        # self.ampl_ctrls.wm_transient(self)  # de-activate all buttons except close on this Toplevel widget
        self.ampl_ctrls.protocol("WM_DELETE_WINDOW", self.no_exit)
        self.ampl_ctrls.title("Amplitude controls")
        self.ampl_ctrls.geometry(f'+{x_shift}+{y_shift}')  # shifting relative to the main window size and positions
        self.master.lift(aboveThis=self.ampl_ctrls)  # makes the main window above the created amplitudes controls
        # Get the (m, n) values from the order specification
        self.orders = []; self.amplitudes_sliders_dict = {}; self.amplitudes = []  # refresh the associated controls
        self.amplitudes_ctrl_boxes_dict = {}; self.amplitudes_labels_dict = {}
        # Construction of amplitude controls
        for order in range(1, n_orders + 1):  # going through all specified orders
            m = -order  # azimuthal order
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                self.orders.append((m, n))  # store the values as tuples
                # Below - initialization of frames for composing controls of polynomials amplitudes
                classical_name = get_classical_polynomial_name((m, n), short_names=True)  # Like Vertical tilt
                self.amplitudes_ctrl_boxes_dict[(m, n)] = Frame(self.ampl_ctrls)  # Frame to hold sliders, labels, etc.
                self.amplitudes_labels_dict[(m, n)] = Label(self.amplitudes_ctrl_boxes_dict[(m, n)],
                                                            text=(f"{(m ,n)} " + classical_name),
                                                            font=(self.default_font.actual()['family'],
                                                                  self.default_font.actual()['size'], 'bold'))
                # Slider below - to control an amplitude of polynomial
                self.amplitudes_labels_dict[(m, n)].grid(row=0, rowspan=1, column=0, columnspan=1)
                self.amplitudes.append(0.0)  # assign all zeros as the flat field, reinitialized each time!
                m += 2  # according to the specification of Zernike polynomial
        # TODO: placing of all controllers is bind to Sliders, make it universal (inputs / sliders)
        self.create_sliders()  # creation of sliders moved to the function
        self.amplitudes_sliders_dict[(1, 1)].update()  # for updating internal geometry properties
        self.amplitudes_labels_dict[(1, 1)].update()   # for updating internal geometry properties
        # Defining the sizes for placement of widgets
        width = int(self.amplitudes_sliders_dict[(1, 1)].winfo_geometry().split("x")[0])
        height = (int(self.amplitudes_sliders_dict[(1, 1)].winfo_geometry().split("x")[1].split("+")[0])
                  + int(self.amplitudes_labels_dict[(1, 1)].winfo_geometry().split("x")[1].split("+")[0]))
        self.slider_intbox_selector = tk.Button(self.ampl_ctrls, text="Active: Sliders", command=self.select_ampl_ctrls,
                                                fg='green', font=(self.default_font.actual()['family'],
                                                                  self.default_font.actual()['size'], 'bold'))
        # Placing the sliders on the window in pyramidal fashion
        y_coordinate = pad  # initial horizontal shift
        if n_orders <= 6:  # seems that up to 6 order, all sliders could be placed inside the window next to a main
            n_max = n_orders  # controls pyramidal placing below
        else:
            n_max = 6  # pyramid only up to 6th order
        for order in range(1, n_max + 1):
            m = -order  # azimuthal order
            n = order  # radial order
            x_coordinate = pad + ((n_max-order)*width)//2  # making pyramid on horizontal placing
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                self.amplitudes_ctrl_boxes_dict[(m, n)].place(x=x_coordinate, y=y_coordinate)
                m += 2
                x_coordinate += width + pad  # adding pad and width of a widget to place next one
            if order == 1:
                if n_orders >= 3:
                    x_coordinate += n_orders*pad + (n_orders-2)*(width//4)
                else:
                    x_coordinate += pad + n_orders*pad
                self.slider_intbox_selector.place(x=x_coordinate, y=(y_coordinate + (height//4)))
                self.slider_intbox_selector.update()  # for updating geometry property of a button
                width_sel = int(self.slider_intbox_selector.winfo_geometry().split("x")[0])
            y_coordinate += height + pad  # adding pad and width of a widget to place next row with
        # 7th order layout differently and manually, it's expected to control not more than 7 orders
        if n_orders == 7:
            m = -7; n = 7; x_coordinate2 = (1/3)*width
            # placing 5 polynomial out of 8
            for polynomial in range(5):  # number of polynomials = order + 1
                self.amplitudes_ctrl_boxes_dict[(m, n)].place(x=x_coordinate2, y=y_coordinate)
                m += 2
                x_coordinate2 += (4/3)*width + pad  # adding pad and width of a widget to place next one
            # placing remaining 7th order polynomials
            y_coordinate += height + pad; x_coordinate2 = width + pad
            for polynomial in range(3):  # number of polynomials = order + 1
                self.amplitudes_ctrl_boxes_dict[(m, n)].place(x=x_coordinate2, y=y_coordinate)
                m += 2
                x_coordinate2 += 2*width + pad  # adding pad and width of a widget to place next one
            y_coordinate += height + pad
        # Below - setting the geometry setting for Toplevel window and command to refresh profile
        # Also, correcting amplitudes controlling window with sliders for adding the button
        if n_orders == 1:
            x_coordinate += width_sel + 2*pad
        elif n_orders == 2:
            x_coordinate += abs(width//2 - width_sel) + 4*pad
        self.ampl_ctrls.geometry(f'{x_coordinate}x{y_coordinate}'); self.ampl_ctrls.update()
        self.plot_zernikes()  # refresh the plot, not retain any values

    def select_ampl_ctrls(self):
        """
        Make different controls representation for Zernike polynomials amplitudes.

        Returns
        -------
        None.

        """
        if self.sliders_shown:
            # Destroy sliders controls
            self.max_order_selector.config(state="disabled")  # disabling refreshing of # of orders
            for key, slider_frame in self.amplitudes_sliders_dict.items():
                slider_frame.destroy()
            self.slider_intbox_selector.config(text="Active: Inputs")
            self.create_amplitudes_inputs()  # Create input controls
        else:
            # Restore sliders controls
            self.max_order_selector.config(state="normal")
            # Destroy associated with input boxes controllers
            for key, input_frame in self.amplitudes_inputs_dict.items():
                input_frame.destroy()
            for key, icon in self.plotWidgets_dict.items():
                icon.destroy()
            # Recreate sliders and restore saved amplitde values
            self.create_sliders()
            i = 0
            for key in self.amplitudes_sliders_dict.keys():  # set previously selected amplitudes
                self.amplitudes_sliders_dict[key].set(self.amplitudes[i]); i += 1
            self.slider_intbox_selector.config(text="Active: Sliders")
        self.sliders_shown = not self.sliders_shown

    def create_amplitudes_inputs(self):
        """
        Create Spinboxes for manual controls of amplitudes.

        Returns
        -------
        None.

        """
        n_orders = int(self.selected_order.get()[0])
        i = 0  # counter for saved list of amplitudes
        for order in range(1, n_orders + 1):  # going through all specified orders
            m = -order  # azimuthal order
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                # Slider below - to control an amplitude of polynomial
                self.amplitudes_doubleVars_dict[(m, n)] = tk.DoubleVar()
                self.amplitudes_doubleVars_dict[(m, n)].set(self.amplitudes[i]); i += 1
                self.amplitudes_doubleVars_dict[(m, n)].trace_add("write", self.amplitude_input_changed)
                self.amplitudes_inputs_dict[(m, n)] = Spinbox(self.amplitudes_ctrl_boxes_dict[(m, n)],
                                                              from_=-1.0, to=1.0, increment=0.01, width=6,
                                                              exportselection=True,
                                                              textvariable=self.amplitudes_doubleVars_dict[(m, n)])
                pad = 2
                self.amplitudes_labels_dict[(m, n)].grid(row=0, rowspan=1, column=0, columnspan=2, padx=pad, pady=pad)
                self.amplitudes_inputs_dict[(m, n)].grid(row=1, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
                self.icons_dict[(m, n)] = plot_figure.Figure(figsize=(self.icon_figure_size,
                                                                      self.icon_figure_size))
                self.canvas_dict[(m, n)] = FigureCanvasTkAgg(self.icons_dict[(m, n)],
                                                             master=self.amplitudes_ctrl_boxes_dict[(m, n)])
                self.plotWidgets_dict[(m, n)] = self.canvas_dict[(m, n)].get_tk_widget()
                self.plotWidgets_dict[(m, n)].grid(row=1, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
                self.amplitudes_ctrl_boxes_dict[(m, n)].update()
                m += 2  # according to the specification of Zernike polynomial
        self.load_icons()  # loading profiles of aberrations

    def load_icons(self):
        """
        Load icons for aberrations if presented in the folder "icons", stored in the project folder.

        Returns
        -------
        None.

        """
        # Construction of a path for each icon corresponding to a polynomial
        for key in self.icons_dict.keys():
            name_file = str(key) + ".jpeg"
            current_path = os.path.dirname(__file__)
            icons_folder_path = os.path.join(current_path, "icons")
            current_icon_path = os.path.join(icons_folder_path, name_file)
            if os.path.isfile(current_icon_path):
                img = skimage.io.imread(current_icon_path)
                axes = self.icons_dict[key].add_axes(rect=[0.0, 0.0, 1.0, 1.0])  # full size figure adding
                axes.imshow(img); axes.axis('off')  # loading the image on the figure.axes
                self.canvas_dict[key].draw()  # refreshing GUI

    def amplitude_input_changed(self, *args):
        """
        Call after some delay in ms function to validate user input.

        Parameters
        ----------
        *args : list.
            Provided by tkinter.

        Returns
        -------
        None.

        """
        self.after(1000, self.validate_input_changed)

    def validate_input_changed(self):
        """
        Validate user input into amplitudes controls.

        Returns
        -------
        None.

        """
        i = 0  # count amplitudes in amplitudes list
        redraw_profile = False  # controlling flag
        for key in self.amplitudes_doubleVars_dict.keys():
            try:
                d_var = self.amplitudes_doubleVars_dict[key].get()
                if d_var < -1.0 or d_var > 1.0:
                    self.amplitudes_doubleVars_dict[key].set(self.amplitudes[i])  # set previous value
                # Update stored amplitudes and call the function for redraw of polynomials sum
                else:
                    if d_var != self.amplitudes[i]:  # only if valid amplitude provided, draw it
                        self.amplitudes[i] = d_var; redraw_profile = True
            except tk.TclError:
                self.amplitudes_doubleVars_dict[key].set(self.amplitudes[i])  # set default value
            i += 1
        if redraw_profile:
            self.plot_zernikes()  # redraw once the profile

    def create_sliders(self):
        """
        Create sliders for controlling polynomials amplitudes.

        Returns
        -------
        None.

        """
        # Construction of sliders
        n_orders = int(self.selected_order.get()[0])
        for order in range(1, n_orders + 1):  # going through all specified orders
            m = -order  # azimuthal order
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                # Slider below - to control an amplitude of polynomial
                self.amplitudes_sliders_dict[(m, n)] = tk.Scale(self.amplitudes_ctrl_boxes_dict[(m, n)],
                                                                from_=-1.0, to=1.0, orient='horizontal',
                                                                resolution=0.01, sliderlength=16,
                                                                tickinterval=0.5, length=self.slider_length,
                                                                command=self.sliderValueChanged,
                                                                repeatinterval=200)
                self.amplitudes_labels_dict[(m, n)].grid(row=0, rowspan=1, column=0, columnspan=1)
                self.amplitudes_sliders_dict[(m, n)].grid(row=1, rowspan=1, column=0, columnspan=1)
                m += 2  # according to the specification of Zernike polynomial

    def no_exit(self):
        """
        Prevent to close the top level window with all the sliders for controlling amplitudes.

        Returns
        -------
        None.

        """
        # self.ampl_ctrls.withdraw()  # delete window from a screen
        pass

    # %% Device ctrl
    def operation_mode_selection(self, operation_mode):
        """
        Handle the UI event of selecting of device.

        Parameters
        ----------
        operation_mode : str
            Selected device type.

        Returns
        -------
        None.

        """
        if operation_mode == "DPP + Simulator":
            # If the user wants to control a device, then try to import all needed libraries (2 of 3, 1 - later)
            if not self.librariesImported:
                try:
                    import serial; global serial  # Serial library (pyserial) for general communication with a device
                    import serial.tools.list_ports as list_ports; global list_ports
                    print("Serial library imported")
                except ImportError:
                    print("Serial library https://pyserial.readthedocs.io/en/latest/index.html is not installed")
                # Below - attempt to import library for voltages calculation (in-house developed)
                try:
                    from getvolt import GetVolt as gv  # import developed in-house library for some calculations
                    global gv  # make the name global for accessibility
                    print("Get volts module imported")
                    self.load_infl_matrix_button.state(['!disabled'])  # activate the influence matrix
                    self.maxV_selector.state(['!disabled'])
                    self.open_serial_communication()  # creates additional controlling window above the main one
                except ImportError:
                    print("The in-house developed controlling library not installed on this computer.\n"
                          "Get it from the maintainers with instructions!")
                    print("The selection of device will go again to the Pure Simulated")
                    self.operation_mode_selector.set(self.operation_modes[0])
                self.librariesImported = True
            else:
                self.load_infl_matrix_button.state(['!disabled'])  # activate the influence matrix
                self.maxV_selector.state(['!disabled'])
                self.open_serial_communication()  # creates additional controlling window above the main one
        else:
            self.load_infl_matrix_button.state(['!disabled', 'disabled'])  # disable it again
            self.maxV_selector.state(['disabled'])
            self.getVoltsButton.state(['disabled'])
            if self.serial_comm_ctrl is not None:
                self.serial_comm_ctrl.destroy()  # close the controlling window in the simulation mode

    def load_influence_matrix(self):
        """
        Load the saved influence (calibration) matrix, handle the according button action.

        Returns
        -------
        None.

        """
        # below - get the path to the influence matrix
        influence_matrix_file_path = tk.filedialog.askopenfilename(filetypes=[("Matlab file", "*.mat"),
                                                                              ("Pickled file", "*.pkl")])
        if influence_matrix_file_path is not None:
            file_opened = False  # preventing mistakes if dialog window is cancelled
            if influence_matrix_file_path[len(influence_matrix_file_path)-3:] == 'mat':
                self.influence_matrix = gv.load_InfMat_matlab(influence_matrix_file_path)
                if isinstance(self.influence_matrix, np.ndarray):
                    file_opened = True
            elif influence_matrix_file_path[len(influence_matrix_file_path)-3:] == 'pkl':
                self.influence_matrix = gv.load_InfMat(influence_matrix_file_path)
                if isinstance(self.influence_matrix, np.ndarray):
                    file_opened = True
            if file_opened:
                rows, cols = self.influence_matrix.shape
                # Influence matrix successfully loaded => activate the possibility to calculate voltages
                print("Influence matrix loaded")
                if (rows > 0) and (cols > 0):
                    self.getVoltsButton.state(['!disabled'])
                else:
                    self.getVoltsButton.state(['disabled'])

    def getVolts(self):
        """
        Calculate the voltages for sending them to the device, using the controlling library.

        Returns
        -------
        None.

        """
        self.zernike_amplitudes = np.zeros(self.influence_matrix.shape[0])  # initial amplitudes of all polynomials = 0
        # Collecting all non-zero selected polynomial amplitudes
        diff_amplitudes_size = 0  # for collecting difference between specified amplitudes and calculation back
        if self.sliders_shown:
            for key in self.amplitudes_sliders_dict.keys():  # loop through all UI ctrls
                if abs(self.amplitudes_sliders_dict[key].get()) > 1.0E-6:  # non-zero amplitude provided by the user
                    (m, n) = key
                    diff_amplitudes_size += 1  # count for non-zero specified amplitudes
                    j = get_osa_standard_index(m, n)  # calculation implemented according to the Wiki
                    self.zernike_amplitudes[j] = self.amplitudes_sliders_dict[key].get()
        else:
            for key in self.amplitudes_doubleVars_dict.keys():
                if abs(self.amplitudes_doubleVars_dict[key].get()) > 1.0E-6:  # non-zero amplitude provided by the user
                    (m, n) = key
                    diff_amplitudes_size += 1  # count for non-zero specified amplitudes
                    j = get_osa_standard_index(m, n)  # calculation implemented according to the Wiki
                    self.zernike_amplitudes[j] = self.amplitudes_doubleVars_dict[key].get()
        # !!! TODO - for now, the direction dictates that the amplitudes should be inverted
        self.zernike_amplitudes = -self.zernike_amplitudes
        # Additional correction of initial deformations on a device
        if self.corrections_loaded:
            self.corrected_zernike_amplitudes = np.zeros(self.zernike_amplitudes.shape[0])
            for j in range((self.zernike_amplitudes.shape[0])):
                # ??? sign in an expression below - check in the legacy code, now because of inversion of sign above!
                self.corrected_zernike_amplitudes[j] = self.zernike_amplitudes[j] - self.flatten_field_coefficients[j][0]
            self.corrected_voltages = gv.solve_InfMat(self.influence_matrix, self.corrected_zernike_amplitudes,
                                                      self.maxV_selector_value.get())
            self.corrected_voltages = np.expand_dims(self.corrected_voltages, axis=1)
        # Calculation of voltages without correction
        else:
            self.voltages = gv.solve_InfMat(self.influence_matrix, self.zernike_amplitudes, self.maxV_selector_value.get())
            self.voltages = np.expand_dims(self.voltages, axis=1)  # explicit making of 2D array by adding additional axis
            # Verification of the proper calculation
            self.check_solution = self.influence_matrix*np.power(self.voltages, 2)
            k = 0  # index of collected amplitudes collected from the UI
            if diff_amplitudes_size > 0:  # only if some non-zero amplitudes specified by a user
                self.diff_amplitudes = np.zeros(diff_amplitudes_size)
                m = 0  # index for collecting calculated differences
                for ampl in self.zernike_amplitudes:
                    if abs(ampl) > 1.0E-6:  # non-zero amplitude provided by the user
                        # print("Difference between amplitude from UI and restored after calculation:",
                        #       np.round(abs(self.check_solution[k, 0] - ampl), 2))
                        self.diff_amplitudes[m] = np.round((self.check_solution[k, 0] - ampl), 2); m += 1
                k += 1
        self.send_voltages_button.state(['!disabled'])  # make possible to send calculated volts to a device
        self.load_zeroed_indices_button.config(state="normal")  # make possible to apply additional conditions

    def maxV_changed(self, *args):
        """
        Call it then the user input some value to the Spinbox field.

        Parameters
        ----------
        *args
            All arguments provided by the add_trace function of tk.IntVar.

        Returns
        -------
        None.

        """
        self.after(1000, self.validateMaxVoltageInput)  # sent request to validate the input value

    def validateMaxVoltageInput(self):
        """
        Validate user input into the Spinbox, that should accept only integer values.

        Returns
        -------
        None.

        """
        try:
            val = self.maxV_selector_value.get()
            if val < self.minV or val > self.maxV:
                self.maxV_selector_value.set(self.minV)  # assign the minimal value if provided is out of range
        except tk.TclError:
            self.maxV_selector_value.set(self.minV)  # assign the minimal value if provided e.g. contain symbols

    def open_serial_communication(self):
        """
        Open the window with serial communication with device controls.

        Returns
        -------
        None.

        """
        # All initialization steps are analogue to the specified for amplitudes controls
        # Add the additional window evoked by the button for communication with the device
        self.serial_comm_ctrl = tk.Toplevel(master=self)  # additional window, master - the main window
        y_shift = self.master.winfo_y() + self.master.winfo_height() + 10*self.pady  # shift of Toplevel window vertically
        x_shift = self.master.winfo_x()  # shift of Toplevel window horizontally
        self.serial_comm_ctrl.geometry(f'+{x_shift}+{y_shift}')
        # Below - rewriting of default destroy event for closing the serial connection - handle closing of COM also
        self.serial_comm_ctrl.protocol("WM_DELETE_WINDOW", self.destroy_serial_ctrl_window)
        # Listing of all available COM ports
        self.ports = []
        for port in list_ports.comports():  # get all ports stored in attributes of the class
            self.ports.append(port.name)
        if len(self.ports) == 0:
            print("No COM ports detected")
            self.destroy_serial_ctrl_window()  # close the additional contolling window for serial connection
            print("The selection of device will go again to the Pure Simulated")
            self.operation_mode_selector.set(self.operation_modes[0])
        else:
            # Creation and placing of controlling buttons because at least 1 COM port available
            pad = 4  # additional border shifts
            self.send_voltages_button = Button(self.serial_comm_ctrl, text="Send voltages", command=self.send_voltages)
            self.clickable_ports = tk.StringVar(); self.clickable_ports.set(self.ports[0])
            self.port_selector = OptionMenu(self.serial_comm_ctrl, self.clickable_ports, self.ports[0],
                                            *self.ports, command=self.port_selected)
            self.connection_status = tk.StringVar(); self.connection_status.set("Not initialized")
            self.connection_label = Label(self.serial_comm_ctrl, textvariable=self.connection_status, foreground='red')
            self.get_device_status_button = Button(self.serial_comm_ctrl, text="Get status", command=self.get_device_status)
            self.zero_amplitudes_button = Button(self.serial_comm_ctrl, text="Zero outputs", command=self.zero_amplitudes)
            self.load_zeroed_indices_button = Button(self.serial_comm_ctrl, text="Load Map",
                                                     command=self.load_zeroed_indices)
            self.load_flat_field_button = Button(self.serial_comm_ctrl, text="Load Flattening",
                                                 command=self.load_flat_field)
            self.visualize_correction_button = Button(self.serial_comm_ctrl, text="Visualize Correction",
                                                      command=self.visualize_correction)
            self.visualize_correction_button.config(state='disabled')

            # Device selector
            self.ctrl_device_names = ["Old (uscope)", "New (63 pins)"]
            self.selected_ctrl_device = tk.StringVar(); self.selected_ctrl_device.set(self.ctrl_device_names[1])
            self.ctrl_device_selector = OptionMenu(self.serial_comm_ctrl, self.selected_ctrl_device,
                                                   self.ctrl_device_names[1], *self.ctrl_device_names,
                                                   command=self.selection_of_device)

            # Placing buttons on the window
            self.port_selector.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.connection_label.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.get_device_status_button.grid(row=0, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.send_voltages_button.grid(row=0, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.zero_amplitudes_button.grid(row=1, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.load_zeroed_indices_button.grid(row=1, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.load_flat_field_button.grid(row=1, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.visualize_correction_button.grid(row=1, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.ctrl_device_selector.grid(row=2, rowspan=1, column=0, columnspan=4, padx=pad, pady=pad)
            self.serial_comm_ctrl.grid()

            # Disabling the buttons before some conditions fulfilled
            self.send_voltages_button.state(['disabled'])  # after opening the window, set to disabled, before voltages
            self.get_device_status_button.config(state="disabled"); self.zero_amplitudes_button.config(state="disabled")
            self.load_zeroed_indices_button.config(state="disabled")
            self.port_selected()  # try to initialize serial connection on COM port

    def port_selected(self, *args):
        """
        Handle of selection of the active COM port AND trying open serial connection on it.

        Parameters
        ----------
        *args : list with arguments provided by tkinter call.
            They describe the function signature call.

        Returns
        -------
        None.

        """
        self.deviceHandle = None
        try:
            self.deviceHandle = serial.Serial(self.clickable_ports.get(), baudrate=self.baudrate,
                                              timeout=0.1, write_timeout=0.08)
            time.sleep(2.2)  # delay needed in any case for establishing serial communication (bad practice)
            #  However, the delay above isn't proven to be equal to this magic number
            self.connection_status.set("Initialized"); self.connection_label.config(foreground='green')
            print("Serial connection initialized with:", self.deviceHandle.port, self.deviceHandle.baudrate)
            self.get_device_status_button.config(state="normal")
            # Trying to import additional dependencies to send device specific commands
            if not self.ampcomImported:
                try:
                    if self.selected_ctrl_device.get() == self.ctrl_device_names[0]:
                        import ampcom_pt; global ampcom_pt  # not uploaded internal library
                    else:
                        import ampcom; global ampcom
                    print("Device controlling library imported")
                    self.ampcomImported = True
                except ModuleNotFoundError:
                    print("Device control communication library are not importable")
            self.zero_amplitudes_button.config(state="normal")
        except serial.SerialException:
            print("Could not connect to the device on the specified COM port")
            self.connection_status.set("Not initialized"); self.connection_label.config(foreground='red')
            self.send_voltages_button.config(state="disabled")
            self.get_device_status_button.config(state="disabled")
            self.zero_amplitudes_button.config(state="disabled")

    def selection_of_device(self, selected_device):
        """
        Old pr new device type for importing approprate controlling library.

        Returns
        -------
        None.

        """
        if selected_device == self.ctrl_device_names[0]:
            try:
                import ampcom_pt; global ampcom_pt  # not uploaded internal library
                print(self.selected_ctrl_device.get(), "- device controlling library imported")
                self.ampcomImported = True
            except ModuleNotFoundError:
                print("Device control communication library are not importable")
                self.ampcomImported = False
        else:
            try:
                import ampcom; global ampcom
                print(self.selected_ctrl_device.get(), "- device controlling library imported")
                self.ampcomImported = True
            except ModuleNotFoundError:
                print("Device control communication library are not importable")
                self.ampcomImported = False
        self.zero_amplitudes_button.config(state="normal")

    def get_device_status(self):
        """
        Print the result after sending hard-coded command "status" and also pending after initialization input messages.

        Returns
        -------
        None.

        """
        if self.deviceHandle.in_waiting > 0:
            print("*****Pending received messages from device:*****")
            print(self.deviceHandle.read(self.deviceHandle.in_waiting).decode("utf-8"))
        # Some hard-coded command upon which the preconfigured device should report the status
        self.deviceHandle.write(b'?'); time.sleep(0.1)  # magic number (bad practice) suspend to receive full response
        self.report = self.deviceHandle.read(self.deviceHandle.in_waiting).decode("utf-8")
        if len(self.report) > 0:
            print("*************Device reports:*************")
            print(self.report)

    def send_voltages(self):
        """
        Handle the clicking of controlling sending voltages button.

        Returns
        -------
        None.

        """
        # Calculate parameters for sending to a device
        if self.ampcomImported:  # flag showing that internal library imported
            VMAX = self.maxV_selector_value.get()
            if self.deviceHandle is not None:  # calculate the proper values to send
                # zero all channels before writing new values - not necessary
                # Calculate and send voltages to a device
                if self.selected_ctrl_device.get() == self.ctrl_device_names[0]:
                    if len(self.parse_indices) > 0:  # checking the loaded ignored pins
                        if self.corrections_loaded:
                            self.voltages = self.corrected_voltages
                        self.converted_voltages = ampcom_pt.AmpCom.create_varr2(self.voltages, VMAX, self.parse_indices)
                        print("Volts Written? :", ampcom_pt.AmpCom.AmpWrite(self.deviceHandle, self.converted_voltages))
                        time.sleep(0.24)  # magic number (bad practice) suspend to receive full response
                        self.get_device_status()  # for debugging
                        print("Device updated?", ampcom_pt.AmpCom.AmpUpdate(self.deviceHandle))
                        self.volts_written = True  # flag tracing that some voltages written into the device
                    else:
                        print("No voltages sent to a device, the ignored indices should be provided, use Load Map")
                else:
                    # Send voltages to a new type of devices
                    if self.corrections_loaded:
                        self.voltages = self.corrected_voltages
                    self.converted_voltages = ampcom.AmpCom.create_varr2(self.voltages, VMAX)
                    print("Volts Written? :", ampcom.AmpCom.AmpWrite(self.deviceHandle, self.converted_voltages))
                    time.sleep(0.24)  # magic number (bad practice) suspend to receive full response
                    self.get_device_status()  # for debugging
                    print("Device updated?", ampcom.AmpCom.AmpUpdate(self.deviceHandle))
                    self.volts_written = True  # flag tracing that some voltages written into the device

    def zero_amplitudes(self):
        """
        Zero all output amplitudes (flatten them).

        Returns
        -------
        None.

        """
        if self.ampcomImported:  # works if the internal library imported
            if self.selected_ctrl_device.get() == self.ctrl_device_names[0]:
                ampcom_pt.AmpCom.AmpZero2(self.deviceHandle); time.sleep(0.2)
                ampcom_pt.AmpCom.AmpUpdate(self.deviceHandle)
            else:
                ampcom.AmpCom.AmpZero(self.deviceHandle); time.sleep(0.2)
                ampcom.AmpCom.AmpUpdate(self.deviceHandle)  # necessary command
            self.get_device_status()  # for debugging
        else:
            print("Device not zeroed, check possibility to import local module")

    def load_zeroed_indices(self):
        """
        Ask user to open text file with stored ignored indices, separated by spacebars.

        Returns
        -------
        None.

        """
        map_file_path = tk.filedialog.askopenfilename(filetypes=[("Text file", "*.txt")])
        if map_file_path is not None:
            if len(map_file_path) > 0:
                file = open(map_file_path, 'r')
                string_indices = file.readline()
                self.parse_indices = string_indices.split(" ")  # the indices supposed to be separated by spacebars
                if len(self.parse_indices) > 0:
                    for i in range(len(self.parse_indices)):
                        self.parse_indices[i] = int(self.parse_indices[i])
                print("Parsed ignored indices: ", self.parse_indices)

    def load_flat_field(self):
        """
        Load corrections for flattening the distribution of aberrations (applied by a device).

        Returns
        -------
        None.

        """
        path_fitted_offses = tk.filedialog.askopenfilename(filetypes=[("Matlab files", "*.mat")],
                                                           title="Open fitted offsets")
        if path_fitted_offses is not None and len(path_fitted_offses) > 0:
            try:
                self.fitted_offsets = loadmat(path_fitted_offses)['Zerns']  # scipy.io.loadmat, specific key for dict
            except KeyError as e:
                print("The pre-coded key " + str(e) + " is not available in opened mat file")
            if isinstance(self.fitted_offsets, np.ndarray):
                print("Fitted offsets for Zernike amplitudes loaded")
                # Proceed for loading next file
                path_offsets = tk.filedialog.askopenfilename(filetypes=[("Matlab files", "*.mat")],
                                                             title="Open Zernikes offsets")
                if path_offsets is not None:
                    try:
                        self.offset_zernikes = loadmat(path_offsets)['Off_Zern_Coeffs']  # scipy.io.loadmat
                    except KeyError as e:
                        print("The pre-coded key " + str(e) + " is not available in opened mat file")
                    if isinstance(self.offset_zernikes, np.ndarray):
                        print("Offsets for Zernike amplitudes loaded")
                        # ??? Below - check the compliance with the legacy code
                        self.flatten_field_coefficients = self.fitted_offsets - self.offset_zernikes
                        self.flatten_field_coefficients = np.round(self.flatten_field_coefficients, 3)
                        self.corrections_loaded = True; self.visualize_correction_button.config(state='normal')
                else:
                    # Disabling now the flattening field coefficients only by not loading any file
                    if self.corrections_loaded:
                        print("Hint: flattening corrections now not accounted, repeat loading")
                    self.corrections_loaded = False; self.flatten_field_coefficients = np.ndarray(shape=1)
        else:
            # Disabling now the flattening field coefficients only by not loading any
            if self.corrections_loaded:
                print("Hint: flattening corrections now not accounted, repeat loading")
            self.corrections_loaded = False; self.flatten_field_coefficients = np.ndarray(shape=1)

    def visualize_correction(self):
        """
        Visualize loaded corrections for Zernike amplitudes for making flatten aberrations field.

        Returns
        -------
        None.

        """
        if self.corrections_loaded:
            for key in self.amplitudes_sliders_dict.keys():  # loop through all UI ctrls
                (m, n) = key; j = get_osa_standard_index(m, n)  # calculation of polynomial index
                if self.flatten_field_coefficients[j][0] > 1E-3:  # depends on precision of amplitude controls
                    self.amplitudes_sliders_dict[key].set(self.flatten_field_coefficients[j][0])

    def always_on_top(self):
        """
        Make always the main controlling window on top of amplitudes controls, if the main window shifted.

        Returns
        -------
        None.

        """
        if self.master_geometry != self.master.winfo_geometry():  # the main window shifted
            self.master.lift(aboveThis=self.ampl_ctrls)  # makes the main window on top of amplitudes ctrls window
        self.after_id = self.after(1000, self.always_on_top)

    def destroy_serial_ctrl_window(self):
        """
        Rewrite the default destroy function for the instance of Toplevel window for closing serial connection.

        Returns
        -------
        None.

        """
        self.serial_comm_ctrl.destroy()  # close the created Toplevel widget
        self.load_infl_matrix_button.config(state="disabled")  # disable possibility to load influence matrix
        if self.deviceHandle is not None:
            if self.ampcomImported:
                try:
                    if self.selected_ctrl_device.get() == self.ctrl_device_names[0]:
                        ampcom_pt.AmpCom.AmpReset(self.deviceHandle)
                    else:
                        ampcom.AmpCom.AmpReset(self.deviceHandle)
                    print("The device should be reset due to the program closing")
                except NameError:
                    print("The device not reset")
                finally:
                    self.deviceHandle.close()  # close the serial connection anyway
                    if not self.deviceHandle.isOpen():
                        print("Serial connection closed due to controlling window closed")
            else:
                self.deviceHandle.close()  # close the serial connection
                if not self.deviceHandle.isOpen():
                    print("Serial connection closed due to controlling window closed")
            self.deviceHandle = None
            self.operation_mode_selector.set(self.operation_modes[0])

    # %% Adjust GUI
    def adjust_gui_sizes(self):
        """
        Fix the issue with font size in the GUI launched by the pure python interpreter.

        Returns
        -------
        None.

        """
        # Specifiyng controls for adjusting font sizes
        self.resizer_ctrl_window = tk.Toplevel(master=self)  # initialize the additional Toplevel window
        self.resizer_ctrl_window.lift(aboveThis=self)  # making a Toplevel window on the top of master one

        # Labels
        self.default_font_size_label = Label(master=self.resizer_ctrl_window, text=" Default font size: ")
        self.entry_font_size_label = Label(master=self.resizer_ctrl_window, text=" Entries / Menu font size: ")
        self.figure_size_label = Label(master=self.resizer_ctrl_window, text="Main figure size (inches):")
        self.icon_figure_size_label = Label(master=self.resizer_ctrl_window, text="Icon figure size (inches):")
        self.slider_length_label = Label(master=self.resizer_ctrl_window, text="Slider length (pixs):")

        # Spinboxes (integer selectors)
        self.default_font_size_int = tk.IntVar(); self.default_font_size_int.set(int(self.default_font.cget("size")))
        self.font_size_ctrl = Spinbox(master=self.resizer_ctrl_window, from_=8, to=16,
                                      increment=1, width=4, textvariable=self.default_font_size_int)
        self.main_figure_size_int = tk.DoubleVar(); self.main_figure_size_int.set(self.main_figure_size)
        self.main_figure_size_ctrl = Spinbox(master=self.resizer_ctrl_window, from_=2, to=8,
                                             increment=0.2, width=5, textvariable=self.main_figure_size_int)
        self.entries_font_size_int = tk.IntVar(); self.entries_font_size_int.set(int(self.default_entry_font.cget("size")))
        self.entries_font_size_ctrl = Spinbox(master=self.resizer_ctrl_window, from_=8, to=16,
                                              increment=1, width=4, textvariable=self.entries_font_size_int)
        self.apply_adjusted_fonts_button = Button(master=self.resizer_ctrl_window, command=self.adjust_fonts_sizes,
                                                  text="Apply Configured Font and Figure Sizes")
        self.icon_figure_size_int = tk.DoubleVar(); self.icon_figure_size_int.set(self.icon_figure_size)
        self.icon_figure_size_ctrl = Spinbox(master=self.resizer_ctrl_window, from_=0.2, to=1.0,
                                             increment=0.01, width=5, textvariable=self.icon_figure_size_int)
        self.slider_length_int = tk.IntVar(); self.slider_length_int.set(self.slider_length)
        self.slider_length_ctrl = Spinbox(master=self.resizer_ctrl_window, from_=150, to=200,
                                          increment=1, width=5, textvariable=self.slider_length_int)

        # Grid layoout of labels and controls on the Toplevel window
        pad = 8
        self.default_font_size_label.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.font_size_ctrl.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
        self.entry_font_size_label.grid(row=1, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.entries_font_size_ctrl.grid(row=1, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
        self.figure_size_label.grid(row=2, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.main_figure_size_ctrl.grid(row=2, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
        self.icon_figure_size_label.grid(row=3, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.icon_figure_size_ctrl.grid(row=3, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
        self.slider_length_label.grid(row=4, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.slider_length_ctrl.grid(row=4, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
        self.apply_adjusted_fonts_button.grid(row=5, rowspan=1, column=0, columnspan=2, padx=pad, pady=pad)
        self.resizer_ctrl_window.grid()

    def adjust_fonts_sizes(self):
        """
        Adjust font, figure and slider sizes.

        Returns
        -------
        None.

        """
        # This function is addressed to the issue of different scaling for launching the script from IDE or from console
        # Checking if new default font size changed (controlling fonts except in entries)
        try:
            font_size = self.default_font_size_int.get()
            if int(self.default_entry_font.cget("size")) != font_size:
                self.default_font.config(size=font_size); self.grid()
        except tk.TclError:
            print("Main font size not adjusted because input is not integer value")
            self.default_font_size_int.set(int(self.default_font.cget("size")))

        # Check if entries font size changed
        try:
            font_size = self.entries_font_size_int.get()
            if int(self.default_entry_font.cget("size")) != font_size:
                self.default_entry_font.config(size=font_size); self.grid()
            if int(self.default_menu_font.cget("size")) != font_size:
                self.default_menu_font.config(size=font_size); self.grid()
            if int(self.tooltip_font.cget("size")) != font_size:
                self.tooltip_font.config(size=font_size); self.grid()
        except tk.TclError:
            print("Entries font size not adjusted because input is not integer value")
            self.entries_font_size_int.set(int(self.default_entry_font.cget("size")))

        # Check if the size of the main figure changed
        try:
            figure_size = self.main_figure_size_int.get()
            if self.main_figure_size != figure_size:
                # Redraw main figure in controlling window with changed size
                self.main_figure_size = figure_size
                self.plotWidget.destroy()  # destroy the associated with figure widget
                self.figure = plot_figure.Figure(figsize=(self.main_figure_size, self.main_figure_size))
                self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.plotWidget = self.canvas.get_tk_widget()
                self.plotWidget.grid(row=1, rowspan=6, column=0, columnspan=5, padx=self.padx, pady=self.pady)
                self.grid(); self.plot_zernikes()
        except tk.TclError:
            print("Some non-double value specified for figure size")
            self.main_figure_size_int.set(self.main_figure_size)

        # Check if the size of the icon figure changed
        try:
            figure_size = self.icon_figure_size_int.get()
            if self.icon_figure_size != figure_size:
                self.icon_figure_size = figure_size
        except tk.TclError:
            print("Some non-double value specified for figure size")
            self.icon_figure_size_int_int.set(self.icon_figure_size)

        # Check if slider length changed
        try:
            slider_length = self.slider_length_int.get()
            if self.slider_length != slider_length:
                self.slider_length = slider_length
                # Redraw sliders (now - call to the function handle # of orders changing)
                if self.sliders_shown:
                    self.number_orders_changed(self.selected_order.get())
        except tk.TclError:
            print("Slider length not updated, input value isn't integer")
            self.slider_length_int.set(self.slider_length)

        # close the adjusting window
        self.resizer_ctrl_window.destroy()

    # %% Close main window
    def destroy(self):
        """
        Handle the closing (destroying) of the main window event.

        Returns
        -------
        None.

        """
        self.after_cancel(self.after_id)  # prevent of attempt to run task in background (access to master geometry)
        time.sleep(0.1)
        if self.deviceHandle is not None:  # close connection to a device
            if self.ampcomImported:
                try:
                    if self.selected_ctrl_device.get() == self.ctrl_device_names[0]:
                        ampcom_pt.AmpCom.AmpReset(self.deviceHandle)
                    else:
                        ampcom.AmpCom.AmpReset(self.deviceHandle)
                    print("The device should be reset due to the program closing")
                except NameError:
                    print("The device not reset")
                finally:
                    self.deviceHandle.close()  # close the serial connection anyway
                    if not self.deviceHandle.isOpen():
                        print("Serial connection closed")
            else:
                self.deviceHandle.close()  # close the serial connection anyway
                if not self.deviceHandle.isOpen():
                    print("Serial connection closed")
        print("The GUI closed")


# %% Launch section
if __name__ == "__main__":
    root = tk.Tk()  # running instance of Tk()
    ui_ctrls = ZernikeCtrlUI(root)  # construction of the main frame
    ui_ctrls.mainloop()
    # Below - get the calculated values during the session for testing (debugging) applied functions
    check_solution = ui_ctrls.check_solution; zernike_amplitudes = ui_ctrls.zernike_amplitudes
    diff_amplitudes = ui_ctrls.diff_amplitudes; voltages_bits = ui_ctrls.voltages_bits
    converted_voltages = ui_ctrls.converted_voltages; inspect_offsets = ui_ctrls.flatten_field_coefficients

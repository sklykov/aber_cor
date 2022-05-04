# -*- coding: utf-8 -*-
"""
GUI for representing and controlling Zernike polynomials.

@author: ssklykov
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
import ctypes  # for fixing of blurred text if lauched by pure Python
import platform


# %% GUI class
class ZernikeCtrlUI(Frame):  # all widgets master class - top level window
    """Class specified the GUI controlling instance."""

    master: tk.Tk
    figure: plt.figure
    plotColorbar: bool; flagFlattened: bool
    amplitudes: list; orders: list; changedSliders: int

    def __init__(self, master):
        # Values initialization
        super().__init__(master)  # initialize the main window (frame) for all widgets
        self.plotColorbar = False; self.master.title("Zernike's polynomials controls and representation")
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
        self.corrected_voltages = np.zeros(shape=1); self.increased_font_size = False

        # Below - matrices placeholders for possible returning some placeholders instead of exception
        self.voltages = np.empty(1); self.check_solution = np.empty(1); self.zernike_amplitudes = np.empty(1)
        self.diff_amplitudes = np.empty(1); self.influence_matrix = np.empty(1)
        # Widgets creation and specification (almost all - buttons)
        self.refreshPlotButton = Button(self, text="Refresh Plot", command=self.plot_zernikes)
        self.zernikesLabel = Label(self, text=" Zernike polynoms ctrls up to:")
        self.figure = plot_figure.Figure(figsize=(4.9, 4.9))  # Default empty figure for phase profile
        self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.plotWidget = self.canvas.get_tk_widget()

        # Below - the way of how associate tkinter buttons with the variables and their states! THEY ARE DECOUPLED!
        self.varPlotColorbarButton = tk.BooleanVar(); self.varPlotColorbarButton.set(False)
        self.plotColorbarButton = Checkbutton(self, text="Colorbar", command=self.colorBarPlotting,
                                              onvalue=True, offvalue=False,
                                              variable=self.varPlotColorbarButton)
        self.loadInflMatrixButton = Button(self, text="Load Infl. Matrix", command=self.load_influence_matrix)
        self.loadInflMatrixButton.state(['!disabled', 'disabled'])  # disable of ttk button
        self.flattenButton = Button(self, text="Flatten all", command=self.flatten_zernike_profile)
        self.getVoltsButton = Button(self, text="Get Volts", command=self.getVolts)
        self.getVoltsButton.state(['disabled'])
        self.increase_font_size_button = Button(self, text="Fix Font Size", command=self.increase_font_size)

        # Below - specification of OptionMenu from ttk for polynomials order selection, fixed thanks to StackOverflow
        self.order_n = ["1st ", "2nd ", "3rd ", "4th ", "5th ", "6th ", "7th "]
        self.order_list = [item + "order" for item in self.order_n]
        self.clickable_list = tk.StringVar(); self.clickable_list.set(self.order_list[0])
        self.max_order_selector = OptionMenu(self, self.clickable_list, self.order_list[0], *self.order_list,
                                             command=self.number_orders_changed)

        # Specification of two case selectors: Simulation / Controlling DPP
        self.listDevices = ["Pure Simulator", "DPP + Simulator"]; self.device_selector = tk.StringVar()
        self.device_selector.set(self.listDevices[0])
        self.deviceSelectorButton = OptionMenu(self, self.device_selector,
                                               self.listDevices[0],
                                               *self.listDevices,
                                               command=self.device_selection)

        # Max voltage control with the named label and Combobox for controlling voltage
        self.holderSelector = Frame(self); textVMaxLabel = Label(self.holderSelector, text=" Max Volts:")
        self.maxV_selector_value = tk.IntVar(); self.maxV_selector_value.set(200)  # initial voltage
        # Below - add the association of updating of integer values of Spinbox input value:
        self.maxV_selector_value.trace_add("write", self.maxV_changed)
        self.maxV_selector = Spinbox(self.holderSelector, from_=self.minV, to=self.maxV,
                                     increment=10, state=tk.DISABLED, width=5,
                                     exportselection=True, textvariable=self.maxV_selector_value)
        textVMaxLabel.pack(side=tk.LEFT); self.maxV_selector.pack(side=tk.LEFT)

        # Below - additional window for holding the sliders with the amplitudes
        self.ampl_ctrls = tk.Toplevel(master=self)  # additional window, master - the main window

        # Placing all created widgets in the grid layout on the main window
        pad = 2  # overall additional border distances for all widgets
        self.zernikesLabel.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.max_order_selector.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
        self.refreshPlotButton.grid(row=0, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
        self.plotColorbarButton.grid(row=0, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
        self.flattenButton.grid(row=0, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
        self.deviceSelectorButton.grid(row=7, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
        self.loadInflMatrixButton.grid(row=7, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
        self.holderSelector.grid(row=7, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
        self.getVoltsButton.grid(row=7, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
        self.plotWidget.grid(row=1, rowspan=6, column=0, columnspan=5, padx=pad, pady=pad)
        self.increase_font_size_button.grid(row=7, rowspan=1, column=4, columnspan=1, padx=pad, pady=pad)
        self.grid(); self.master.update()  # for updating associate with master properties (geometry)

        # Issue with font for buttons and menu entries
        self.default_font = tk.font.nametofont("TkDefaultFont")
        # ??? because now my screen resolution is set to 125%, changing fonts are not practical
        # print("Default font: ", self.default_font.actual())

        # set the value for the OptionMenu and call function for construction of ctrls
        self.clickable_list.set(self.order_list[0]); self.after(0, self.number_orders_changed(self.order_list[0]))
        self.master_geometry = self.master.winfo_geometry()  # saves the main window geometry
        self.after_id = self.after(1000, self.always_on_top)  # launch the function bringing the main window always on top

        # Below - blurred text fixing for launching this script by Python console
        print("Script launched on:", platform.system())
        if platform.system() == "Windows":
            ctypes.windll.shcore.SetProcessDpiAwareness(1)

    def plot_zernikes(self):
        """
        Plot the sum of specified Zernike's polynomials amplitudes.

        Returns
        -------
        None.

        """
        # t1 = time.time()
        # below: update the plot
        self.figure = get_plot_zps_polar(self.figure, orders=self.orders, step_r=0.005, step_theta=0.9,
                                         alpha_coefficients=self.amplitudes, show_amplitudes=self.plotColorbar)
        self.canvas.draw()  # redraw the figure
        # t2 = time.time(); print("redraw time(ms):", int(np.round((t2-t1)*1000, 0)))  # for debugging

    def colorBarPlotting(self):
        """
        Redraw of colormap with the Zernike's polynomials sum on the unit radius aperture.

        Returns
        -------
        None.

        """
        self.plotColorbar = self.varPlotColorbarButton.get(); self.plot_zernikes()

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
        for key in self.amplitudes_sliders_dict.keys():
            self.amplitudes[i] = self.amplitudes_sliders_dict[key].get(); i += 1
        if self.changedSliders > 1:  # more than one slider changed => flatten operation
            self.changedSliders -= 1
        if not self.flagFlattened:  # if no flatten flag, redraw the plot
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
        for key in self.amplitudes_sliders_dict.keys():
            if abs(self.amplitudes_sliders_dict[key].get()) > 0.0001:  # if not actually equal to zero
                self.amplitudes_sliders_dict[key].set(0.0)
                self.changedSliders += 1  # counting number of zeroed sliders

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
        y_shift = self.master.winfo_geometry().split("+")[2]  # shift of Toplevel window horizontally
        x_shift = self.master.winfo_x() + self.master.winfo_width() + 2  # shift of Toplevel window vertically
        self.ampl_ctrls.destroy(); self.ampl_ctrls = tk.Toplevel(master=self)
        self.ampl_ctrls.wm_transient(self); self.ampl_ctrls.protocol("WM_DELETE_WINDOW", self.no_exit)
        self.ampl_ctrls.title("Amplitude controls")
        self.ampl_ctrls.geometry(f'+{x_shift}+{y_shift}')  # shifting relative to the main window size and positions
        self.master.lift(aboveThis=self.ampl_ctrls)  # makes the main window above the created amplitudes controls
        # Get the (m, n) values from the order specification
        self.orders = []; self.amplitudes_sliders_dict = {}; self.amplitudes = []  # refresh the associated controls
        self.amplitudes_ctrl_boxes_dict = {}; self.amplitudes_labels_dict = {}
        # Construction of sliders
        for order in range(1, n_orders + 1):  # going through all specified orders
            m = -order  # azimuthal order
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                self.orders.append((m, n))  # store the values as tuples
                # Below - initialization of sliders for controlling amplitudes of Zernike's polynomials
                classical_name = get_classical_polynomial_name((m, n), short_names=True)  # Like Vertical tilt
                # label=(f"Z{(m ,n)} " + classical_name)
                self.amplitudes_ctrl_boxes_dict[(m, n)] = Frame(self.ampl_ctrls)  # Frame to hold sliders, labels, etc.
                self.amplitudes_labels_dict[(m, n)] = Label(self.amplitudes_ctrl_boxes_dict[(m, n)],
                                                            text=(f"{(m ,n)} " + classical_name),
                                                            font=(self.default_font.actual()['family'],
                                                                  self.default_font.actual()['size'],
                                                                  'bold'))
                # Slider below - to control an amplitude of polynomial
                self.amplitudes_sliders_dict[(m, n)] = tk.Scale(self.amplitudes_ctrl_boxes_dict[(m, n)],
                                                                from_=-1.0, to=1.0, orient='horizontal',
                                                                resolution=0.02, sliderlength=16,
                                                                tickinterval=0.5, length=156,
                                                                command=self.sliderValueChanged,
                                                                repeatinterval=220)
                self.amplitudes_labels_dict[(m, n)].grid(row=0, rowspan=1, column=0, columnspan=1)
                self.amplitudes_sliders_dict[(m, n)].grid(row=1, rowspan=1, column=0, columnspan=1)
                # simplest way of packing - adding buttons on top of each other
                self.amplitudes.append(0.0)  # assign all zeros as the flat field
                m += 2  # according to the specification
        self.amplitudes_sliders_dict[(1, 1)].update()  # for updating internal geometry properties
        self.amplitudes_labels_dict[(1, 1)].update()   # for updating internal geometry properties
        # Defying the sizes for placement of widgets
        width = int(self.amplitudes_sliders_dict[(1, 1)].winfo_geometry().split("x")[0])
        height = (int(self.amplitudes_sliders_dict[(1, 1)].winfo_geometry().split("x")[1].split("+")[0])
                  + int(self.amplitudes_labels_dict[(1, 1)].winfo_geometry().split("x")[1].split("+")[0]))
        # Placing the sliders on the window
        y_coordinate = pad  # initial horizontal shift
        if n_orders <= 6:
            for order in range(1, n_orders + 1):
                m = -order  # azimuthal order
                n = order  # radial order
                x_coordinate = pad + ((n_orders-order)*width)//2  # making pyramid on horizontal placing
                for polynomial in range(order + 1):  # number of polynomials = order + 1
                    # self.amplitudes_ctrl_boxes_dict[(m, n)].grid(row=(order-1), rowspan=1, column=row_cursor,
                    #                                              columnspan=1, padx=pad, pady=pad)
                    self.amplitudes_ctrl_boxes_dict[(m, n)].place(x=x_coordinate, y=y_coordinate)
                    m += 2
                    x_coordinate += width + pad  # adding pad and width of a widget to place next one
                y_coordinate += height + pad  # adding pad and width of a widget to place next row with
        # 7th order layout differently
        # FIXME - make better layout
        else:
            for order in range(1, 7):
                m = -order  # azimuthal order
                n = order  # radial order
                x_coordinate = pad + ((n_orders-1-order)*width)//2  # making pyramid on horizontal placing
                for polynomial in range(order + 1):  # number of polynomials = order + 1
                    self.amplitudes_ctrl_boxes_dict[(m, n)].place(x=x_coordinate, y=y_coordinate)
                    m += 2
                    x_coordinate += width + pad  # adding pad and width of a widget to place next one
                    if order == 6:
                        print(x_coordinate, y_coordinate)
                y_coordinate += height + pad  # adding pad and width of a widget to place next row with widgets
            y_coordinate += pad
            m = -7; n = 7; x_coordinate2 = pad
            for polynomial in range(3):  # number of polynomials = order + 1
                self.amplitudes_ctrl_boxes_dict[(m, n)].place(x=x_coordinate2, y=y_coordinate)
                print(x_coordinate2, y_coordinate)
                m += 2
                x_coordinate2 += 2*width + pad  # adding pad and width of a widget to place next one
            y_coordinate += height + pad
        self.ampl_ctrls.geometry(f'{x_coordinate}x{y_coordinate}'); self.ampl_ctrls.update()
        self.plot_zernikes()  # refresh the plot, not retain any values

    def device_selection(self, new_device):
        """
        Handle the UI event of selecting of device.

        Parameters
        ----------
        new_device : str
            Selected device type.

        Returns
        -------
        None.

        """
        if new_device == "DPP + Simulator":
            # If the user wants to control a device, then try to import all needed libraries (2 of 3, 1 - later)
            if not self.librariesImported:
                try:
                    import serial; global serial  # Serial library (pyserial) for general communication with a device
                    import serial.tools.list_ports as list_ports; global list_ports
                    print("Serial library imported")
                except ValueError:
                    print("Serial library https://pyserial.readthedocs.io/en/latest/index.html is not installed")
                # Below - attempt to import library for voltages calculation (in-house developed)
                try:
                    import getvolt as gv  # import developed in-house library available for the other parts of program
                    global gv  # make the name global for accessibility
                    print("Get volts module imported")
                    self.loadInflMatrixButton.state(['!disabled'])  # activate the influence matrix
                    self.maxV_selector.state(['!disabled'])
                    self.open_serial_cimmunication()  # creates additional controlling window above the main one
                except ImportError:
                    print("The in-house developed controlling library not installed on this computer.\n"
                          "Get it for maintainers with instructions!")
                    print("The selection of device will go again to the Pure Simulated")
                    self.device_selector.set(self.listDevices[0])
                self.librariesImported = True
            else:
                self.loadInflMatrixButton.state(['!disabled'])  # activate the influence matrix
                self.maxV_selector.state(['!disabled'])
                self.open_serial_cimmunication()  # creates additional controlling window above the main one
        else:
            self.loadInflMatrixButton.state(['!disabled', 'disabled'])  # disable it again
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
        # According to the documentation, piston is included
        diff_amplitudes_size = 0  # for collecting difference between specified amplitudes and calculation back
        for key in self.amplitudes_sliders_dict.keys():  # loop through all UI ctrls
            if abs(self.amplitudes_sliders_dict[key].get()) > 1.0E-6:  # non-zero amplitude provided by the user
                (m, n) = key
                diff_amplitudes_size += 1  # count for non-zero specified amplitudes
                j = get_osa_standard_index(m, n)  # calculation implemented according to the Wiki
                self.zernike_amplitudes[j] = self.amplitudes_sliders_dict[key].get()
        # Additional correction of initial deformations (aberrations)
        if self.corrections_loaded:
            self.corrected_zernike_amplitudes = np.zeros(self.zernike_amplitudes.shape[0])
            for j in range((self.zernike_amplitudes.shape[0])):
                # ??? sign - check in the legacy code
                self.corrected_zernike_amplitudes[j] = self.zernike_amplitudes[j] + self.flatten_field_coefficients[j][0]
            self.corrected_voltages = gv.solve_InfMat(self.influence_matrix, self.zernike_amplitudes,
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
        self.load_zeroed_indicies_button.config(state="normal")  # make possible to apply additional conditions

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
        except Exception:
            self.maxV_selector_value.set(self.minV)  # assign the minimal value if provided e.g. contain symbols

    def no_exit(self):
        """
        Prevent to close the top level window with all the sliders for controlling amplitudes.

        Returns
        -------
        None.

        """
        # self.ampl_ctrls.withdraw()  # delete window from a screen
        pass

    def destroy_serial_ctrl_window(self):
        """
        Rewrite the default destroy function for the instance of Toplevel window for closing serial connection.

        Returns
        -------
        None.

        """
        self.serial_comm_ctrl.destroy()  # close the created Toplevel widget
        self.loadInflMatrixButton.config(state="disabled")  # disable possibility to load influence matrix
        if self.deviceHandle is not None:
            if self.ampcomImported:
                try:
                    ampcom_pt.AmpCom.AmpZero2(self.deviceHandle); time.sleep(0.25)
                    print("The device should be zeroed due to ctrl window closing")
                except NameError:
                    print("The device not zeroed")
                finally:
                    self.deviceHandle.close()  # close the serial connection anyway
                    if not self.deviceHandle.isOpen():
                        print("Serial connection closed due to controlling window closed")
            else:
                self.deviceHandle.close()  # close the serial connection
                if not self.deviceHandle.isOpen():
                    print("Serial connection closed due to controlling window closed")
            self.deviceHandle = None
            self.device_selector.set(self.listDevices[0])

    def open_serial_cimmunication(self):
        """
        Open the window with serial communication with device controls.

        Returns
        -------
        None.

        """
        # All initialization steps are analogue to the specified for amplitudes controls
        # Add the additional window evoked by the button for communication with the device
        self.serial_comm_ctrl = tk.Toplevel(master=self)  # additional window, master - the main window
        self.serial_comm_ctrl_offsets = "+2+765"; self.serial_comm_ctrl.wm_transient(self)
        self.serial_comm_ctrl.geometry(self.serial_comm_ctrl_offsets)
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
            self.device_selector.set(self.listDevices[0])
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
            self.load_zeroed_indicies_button = Button(self.serial_comm_ctrl, text="Load Map", command=self.load_zeroed_indices)
            self.load_flat_field_button = Button(self.serial_comm_ctrl, text="Load Flattening", command=self.load_flat_field)
            self.visualize_correction_button = Button(self.serial_comm_ctrl, text="Visualize Correction",
                                                      command=self.visualize_correction)
            self.visualize_correction_button.config(state='disabled')

            # Placing buttons on the window
            self.port_selector.grid(row=0, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.connection_label.grid(row=0, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.get_device_status_button.grid(row=0, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.send_voltages_button.grid(row=0, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.zero_amplitudes_button.grid(row=1, rowspan=1, column=2, columnspan=1, padx=pad, pady=pad)
            self.load_zeroed_indicies_button.grid(row=1, rowspan=1, column=3, columnspan=1, padx=pad, pady=pad)
            self.load_flat_field_button.grid(row=1, rowspan=1, column=0, columnspan=1, padx=pad, pady=pad)
            self.visualize_correction_button.grid(row=1, rowspan=1, column=1, columnspan=1, padx=pad, pady=pad)
            self.serial_comm_ctrl.grid()

            # Disabling the buttons before some conditions fulfilled
            self.send_voltages_button.state(['disabled'])  # after opening the window, set to disabled, before voltages
            self.get_device_status_button.config(state="disabled"); self.zero_amplitudes_button.config(state="disabled")
            self.load_zeroed_indicies_button.config(state="disabled")
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
        # print("Selected COM port:", self.clickable_ports.get())
        self.deviceHandle = None
        try:
            self.deviceHandle = serial.Serial(self.clickable_ports.get(), baudrate=self.baudrate,
                                              timeout=0.1, write_timeout=0.08)
            time.sleep(2.2)  # delay needed in any case for establishing serial communication (bad)
            #  However, the delay above isn't proven to be equal to this magic number
            self.connection_status.set("Initialized"); self.connection_label.config(foreground='green')
            print("Serial connection initialized with:", self.deviceHandle.port, self.deviceHandle.baudrate)
            self.get_device_status_button.config(state="normal")
            # Trying to import additional dependencies to send device specific commands
            if not self.ampcomImported:
                try:
                    import ampcom_pt; global ampcom_pt  # not uploaded internal libra
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
                if self.volts_written:
                    ampcom_pt.AmpCom.AmpZero2(self.deviceHandle); time.sleep(0.2)  # bad practice
                    self.get_device_status()  # for debugging
                if len(self.parse_indices) > 0:  # checking the loaded ignored pins
                    if not self.corrections_loaded:  # use only primarly calculated voltages, without corrections
                        self.converted_voltages = ampcom_pt.AmpCom.create_varr2(self.voltages, VMAX, self.parse_indices)
                        print("Volts Written? :", ampcom_pt.AmpCom.AmpWrite(self.deviceHandle, self.converted_voltages))
                        time.sleep(0.25)  # magic number (bad practice) suspend to receive full response
                        self.get_device_status()  # for debugging
                        print("Device updated?", ampcom_pt.AmpCom.AmpUpdate(self.deviceHandle))
                        self.volts_written = True  # flag tracing that some voltages written into the device
                else:
                    print("No voltages sent to a device, the ignored indices should be provided, use Load Map")

    def zero_amplitudes(self):
        """
        Zero all output amplitudes (flatten them).

        Returns
        -------
        None.

        """
        if self.ampcomImported:  # works if the internal library imported
            ampcom_pt.AmpCom.AmpZero2(self.deviceHandle); time.sleep(0.2)
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
        Load corrections for flattening the aberrations distribution (applied by a device).

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
                print("The precoded key " + str(e) + " is not available in opened mat file")
            if isinstance(self.fitted_offsets, np.ndarray):
                print("Fitted offsets for Zernike amplitudes loaded")
                # Proceed for loading next file
                path_offsets = tk.filedialog.askopenfilename(filetypes=[("Matlab files", "*.mat")],
                                                             title="Open Zernikes offsets")
                if path_offsets is not None and len(path_offsets) > 0:
                    try:
                        self.offset_zernikes = loadmat(path_offsets)['Off_Zern_Coeffs']  # scipy.io.loadmat
                    except KeyError as e:
                        print("The precoded key " + str(e) + " is not available in opened mat file")
                    if isinstance(self.offset_zernikes, np.ndarray):
                        print("Offsets for Zernike amplitudes loaded")
                        # ??? Below - check the compliance with the legacy code
                        self.flatten_field_coefficients = self.fitted_offsets - self.offset_zernikes
                        self.flatten_field_coefficients = np.round(self.flatten_field_coefficients, 3)
                        self.corrections_loaded = True; self.visualize_correction_button.config(state='normal')
                else:
                    # Disabling now the flattening field coefficients only by not loading any file
                    self.corrections_loaded = False; self.flatten_field_coefficients = np.ndarray(size=1)
        else:
            # Disabling now the flattening field coefficients only by not loading any file
            self.corrections_loaded = False; self.flatten_field_coefficients = np.ndarray(size=1)

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
                if self.flatten_field_coefficients[j][0] > 1E-3:  # depends of precision of amplitude controls
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

    def increase_font_size(self):
        """
        Fix the issue with font size in the GUI launched by the pure python interepreter.

        Returns
        -------
        None.

        """
        if not self.increased_font_size:
            self.default_font.config(size=11)  # FIXME: actually, not enough for proper scaling of all widgets
            self.increased_font_size = True
        else:
            self.default_font.config(size=9)
            self.increased_font_size = False

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
                    ampcom_pt.AmpCom.AmpZero(self.deviceHandle)
                    print("The device should be zeroed due to the program closing")
                except NameError:
                    print("The device not zeroed")
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

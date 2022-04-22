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
# import time
import numpy as np


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
        self.plotColorbar = False
        self.master.title("Zernike's polynomials controls and representation")
        master.geometry("+5+50")  # put the main window on the (+x, +y) coordinate away from the top left monitor coordinate
        self.amplitudes = [0.0, 0.0]  # default amplitudes for the 1st order
        self.orders = [(-1, 1), (1, 1)]; self.flagFlattened = False; self.changedSliders = 1
        self.amplitudes_sliders_dict = {}
        self.minV = 200; self.maxV = 400
        self.deviceHandle = None  # holder for the opened serial communication handle
        self.serial_comm_ctrl = None  # empty holder for serial communication ctrl
        self.voltages_bits = None  # empty holder because of asking in the end of the script to return this value
        self.librariesImported = False  # libraries for import - PySeriral and local device controlling library
        # Below - matrices placeholders for possible returning some placeholders instead of exception
        self.voltages = np.empty(1); self.check_solution = np.empty(1); self.zernike_amplitudes = np.empty(1)
        self.diff_amplitudes = np.empty(1); self.influence_matrix = np.empty(1)
        # Widgets creation and specification (almost all - buttons)
        self.refreshPlotButton = Button(self, text="Refresh Plot", command=self.plot_zernikes)
        self.zernikesLabel = Label(self, text=" Zernike polynoms ctrls up to:")
        self.figure = plot_figure.Figure(figsize=(5, 5))  # Default empty figure for phase profile
        self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.plotWidget = self.canvas.get_tk_widget()
        # !!! Below - the way of how associate tkinter buttons with the variables and their states! THEY ARE DECOUPLED!
        self.varPlotColorbarButton = tk.BooleanVar(); self.varPlotColorbarButton.set(False)
        self.plotColorbarButton = Checkbutton(self, text="Colorbar", command=self.colorBarPlotting,
                                              onvalue=True, offvalue=False,
                                              variable=self.varPlotColorbarButton)
        self.loadInflMatrixButton = Button(self, text="Load Infl. Matrix", command=self.load_influence_matrix)
        self.loadInflMatrixButton.state(['!disabled', 'disabled'])  # disable of ttk button
        self.flattenButton = Button(self, text="Flatten all", command=self.flattenAll)
        self.getVoltsButton = Button(self, text="Get Volts", command=self.getVolts)
        self.getVoltsButton.state(['disabled'])
        # Below - specification of OptionMenu from ttk for polynomials order selection, fixed thanks to StackOverflow
        self.order_n = ["1st ", "2nd ", "3rd ", "4th ", "5th ", "6th ", "7th "]
        self.order_list = [item + "order" for item in self.order_n]
        self.clickable_list = tk.StringVar(); self.clickable_list.set(self.order_list[0])
        self.max_order_selector = OptionMenu(self, self.clickable_list, self.order_list[0], *self.order_list,
                                             command=self.numberOrdersChanged)
        # Specification of two case selectors: Simulation / Controlling DPP
        self.listDevices = ["Pure Simulator", "DPP + Simulator"]; self.device_selector = tk.StringVar()
        self.device_selector.set(self.listDevices[0])
        self.deviceSelectorButton = OptionMenu(self, self.device_selector,
                                               self.listDevices[0],
                                               *self.listDevices,
                                               command=self.deviceSelected)
        # Max voltage control with the named label and Combobox for controlling voltage
        self.holderSelector = Frame(self); textVMaxLabel = Label(self.holderSelector, text=" Max Volts:")
        self.maxV_selector_value = tk.IntVar(); self.maxV_selector_value.set(200)  # initial voltage
        # !!! Below - add the association of updating of integer values of Spinbox input value:
        self.maxV_selector_value.trace_add("write", self.maxV_changed)
        self.maxV_selector = Spinbox(self.holderSelector, from_=self.minV, to=self.maxV,
                                     increment=10, state=tk.DISABLED, width=5,
                                     exportselection=True, textvariable=self.maxV_selector_value)
        textVMaxLabel.pack(side=tk.LEFT); self.maxV_selector.pack(side=tk.LEFT)
        # Below - additional window for holding the sliders with the amplitudes
        self.ampl_ctrls = tk.Toplevel(master=self)  # additional window, master - the main window
        # put this additional window with some offset for the representing it next to the main
        self.ampl_ctrls_offsets = "+658+50"; self.ampl_ctrls.geometry(self.ampl_ctrls_offsets)
        # Seems, that command below makes accessible the button values from this window to the main
        self.ampl_ctrls.wm_transient(self); self.ampl_ctrls.title("Amplitude controls")
        self.ampl_ctrls.protocol("WM_DELETE_WINDOW", self.no_exit)  # !!! Associate clicking on quit button (X) with the function
        # Placing all created widgets in the grid layout on the main window
        self.zernikesLabel.grid(row=0, rowspan=1, column=0, columnspan=1)
        self.max_order_selector.grid(row=0, rowspan=1, column=1, columnspan=1)
        self.refreshPlotButton.grid(row=0, rowspan=1, column=2, columnspan=1)
        self.plotColorbarButton.grid(row=0, rowspan=1, column=3, columnspan=1)
        self.flattenButton.grid(row=0, rowspan=1, column=4, columnspan=1)
        self.deviceSelectorButton.grid(row=7, rowspan=1, column=0, columnspan=1)
        self.loadInflMatrixButton.grid(row=7, rowspan=1, column=1, columnspan=1)
        self.holderSelector.grid(row=7, rowspan=1, column=2, columnspan=1, padx=1)
        self.getVoltsButton.grid(row=7, rowspan=1, column=3, columnspan=1)
        self.plotWidget.grid(row=1, rowspan=6, column=0, columnspan=5)
        self.grid()
        # self.grid_propagate(False)  # Preventing of shrinking of windows to conform with all placed widgets
        # set the value for the OptionMenu and call function for construction of ctrls
        self.clickable_list.set(self.order_list[3]); self.after(0, self.numberOrdersChanged(self.order_list[3]))

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
            # print(self.amplitudes_sliders_dict[key].get(), end=" ")  # FOR DEBUG
            self.amplitudes[i] = self.amplitudes_sliders_dict[key].get(); i += 1
        # print("Recorded amplitudes:", self.amplitudes)  # FOR
        if self.changedSliders > 1:  # more than one slider changed => flatten operation
            self.changedSliders -= 1
        if not self.flagFlattened:  # if no flatten flag, redraw the plot
            self.plot_zernikes()
        else:
            if self.changedSliders == 1:  # if all sliders finally zeroed, redraw the plot
                self.flagFlattened = False; self.plot_zernikes()

    def flattenAll(self):
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

    def numberOrdersChanged(self, selected_order: str):
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
        # print(selected_order)  # FOR DEBUG
        n_orders = int(selected_order[0])
        # Refresh the TopLevel window and the associated dictionary with buttons
        self.ampl_ctrls.destroy(); self.ampl_ctrls = tk.Toplevel(master=self)
        self.ampl_ctrls.wm_transient(self); self.ampl_ctrls.protocol("WM_DELETE_WINDOW", self.no_exit)
        self.ampl_ctrls.title("Amplitude controls"); self.ampl_ctrls.geometry(self.ampl_ctrls_offsets)
        # Get the (m, n) values from the order specification
        self.orders = []; self.amplitudes_sliders_dict = {}; self.amplitudes = []  # refresh the associated controls
        for order in range(1, n_orders + 1):  # going through all specified orders
            m = -order  # azimuthal order
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                self.orders.append((m, n))  # store the values as tuples
                # Below - initialization of sliders for controlling amplitudes of Zernike's polynomials
                classical_name = get_classical_polynomial_name((m, n), short_names=True)
                self.amplitudes_sliders_dict[(m, n)] = tk.Scale(self.ampl_ctrls, from_=-1.0, to=1.0, orient='horizontal',
                                                                resolution=0.05, sliderlength=20,
                                                                label=(f"Z{(m ,n)} " + classical_name),
                                                                tickinterval=0.5, length=152,
                                                                command=self.sliderValueChanged,
                                                                repeatinterval=220)
                # simplest way of packing - adding buttons on top of each other
                # self.amplitudes_sliders_dict[(m, n)].pack(side=tk.TOP)
                self.amplitudes.append(0.0)  # assign all zeros as the flat field
                m += 2  # according to the specification
        # Placing the sliders on the window
        for order in range(1, n_orders + 1):
            # rows_offset = n_orders - order
            rows_offset = 0
            m = -order  # azimuthal order
            row_cursor = 0
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                # self.amplitudes_sliders_dict[(m, n)].grid(row=(rows_offset + row_cursor), rowspan=1,
                #                                           column=(order-1), columnspan=1)
                self.amplitudes_sliders_dict[(m, n)].grid(row=(order-1), rowspan=1,
                                                          column=(rows_offset + row_cursor), columnspan=1)
                m += 2; row_cursor += 1
        self.plot_zernikes()  # refresh the plot, not retain any values
        # print(self.orders)

    def deviceSelected(self, new_device):
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
            try:
                import getvolt as gv  # import developed in-house library available for the other parts of program
                global gv  # make the name global for accessibility
                self.loadInflMatrixButton.state(['!disabled'])  # activate the influence matrix
                self.maxV_selector.state(['!disabled'])
                self.openSerialCommunication()  # creates additional controlling window above the main one
            except ImportError:
                print("The in-house developed controlling library not installed on this computer.\n"
                      "Get it for maintainers with instructions!")
                print("The selection of device will go again to the Pure Simulated")
                self.device_selector.set(self.listDevices[0])
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
        # print(influence_matrix_file_path[len(influence_matrix_file_path)-3:])
        if influence_matrix_file_path[len(influence_matrix_file_path)-3:] == 'mat':
            self.influence_matrix = gv.load_InfMat_matlab(influence_matrix_file_path)
        elif influence_matrix_file_path[len(influence_matrix_file_path)-3:] == 'pkl':
            self.influence_matrix = gv.load_InfMat(influence_matrix_file_path)
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
        # According to the documentation, piston is included, OSA index -= 1 from the definition
        diff_amplitudes_size = 0  # for collecting difference between specified amplitudes and back calculated after volts calc.
        for key in self.amplitudes_sliders_dict.keys():  # loop through all UI ctrls
            if abs(self.amplitudes_sliders_dict[key].get()) > 1.0E-6:  # non-zero amplitude provided by the user
                (m, n) = key
                # print("Defined orders:", (m, n))
                diff_amplitudes_size += 1  # count for non-zero specified amplitudes
                j = get_osa_standard_index(m, n)  # according to Wiki
                # j -= 1  # according to the specification, but seems a mistake in docs ???
                # print("Defined index: ", (j+1))
                self.zernike_amplitudes[j] = self.amplitudes_sliders_dict[key].get()
        self.voltages = gv.solve_InfMat(self.influence_matrix, self.zernike_amplitudes, self.maxV_selector_value.get())
        self.voltages = np.expand_dims(self.voltages, axis=1)  # explicit making of 2D array by adding additional axis
        # print("Volt sizes: ", np.shape(self.voltages))
        # print("Influence matrix: ", np.shape(self.influence_matrix))
        # Verification of the proper calculation
        # print("Solution :", (self.influence_matrix*np.power(self.voltages, 2)))
        self.check_solution = self.influence_matrix*np.power(self.voltages, 2)
        k = 0  # index of collected amplitudes collected from the UI
        if diff_amplitudes_size > 0:  # only if some non-zero amplitudes specified by a user
            self.diff_amplitudes = np.zeros(diff_amplitudes_size)
            m = 0  # index for collecting calculated differences
        for ampl in self.zernike_amplitudes:
            if abs(ampl) > 1.0E-6:  # non-zero amplitude provided by the user
                # print("Difference between amplitude from UI and restored after calculation:",
                #       np.round(abs(self.check_solution[k, 0] - ampl), 2))
                if diff_amplitudes_size > 0:
                    self.diff_amplitudes[m] = np.round((self.check_solution[k, 0] - ampl), 2); m += 1
            k += 1
        self.openSerialCommButton.state(['!disabled'])
        # TODO - send the voltages using AmpCom to the device

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
        pass

    def destroySerialCtrlWindow(self):
        """
        Rewrite the default destroy function for the instance of Toplevel window for closing serial connection.

        Returns
        -------
        None.

        """
        self.serial_comm_ctrl.destroy()
        if self.deviceHandle is not None:
            try:
                ampcom.AmpCom.AmpZero(self.deviceHandle)
            except Exception:
                print("The device not zeroed")
            finally:
                self.deviceHandle.close()  # close the serial connection anyway
                self.deviceHandle = None
        print("Serial connection closed due to controlling window closing")

    def openSerialCommunication(self):
        """
        Open the window with serial communication with device controls.

        Returns
        -------
        None.

        """
        # All initialization steps are analogue to the specified for amplitudes controls
        self.serial_comm_ctrl = tk.Toplevel(master=self)  # additional window, master - the main window
        self.serial_comm_ctrl_offsets = "+5+775"; self.serial_comm_ctrl.wm_transient(self)
        self.serial_comm_ctrl.geometry(self.serial_comm_ctrl_offsets)
        # !!! Below - rewriting of default destroy event for closing the serial connection
        self.serial_comm_ctrl.protocol("WM_DELETE_WINDOW", self.destroySerialCtrlWindow)
        # Add the additional window evoked by the button for communication with the device
        self.openSerialCommButton = Button(self.serial_comm_ctrl, text="Send voltages", command=self.send_voltages)
        self.serialCommLabel = Label(self.serial_comm_ctrl, text="Experimental, sends the calculated voltages: ")
        self.serialCommLabel.pack(side=tk.LEFT); self.openSerialCommButton.pack(side=tk.LEFT)
        self.openSerialCommButton.state(['disabled'])  # after opening the window, set to disabled, before voltages

    def send_voltages(self):
        """
        Handle the clicking of controlling sending voltages button.

        Returns
        -------
        None.

        """
        # Trying to import additional dependencies
        if not self.librariesImported:
            try:
                import serial; global serial
                print("Serial library imported")
                import ampcom; global ampcom
                print("Device controlling library imported")
                self.librariesImported = True
            except ValueError:
                print("Serial library https://pyserial.readthedocs.io/en/latest/index.html is not installed")
            except ModuleNotFoundError:
                print("Device control communication library are not importable")
                self.serial_comm_ctrl.destroy()  # close the controlling window
        if self.librariesImported:
            VMAX = self.maxV_selector_value.get(); print("Used maximum voltage:", VMAX)
            self.voltages_bits = ampcom.AmpCom.create_varr2(self.voltages, VMAX)
            if self.deviceHandle is None:  # open serial communication and send voltages
                try:
                    self.deviceHandle = ampcom.AmpCom.AmpConnect()  # connect to the device
                    ampcom.AmpCom.AmpStatus(self.deviceHandle)  # should print the device status
                    converted_voltages = ampcom.AmpCom.create_varr2(self.voltages, VMAX)  # made out of found voltages bits
                    ampcom.AmpCom.AmpWrite(self.deviceHandle, converted_voltages)  # send converted V to bits to a device
                    ampcom.AmpCom.AmpStatus(self.deviceHandle)  # should print the device status
                except Exception:
                    print("Connection hasn't been established, check connection settings")
            else:  # the connection was established already
                ampcom.AmpCom.AmpStatus(self.deviceHandle)  # should print the device status - ??? (reimplement it)
                converted_voltages = ampcom.AmpCom.create_varr2(self.voltages, VMAX)  # made out of found voltages bits
                ampcom.AmpCom.AmpWrite(self.deviceHandle, converted_voltages)  # send converted V to bits to a device
                ampcom.AmpCom.AmpUpdate(self.deviceHandle)  # always update after write
                ampcom.AmpCom.AmpStatus(self.deviceHandle)  # should print the device status

    def destroy(self):
        """
        Handle the closing (destroying) of the main window event.

        Returns
        -------
        None.

        """
        if self.deviceHandle is not None:
            try:
                ampcom.AmpCom.AmpZero(self.deviceHandle)
            except Exception:
                print("The device not zeroed")
            finally:
                self.deviceHandle.close()  # close the serial connection anyway
        print("The GUI closed")


# %% Launch section
if __name__ == "__main__":
    root = tk.Tk()  # running instance of Tk()
    ui_ctrls = ZernikeCtrlUI(root)  # construction of the main frame
    ui_ctrls.mainloop()
    # Below - get the calculated values during the session for testing applied functions
    check_solution = ui_ctrls.check_solution; zernike_amplitudes = ui_ctrls.zernike_amplitudes
    diff_amplitudes = ui_ctrls.diff_amplitudes; voltages_bits = ui_ctrls.voltages_bits

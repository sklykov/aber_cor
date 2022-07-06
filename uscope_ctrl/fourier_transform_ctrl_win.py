# -*- coding: utf-8 -*-
"""
Rewrite methods of QMainWindow from PyQT5 for further using it for FFT calculation and re-usage.

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3
"""
# %% Import section
from PyQt5.QtWidgets import (QMainWindow, QWidget, QGridLayout, QPushButton, QSpinBox,
                             QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit)
import numpy as np
from numpy import fft
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from multiprocessing import Queue as mpQueue
import time


# %% Class def
class FourierTransformCtrlWindow(QMainWindow):
    """Reimplement of a few QMainWindow methods."""

    def __init__(self, main_class_instance):
        super().__init__()
        self.main_class = main_class_instance
        self.setWindowTitle("Fourier Transform and Processing")
        self.image_quality_metric = 0.0; self.aberrations_selectors = {}
        self.ctrl_script_handle = None; self.aberrations_selector_window = None

        # Buttons specification
        self.recalculate_button = QPushButton("Recalculate metric")
        self.recalculate_button.clicked.connect(self.calculate_image_quality_metric)
        self.open_ctrl_button = QPushButton("Open DPP ctrl")
        self.open_ctrl_button.clicked.connect(self.open_ctrl_program)
        self.applied_aberrations_ctrl_button = QPushButton("Select aberrations")
        self.applied_aberrations_ctrl_button.clicked.connect(self.open_aberrations_selector)

        # Radiuses for metric calculation selectors specification
        self.r_min = QSpinBox(); self.r_min.setSingleStep(1); self.r_min.setMaximum(98)
        self.r_max = QSpinBox(); self.r_max.setSingleStep(1); self.r_max.setMaximum(100)
        self.r_min.setMinimum(1); self.r_max.setMinimum(2); self.r_min.setValue(1); self.r_max.setValue(11)
        self.r_min.setAlignment(Qt.AlignCenter); self.r_max.setAlignment(Qt.AlignCenter)
        self.r_min.setPrefix("R min %: "); self.r_max.setPrefix("R max %: ")
        self.r_min.valueChanged.connect(self.draw_metric_calculation_ranges)
        self.r_max.valueChanged.connect(self.draw_metric_calculation_ranges)

        # Biases for aberrations correction controls specification
        self.bias_min = QDoubleSpinBox(); self.bias_min.setSingleStep(0.1); self.bias_min.setMaximum(3.0)
        self.bias_max = QDoubleSpinBox(); self.bias_max.setSingleStep(0.1); self.bias_max.setMaximum(4.0)
        self.bias_min.setMinimum(-4.0); self.bias_max.setMinimum(-3.0)
        self.bias_min.setValue(-1.0); self.bias_max.setValue(1.0)
        self.bias_min.setAlignment(Qt.AlignCenter); self.bias_max.setAlignment(Qt.AlignCenter)
        self.bias_min.setPrefix("Bias min: "); self.bias_max.setPrefix("Bias max: ")
        self.bias_min.setDecimals(1); self.bias_max.setDecimals(1)
        self.bias_min.valueChanged.connect(self.min_bias_value_changed)
        self.bias_max.valueChanged.connect(self.max_bias_value_changed)
        self.bias_min.setEnabled(False); self.bias_max.setEnabled(False)  # disable ctrls until the device opened

        # Maximum order selector
        self.orders_names = ["1st order", "2nd order", "3rd order", "4th order",
                             "5th order", "6th order", "7th order"]
        self.max_order_selector = QComboBox(); self.max_order_selector.addItems(self.orders_names)
        self.max_order_selector.setCurrentText(self.orders_names[1])  # default maximum order
        self.max_order_selector.currentTextChanged.connect(self.max_order_changed)  # handling max order choosing
        self.max_order_selector.setEditable(True)  # setEditable is for setAlignment
        self.max_order_selector.lineEdit().setAlignment(Qt.AlignCenter)
        self.max_order_selector.lineEdit().setReadOnly(True)

        # Embedding the matplolib graph into the qt window
        # self.default_plot_figure = 6.0  # default figure size (in inches)
        # self.plot_figure = Figure(figsize=(self.default_plot_figure, self.default_plot_figure))
        self.plot_figure = Figure()  # actual size defined by the QWindow geometry settings previously
        self.canvas_widget = FigureCanvas(self.plot_figure); self.canvas_widget.draw()
        self.plot_figure_axes = self.plot_figure.add_subplot()
        self.navigation_toolbar = NavigationToolbar(self.canvas_widget, self)
        self.plot_figure_axes.autoscale(enable=True, axis='both', tight=True)
        # self.plot_figure_axes.axis('off')
        self.plot_figure.tight_layout()
        self.plot_figure.subplots_adjust(left=0.08, bottom=0.02, right=0.97, top=1)
        self.imshowing = None; self.fft_transformed_image = None

        # Grid layout
        self.qwindow_fft = QWidget()  # The composing of all widgets for image representation into one main widget
        self.setCentralWidget(self.qwindow_fft)
        grid = QGridLayout(self.qwindow_fft)  # grid layout allows better layout of buttons and frames
        # grid.addWidget(self.fft_plot_widget, 0, 0, 4, 4)  # add image representation widget
        grid.addWidget(self.canvas_widget, 0, 0, 4, 4); grid.addWidget(self.navigation_toolbar, 4, 0, 1, 4)
        grid.addWidget(self.recalculate_button, 5, 0, 1, 1); grid.addWidget(self.r_min, 5, 1, 1, 1)
        grid.addWidget(self.r_max, 5, 2, 1, 1); grid.addWidget(self.open_ctrl_button, 6, 0, 1, 1)
        grid.addWidget(self.bias_min, 6, 1, 1, 1); grid.addWidget(self.bias_max, 6, 2, 1, 1)
        grid.addWidget(self.max_order_selector, 7, 0, 1, 1)
        grid.addWidget(self.applied_aberrations_ctrl_button, 7, 1, 1, 1)

        self.plot_fourier_transform()  # call the plotting function

        if hasattr(self.main_class, "flag_live_stream"):
            if self.main_class.flag_live_stream:
                print("Launch continuous Fourier transform calculation")

    def plot_fourier_transform(self):
        """
        Calculate and plot Fourier transform of the current image.

        Returns
        -------
        None.

        """
        self.current_image = self.main_class.imageWidget.getImageItem().image  # get the current displayed image
        if self.current_image is not None and isinstance(self.current_image, np.ndarray):
            maxI = np.max(self.current_image); minI = np.min(self.current_image)
            if maxI > minI:  # simple check that something is drawn on an image
                self.fft_transformed_image = fft.fft2(self.current_image.astype(dtype='float'))
                self.fft_transformed_image = np.abs(self.fft_transformed_image)
                # Below - shift all frequencies from corners to the center (shift of origin)
                self.fft_transformed_image = fft.fftshift(self.fft_transformed_image)
                # Normalization of the Fourier transformed image (???):
                self.fft_transformed_image /= 2*np.pi*maxI
                # Below - LogNorm colors for representation of logarithmically scaled amplitudes
                minFFTvalue = np.min(self.fft_transformed_image); maxFFTvalue = np.max(self.fft_transformed_image)
                if maxFFTvalue <= minFFTvalue:
                    maxFFTvalue = 1; minFFTvalue = 0
                if self.imshowing is None:
                    self.imshowing = self.plot_figure_axes.imshow(self.fft_transformed_image,
                                                                  norm=LogNorm(vmin=minFFTvalue, vmax=maxFFTvalue))
                else:
                    self.imshowing.set_data(self.fft_transformed_image)
                    m, n = self.fft_transformed_image.shape
                    self.imshowing.set_extent((0, m, n, 0))  # update the range (now - always, need only if image size changed)
                # self.plot_figure.colorbar(self.imshowing)
                self.canvas_widget.draw_idle()
                # Draw the region selection of accounted values for metrics:
                self.draw_metric_calculation_ranges()
                # Re-calculate the image quality metric
                self.calculate_image_quality_metric()

    def refresh_plot(self):
        """
        Refresh plotted Fourier transform.

        Returns
        -------
        None.

        """
        self.plot_fourier_transform()  # call the function for plotting

    def draw_metric_calculation_ranges(self):
        """
        Draw circles on the Fourier transformed image for accounting of an image metric.

        Returns
        -------
        None.

        """
        # Check consistency of input values
        if self.r_min.value() >= self.r_max.value():
            self.r_max.setValue(self.r_min.value() + 1)
        # Draw the selected ranges on Fourier-spectrum
        if self.fft_transformed_image is not None:
            # Draw the region selection of accounted values for metrics:
            m, n = self.fft_transformed_image.shape
            r1_percent = self.r_min.value()/100; r2_percent = self.r_max.value()/100
            # ??? In general, image should be with the same width and height
            if m >= n:
                dim = m
            else:
                dim = n
            self.r1 = int(np.round(r1_percent*dim, 0))  # 1% of the fft transformed image
            self.r2 = int(np.round(r2_percent*dim, 0))  # 11% of the fft transformed image
            if self.plot_figure_axes is not None:
                # Cleaning previous circles
                if len(self.plot_figure_axes.patches) >= 1:
                    self.plot_figure_axes.patches[0].remove()
                    self.plot_figure_axes.patches[0].remove()
                self.plot_figure_axes.add_patch(Circle((m//2, n//2), self.r1, edgecolor='red', facecolor='none'))
                self.plot_figure_axes.add_patch(Circle((m//2, n//2), self.r2, edgecolor='red', facecolor='none'))
                self.canvas_widget.draw_idle()

    def calculate_image_quality_metric(self):
        """
        Calculate the image quality metric (sum of spectrum values between two circles).

        Returns
        -------
        None.

        """
        self.image_quality_metric = 0.0
        if (self.fft_transformed_image is not None and isinstance(self.fft_transformed_image, np.ndarray)):
            m, n = self.fft_transformed_image.shape
            r1_percent = self.r_min.value()/100; r2_percent = self.r_max.value()/100
            if m >= n:
                dim = m
            else:
                dim = n
            r1 = r1_percent*dim; r2 = r2_percent*dim
            # For speeding up the calculations, below - calculation starting coordinates
            i_min = m//2 - int(np.round(r2, 0)); j_min = n//2 - int(np.round(r2, 0))
            i_max = m//2 + int(np.round(r2, 0)); j_max = n//2 + int(np.round(r2, 0))
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    distance = np.sqrt(np.power(i-m//2, 2) + np.power(j-n//2, 2))
                    if r1 <= distance <= r2:
                        self.image_quality_metric += self.fft_transformed_image[i, j]
            self.image_quality_metric = np.round(self.image_quality_metric, 0)
            print("Calculated metric:", self.image_quality_metric)

    def min_bias_value_changed(self):
        """
        Check consistency of min and max biases controls, if min bias value changed.

        Returns
        -------
        None.

        """
        if self.bias_min.value() >= self.bias_max.value():
            self.bias_max.setValue(self.bias_min.value() + 0.2)

    def max_bias_value_changed(self):
        """
        Check consistency of min and max biases controls, if max bias value changed.

        Returns
        -------
        None.

        """
        if self.bias_max.value() <= self.bias_min.value():
            self.bias_min.setValue(self.bias_min.value() - 0.2)

    def open_ctrl_program(self):
        """
        Open the controlling of DPP program.

        Returns
        -------
        None.

        """
        try:
            from dpp_ctrl import gui_dpp_ctrl
            try:
                aberrations_queue = mpQueue(maxsize=5)
                self.ctrl_dpp_prc = gui_dpp_ctrl.IndPrcLauncher(aberrations_queue)
                self.ctrl_dpp_prc.start()
                time.sleep(5); aberrations = {"0, 2": 0.5, "1,1": -0.5}
                aberrations_queue.put_nowait(aberrations)
            except AttributeError:
                # Use the general launcher of the controlling program
                print("External aberrations specifying not launched")
                gui_dpp_ctrl.external_default_launch()
            self.bias_min.setEnabled(True); self.bias_max.setEnabled(True)  # enable bias ctrls
        except ModuleNotFoundError:
            print("Check that the DPP controlling program is installed in the active environment")

    def open_aberrations_selector(self):
        """
        Create / show the QMainWindow with the selectors of aberrations to correct.

        Returns
        -------
        None.

        """
        if self.aberrations_selector_window is None:
            self.aberrations_selector_window = QMainWindow(parent=self)
            self.aberrations_selector_window.show()
            # widget below is needed for laying out children widgets
            self.aberrations_sel_widget = QWidget(parent=self.aberrations_selector_window)
            self.aberrations_selector_window.setCentralWidget(self.aberrations_sel_widget)
            self.aberrations_selector_window.setWindowTitle("Select correcting aberrations")
            self.aberrations_selector_window_w = 580; self.aberrations_selector_window_h = 500
            self.aberrations_selector_window.setGeometry(300, 300, self.aberrations_selector_window_w,
                                                         self.aberrations_selector_window_h)
            self.create_check_boxes_for_aberrations()
        elif not self.aberrations_selector_window.isVisible():
            self.aberrations_selector_window.show()

    def create_check_boxes_for_aberrations(self):
        try:
            from dpp_ctrl import zernike_pol_calc
        except ModuleNotFoundError:
            print("Check that dpp-ctrl module installed by pip")
        # Create single check box with name (picture?)
        self.aberrations_selectors[(-1, 1)] = QCheckBox((str((-1, 1)) + " "
                                                         + zernike_pol_calc.get_classical_polynomial_name(mode=(-1, 1),
                                                                                                          short_names=True)),
                                                        parent=self.aberrations_sel_widget)
        self.aberrations_selectors[(-1, 1)].setFont(QFont("Liberation Sans", 9, QFont.Bold))
        self.aberration_selector_width = self.aberrations_selectors[(-1, 1)].geometry().width()
        self.aberration_selector_width = self.aberrations_selectors[(-1, 1)].geometry().height()
        # self.aberrations_selectors[(-1, 1)].setChecked(False); self.aberrations_selectors[(-1, 1)].move(25, 25)
        # Define the selected max order
        for i, order in enumerate(self.orders_names):
            if order == self.max_order_selector.currentText():
                break
        max_order = i+1
        # self.aberrations_selector_window.update()

    def max_order_changed(self):
        pass

    def closeEvent(self, closing_event):
        """
        Rewrite standard handling of a click on the Close button.

        Parameters
        ----------
        closing_event : PyQt5.QtGui.QCloseEvent
            provided by calling signature of this function by PyQt.

        Returns
        -------
        None.

        """
        if self.ctrl_script_handle is not None:
            self.ctrl_script_handle.destroy()
        self.main_class.fft_transform_window = None
        self.close()

# -*- coding: utf-8 -*-
"""
Rewrite methods of QMainWindow from PyQT5 for further using it for FFT calculation and re-usage.

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3
"""
# %% Import section
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QPushButton
# import pyqtgraph
import numpy as np
from numpy import fft
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm


# %% Class def
class FourierTransformCtrlWindow(QMainWindow):
    """Reimplementate of a few QMainWindow methods."""

    def __init__(self, main_class_instance):
        super().__init__()
        self.main_class = main_class_instance
        self.setWindowTitle("Fourier Transform and Processing")

        # Construction of the pyqtgraph widget for showing the fft transform graph
        # self.plot_fft = pyqtgraph.PlotItem()
        # # self.plot_fft.getViewBox().setMouseEnabled(False, False)
        # self.fft_plot_widget = pyqtgraph.ImageView(view=self.plot_fft)  # The main widget for image showing
        # self.fft_plot_widget.ui.roiBtn.hide(); self.fft_plot_widget.ui.menuBtn.hide()   # Hide ROI, Norm buttons
        # self.fft_plot_widget.ui.histogram.hide()  # hide histogram graph
        # self.fft_plot_widget.getView().showAxis("left", show=False)
        # self.fft_plot_widget.getView().showAxis("bottom", show=False)
        self.ctrl_button = QPushButton("ctrl button")

        # Embedding the matplolib graph into the qt window
        # self.default_plot_figure = 6.0  # default figure size (in inches)
        # self.plot_figure = Figure(figsize=(self.default_plot_figure, self.default_plot_figure))
        self.plot_figure = Figure()  # actual size defined by the QWindow geometry settings previously
        self.canvas_widget = FigureCanvas(self.plot_figure); self.canvas_widget.draw()
        self.plot_figure_axes = self.plot_figure.add_subplot()
        self.navigation_toolbar = NavigationToolbar(self.canvas_widget, self)
        # self.plot_figure_axes.axis('off')
        self.plot_figure.tight_layout()
        self.plot_figure.subplots_adjust(left=0.07, bottom=0.03, right=0.99, top=1)

        # Grid layout
        self.qwindow_fft = QWidget()  # The composing of all widgets for image representation into one main widget
        self.setCentralWidget(self.qwindow_fft)
        grid = QGridLayout(self.qwindow_fft)  # grid layout allows better layout of buttons and frames
        # grid.addWidget(self.fft_plot_widget, 0, 0, 4, 4)  # add image representation widget
        grid.addWidget(self.canvas_widget, 0, 0, 4, 4); grid.addWidget(self.navigation_toolbar, 4, 0, 1, 4)
        grid.addWidget(self.ctrl_button, 5, 0, 1, 1)

        self.plot_fourier_trasnform()  # call the plotting function

        if hasattr(self.main_class, "flag_live_stream"):
            if self.main_class.flag_live_stream:
                print("Launch continiuos Fourier transform calculation")

    def plot_fourier_trasnform(self):
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
                # Below - shift all frequences from corners to the center (shift of origin)
                self.fft_transformed_image = fft.fftshift(self.fft_transformed_image)
                # Normalization of the Fourier transformed image (???):
                self.fft_transformed_image /= 2*np.pi*maxI
                # self.fft_transformed_image = ((self.fft_transformed_image/np.max(self.fft_transformed_image))*255)
                # self.fft_plot_widget.setImage(self.fft_transformed_image)
                # Below - LogNorm colors for representation of logariphmically scaled amplitudes
                minFFTvalue = np.min(self.fft_transformed_image); maxFFTvalue = np.max(self.fft_transformed_image)
                if maxFFTvalue <= minFFTvalue:
                    maxFFTvalue = 1; minFFTvalue = 0
                self.imshowing = self.plot_figure_axes.imshow(self.fft_transformed_image,
                                                              norm=LogNorm(vmin=minFFTvalue, vmax=maxFFTvalue))
                self.canvas_widget.draw_idle()
            # self.plot_figure.colorbar(self.imshowing)

    def refresh_plot(self):
        """
        Refresh plotted Fourier transform.

        Returns
        -------
        None.

        """
        self.plot_fourier_trasnform()

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
        self.main_class.fft_transform_window = None
        self.close()

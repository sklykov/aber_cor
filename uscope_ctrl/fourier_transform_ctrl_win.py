# -*- coding: utf-8 -*-
"""
Rewrite methods of QMainWindow from PyQT5 for further using it for FFT calculation and re-usage.

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3
"""
# %% Import section
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QPushButton
import pyqtgraph
import numpy as np
from scipy import fftpack


# %% Class def
class FourierTransformCtrlWindow(QMainWindow):
    """Reimplementate of a few QMainWindow methods."""

    def __init__(self, main_class_instance):
        super().__init__()
        self.main_class = main_class_instance
        self.setWindowTitle("Fourier Transform and Processing")

        # Construction of the widget for showing the fft transform graph
        self.plot_fft = pyqtgraph.PlotItem()
        # self.plot_fft.getViewBox().setMouseEnabled(False, False)
        self.fft_plot_widget = pyqtgraph.ImageView(view=self.plot_fft)  # The main widget for image showing
        self.fft_plot_widget.ui.roiBtn.hide(); self.fft_plot_widget.ui.menuBtn.hide()   # Hide ROI, Norm buttons
        # self.fft_plot_widget.ui.histogram.hide()  # hide histogram graph
        # self.fft_plot_widget.getView().showAxis("left", show=False)
        # self.fft_plot_widget.getView().showAxis("bottom", show=False)
        self.ctrl_button = QPushButton("ctrl button")

        # Grid layout
        self.qwindow_fft = QWidget()  # The composing of all widgets for image representation into one main widget
        self.setCentralWidget(self.qwindow_fft)
        grid = QGridLayout(self.qwindow_fft)  # grid layout allows better layout of buttons and frames
        grid.addWidget(self.fft_plot_widget, 0, 0, 4, 4)  # add image representation widget
        grid.addWidget(self.ctrl_button, 4, 0, 1, 1)

        self.current_image = main_class_instance.imageWidget.getImageItem().image  # get the current displayed image
        if self.current_image is not None:
            self.fft_transformed_image = fftpack.fft2(self.current_image.astype(dtype='float'))
            self.fft_transformed_image = np.abs(self.fft_transformed_image)
            # self.fft_transformed_image = ((self.fft_transformed_image/np.max(self.fft_transformed_image))*255)
            self.fft_plot_widget.setImage(self.fft_transformed_image)
            print(np.max(self.fft_transformed_image), np.min(self.fft_transformed_image))

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

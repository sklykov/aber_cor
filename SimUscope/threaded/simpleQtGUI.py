# -*- coding: utf-8 -*-
"""
Simple GUI for representing of generated noisy image using PyQT.

@author: ssklykov
"""
# %% Imports
# import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QGridLayout, QSpinBox, QCheckBox, QVBoxLayout
import numpy as np
from generate_noise_pic import generate_noise_picture
from threading import Thread
# import matplotlib.pyplot as plt
import time
import pyqtgraph

# %% Global variables as simple method for synchronization
flag_generation = False
width_default = 1000; height_default = 1000  # Default width and height for generation of images


# %% Class wrapper for threaded noisy single picture generation
class SingleImageGenerator(Thread):
    """Threaded class for generation of single noisy image."""

    height = 100; width = 100; image = np.zeros((height, width), dtype='uint8')

    def __init__(self, height: int = 100, width: int = 100):
        Thread.__init__(self); self.height = height; self.width = width

    def run(self):
        """
        Generate of single image and storing it inside the class variable.

        Returns
        -------
        None.

        """
        self.image = generate_noise_picture(self.width, self.height)  # Width and height order re-defined in the subfunction as well


# %% Class wrapper for threaded noisy continuous pictures generation for checking the performance of Python representation
class ContinuousImageGenerator(Thread):
    """Threaded class for generation of continuous stream of noisy images and updating the ImageView widget."""

    meanGenTimes = np.zeros(200, dtype="uint")

    def __init__(self, imageWidget, refresh_delay_ms: int = 100, height: int = 100, width: int = 100, testPerformance: bool = False):
        Thread.__init__(self); self.height = height; self.width = width; self.testPerformance = testPerformance
        self.imageWidget = imageWidget; self.refresh_delay_ms = refresh_delay_ms

    def run(self):
        """
        Make continuous generation of noisy pictures and updating the ImageView widget from pyqtgraph for their showing.

        Returns
        -------
        None.

        """
        global flag_generation  # Dirty way of synchronizing of button clicked in the GUI with this indepent thread process
        i = 0  # for adding the elements into preinitilized array for further mean generation time calculation
        while(flag_generation):
            if self.testPerformance:
                t1 = time.time()
                # Below - the workaround for preventing kernel diying during continuous generation without any delays
                if self.refresh_delay_ms == 0:  # if the delay between frames is 0, than the generation is unstable
                    self.refresh_delay_ms += 1  # make the delay at least 1 ms
            image = generate_noise_picture(self.width, self.height)  # Get the noisy picture, width and height swapped
            self.imageWidget.setImage(image)  # Set the image for representation by passed ImageView pyqtgraph widget
            time.sleep(self.refresh_delay_ms/1000)  # Applying artificial delays between each image generation
            # If testing of Performance requested, then accumulating of passed times in the array performed
            if self.testPerformance:
                t2 = time.time()
                if i < np.size(self.meanGenTimes):
                    self.meanGenTimes[i] = np.uint(np.round((t2-t1)*1000, 0)); i += 1  # Add the passed time to the array
        # If generation stopped and the test of performance was asked, then print out the mean generation time
        if self.testPerformance:
            # Calculation of the final element that in the array is zero (passed time not saved)
            for j in range(np.size(self.meanGenTimes)):
                if self.meanGenTimes[j] == 0:
                    break
            self.meanGenTimes = self.meanGenTimes[0:j]  # Truckate array till the non-zerp element for mean value calculation
            mean_gen_t = np.uint(np.round(np.mean(self.meanGenTimes), 0))
            print("Mean generation time is:", mean_gen_t, "ms")


# %% Implementation of all windows inside the child class
class SimUscope(QMainWindow):
    """Create the GUI with buttons for testing image acquisition from the camera and making some image processing."""

    __flagGeneration = False  # Private class variable for recording state of continuous generation
    __flagTestPerformance = False  # Private class variable for switching between test state by using QCheckBox

    def __init__(self, img_height, img_width):
        """Create overall UI inside the QMainWindow widget."""
        super().__init__()
        self.imageGenerator = SingleImageGenerator(img_height, img_width); self.img_height = img_height; self.img_width = img_width
        self.img = np.zeros((self.img_height, self.img_width), dtype='uint8')  # Black initial image
        self.setWindowTitle("Simulation of uscope camera"); self.setGeometry(200, 200, 800, 700)
        # Buttons and ImageView setting on the main window
        self.plot = pyqtgraph.PlotItem(); self.plot.setXRange(0, self.img_width); self.plot.setYRange(0, self.img_height)
        self.imageWidget = pyqtgraph.ImageView(view=self.plot)  # The main widget for image showing
        self.imageWidget.ui.roiBtn.hide(); self.imageWidget.ui.menuBtn.hide()   # Hide ROI, Norm buttons from the ImageView
        self.imageWidget.setImage(self.img)  # Set image for representation in the ImageView widget
        # self.roi = pyqtgraph.ROI((1, 1), size=(100, 100), rotatable=False, removable=True); self.plot.addItem(self.roi)
        self.qwindow = QWidget()  # The composing of all buttons and frame for image representation into one main widget
        self.buttonGenSingleImg = QPushButton("Generate Single Pic"); self.buttonGenSingleImg.clicked.connect(self.generate_single_pic)
        self.buttonContinuousGen = QPushButton("Continuous Generation")  # Switches on/off continuous generation
        self.toggleTestPerformance = QCheckBox("Test Performance"); self.toggleTestPerformance.setEnabled(True)
        self.toggleTestPerformance.setChecked(False)  # setChecked - set the state of a button
        self.buttonContinuousGen.clicked.connect(self.generate_continuous_pics); self.buttonContinuousGen.setCheckable(True)
        self.exposureTime = QSpinBox(); self.exposureTime.setSingleStep(1); self.exposureTime.setSuffix(" ms")
        self.exposureTime.setPrefix("Exposure time: "); self.exposureTime.setMinimum(1); self.exposureTime.setMaximum(1000)
        self.exposureTime.setValue(100); self.exposureTime.adjustSize()
        self.quitButton = QPushButton("Quit"); self.quitButton.setStyleSheet("color: red")
        self.quitButton.clicked.connect(self.quitClicked)
        # Grid layout below - the main layout pattern for all buttons and windos on the Main Window
        grid = QGridLayout(self.qwindow); self.setLayout(grid)  # grid layout allows better layout of buttons and frames
        grid.addWidget(self.buttonGenSingleImg, 0, 0, 1, 1); grid.addWidget(self.buttonContinuousGen, 0, 1, 1, 1)
        grid.addWidget(self.toggleTestPerformance, 0, 2, 1, 1); grid.addWidget(self.exposureTime, 0, 3, 1, 1)
        grid.addWidget(self.quitButton, 0, 5, 1, 1)
        # vbox below - container for Height / Width buttons
        vbox = QVBoxLayout(self.qwindow); self.widthButton = QSpinBox(); self.heightButton = QSpinBox(); vbox.addWidget(self.widthButton)
        self.heightButton.setPrefix("Height: "); self.widthButton.setPrefix("Width: "); vbox.addWidget(self.heightButton)
        grid.addLayout(vbox, 0, 4, 1, 1); self.widthButton.setSingleStep(2); self.heightButton.setSingleStep(2)
        self.widthButton.setMinimum(50); self.heightButton.setMinimum(50)
        self.widthButton.setMaximum(3000); self.heightButton.setMaximum(3000)
        self.widthButton.setValue(self.img_width); self.heightButton.setValue(self.img_height)
        # Set valueChanged event handlers
        self.widthButton.valueChanged.connect(self.imgSizeChanged); self.heightButton.valueChanged.connect(self.imgSizeChanged)
        # ImageWidget should be central - for better representation of generated images
        grid.addWidget(self.imageWidget, 1, 0, 5, 6)  # the ImageView widget spans on ... rows and ... columns (2 values in the end)
        self.setCentralWidget(self.qwindow)  # Actually, allows to make both buttons and ImageView visible

    def generate_single_pic(self):
        """
        Handle clicking of Generate Single Picture. This method updates the image associated with ImageView widget.

        Returns
        -------
        None.

        """
        # !!!: Don't forget that waiting that thread complete its task guarantees all proper assignments (below)
        self.imageGenerator.start(); self.imageGenerator.join(); self.imageWidget.setImage(self.imageGenerator.image)
        self.imageGenerator = SingleImageGenerator(self.img_height, self.img_width)

    def generate_continuous_pics(self):
        """
        Handle clicking of Continuous Generation button. Generates continuously and updates the ImageView widget.

        Returns
        -------
        None.

        """
        self.__flagGeneration = not(self.__flagGeneration)  # changing the state of generation
        self.buttonContinuousGen.setDown(self.__flagGeneration)  # changing the visible state of button (clicked or not)
        global flag_generation
        flag_generation = self.__flagGeneration
        if (self.__flagGeneration):
            self.toggleTestPerformance.setDisabled(True)  # Disable the check box for preventing test on during continuous generation
            self.exposureTime.setDisabled(True)  # Disable the exposure time control
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True)
            continuousImageGen = ContinuousImageGenerator(self.imageWidget, self.exposureTime.value(),
                                                          self.img_height, self.img_width,
                                                          self.toggleTestPerformance.isChecked())
            continuousImageGen.start()  # Run the threaded code
        else:
            self.toggleTestPerformance.setEnabled(True); self.exposureTime.setEnabled(True)
            self.widthButton.setEnabled(True); self.heightButton.setEnabled(True)

    def closeEvent(self, closeEvent):
        """
        Rewrites the default behaviour of clicking on the X button on the main window GUI.

        Parameters
        ----------
        closeEvent : QWidget Close Event
            Needed by the API.

        Returns
        -------
        None.

        """
        global flag_generation
        if flag_generation:
            flag_generation = False  # For notifiyng of Generation Thread to stop generation
            exp_time = self.exposureTime.value(); time.sleep((exp_time*3)/1000)  # Artificial delay for waiting the Generation Thread ended
            print("Waited", exp_time*3, "ms for closing the Generation Thread")
            closeEvent.accept()  # Maybe redundant, but this is explicit accepting quit event (could be refused if asked for example)

    def quitClicked(self):
        """
        Handle the clicking event on the Quit Button.

        Sets the global variables to False state. Waits for threads stop running. Quits the Main window.

        Returns
        -------
        None.

        """
        global flag_generation
        if flag_generation:
            flag_generation = False  # For notifiyng of Generation Thread to stop generation
            exp_time = self.exposureTime.value(); time.sleep((exp_time*3)/1000)  # Artificial delay for waiting the Generation Thread ended
        self.close()  # Calls the closing event for QMainWindow

    def imgSizeChanged(self):
        """
        Handle changing of image width or height. Allows to pick up values for single image generation and continuous one.

        Returns
        -------
        None.

        """
        self.img_width = self.widthButton.value(); self.img_height = self.heightButton.value()
        self.imageGenerator = SingleImageGenerator(self.img_height, self.img_width)


# %% Tests
if __name__ == "__main__":
    my_app = QApplication([])  # application without any command-line arguments
    my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
    main_window = SimUscope(width_default, height_default); main_window.show()
    my_app.exec()  # Exit of the main program

    # Simple check if the continuous generation still could be runned in the background and stopping it by assigning the global variable to False
    if flag_generation:
        print("Generation is still happening and will be handled by the main program")
        flag_generation = False; time.sleep(0.3)

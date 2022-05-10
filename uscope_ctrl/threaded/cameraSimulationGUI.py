# -*- coding: utf-8 -*-
"""
Simple GUI for representing of generated noisy image using PyQT.

@author: ssklykov
"""
# %% Imports
# import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, QGridLayout,
                             QSpinBox, QCheckBox, QVBoxLayout, QComboBox, QLabel)
from PyQt5.QtCore import Qt, QRect
import time
import numpy as np
import pyqtgraph
from cameraLiveStreamSimulation import SingleImageThreadedGenerator, ContinuousImageThreadedGenerator
from queue import Queue, Empty
# from multiprocessing import Queue as ProcessQueue
# from cameraCtrlProcess import PCOcamera as PCOcameraProcess
import os
from skimage import io
from checkExceptionsInLive import CheckMessagesForExceptions
from cameraCtrl import PCOcamera
from threading import Thread

# %% Some default values. The global value is omitted as the non-reliable for communication with the functions imported from modules
width_default = 1000; height_default = 1000  # Default width and height for generation of images


# %% GUI class
class SimUscope(QMainWindow):
    """Create the GUI with buttons for testing image acquisition from the camera and making some image processing."""

    __flagGeneration = False  # Private class variable for recording state of continuous generation
    __flagTestPerformance = False  # Private class variable for switching between test state by using
    autoRange = True  # flag for handling user preference about possibility to zoom in/out to images

    def __init__(self, img_height, img_width, applicationHandle: QApplication):
        """Create overall UI inside the QMainWindow widget."""
        super().__init__()
        self.applicationHandle = applicationHandle  # handle to the main application for exit it in the appropriate place
        self.messages2Camera = Queue(maxsize=5)  # Initialize message queue for communication with the camera (simulated or not)
        self.exceptionsQueue = Queue(maxsize=5)  # Initialize separate queue for spreading and handling Exceptions occured within modules
        self.imagesQueue = Queue(maxsize=50)  # Initialize the queue for holding acquired images and acessing them from the GUI thread
        self.gui_refresh_rate_ms = 10  # The constant time pause between each attempt to retrieve the image
        self.imageGenerator = SingleImageThreadedGenerator(img_height, img_width); self.img_height = img_height; self.img_width = img_width
        self.img_width_default = img_width; self.img_height_default = img_height
        self.img = np.zeros((self.img_height, self.img_width), dtype='uint8')  # Black initial image
        self.setWindowTitle("Camera control / simulation GUI"); self.setGeometry(200, 200, 840, 800)
        # PlotItem allows showing the axes and restrict the mouse usage over the image
        self.plot = pyqtgraph.PlotItem()
        self.plot.getViewBox().setMouseEnabled(False, False)  # !!!: Disable the possibility of move image by mouse
        # ImageView - for showing the generated image
        self.imageWidget = pyqtgraph.ImageView(view=self.plot)  # The main widget for image showing
        self.imageWidget.ui.roiBtn.hide(); self.imageWidget.ui.menuBtn.hide()   # Hide ROI, Norm buttons from the ImageView
        self.imageWidget.setImage(self.img)  # Set image for representation in the ImageView widget
        # connecting user interaction with levels buttons - below
        self.imageWidget.ui.histogram.sigLevelChangeFinished.connect(self.histogramLevelsChanged)
        # QWidget - main widget window for grid layout
        self.qwindow = QWidget()  # The composing of all buttons and frame for image representation into one main widget
        # Selector of type of a camera - Simulated or PCO one
        self.cameraSelector = QComboBox(); self.cameraSelector.addItems(["Simulated Threaded", "PCO"])
        # self.cameraSelector.addItems(["Simulated Threaded", "PCO", "PCO Process"])
        self.cameraSelector.setCurrentText("Simulated Threaded")  # Deafult camera for initialization - the simulated one
        self.cameraSelector.currentTextChanged.connect(self.activeCameraChanged)  # Attach handlers for camera choosing
        self.cameraSelLabel = QLabel("Camera Type"); self.cameraSelector.setEditable(True)  # setEditable is needed for setAlignment
        self.cameraSelector.lineEdit().setAlignment(Qt.AlignCenter); self.cameraSelector.lineEdit().setReadOnly(True)
        vboxSelector = QVBoxLayout(); vboxSelector.addWidget(self.cameraSelLabel); vboxSelector.addWidget(self.cameraSelector)
        self.cameraSelLabel.setAlignment(Qt.AlignCenter)  # Align the label text on the center
        # ROI size selectors
        self.heightROI = QSpinBox(); self.heightROI.setSingleStep(2); self.heightROI.setMaximum(self.img_width)  # swapped
        self.widthROI = QSpinBox(); self.widthROI.setSingleStep(2); self.widthROI.setMaximum(self.img_height)  # swapped
        self.heightROI.setMinimum(2); self.widthROI.setMinimum(2); self.heightROI.setValue(100); self.widthROI.setValue(100)
        self.widthROI.setAlignment(Qt.AlignCenter); self.heightROI.setAlignment(Qt.AlignCenter)
        self.widthROI.setPrefix("ROI width: "); self.heightROI.setPrefix("ROI height: ")
        vboxROI = QVBoxLayout(); vboxROI.addWidget(self.widthROI); vboxROI.addWidget(self.heightROI)
        self.widthROI.valueChanged.connect(self.roiSizesSpecified); self.heightROI.valueChanged.connect(self.roiSizesSpecified)
        # Push buttons for events evoking
        self.snapSingleImgButton = QPushButton("Generate Single Pic"); self.snapSingleImgButton.clicked.connect(self.snap_single_img)
        self.continuousStreamButton = QPushButton("Continuous Generation")  # Switches on/off continuous generation
        self.toggleTestPerformance = QCheckBox("Test Performance"); self.toggleTestPerformance.setEnabled(True)
        self.toggleTestPerformance.setChecked(False)  # setChecked - set the state of a button
        self.continuousStreamButton.clicked.connect(self.continuous_stream); self.continuousStreamButton.setCheckable(True)
        self.disableAxesOnImageButton = QPushButton("Disable axis on image"); self.disableAxesOnImageButton.setCheckable(True)
        self.disableAxesOnImageButton.clicked.connect(self.disableAxesOnImage)
        self.disableAutoLevelsButton = QPushButton("Disable pixels leveling"); self.disableAutoLevelsButton.setCheckable(True)
        self.disableAutoLevelsButton.clicked.connect(self.disableAutoLevelsCalculation)
        self.exposureTimeButton = QSpinBox(); self.exposureTimeButton.setSingleStep(1); self.exposureTimeButton.setSuffix(" ms")
        self.exposureTimeButton.setPrefix("Exposure time: "); self.exposureTimeButton.setMinimum(1); self.exposureTimeButton.setMaximum(1000)
        self.exposureTimeButton.setValue(100); self.exposureTimeButton.adjustSize()
        self.exposureTimeButton.setKeyboardTracking(False)  # special function to disable emtting of signals for each typed value
        # E.g., when the user is typing "100" => emit 3 signals for 3 numbers, if keyboardTracking(True)
        self.exposureTimeButton.valueChanged.connect(self.exposureTimeChanged)
        self.switchMouseCtrlImage = QCheckBox("Handling Image by mouse"); self.switchMouseCtrlImage.setChecked(False)
        self.switchMouseCtrlImage.stateChanged.connect(self.activateMouseOnImage)
        self.saveSnapImg = QPushButton("Save Single Image"); self.saveSnapImg.clicked.connect(self.saveSingleSnapImg)
        self.saveSnapImg.setDisabled(True)  # disable the saving image before any image generated / displayed
        self.putROI = QPushButton("ROI selector"); self.putROI.clicked.connect(self.putROIonImage)
        self.generateException = QPushButton("Generate Exception"); self.generateException.clicked.connect(self.generateMessageWithException)
        self.cropImageButton = QPushButton("Crop Image"); self.cropImageButton.clicked.connect(self.cropImage)
        self.restoreFullImgButton = QPushButton("Restore Full Image"); self.restoreFullImgButton.clicked.connect(self.restoreFullImage)
        self.cropImageButton.setDisabled(True); self.restoreFullImgButton.setDisabled(True)  # until some roi selected
        self.quitButton = QPushButton("Quit"); self.quitButton.setStyleSheet("color: red")
        self.quitButton.clicked.connect(self.quitClicked)
        # Manual image level inputs and update them from the slider control on histogram viewer from the ImageWidget
        self.minLevelButton = QSpinBox(); self.maxLevelButton = QSpinBox()
        self.minLevelButton.valueChanged.connect(self.pixelValuesChanged)
        self.maxLevelButton.valueChanged.connect(self.pixelValuesChanged)
        self.minLevelButton.setSingleStep(1); self.maxLevelButton.setSingleStep(1)
        self.minLevelButton.setMinimum(0); self.maxLevelButton.setMinimum(1)
        self.maxLevelButton.setMaximum(65535); self.minLevelButton.setMaximum(65534)
        self.minLevelButton.setPrefix("Min px: "); self.maxLevelButton.setPrefix("Max px: ")
        self.minLevelButton.setMaximumWidth(120); self.maxLevelButton.setMaximumWidth(120)
        self.minLevelButton.setKeyboardTracking(False); self.maxLevelButton.setKeyboardTracking(False)
        vboxLevels = QVBoxLayout(); vboxLevels.addWidget(self.minLevelButton); vboxLevels.addWidget(self.maxLevelButton)
        vboxLevels.setAlignment(Qt.AlignCenter)
        # Grid layout below - the main layout pattern for all buttons and windows put on the Main Window (GUI)
        grid = QGridLayout(self.qwindow)  # grid layout allows better layout of buttons and frames
        grid.addLayout(vboxSelector, 0, 0, 1, 1)  # Add selector of a camera
        grid.addWidget(self.snapSingleImgButton, 0, 1, 1, 1); grid.addWidget(self.continuousStreamButton, 0, 2, 1, 1)
        grid.addWidget(self.toggleTestPerformance, 0, 3, 1, 1); grid.addWidget(self.exposureTimeButton, 0, 4, 1, 1)
        # vbox below - container for Height / Width buttons
        vbox = QVBoxLayout(); self.widthButton = QSpinBox(); self.heightButton = QSpinBox(); vbox.addWidget(self.widthButton)
        self.heightButton.setPrefix("Height: "); self.widthButton.setPrefix("Width: "); vbox.addWidget(self.heightButton,)
        grid.addLayout(vbox, 0, 5, 1, 1); self.widthButton.setSingleStep(2); self.heightButton.setSingleStep(2)
        self.widthButton.setMinimum(50); self.heightButton.setMinimum(50); self.widthButton.setMaximum(3000)
        self.heightButton.setMaximum(3000); self.widthButton.setValue(self.img_width); self.heightButton.setValue(self.img_height)
        grid.addWidget(self.quitButton, 0, 6, 1, 1); grid.addWidget(self.switchMouseCtrlImage, 7, 0, 1, 1)
        grid.addWidget(self.saveSnapImg, 7, 1, 1, 1); grid.addWidget(self.putROI, 7, 2, 1, 1)
        grid.addLayout(vboxROI, 7, 3, 1, 1); grid.addWidget(self.generateException, 7, 6, 1, 1)
        grid.addWidget(self.cropImageButton, 7, 4, 1, 1); grid.addWidget(self.restoreFullImgButton, 7, 5, 1, 1)
        grid.addWidget(self.disableAxesOnImageButton, 1, 6, 1, 1); grid.addWidget(self.disableAutoLevelsButton, 2, 6, 1, 1)
        grid.addLayout(vboxLevels, 3, 6, 1, 1, Qt.AlignCenter)  # adding the min / max pixel values
        # Set valueChanged event handlers
        self.widthButton.valueChanged.connect(self.imageSizeChanged); self.heightButton.valueChanged.connect(self.imageSizeChanged)
        # ImageWidget should be central - for better representation of generated images
        grid.addWidget(self.imageWidget, 1, 0, 6, 6)  # the ImageView widget spans on ... rows and ... columns (2 values in the end)
        self.setCentralWidget(self.qwindow)  # Actually, allows to make both buttons and ImageView visible
        # Initiliaze and start the Exception checker and associate it with the initialized Quit button
        self.exceptionChecker = CheckMessagesForExceptions(self.exceptionsQueue, self.quitButton, period_checks_ms=25)
        self.exceptionChecker.start()  # Start the Exception checker
        self.initializeCamera()  # Initializes default camera defined by the default value of cameraSelector button

    def initializeCamera(self):
        """
        Initialize the default camera.

        Returns
        -------
        None.

        """
        if self.cameraSelector.currentText() == "Simulated Threaded":
            # Initialize the simulated camera
            pyqtgraph.setConfigOptions(imageAxisOrder='col-major')  # Set for conforming with initial development
            self.continuousImageGen = ContinuousImageThreadedGenerator(self.imageWidget, self.messages2Camera,
                                                                       self.exposureTimeButton.value(), self.img_height, self.img_width,
                                                                       self.toggleTestPerformance.isChecked(), self.autoRange)
        elif self.cameraSelector.currentText() == "PCO":
            pyqtgraph.setConfigOptions(imageAxisOrder='row-major')  # Set for conforming with images from the camera
            # Initialize the PCO camera
            self.cameraHandle = PCOcamera(self.messages2Camera, self.exceptionsQueue, self.imagesQueue, self.exposureTimeButton.value())

    def snap_single_img(self):
        """
        Handle clicking of Generate Single Picture. This method updates the image associated with ImageView widget.

        Returns
        -------
        None.

        """
        if self.cameraSelector.currentText() == "Simulated Threaded":
            # !!!: Don't forget that waiting that thread complete its task guarantees all proper assignments (below)
            self.imageGenerator.start(); self.imageGenerator.join()
            self.imageWidget.setImage(self.imageGenerator.image, autoRange=self.autoRange,
                                      autoLevels=not(self.disableAutoLevelsButton.isChecked()))
            self.imageGenerator = SingleImageThreadedGenerator(self.img_height, self.img_width)  # Re-initialize the class
            if not(self.saveSnapImg.isEnabled()):   # activate saving images, since some image generated and displayed
                self.saveSnapImg.setEnabled(True)
        elif self.cameraSelector.currentText() == "PCO":
            if not(self.messages2Camera.full()):
                self.messages2Camera.put_nowait("Snap single image")  # Send the command for acquiring single image
                image = self.imagesQueue.get(block=True)  # Waiting then image will be available
                if not(isinstance(image, str)) and (image is not None):
                    self.imageWidget.setImage(image, autoLevels=not(self.disableAutoLevelsButton.isChecked()))  # Represent acquired image
                else:
                    print(image)  # Printing substitution of an image by the string

    def continuous_stream(self):
        """
        Handle clicking of Continuous Generation button. Generates continuously and updates the ImageView widget.

        Returns
        -------
        None.

        """
        self.__flagGeneration = not(self.__flagGeneration)  # changing the state of generation
        self.continuousStreamButton.setDown(self.__flagGeneration)  # changing the visible state of button (clicked or not)
        if (self.__flagGeneration):
            # Activate generation or Live imaging
            self.toggleTestPerformance.setDisabled(True)  # Disable the check box for preventing test on during continuous generation
            self.exposureTimeButton.setDisabled(True)  # Disable the exposure time control
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True); self.cameraSelector.setDisabled(True)
            self.snapSingleImgButton.setDisabled(True)
            # Below - initialization of the imported class for continuous image generation
            if self.cameraSelector.currentText() == "Simulated Threaded":
                self.continuousImageGen.start()  # Run the threaded code for continuous updating of showing image
            elif self.cameraSelector.currentText() == "PCO":
                self.messages2Camera.put_nowait("Start Live Stream")  # Send this command to the wrapper class
                imageUpdater = Thread(target=self.update_image, args=())  # assign updating of images to the evoked Thread
                imageUpdater.start()  # start the Thread and assigned to it task

            # Below - activation of save single image button
            if not(self.saveSnapImg.isEnabled()):   # activate saving single image button, since some image generated and displayed
                time.sleep(self.exposureTimeButton.value()/1000)  # wait at least 1 exposure time before activate the saving button
                self.saveSnapImg.setEnabled(True)
        else:
            # Deactivate generation or Live imaging
            if not(self.messages2Camera.full()) and (self.cameraSelector.currentText() == "Simulated Threaded"):
                self.messages2Camera.put_nowait("Stop Generation")  # Send the message to stop continuous generation
                if hasattr(self, "continuousImageGen"):
                    if (self.continuousImageGen.is_alive()):  # if the threaded process is still running
                        self.continuousImageGen.join()  # wait the stop of threaded process ending
                # Drawback of the continuous image generator - upon the receiving the Stop generation command it exits the thread
                # So, re-initialize the simulation camera below
                self.continuousImageGen = ContinuousImageThreadedGenerator(self.imageWidget, self.messages2Camera,
                                                                           self.exposureTimeButton.value(), self.img_height, self.img_width,
                                                                           self.toggleTestPerformance.isChecked(), self.autoRange)
            # For stopping the live stream from the PCO camera - send the command to the wrapper class
            if not(self.messages2Camera.full()) and (self.cameraSelector.currentText() == "PCO"):
                self.messages2Camera.put_nowait("Stop Live Stream")  # Send the message to stop live stream
            # Returns buttons to active state below
            self.snapSingleImgButton.setEnabled(True)
            self.toggleTestPerformance.setEnabled(True); self.exposureTimeButton.setEnabled(True)
            self.widthButton.setEnabled(True); self.heightButton.setEnabled(True); self.cameraSelector.setEnabled(True)

    def update_image(self):
        """
        Update the received from the camera via queue image which is shown on the GUI.

        Returns
        -------
        None.

        """
        timeoutWait = round(6*self.exposureTimeButton.value())  # integer number of 6*exposure time for a single frame
        try:
            image = self.imagesQueue.get(block=True, timeout=(timeoutWait/1000))  # Waiting then image will be
        except Empty:
            print("The first image not acquired, timeout reached")
            image = None
            self.continuousStreamButton.click()  # should deactivate the live stream
        timeoutWait = round(1.5*self.exposureTimeButton.value())  # integer number of 1.5*exposure time for a single frame
        # If instead of image generated only string, then the PCO (or other) camera hasn't been initialized
        # And below the if statement checks that the first image properly acquired
        if not(isinstance(image, str)) and (image is not None) and (isinstance(image, np.ndarray)):
            while(self.__flagGeneration):
                self.imageWidget.setImage(image, autoLevels=not(self.disableAutoLevelsButton.isChecked()))  # Represent acquired image
                # time.sleep(0.01)  # wait 10 ms before querying the new image from the queue
                try:
                    # Waiting then image will be available at least 1.5 of exposure time
                    # image = self.imagesQueue.get(block=True, timeout=(timeoutWait/1000))
                    time.sleep((self.exposureTimeButton.value() + 10)/1000)  # delay before querying new image
                    image = self.imagesQueue.get_nowait()  # non-blocking of the queue request for an image
                except Empty:
                    pass  # just pass the Empty exception if the live stream cancelled in between of waiting time
        elif isinstance(image, str):  # image substitution by the string
            while(self.__flagGeneration):
                print(image)  # Printing substitution of an image by the string
                time.sleep(self.exposureTimeButton.value()/1000)  # wait at least the exposure time
                try:
                    timeoutWait = round(1.5*self.exposureTimeButton.value())  # integer number of 1.5*exposure time for a single frame
                    # Waiting then image will be available at least 1.5 of exposure time
                    image = self.imagesQueue.get(block=True, timeout=(timeoutWait/1000))
                    # time.sleep((self.exposureTimeButton.value() + 10)/1000)
                    # image = self.imagesQueue.get_nowait()
                except Empty:
                    pass  # just pass the Empty exception if the live stream cancelled in between of waiting time

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
        # Below - sending message to stop continuous generation
        if self.cameraSelector.currentText() == "Simulated Threaded":
            if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and self.__flagGeneration:
                self.messages2Camera.put_nowait("Stop Generation")  # Send the message to stop continuous generation
                if hasattr(self, "continuousImageGen"):
                    if (self.continuousImageGen.is_alive()):  # if the threaded generation process is still running
                        self.continuousImageGen.join()  # wait the stop of threaded process ending
        elif self.cameraSelector.currentText() == "PCO":
            if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and hasattr(self, "cameraHandle"):
                if (self.cameraHandle.is_alive()):  # if the threaded associated with the camera process hasn't been finished
                    self.messages2Camera.put_nowait("Close the camera")  # Send the message to stop the imaging and deinitialize the camera
                    self.cameraHandle.join()  # wait the camera closing / deinitializing
        if self.exceptionChecker.is_alive():
            self.exceptionsQueue.put_nowait("Stop Exception Checker")
            self.exceptionChecker.join()
        closeEvent.accept()  # Maybe redundant, but this is explicit accepting quit event (could be refused if asked for example)
        self.applicationHandle.exit()  # Exit the main application, returning to the calling it kernel

    def quitClicked(self):
        """
        Handle the clicking event on the Quit Button.

        Sets the global variables to False state. Waits for threads stop running. Quits the Main window.

        Returns
        -------
        None.

        """
        if self.cameraSelector.currentText() == "Simulated Threaded":
            if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and self.__flagGeneration:
                self.messages2Camera.put_nowait("Stop Generation")  # Send the message to stop continuous generation
                if hasattr(self, "continuousImageGen"):
                    if (self.continuousImageGen.is_alive()):  # if the threaded generation process is still running
                        self.continuousImageGen.join()  # wait the stop of threaded process ending
        elif self.cameraSelector.currentText() == "PCO":
            if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and hasattr(self, "cameraHandle"):
                self.messages2Camera.put_nowait("Close the camera")  # Send the message to stop the imaging and deinitialize the camera:
                if (self.cameraHandle.is_alive()):  # if the threaded associated with the camera process hasn't been finished
                    self.cameraHandle.join()  # wait the camera closing / deinitializing
        if self.exceptionChecker.is_alive():
            self.exceptionsQueue.put_nowait("Stop Exception Checker")
            self.exceptionChecker.join()
        self.close()  # Calls the closing event for QMainWindow

    def imageSizeChanged(self):
        """
        Handle changing of image width or height. Allows to pick up values for single image generation and continuous one.

        Returns
        -------
        None.

        """
        self.img_width = self.widthButton.value(); self.img_height = self.heightButton.value()
        self.imageGenerator = SingleImageThreadedGenerator(self.img_height, self.img_width)  # reinitialize single generator

    def activateMouseOnImage(self):
        """
        Activate or Deactivate mouse handling of images - zooming in / out.

        Returns
        -------
        None.

        """
        self.plot.getViewBox().setMouseEnabled(self.switchMouseCtrlImage.isChecked(), self.switchMouseCtrlImage.isChecked())
        if not(self.switchMouseCtrlImage.isChecked()):  # switched from enabled to disabled state
            self.autoRange = True  # All new images will be auto-centralized
            self.plot.getViewBox().enableAutoRange(enable=True)  # "Centralize" the image by auto range on axes
        else:
            self.autoRange = False

    def saveSingleSnapImg(self):
        """
        Save the current displayed image (but it isn't thread-save!).

        Returns
        -------
        None.

        """
        snap_img = self.imageWidget.getImageItem().image  # get the current displayed image
        composedName = "SnapImage.tiff"; composedPath = os.path.join(os.getcwd(), composedName)  # getting some default image name
        io.imsave(composedPath, snap_img, plugin='tifffile')  # save tiff file by using embedded tifffile plugin

    def putROIonImage(self):
        """
        Put ROI on the displayed image.

        Returns
        -------
        None.

        """
        qrect = QRect(2, 2, self.img_height, self.img_width)  # Bounding of ROI to the image (ROI not movable outside of an image)
        if hasattr(self, "roi"):  # Checking of the class has the attribute already "roi" => some ROI already drawn
            self.removeROI()  # Cleaning existed (drawn) roi from the image for refreshing it
        # ROI will be put in the middle of an image, assuming it has size more than 100 pixels, width and height swapped again
        # Create the ROI object that is non rotatable, removable and expanding only evenly (snapSize and scaleSnap)
        self.roi = pyqtgraph.ROI((self.img_width//2 - self.img_width//20, self.img_height//2 - self.img_height//20),
                                 size=(self.img_width//10, self.img_height//10), snapSize=2.0,
                                 scaleSnap=True, rotatable=False, removable=True, maxBounds=qrect)
        self.plot.addItem(self.roi)  # Add ROI object on the image
        self.roi.sigRemoveRequested.connect(self.removeROI)  # Register handling of removing of ROI
        self.roi.sigRegionChangeFinished.connect(self.roiSizeChanged)
        self.heightROI.setValue(self.img_height//10); self.widthROI.setValue(self.img_width//10)  # Update ROI sizes values on the GUI
        self.cropImageButton.setEnabled(True)  # some ROI specified => cropping is possible

    def removeROI(self):
        """
        Remove created ROI from the displayed image.

        Returns
        -------
        None.

        """
        self.plot.removeItem(self.roi)  # Remove added roi
        self.cropImageButton.setDisabled(True)  # cropiing is impossible

    def roiSizesSpecified(self):
        """
        Adjust input value for preventing odd values specification within the QSpinBoxes.

        Returns
        -------
        None.

        """
        if self.heightROI.value() % 2 != 0:
            self.heightROI.setValue(self.heightROI.value()+1)
        if self.widthROI.value() % 2 != 0:
            self.widthROI.setValue(self.widthROI.value()+1)
        if hasattr(self, "roi"):
            self.roi.setSize((self.widthROI.value(), self.heightROI.value()))

    def roiSizeChanged(self):
        """
        Transfer changed by the user sizes of ROI on the image to the buttons representing them on GUI.

        Returns
        -------
        None.

        """
        (w, h) = self.roi.size(); w = int(w); h = int(h)
        self.widthROI.setValue(w); self.heightROI.setValue(h)

    def generateMessageWithException(self):
        """
        Put the exception into the Queue used for communication with the camera for testing quit/stop handling.

        Returns
        -------
        None.

        """
        if not(self.exceptionsQueue.full()):
            self.exceptionsQueue.put_nowait(Exception("Generated Test Exception"))

    def exposureTimeChanged(self):
        """
        Handle changing of exposure time by the user.

        Returns
        -------
        None.

        """
        if self.cameraSelector.currentText() == "PCO":
            # below - send the tuple with string command and exposure time value
            self.messages2Camera.put_nowait(("Set exposure time", self.exposureTimeButton.value()))
        elif self.cameraSelector.currentText() == "Simulated Threaded":
            # Re-initialize the instance of continuous image generator with new exposure time
            self.continuousImageGen = ContinuousImageThreadedGenerator(self.imageWidget, self.messages2Camera,
                                                                       self.exposureTimeButton.value(), self.img_height, self.img_width,
                                                                       self.toggleTestPerformance.isChecked(), self.autoRange)

    def activeCameraChanged(self):
        """
        Handle changing of active (selected) camera.

        Returns
        -------
        None.

        """
        if self.cameraSelector.currentText() == "PCO":
            pyqtgraph.setConfigOptions(imageAxisOrder='row-major')  # Set for conforming with images from the camera
            # No need to deinitialize the simulated camera now
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True)
            self.snapSingleImgButton.setDisabled(True); self.continuousStreamButton.setDisabled(True)
            self.generateException.setVisible(False)  # Remove button for testing of handling of generated Exceptions
            # Changing the titles of the buttons for controlling getting the images (from the camera or generated ones)
            self.snapSingleImgButton.setText("Single Snap Image"); self.continuousStreamButton.setText("Live Stream")
            # Initialize the PCO camera
            self.cameraHandle = PCOcamera(self.messages2Camera, self.exceptionsQueue, self.imagesQueue, self.exposureTimeButton.value())
            if not(self.cameraHandle.is_alive()):
                self.cameraHandle.start()   # Start the main loop for receiving the commands
            self.snapSingleImgButton.setEnabled(True); self.continuousStreamButton.setEnabled(True)
        elif self.cameraSelector.currentText() == "Simulated Threaded":
            pyqtgraph.setConfigOptions(imageAxisOrder='col-major')  # Set for conforming with initial development
            if (hasattr(self, "cameraHandle")):  # Check that the PCO camera was initialized
                if self.cameraHandle.is_alive():  # Check that the PCO camera is still running in the separate thread
                    self.messages2Camera.put_nowait("Close the camera")  # for stopping the imaging and deinitialize the camera
                    self.cameraHandle.join()  # wait the camera closing / deinitializing
            self.widthButton.setEnabled(True); self.heightButton.setEnabled(True)
            if not(self.generateException.isVisible()):  # return the visibility of the button
                self.generateException.setVisible(True)
            self.snapSingleImgButton.setText("Generate Single Pic"); self.continuousStreamButton.setText("Continuous Generation")

    def cropImage(self):
        """
        Crop the selected ROI and send it this region for imaging.

        Returns
        -------
        None.

        """
        # Implementation for the simulated camera
        if self.cameraSelector.currentText() == "Simulated Threaded":
            self.img_height = self.heightROI.value(); self.img_width = self.widthROI.value()
            self.imageGenerator = SingleImageThreadedGenerator(self.img_height, self.img_width); self.snapSingleImgButton.click()
            self.widthButton.setValue(self.widthROI.value()); self.heightButton.setValue(self.heightROI.value())
            if self.__flagGeneration:
                # Refreshing of continuous generation by the clicking on the button for evoking continuous stream - below
                self.continuousStreamButton.click(); time.sleep(0.05); self.continuousStreamButton.click()
            else:
                # Reinitialize the simulation class below for the getting new image sizes
                self.continuousImageGen = ContinuousImageThreadedGenerator(self.imageWidget, self.messages2Camera,
                                                                           self.exposureTime.value(), self.img_height, self.img_width,
                                                                           self.toggleTestPerformance.isChecked(), self.autoRange)
        # Implementation for the actual camera
        elif self.cameraSelector.currentText() == "PCO":
            if not(self.messages2Camera.full()):
                # Message in the format (sting_command, (paramaters))
                (y, x) = self.roi.pos()  # get the ROI's origin
                y = int(y); x = int(x)
                self.messages2Camera.put_nowait(("Crop Image", (y, x, self.heightROI.value(), self.widthROI.value())))

        self.removeROI()  # remove the ROI because it was used for cropping and not actual anymore
        self.restoreFullImgButton.setEnabled(True)  # Allowing the restoration of an image

    def restoreFullImage(self):
        """
        Restore full frame image (predefined call for camera simulation and the internal property of the camera).

        Returns
        -------
        None.

        """
        # Implementation for the simulated camera
        if self.cameraSelector.currentText() == "Simulated Threaded":
            self.img_height = self.img_height_default; self.img_width = self.img_width_default
            self.widthButton.setValue(self.img_width_default); self.heightButton.setValue(self.img_height_default)
            self.imageGenerator = SingleImageThreadedGenerator(self.img_height, self.img_width); self.snapSingleImgButton.click()
            if self.__flagGeneration:
                # Refreshing of continuous generation by the clicking on the button for evoking continuous stream - below
                self.continuousStreamButton.click(); time.sleep(0.05); self.continuousStreamButton.click()
            else:
                # Reinitialize the simulation class below for the getting new image sizes
                self.continuousImageGen = ContinuousImageThreadedGenerator(self.imageWidget, self.messages2Camera, self.exposureTime.value(),
                                                                           self.img_height_default, self.img_width_default,
                                                                           self.toggleTestPerformance.isChecked(), self.autoRange)
        # Implementation for the actual camera
        elif self.cameraSelector.currentText() == "PCO":
            if not(self.messages2Camera.full()):
                self.messages2Camera.put_nowait("Restore Full Frame")
        self.restoreFullImgButton.setDisabled(True)

    def disableAxesOnImage(self):
        """
        Disable / enable representation of axes of generated / acquired images.

        Returns
        -------
        None.

        """
        self.imageWidget.getView().showAxis("left", show=not(self.disableAxesOnImageButton.isChecked()))
        self.imageWidget.getView().showAxis("bottom", show=not(self.disableAxesOnImageButton.isChecked()))

    def disableAutoLevelsCalculation(self):
        if self.disableAutoLevelsButton.isChecked():
            self.imageWidget.ui.histogram.hide()  # hide the side histogram with sliders
            self.imageWidget.getImageItem().setLevels([self.minLevelButton.value(), self.maxLevelButton.value()], update=False)
            # if self.cameraSelector.currentText() == "Simulated Threaded":
            #     # set autolevels for a generated U8 image
            #     self.imageWidget.setLevels(min=self.minLevelButton.value(), max=self.maxLevelButton.value())
        else:
            self.imageWidget.ui.histogram.show()

    def histogramLevelsChanged(self):
        levels = self.imageWidget.ui.histogram.getLevels()
        minPixelLevel = int(round(levels[0], 0)); maxPixelLevel = int(round(levels[1], 0))
        self.minLevelButton.setValue(minPixelLevel); self.maxLevelButton.setValue(maxPixelLevel)

    def pixelValuesChanged(self):
        if self.disableAutoLevelsButton.isChecked():
            if self.minLevelButton.value() > self.maxLevelButton.value():
                self.maxLevelButton.setValue(self.minLevelButton.value() + 1)
            self.imageWidget.getImageItem().setLevels([self.minLevelButton.value(), self.maxLevelButton.value()], update=False)


# %% Tests
if __name__ == "__main__":
    my_app = QApplication([])  # application without any command-line arguments
    # my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
    main_window = SimUscope(width_default, height_default, my_app); main_window.show()
    my_app.exec()  # Execute the application in the main kernel

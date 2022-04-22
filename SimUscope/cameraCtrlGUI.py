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
from queue import Empty
from multiprocessing import Queue
from cameraCtrlWrapper import CameraWrapper
import os
from skimage import io
from checkExceptionsMessagesLive import CheckMessagesForExceptions, MessagesPrinter
from threading import Thread

# %% Some default values, used for the initialization of GUI
width_default = 2000; height_default = 2000  # Default width and height for generation of images


# %% GUI controls
class SimUscope(QMainWindow):
    """Create the GUI with buttons for testing image acquisition from the camera and making some image processing."""

    wait_multiplicator: int
    __flagLiveStream = False  # Private class variable for recording state of continuous generation
    __flagTestPerformance = False  # Private class variable for switching between test state by using
    autoRange = True  # flag for handling user preference about possibility to zoom in/out to images
    __flagRealCameraInitialized = False  # For further usage it for making GUI more responsive and stable
    globalTimeout = 1  # in seconds, to wait for all join() commands with processes

    # noinspection PyUnresolvedReferences
    def __init__(self, img_height, img_width, applicationHandle: QApplication):
        """Create overall UI inside the QMainWindow widget."""
        super().__init__()
        self.imageUpdater = None  # Added suggested definition at initialization stage
        self.roi = None
        self.cameraHandle = None  # Added suggested definition at initialization stage
        pyqtgraph.setConfigOptions(imageAxisOrder='row-major')  # Set for conforming with images from the camera
        self.applicationHandle = applicationHandle  # handle to the main application for exit it in the appropriate place
        self.messages2Camera = Queue(maxsize=10)  # Initialize message queue for communication with the camera (simulated or not)
        self.messagesFromCamera = Queue(maxsize=10)  # Initialize message queue for listening messages from the camera
        self.exceptionsQueue = Queue(maxsize=5)  # Initialize separate queue for spreading and handling Exceptions occurred within modules
        self.imagesQueue = Queue(maxsize=40)  # Initialize the queue for holding acquired images and accessing them from the GUI thread
        self.gui_refresh_rate_ms = 10  # The constant time pause between each attempt to retrieve the image
        self.wait_multiplicator = 2  # The amount for calculation of timeout for retrieving image from the imagesQueue
        self.img_height = img_height; self.img_width = img_width
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
        self.cameraSelector = QComboBox(); self.cameraSelector.addItems(["Simulated", "PCO"])
        # self.cameraSelector.addItems(["Simulated Threaded", "PCO", "PCO Process"])
        self.cameraSelector.setCurrentText("PCO")  # Default camera for initialization - the simulated one
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
        self.widthROI.setKeyboardTracking(False); self.heightROI.setKeyboardTracking(False)  # Disable instant tracking of any changes
        # Push buttons for events evoking
        self.snapSingleImgButton = QPushButton("Generate Single Pic"); self.snapSingleImgButton.clicked.connect(self.snap_single_img)
        self.continuousStreamButton = QPushButton("Continuous Generation")  # Switches on/off continuous generation
        self.toggleTestPerformance = QCheckBox("Test Performance"); self.toggleTestPerformance.setEnabled(True)
        self.toggleTestPerformance.setChecked(False)  # setChecked - set the state of a button
        self.continuousStreamButton.clicked.connect(self.continuous_stream); self.continuousStreamButton.setCheckable(True)
        self.disableAxesOnImageButton = QPushButton("Disable axes on image"); self.disableAxesOnImageButton.setCheckable(True)
        self.disableAxesOnImageButton.clicked.connect(self.disableAxesOnImage)
        self.disableAutoLevelsButton = QPushButton("Disable pixels leveling"); self.disableAutoLevelsButton.setCheckable(True)
        self.disableAutoLevelsButton.clicked.connect(self.disableAutoLevelsCalculation)
        self.exposureTimeButton = QSpinBox(); self.exposureTimeButton.setSingleStep(1); self.exposureTimeButton.setSuffix(" ms")
        self.exposureTimeButton.setPrefix("Exposure time: "); self.exposureTimeButton.setMinimum(1); self.exposureTimeButton.setMaximum(1000)
        self.exposureTimeButton.setValue(100); self.exposureTimeButton.adjustSize()
        self.checkPCOcameraStatus = QPushButton("Check camera status"); self.checkPCOcameraStatus.setVisible(False)
        self.checkPCOcameraStatus.clicked.connect(self.checkCameraStatus)
        self.exposureTimeButton.setKeyboardTracking(False)  # !!! special function to disable emitting of signals for each typed value
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
        grid.addWidget(self.checkPCOcameraStatus, 7, 6, 1, 1)
        # Set valueChanged event handlers
        self.widthButton.valueChanged.connect(self.imageSizeChanged); self.heightButton.valueChanged.connect(self.imageSizeChanged)
        # ImageWidget should be central - for better representation of generated images
        grid.addWidget(self.imageWidget, 1, 0, 6, 6)  # the ImageView widget spans on ... rows and ... columns (2 values in the end)
        self.setCentralWidget(self.qwindow)  # Actually, allows to make both buttons and ImageView visible
        # Initialize and start the Exception checker and associate it with the initialized Quit button
        self.exceptionChecker = CheckMessagesForExceptions(self.exceptionsQueue, self.quitButton, period_checks_ms=30)
        self.messagesPrinter = MessagesPrinter(self.messagesFromCamera, period_checks_ms=40)
        self.exceptionChecker.start(); self.messagesPrinter.start()  # Start the Exception Checker and Messages Printer
        self.initializeCamera()  # Initializes default camera defined by the default value of cameraSelector button

    def initializeCamera(self):
        """
        Initialize the default (selected above in the __init__ method) camera.

        Returns
        -------
        None.

        """
        # Initialize the Camera class
        self.cameraHandle = CameraWrapper(self.messages2Camera, self.exceptionsQueue, self.imagesQueue, self.messagesFromCamera,
                                          self.exposureTimeButton.value(), self.img_width_default, self.img_height_default,
                                          camera_type=self.cameraSelector.currentText())
        self.cameraHandle.start()  # start the main loop of the controlling class
        # PCO camera demands long time for initialization, if below - for waiting it to finish everything
        if self.cameraSelector.currentText() == "PCO":
            # Below - the flag to delay this code to wait of confirmation that camera initialized
            received_initialization_confirmation = False
            message = "-"  # empty message
            # The loop is waiting for notification about the camera initialization
            while not received_initialization_confirmation:
                try:
                    message = self.imagesQueue.get_nowait()
                    if (message == "The PCO camera initialized") or (message == "The Simulated PCO camera initialized"):
                        if message == "The PCO camera initialized":
                            self.__flagRealCameraInitialized = True   # Internal flag for further usage
                        received_initialization_confirmation = True; break
                except Empty:
                    pass
                time.sleep(0.08)  # sleep before checks for receiving the confirmation about initialization
            print("Confirmation received:", message)
            self.checkPCOcameraStatus.setVisible(True)
        # Below - associated with the selected camera peculiarities
        if self.cameraSelector.currentText() == "PCO":
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True)  # PCO camera cannot support arbitrary size changes
            self.generateException.setVisible(False)  # Remove button for testing of handling of generated Exceptions
            # Changing the titles of the buttons for controlling getting the images (from the camera or generated ones)
            self.snapSingleImgButton.setText("Single Snap Image"); self.continuousStreamButton.setText("Live Stream")
            self.checkPCOcameraStatus.setVisible(True)
        elif self.cameraSelector.currentText() == "Simulated":
            self.checkPCOcameraStatus.setVisible(False)
            self.widthButton.setEnabled(True); self.heightButton.setEnabled(True)
            if not(self.generateException.isVisible()):  # return the visibility of the button
                self.generateException.setVisible(True)
            self.snapSingleImgButton.setText("Generate Single Pic"); self.continuousStreamButton.setText("Continuous Generation")

    def activeCameraChanged(self):
        """
        Handle changing of active (selected) camera.

        Returns
        -------
        None.

        """
        # General changing of type of the camera
        # Deinitialize previous initialized process bundled with the Camera class
        if self.cameraHandle.is_alive():
            self.messages2Camera.put_nowait("Close the camera")
            self.cameraHandle.join(timeout=self.globalTimeout)  # wait for exiting from the evoked process
        print("Previous camera deinitialized")
        # Initialize the Camera class with the selected camera type by the user (button)
        self.snapSingleImgButton.setDisabled(True); self.continuousStreamButton.setDisabled(True)  # Cannot be used before camera initialized
        self.cameraHandle = CameraWrapper(self.messages2Camera, self.exceptionsQueue, self.imagesQueue, self.messagesFromCamera,
                                          self.exposureTimeButton.value(), self.img_width_default, self.img_height_default,
                                          camera_type=self.cameraSelector.currentText())
        self.cameraHandle.start()   # Start the main loop in the Camera for receiving the commands
        # PCO camera demands long time for initialization, if below - for waiting it to finish everything
        if self.cameraSelector.currentText() == "PCO":
            # Below - the flag to delay this code to wait of confirmation that camera initialized
            received_initialization_confirmation = False
            message = "-"  # empty message
            # The loop is waiting for notification about the camera initialization
            while not received_initialization_confirmation:
                try:
                    message = self.imagesQueue.get_nowait()
                    if (message == "The PCO camera initialized") or (message == "The Simulated PCO camera initialized"):
                        if message == "The PCO camera initialized":
                            self.__flagRealCameraInitialized = True   # Internal flag for further usage
                        received_initialization_confirmation = True; break
                except Empty:
                    pass
                time.sleep(0.08)  # sleep before checks for receiving the confirmation about initialization
            print("Confirmation received:", message)
        self.snapSingleImgButton.setEnabled(True); self.continuousStreamButton.setEnabled(True)
        # Below - associated with the selected camera pecularities
        if self.cameraSelector.currentText() == "PCO":
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True)  # PCO camera cannot support arbitrary size changes
            self.generateException.setVisible(False)  # Remove button for testing of handling of generated Exceptions
            # Changing the titles of the buttons for controlling getting the images (from the camera or generated ones)
            self.snapSingleImgButton.setText("Single Snap Image"); self.continuousStreamButton.setText("Live Stream")
            self.checkPCOcameraStatus.setVisible(True)
        elif self.cameraSelector.currentText() == "Simulated":
            self.checkPCOcameraStatus.setVisible(False)
            self.widthButton.setEnabled(True); self.heightButton.setEnabled(True)
            if not(self.generateException.isVisible()):  # return the visibility of the button
                self.generateException.setVisible(True)
            self.snapSingleImgButton.setText("Generate Single Pic"); self.continuousStreamButton.setText("Continuous Generation")

    def snap_single_img(self):
        """
        Handle clicking of Generate Single Picture. This method updates the image associated with ImageView widget.

        Returns
        -------
        None.

        """
        if not(self.messages2Camera.full()):
            self.messages2Camera.put_nowait("Snap single image")  # Send the command for acquiring single image
            timeoutWait = round((3*self.wait_multiplicator)*self.exposureTimeButton.value())  # timeout to wait the image on the imagesQueue
            try:
                image = self.imagesQueue.get(block=True, timeout=(timeoutWait/1000))  # Waiting then image will be available
            except Empty:
                image = None
                print("The snap image not acquired, timeout reached")
            if not(isinstance(image, str)) and (image is not None):
                self.imageWidget.setImage(image, autoLevels=not(self.disableAutoLevelsButton.isChecked()))  # Represent acquired image
            if isinstance(image, str):
                print("Image: ", image)  # replacer for PCO simulation

    def continuous_stream(self):
        """
        Handle clicking of Continuous Generation button. Generates continuously and updates the ImageView widget.

        Returns
        -------
        None.

        """
        self.__flagLiveStream = not self.__flagLiveStream  # changing the state of generation
        self.continuousStreamButton.setDown(self.__flagLiveStream)  # changing the visible state of button (clicked or not)
        if self.__flagLiveStream:
            # Activate generation or Live imaging
            self.toggleTestPerformance.setDisabled(True)  # Disable the checkbox for preventing test on during continuous generation
            self.exposureTimeButton.setDisabled(True)  # Disable the exposure time control
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True); self.cameraSelector.setDisabled(True)
            self.snapSingleImgButton.setDisabled(True)
            # Below - initialization of the imported class for continuous image generation
            self.messages2Camera.put_nowait("Start Live Stream")  # Send this command to the wrapper class
            self.imageUpdater = Thread(target=self.update_image, args=())  # assign updating of images to the evoked Thread
            self.imageUpdater.start()  # start the Thread and assigned to it task
            # Below - activation of save single image button
            if not(self.saveSnapImg.isEnabled()):   # activate saving single image button, since some image generated and displayed
                time.sleep(self.exposureTimeButton.value()/1000)  # wait at least 1 exposure time before activate the saving button
                self.saveSnapImg.setEnabled(True)
        else:
            # For stopping the live stream from the Camera - send the command to it through messages queue
            if not(self.messages2Camera.full()):
                self.messages2Camera.put_nowait("Stop Live Stream")  # Send the message to stop live stream
            # Return buttons to active state below
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
        timeoutWait = round((self.wait_multiplicator*5)*self.exposureTimeButton.value())  # timeout to wait the first image - wait more
        try:
            image = self.imagesQueue.get(block=True, timeout=(timeoutWait/1000))  # Waiting then image will be
        except Empty:
            print("The first image not acquired, timeout reached")
            image = None
            self.continuousStreamButton.click()  # should deactivate the live stream
        timeoutWait = round(self.wait_multiplicator*self.exposureTimeButton.value())  # timeout to wait the image on the imagesQueue
        # For assessing the performance - calculate the mean time of representation
        if self.toggleTestPerformance.isChecked():
            measured_passed_t = np.zeros(101, dtype='int')
        # If instead of image generated only string, then the PCO (or other) camera hasn't been initialized
        # And below the if statement checks that the first image properly acquired
        if not(isinstance(image, str)) and (image is not None) and (isinstance(image, np.ndarray)):
            if self.toggleTestPerformance.isChecked():
                j = 0  # index for putting the measured times into the array for mean passed time calculation
            flagGetNewTime = True  # If the image not retrieved from the queue, not refresh the time, keep counting from the attempt
            while self.__flagLiveStream:
                if self.toggleTestPerformance.isChecked() and flagGetNewTime:
                    t1 = time.time()
                self.imageWidget.setImage(image, autoLevels=not(self.disableAutoLevelsButton.isChecked()))  # Represent acquired image
                try:
                    # Waiting then image will be available at least some more time than exposure time
                    image = self.imagesQueue.get(block=True, timeout=(timeoutWait/1000))
                    flagGetNewTime = True  # For refreshing entry time measurement t1
                except Empty:
                    flagGetNewTime = False  # keep the time measured above for accounting the performance
                    continue  # proceed to the next attempt to retrieve the image from the queue
                # Attempt to make the GUI more responsive to the user input => providing artificial delays for button (simulation mode)
                if not self.__flagRealCameraInitialized:
                    # Simulation and recalculate min/max pixel value for each image => bigger delays
                    if (self.exposureTimeButton.value() < 50) and not(self.disableAutoLevelsButton.isChecked()):
                        time.sleep(0.015)  # freeze this loop for some ms if exposure time is low and autoLevels are active
                    elif (self.exposureTimeButton.value() < 25) and not(self.disableAutoLevelsButton.isChecked()):
                        time.sleep(0.025)  # freeze this loop for some ms if exposure time is low and autoLevels are active
                    elif (self.exposureTimeButton.value() < 10) and not(self.disableAutoLevelsButton.isChecked()):
                        time.sleep(0.035)  # freeze this loop for some ms if exposure time is low and autoLevels are active
                    elif (self.exposureTimeButton.value() < 5) and not(self.disableAutoLevelsButton.isChecked()):
                        time.sleep(0.045)  # freeze this loop for some ms if exposure time is low and autoLevels are active
                    # Simulation and assign the specified min/max pixel values for each image => fewer delays
                    elif (self.exposureTimeButton.value() < 25) and (self.disableAutoLevelsButton.isChecked()):
                        time.sleep(0.005)  # freeze this loop for some ms if exposure time is low and autoLevels are active
                    elif (self.exposureTimeButton.value() < 10) and (self.disableAutoLevelsButton.isChecked()):
                        time.sleep(0.010)  # freeze this loop for some ms if exposure time is low and autoLevels are active
                    elif (self.exposureTimeButton.value() < 5) and (self.disableAutoLevelsButton.isChecked()):
                        time.sleep(0.015)  # freeze this loop for some ms if exposure time is low and autoLevels are active
                # Another attempt to make more responsive GUI if using the real camera as the controlling device
                if self.__flagRealCameraInitialized:
                    if 200 > self.exposureTimeButton.value() >= 100:
                        time.sleep(1/1000)  # delays in ms in explicit form conversion to seconds!
                    elif 100 > self.exposureTimeButton.value() >= 50:
                        time.sleep(2/1000)  # delays in ms in explicit form conversion to seconds!
                    elif 50 > self.exposureTimeButton.value() >= 25:
                        time.sleep(4/1000)  # delays in ms in explicit form conversion to seconds!
                    elif 25 > self.exposureTimeButton.value() >= 10:
                        time.sleep(8/1000)  # delays in ms in explicit form conversion to seconds!
                    elif self.exposureTimeButton.value() < 10:
                        time.sleep(12/1000)  # delays in ms in explicit form conversion to seconds!
                # Below - recording the passed time for calculation of mean passed time after
                if self.toggleTestPerformance.isChecked():
                    # Putting the measured times in Ring buffer manner
                    t2 = time.time(); measured_passed_t[j] = np.round(((t2-t1)*1000), 0)
                    if j < np.size(measured_passed_t)-1:
                        j += 1
                    else:
                        j = 0
            # Ending the while loop => the measured performance could be calculated
            if self.toggleTestPerformance.isChecked():
                for i in range(np.size(measured_passed_t)):
                    if measured_passed_t[i] == 0:
                        break
                measured_passed_t = measured_passed_t[0:i]  # remove all zero values
                print("Mean time for image refreshing", int(np.round(np.mean(measured_passed_t), 0)), "ms")
        # If the image coming from the process as the string (PCO simulated), then just print it below
        elif isinstance(image, str):  # image substitution by the string
            while self.__flagLiveStream:
                print(image)  # Printing substitution of an image by the string
                time.sleep(self.exposureTimeButton.value()/1000)  # wait at least the exposure time
                try:
                    # Waiting then image will be available at least 1.5 of exposure time
                    image = self.imagesQueue.get(block=True, timeout=(timeoutWait/1000))
                except Empty:
                    pass  # just pass the Empty exception if the live stream cancelled in between of waiting time
                # Making the GUI more responsive for the low exposure times
                if self.exposureTimeButton.value() < 10:
                    time.sleep(0.01)  # add some overhead for better checking of the user input

    def imageSizeChanged(self):
        """
        Handle changing of image width or height. Allows picking up values for single image generation and continuous one.

        Returns
        -------
        None.

        """
        self.img_width = self.widthButton.value(); self.img_height = self.heightButton.value()
        if self.cameraSelector.currentText() == "Simulated":
            self.messages2Camera.put_nowait(("Change simulate picture sizes to:", (self.img_width, self.img_height)))

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
        qrect = QRect(2, 2, self.img_height, self.img_width)  # Bounding of ROI to the image (ROI not movable outside an image)
        if hasattr(self, "roi"):  # Checking of the class has the attribute already "roi" => some ROI already drawn
            self.removeROI()  # Cleaning existed (drawn) roi from the image for refreshing it
        # ROI will be put in the middle of an image, assuming it has size more than 100 pixels, width and height swapped again
        # Create the ROI object that is non-rotatable, removable and expanding only evenly (snapSize and scaleSnap)
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
        self.cropImageButton.setDisabled(True)  # cropping is impossible

    def roiSizesSpecified(self):
        """
        Adjust input value for preventing odd values specification within the QSpinBoxes.

        Returns
        -------
        None.

        """
        if self.heightROI.value() % 2 != 0:  # assuming that ROI size selection (height) can be only even number
            self.heightROI.setValue(self.heightROI.value()+1)
        if self.widthROI.value() % 2 != 0:  # assuming that ROI size selection (width) can be only even number
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
        self.widthROI.setValue(w); self.heightROI.setValue(h)  # assign selected by the ROI ctrl sizes to buttons

    def generateMessageWithException(self):
        """
        Put the exception into the Queue used for communication with the camera for testing quit/stop handling.

        Returns
        -------
        None.

        """
        if not(self.exceptionsQueue.full()):
            # Generate Exceptiom for testing its handling and quit() function
            self.exceptionsQueue.put_nowait(Exception("Generated Test Exception"))

    def exposureTimeChanged(self):
        """
        Handle changing of exposure time by the user.

        Returns
        -------
        None.

        """
        # below - send the tuple with string command and exposure time value
        self.messages2Camera.put_nowait(("Set exposure time", self.exposureTimeButton.value()))
        # Assign multiplicator for letting the software wait more for the image depending on the exposure time:
        if self.exposureTimeButton.value() <= 50:
            self.wait_multiplicator = 3
        if self.exposureTimeButton.value() <= 25:
            self.wait_multiplicator = 4
        if self.exposureTimeButton.value() <= 10:
            self.wait_multiplicator = 5
        if self.__flagRealCameraInitialized:  # ??? if it helps to acquire the images more stable
            self.wait_multiplicator += 2

    def cropImage(self):
        """
        Crop the selected ROI and send it this region for imaging.

        Returns
        -------
        None.

        """
        # Sending the image and height as the tuple along with the command to the camera
        self.img_height = self.heightROI.value(); self.img_width = self.widthROI.value()
        self.widthButton.setValue(self.widthROI.value()); self.heightButton.setValue(self.heightROI.value())
        # Stop Live Stream if it's active - below
        if self.__flagLiveStream:
            self.continuousStreamButton.click()  # Should disable automatically Live Stream
            time.sleep(self.exposureTimeButton.value()/1000)  # wait (main thread) at least the image finish acquiring
            if self.imageUpdater.is_alive():
                self.imageUpdater.join((2*self.exposureTimeButton.value())/1000)  # wait the image updater thread stopped
        # Implementation for the actual camera
        if not(self.messages2Camera.full()):
            # Message in the format (sting_command, (parameters))
            (y, x) = self.roi.pos()  # get the ROI's origin
            y = int(y); x = int(x)
            self.messages2Camera.put_nowait(("Crop Image", (y, x, self.widthROI.value(), self.heightROI.value())))
        self.removeROI()  # remove the ROI because it was used for cropping and not actual anymore
        self.restoreFullImgButton.setEnabled(True)  # Allowing the restoration of an image

    def restoreFullImage(self):
        """
        Restore full frame image (predefined call for camera simulation and the internal property of the camera).

        Returns
        -------
        None.

        """
        # Assign appropriate values to the buttons and send the commands
        self.img_height = self.img_height_default; self.img_width = self.img_width_default
        self.widthButton.setValue(self.img_width_default); self.heightButton.setValue(self.img_height_default)
        # Stop Live Stream if it's active - below
        if self.__flagLiveStream:
            self.continuousStreamButton.click()  # Should disable automatically Live Stream
            time.sleep(self.exposureTimeButton.value()/1000)  # wait (main thread) at least the image finish acquiring
            if self.imageUpdater.is_alive():
                self.imageUpdater.join((2*self.exposureTimeButton.value())/1000)  # wait the image updater thread stopped
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
        # Disable showing the axes on the image
        self.imageWidget.getView().showAxis("left", show=not(self.disableAxesOnImageButton.isChecked()))
        self.imageWidget.getView().showAxis("bottom", show=not(self.disableAxesOnImageButton.isChecked()))

    def disableAutoLevelsCalculation(self):
        """
        Disable auto-leveling of pixel values of shown on the main GUI window image.

        Returns
        -------
        None.

        """
        if self.disableAutoLevelsButton.isChecked():  # isChecked() = Button pressed (True state)
            self.imageWidget.ui.histogram.hide()  # hide the side histogram with sliders
            self.imageWidget.getImageItem().setLevels([self.minLevelButton.value(), self.maxLevelButton.value()], update=False)
        else:
            self.imageWidget.ui.histogram.show()  # restore the histogram bar and controls

    def histogramLevelsChanged(self):
        """
        Handle changing on the GUI the dragging pixel levels (minimum and maximum ones).

        Returns
        -------
        None.

        """
        # Assign the selected on histogram bar values to the buttons
        levels = self.imageWidget.ui.histogram.getLevels()
        minPixelLevel = int(round(levels[0], 0)); maxPixelLevel = int(round(levels[1], 0))
        self.minLevelButton.setValue(minPixelLevel); self.maxLevelButton.setValue(maxPixelLevel)

    def pixelValuesChanged(self):
        """
        Handle manual input of minimal or maximal pixel value on the GUI.

        Returns
        -------
        None.

        """
        if self.disableAutoLevelsButton.isChecked():
            if self.minLevelButton.value() > self.maxLevelButton.value():
                self.maxLevelButton.setValue(self.minLevelButton.value() + 1)
            self.imageWidget.getImageItem().setLevels([self.minLevelButton.value(), self.maxLevelButton.value()], update=False)

    def checkCameraStatus(self):
        """
        Ask for the current camera status for the real PCO camera.

        Returns
        -------
        None.

        """
        if self.__flagRealCameraInitialized:
            self.messages2Camera.put_nowait("Get the PCO camera status")

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
        # Stop the camera - below
        if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and hasattr(self, "cameraHandle"):
            if self.__flagLiveStream:
                self.continuousStreamButton.click()   # Emulate disactivation of the Live Stream by clicking the button
                time.sleep((4*self.exposureTimeButton.value())/1000)  # wait (main thread) at least the image finish acquiring
                if self.imageUpdater.is_alive():
                    self.imageUpdater.join(timeout=((2*self.exposureTimeButton.value())/1000))  # wait the image updater thread stopped
                    print("Image Updater stopped")
            self.messages2Camera.put_nowait("Stop Program")  # Send the message to stop the imaging and deinitialize the camera:
            if self.cameraHandle is not None:
                if self.cameraHandle.is_alive():  # if the threaded associated with the camera process hasn't been finished
                    self.cameraHandle.join(timeout=self.globalTimeout)  # wait the camera closing / deinitializing
                    print("Camera process released")
        # Stop Exceptions checker and Messages Printer
        if self.exceptionChecker.is_alive():
            self.exceptionsQueue.put_nowait("Stop Exception Checker")
            self.exceptionChecker.join()
            print("Exception checker stopped")
        if self.messagesPrinter.is_alive():
            self.messagesFromCamera.put_nowait("Stop Messages Printer")
            self.messagesPrinter.join()
            print("Messages Printer stopped")
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
        if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and hasattr(self, "cameraHandle"):
            # __flagLiveStream - if true, then the live stream is on
            if self.__flagLiveStream:
                self.continuousStreamButton.click()   # Emulate disactivation of the Live Stream by clicking the button
                time.sleep((4*self.exposureTimeButton.value())/1000)  # wait (main thread) at least the image finish acquiring
                if self.imageUpdater.is_alive():
                    self.imageUpdater.join(timeout=((2*self.exposureTimeButton.value())/1000))  # wait the image updater thread stopped
                    print("Image Updater stopped")
            self.messages2Camera.put_nowait("Close the camera")  # Send the message to stop the imaging and deinitialize the camera:
            if self.cameraHandle.is_alive():  # if the threaded associated with the camera process hasn't been finished
                self.cameraHandle.join(timeout=self.globalTimeout)  # wait the camera closing / deinitializing
                self.cameraHandle = None  # explicitly setting the handle to None for preventing again checking if it's alive
                print("Camera process released")
        if self.exceptionChecker.is_alive():
            self.exceptionsQueue.put_nowait("Stop Exception Checker")
            self.exceptionChecker.join()
            print("Exception checker stopped")
        if self.messagesPrinter.is_alive():
            self.messagesFromCamera.put_nowait("Stop Messages Printer")
            self.messagesPrinter.join()
            print("Messages Printer stopped")
        self.close()  # Calls the closing event for QMainWindow


# %% GUI Launch
if __name__ == "__main__":
    my_app = QApplication([])  # application without any command-line arguments
    # my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
    main_window = SimUscope(width_default, height_default, my_app); main_window.show()
    my_app.exec()  # Execute the application in the main kernel

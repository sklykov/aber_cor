# -*- coding: utf-8 -*-
"""
Module for import into the camera GUI for performing periodically checks that there are no exceptions happened and for printing messages.

@author: ssklykov
"""
# %% Imports
from threading import Thread
from queue import Queue, Empty
import time
from multiprocessing import Queue as ProcessQueue


# %% Exception checker
class CheckMessagesForExceptions(Thread):
    """
    Threaded class for continuous and independent loop running for checking that any Exception reported anythere in the GUI program.

    If any exception is caught, then the Quit or Exit button will be clicked.
    """

    def __init__(self, messages_queue: Queue, quitButton, period_checks_ms: int = 100):
        self.messages_queue = messages_queue; self.period_checks_ms = period_checks_ms
        if self.period_checks_ms < 5:
            self.period_checks_ms = 5  # minimal delay between checks = 5 ms
        self.quitButton = quitButton
        Thread.__init__(self)

    def run(self):
        """
        Check constantly in the while loop the Queue for presence of exceptions and if found, call the "Quit" function of the main window.

        Returns
        -------
        None.

        """
        running = True; quitMainProgram = False
        while running:
            if not(self.messages_queue.empty()) and (self.messages_queue.qsize() > 0):
                try:
                    message = self.messages_queue.get_nowait()  # Getting immediately the message
                    if isinstance(message, Exception):  # caught the exception
                        print("Encountered and handled exception: ", message)
                        # Should evoke all operations associated with clicked Quit button on the main window
                        running = False; quitMainProgram = True
                        break
                    if isinstance(message, str):  # normal ending the running task
                        if message == "Stop Exception Checker" or message == "Stop" or message == "Stop Program":
                            # print("Exception checker stopped")
                            running = False; break
                        else:
                            print("Some message caught by the Exception checker but not recognized")
                except Empty:
                    pass
            time.sleep(self.period_checks_ms/1000)  # Artificial delays between each loop iteration
        # Only now, if the loop has been ended because of caught Exception, call from the main window quit action
        if quitMainProgram:
            self.quitButton.click()  # Simulate clicking on the button on the main window for stopping the GUI program


# %% Message printer
class MessagesPrinter(Thread):
    """
    Check and print messages from some Queue (for interoperability for calling from IPython console and normal one).

    This class is threaded for allowing simple printing in the IPython console of Spyder IDE.
    """

    def __init__(self, messages_queue: ProcessQueue, period_checks_ms: int = 100):
        self.messages_queue = messages_queue; self.period_checks_ms = period_checks_ms
        if self.period_checks_ms < 5:
            self.period_checks_ms = 5  # minimal delay between checks = 5 ms
        Thread.__init__(self)

    def run(self):
        """
        Periodically check and, if found, print the messages put on the input Queue (from multiprocessing).

        Returns
        -------
        None.

        """
        running = True
        while running:
            if not(self.messages_queue.empty()) and (self.messages_queue.qsize() > 0):
                try:
                    message = self.messages_queue.get_nowait()  # Getting immediately the message
                    if isinstance(message, str):  # normal ending the running task
                        if message == "Stop Messages Printer" or message == "Stop" or message == "Stop Program":
                            # print("Messages Printer stopped")
                            running = False; break
                        else:
                            print(message)

                except Empty:
                    pass

            time.sleep(self.period_checks_ms/1000)  # Artificial delays between each loop iteration

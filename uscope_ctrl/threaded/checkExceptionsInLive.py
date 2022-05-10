# -*- coding: utf-8 -*-
"""
Module for import into the camera GUI for performing periodically checks that there are no exceptions happened.

@author: ssklykov
"""
# %% Imports
from threading import Thread
from queue import Queue, Empty
import time


# %% Class implementation of Exceptions checker and evoking clicking of Exit or Quit button
class CheckMessagesForExceptions(Thread):
    """Threaded class for continuous and independent loop running for checking that any Exception reported anythere in the GUI program."""

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
                            running = False
                        else:
                            print("Some message caught by the Exception checker but not recognized")
                except Empty:
                    pass
            time.sleep(self.period_checks_ms/1000)  # Artificial delays between each loop iteration
        # Only now, if the loop has been ended because of caught Exception, call from the main window quit action
        if quitMainProgram:
            self.quitButton.click()  # Simulate clicking on the button on the main window for stopping the program

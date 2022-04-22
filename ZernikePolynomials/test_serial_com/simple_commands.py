# -*- coding: utf-8 -*-
"""
Script for testing preconfigured Arduino board opened for serial communictation and connected on the specified port.

@author: ssklykov
"""
# %% Imports
import serial

# %% Properties
port = "COM4"
baudrate = 115200

# %% Functions

# %% Testing
if __name__ == '__main__':
    # below - create and open serial communication
    ser_con = serial.Serial(port=port, baudrate=baudrate, timeout=1.6, write_timeout=0.05)
    # Check if the communication is opened and print the responce from the board
    if ser_con.isOpen():
        print("Port opened upon the creation:", ser_con.isOpen())
        respond = ser_con.readall(); respond = respond.decode("utf-8")
        print(respond)

    # Send some special command
    ser_con.write(b'?')
    data = ser_con.readall()
    print(data.decode("utf-8"))

    # Send some command and receive it echoed back
    ser_con.write(b'test')
    data = ser_con.readall()
    print(data.decode("utf-8"))

    # Close the serial communication
    ser_con.close()
    print("Is connection closed?:", not ser_con.isOpen())

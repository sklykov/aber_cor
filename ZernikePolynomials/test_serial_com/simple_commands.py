# -*- coding: utf-8 -*-
"""
Script for testing preconfigured Arduino board opened for serial communictation and connected on the specified port.

@author: ssklykov
"""
# %% Imports
import serial
import time

# %% Properties
port = "COM4"
baudrate = 115200

# %% Functions

# %% Testing
if __name__ == '__main__':
    # below - create and open serial communication
    ser_con = serial.Serial(port=port, baudrate=baudrate, timeout=2, write_timeout=0.02)
    # Check if the communication is opened and print the responce from the board
    if ser_con.isOpen():
        print("Port opened upon the creation:", ser_con.isOpen())
        respond = ser_con.readlines()[0]; respond = respond.decode("utf-8")
        print(respond)

    time.sleep(0.02)
    # Send some special command
    t1 = time.perf_counter()
    ser_con.write(b'?')
    # data = ser_con.readall()
    data1 = ser_con.readlines()[0].decode("utf-8")
    print(data1)
    t2 = time.perf_counter(); print("Respond to simple command takes: ", round(t2-t1, 2), "s")

    time.sleep(0.02)
    # Send some command and receive it echoed back
    # t1 = time.perf_counter()
    # ser_con.writelines(b'test')
    # data = ser_con.readall()
    # print(data.decode("utf-8"))
    # t2 = time.perf_counter(); print("Echoing of command takes: ", round(t2-t1, 2), "s")

    t1 = time.time()
    ser_con.write(b'!')
    time.sleep(0.01)
    # data = ser_con.readall()
    # data2 = ser_con.readall().decode("utf-8")
    data2 = ser_con.read(ser_con.inWaiting()).decode("utf-8")
    print("Getting back: ", data2)
    t2 = time.time(); print("Echoing of command takes: ", round(t2-t1, 2), "s")

    # Close the serial communication
    ser_con.close()
    print("Is connection closed?:", not ser_con.isOpen())

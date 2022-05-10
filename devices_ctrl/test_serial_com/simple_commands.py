# -*- coding: utf-8 -*-
"""
Script for testing preconfigured Arduino board opened for serial communictation and connected on the specified port.

@author: ssklykov
"""
# %% Imports
import serial
import serial.tools.list_ports as list_ports  # needed explicit import, not imported automatically along with the module
import time

# %% Properties
# port = "COM4"  # defined below
baudrate = 115200

# %% Functions

# %% Testing
if __name__ == '__main__':
    ports = []
    for port in list_ports.comports():  # get all ports stored in attributes of the class
        ports.append(port.name)
    print("All names of ports:", ports)
    # below - create and open serial communication
    ser_con = serial.Serial(port=ports[0], baudrate=baudrate, timeout=0.1, write_timeout=0.01)
    # Check if the communication is opened and print the responce from the board
    if ser_con.isOpen():
        print("Port opened upon the creation:", ser_con.isOpen())
        # !!! The repeated issue with reading data fast was good described in the following comment:
        # https://github.com/pyserial/pyserial/issues/216#issuecomment-609669312
        # The workaround for fast reading the stored lines in buffer is described in:
        # https://stackoverflow.com/questions/47235381/python-serial-read-is-extremely-slow-how-to-speed-up
        time.sleep(2)  # for start receiving written in buffer strings
        print("Number of bytes in buffer: ", ser_con.in_waiting)
        # respond = ser_con.readlines()[0]  # readlines is not efficient
        # respond = ser_con.readall()  # readall is not efficient
        respond = ser_con.read(ser_con.in_waiting)
        respond = respond.decode("utf-8")
        print(respond)

        time.sleep(0.02)
        # Send some special command for receiving device state
        t1 = time.perf_counter(); ser_con.write(b'?')
        time.sleep(0.01)
        print(ser_con.read(ser_con.in_waiting).decode("utf-8"))  # echoed back command
        time.sleep(0.02)
        # data = ser_con.readall()  # not effective
        print("Number of bytes in buffer: ", ser_con.in_waiting)
        data1 = ser_con.read(ser_con.in_waiting).decode("utf-8"); print(data1)
        t2 = time.perf_counter(); print("Respond to simple command takes: ", round(t2-t1, 2), "s")

        time.sleep(0.2)
        # Send some command and receive it echoed back
        t1 = time.perf_counter(); ser_con.write(b'test')
        time.sleep(0.02); print(ser_con.read(ser_con.in_waiting).decode("utf-8"))  # echoed back command
        t2 = time.perf_counter(); print("Echoing of command takes: ", round(t2-t1, 2), "s")

        time.sleep(0.2)
        t1 = time.perf_counter(); ser_con.write(b'!'); time.sleep(0.02)
        data2 = ser_con.read(ser_con.in_waiting).decode("utf-8")
        print(data2)
        t2 = time.perf_counter(); print("Echoing of character takes: ", round(t2-t1, 2), "s")

        # Close the serial communication
        time.sleep(0.02); ser_con.close()
        print("Is connection closed?:", not ser_con.isOpen())

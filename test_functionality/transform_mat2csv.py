# -*- coding: utf-8 -*-
"""
Transform a *mat file to a *csv file.

@author: sklykov

@license: GPLv3, general terms on: https://www.gnu.org/licenses/gpl-3.0.en.html

"""
# %% Imports
import csv
from scipy.io import loadmat
import os
import numpy as np

# %% Transform a *mat file to a *csv file
mat_file_name = "test.mat"
test_file_path = os.path.join(os.path.dirname(__file__), mat_file_name)
matrix = None
if os.path.isfile(test_file_path):
    for key in loadmat(test_file_path).keys():
        if '__' not in key:
            matrix = loadmat(test_file_path)[key]
    # checking that matrix is not empty
    if matrix is not None:
        shape = np.shape(matrix)
        # Suppose that matrix maximum size is 2D
        if len(shape) > 0:
            m = shape[0]
            if len(shape) > 1:
                n = shape[1]
            else:
                n = -1
            if m > 0:
                with open('test.csv', 'w', newline='') as test_csv_file:
                    test_writer = csv.writer(test_csv_file, delimiter='\t')
                    for i in range(m):
                        if n > 1:
                            test_writer.writerow(matrix[i, 0:n])
                        else:
                            test_writer.writerow(matrix[i])

# %% Read a matrix from a *csv file
csv_file_name = "test2.csv"
test_csv_path = os.path.join(os.path.dirname(__file__), csv_file_name)
read_matrix = None
if os.path.isfile(test_csv_path):
    line_count = 0
    with open(csv_file_name, 'r', newline='') as test_csv_file:
        csv_reader = csv.reader(test_csv_file, delimiter='\t')
        lines_number = len(list(csv_reader))  # get number of lines in csv files
        test_csv_file.seek(0)  # return back to the start of the csv file
        for row in csv_reader:
            if line_count == 0:
                # initialize matrix for storing values (assuming that matrix has 2 dimensions)
                read_matrix = np.zeros(shape=(lines_number, len(row)))
            try:
                for i in range(len(row)):
                    row[i] = row[i].replace(',', '.')  # if the float number separated by comma not dot in a file
                    try:
                        read_matrix[line_count, i] = float(row[i])
                    except ValueError:
                        raise ValueError("Error raised during attempt to convert '" + row[i]
                                         + "' - read value on row, column: " + f"{line_count, i}")
            except ValueError as e:
                print(e)
            line_count += 1

# %% Check difference between write and read matrices
if matrix is not None and read_matrix is not None:
    if matrix.shape == read_matrix.shape:
        diff = matrix - read_matrix

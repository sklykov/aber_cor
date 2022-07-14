# -*- coding: utf-8 -*-
"""
Parse the tab-delimeted *csv to the numpy matrix.

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3
"""
# %% Imports
import csv
import os
import numpy as np

# %% Test parameters specification
csv_file_name = "test2.csv"
test_csv_path = os.path.join(os.path.dirname(__file__), csv_file_name)


# %% Read (parsing) function
def read_matrix_csv(abs_path_csv_file: str, delimeter: str = '\t') -> np.ndarray:
    r"""
    Read matrix from csv file on the specified absolute path.

    Parameters
    ----------
    abs_path_csv_file : str
        Absolute path to the file for reading from.
    delimeter : str, optional
        Delimiter between values. The default is '\t' (tab-separated values).

    Raises
    ------
    FileNotFoundError
        If the check os.path.isfile(abs_path_csv_file) provides false.
    ValueError
        If the attempt to convert read value from the *csv file to 'float' was failed.

    Returns
    -------
    read_matrix : np.ndarray
        Read matrix stored in *csv file.

    """
    read_matrix = None
    if os.path.isfile(abs_path_csv_file):
        line_count = 0
        with open(abs_path_csv_file, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimeter)
            lines_number = len(list(csv_reader))  # get number of lines in csv files
            csv_file.seek(0)  # return back to the start of the csv file
            for row in csv_reader:
                if line_count == 0:
                    # initialize matrix for storing values (assuming that matrix has 2 dimensions)
                    read_matrix = np.zeros(shape=(lines_number, len(row)))
                for i in range(len(row)):
                    row[i] = row[i].replace(',', '.')  # if the float number separated by comma not dot in a file
                    try:
                        read_matrix[line_count, i] = float(row[i])
                    except ValueError:
                        raise(ValueError("Error raised during attempt to convert to float number the value: '"
                                         + row[i] + "' on row, column: " + f"{line_count, i}"))
                line_count += 1
        return read_matrix
    else:
        raise FileNotFoundError("Provided path (" + abs_path_csv_file + ") guides not a file")


# %% Simple test
if __name__ == '__main__':
    M = read_matrix_csv(test_csv_path)

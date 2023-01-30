# -*- coding: utf-8 -*-
"""
Create a *json file using different keys, including saving huge data (numeric matrix).

@author: sklykov

@license: GPLv3, general terms on: https://www.gnu.org/licenses/gpl-3.0.en.html

"""
# %% Imports
import json
from read_data_from_csv import read_matrix_csv
import os

# %% Prepare data for saving in a *json file
csv_file_name = "test2.csv"; read_matrix = None
test_csv_path = os.path.join(os.path.dirname(__file__), csv_file_name)
read_matrix = read_matrix_csv(test_csv_path)
read_matrix = read_matrix.tolist()  # good for the 2D matrix
# below is conversion for 1D np.ndarray but with sizes (length, 1)
# read_matrix = list(np.squeeze(read_matrix))
data4serialization = {}; data4serialization['matrix'] = read_matrix
data4serialization['type'] = "Demo"; data4serialization['ID'] = 20220715

# %% Saving the *json file
json_file_name = "test.json"
with open(json_file_name, 'w') as json_write_file:
    json.dump(data4serialization, json_write_file)

# %% Reading data from the *json file
with open(json_file_name, 'r') as json_read_file:
    deserialized_data = json.load(json_read_file)
    print(deserialized_data.keys())

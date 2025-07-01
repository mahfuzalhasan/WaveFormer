import numpy as np
import pickle
import os

fixed_file_list_path = "./fixed_test_list.txt"
file_path = './test_list.txt'
save_file = './test_list.pkl'

with open(file_path, 'r') as file:
    file_data = file.read()

# Split the original file content by commas and clean up formatting
file_list = [name.strip().strip("'") for name in file_data.split(',')]
print(f'{len(file_list)}')

# Write the corrected format to a new file
with open(fixed_file_list_path, 'w') as fixed_file:
    fixed_file.write("\n".join(file_list))

if not os.path.exists(save_file):
    with open(save_file, 'wb') as f:
        # print(f'### Saving {file_name} ##')
        pickle.dump(file_list, f)

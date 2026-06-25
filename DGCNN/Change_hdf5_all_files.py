

y_hat_exp_name = "exp_name"

input_file = f'../DGCNN/data/s3dis_hdf5_data_{y_hat_exp_name}/all_files.txt'
output_file = f'../DGCNN/data/s3dis_hdf5_data_{y_hat_exp_name}/all_files.txt' 

with open(input_file, 'r') as file:
    lines = file.readlines()

updated_lines = [line.replace('s3dis_hdf5_data_labeled_points', f's3dis_hdf5_data_{y_hat_exp_name}') for line in lines]


with open(output_file, 'w') as file:
    file.writelines(updated_lines)

print(f"Updated file paths have been saved to {output_file}")

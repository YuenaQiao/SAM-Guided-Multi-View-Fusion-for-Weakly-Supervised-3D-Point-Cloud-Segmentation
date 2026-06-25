import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


print("BASE_DIR: ", BASE_DIR)
print("ROOT_DIR: ", ROOT_DIR)

sys.path.append(BASE_DIR)
import indoor3d_util

parser = argparse.ArgumentParser(description="Collect S3DIS room data with HLP pseudo labels.")
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="Path to Stanford3dDataset_v1.2_Aligned_Version.",
)
parser.add_argument(
    "--hlp_data_path",
    type=str,
    default=None,
    help="Path to the hlp_data folder containing HLP predicted labels.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default=None,
    help="Output folder for generated numpy files. Default: <ROOT_DIR>/data/s3dis_numpy_for_hdf5_<y_hat_exp_name>.",
)
args = parser.parse_args()

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(args.data_path, p) for p in anno_paths]

process_num=0

y_hat_exp_name = "exp_name1" #exp_name

output_folder = args.output_folder if args.output_folder is not None else os.path.join(ROOT_DIR, f'data/s3dis_numpy_for_hdf5_{y_hat_exp_name}')


print(output_folder)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.

for anno_path in anno_paths:
    # print(anno_path)

    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' 

        output_path = os.path.join(output_folder, out_filename)
        if os.path.exists(output_path):
                print("Saved already, skip.")
                process_num = process_num + 1
                continue
        
        indoor3d_util.collect_point_label(anno_path, output_path, 'numpy', args.data_path, args.hlp_data_path)

        print(process_num, " anno_path:", anno_path)
        process_num += 1
    except:
        print(anno_path, 'ERROR!!')

print(process_num)

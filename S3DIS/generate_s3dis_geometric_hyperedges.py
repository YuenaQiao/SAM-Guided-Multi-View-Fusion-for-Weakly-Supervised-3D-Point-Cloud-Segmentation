import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

def getMaskNum(superpointGraph_path):
    # Initialize an empty set to store all unique edge indices.
    unique_edge_indices = set()

    # Read the txt file.
    with open(superpointGraph_path, "r") as file:
        # Iterate over each line in the file.
        for line in file:
            # Split each line to get the edge index.
            edge_index = line.strip().split()[-1]
            # Add the edge index to the set.
            unique_edge_indices.add(edge_index)

    superpoint_maskNum = len(unique_edge_indices)
    return superpoint_maskNum

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

superpoint_graph_PATH = '../dataset/superpoint/superpoint_graph'

areas_names = sorted(os.listdir(superpoint_graph_PATH))
for area_name in areas_names:
    rooms_folder_path = os.path.join(superpoint_graph_PATH, area_name)
    # print("rooms_folder_path: ", rooms_folder_path)
    if os.path.isdir(rooms_folder_path):
        rooms_txt = os.listdir(rooms_folder_path)
        for room_txt in rooms_txt:
            txt_name_info = room_txt.split('_')
            room_name = txt_name_info[2] + "_" +txt_name_info[3]

            print(f"{area_name} {room_name}")
            superpoint_MPPR_root = f"../dataset/superpoint/superpoint_hyperedges/{area_name}" # output_path(s3dis_geometric_hyperedges path)
            create_folder_if_not_exists(superpoint_MPPR_root)

            # Read the txt file, where each line is assumed to contain space-separated xyz coordinates and a label.
            superpointGraph_File_path = os.path.join(rooms_folder_path, room_txt)
            superpointGraph_Points_data = np.loadtxt(superpointGraph_File_path)

            # Extract xyz coordinates as features and mask labels as labels.
            features = superpointGraph_Points_data[:, :3]
            labels = superpointGraph_Points_data[:, 3]

            superpoint_maskNum = getMaskNum(superpointGraph_File_path)
            # print(f"superPoint graph's total edges num: {superpoint_maskNum}")

            # Create a KNN classifier.
            k_neighbors = 1
            knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
            knn_classifier.fit(features, labels)

            # Given an S3DIS scene txt file, predict the possible mask label for each point and generate a new txt file to complete mask propagation.
            s3dis_file_path = f"../dataset/Stanford3dDataset_v1.2_Aligned_Version/{area_name}/{room_name}/{room_name}.txt"
            
            s3dis_points_data = np.loadtxt(s3dis_file_path)

            # Perform KNN prediction for each point and write the results to a new txt file.
            output_file_path = os.path.join(superpoint_MPPR_root, f'{area_name}_{room_name}_superpoint_maskPropagation.txt')

            # Create a dictionary to store the point IDs corresponding to each mask label (line number - 1).
            mask_label_dict = {}

            s3dis_points = s3dis_points_data[:,:3]
            predicted_labels = knn_classifier.predict(s3dis_points)
            with open(output_file_path, 'w') as output_file:
                for point_ID in range(len(s3dis_points)):
                    predicted_label = predicted_labels[point_ID]
                    if point_ID == 0:  # If this is the first line, write maskNum.
                        output_file.write(f"{point_ID} {predicted_label} {superpoint_maskNum}\n")
                    else:
                        output_file.write(f"{point_ID} {predicted_label}\n")
            print(f"Prediction results have been written to {output_file_path}")





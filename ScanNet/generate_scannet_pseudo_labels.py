"""

Post-process ScanNet HLP results into full-scene pseudo labels.

"""

import os
import json
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from plyfile import PlyData


TEST_CLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
              10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def gen_label_map():
    label_map = np.zeros(41, dtype=np.int32)
    for i in range(41):
        if i in TEST_CLASS:
            label_map[i] = TEST_CLASS.index(i)
        else:
            label_map[i] = 0
    return label_map


def get_scene_data_and_labels(data_root, scene_name):
    scans_dir = os.path.join(data_root, "scans")
    label_map = gen_label_map()

    scene_rgb_path = os.path.join(scans_dir, scene_name, f"{scene_name}_vh_clean_2.ply")
    scene_label_path = os.path.join(scans_dir, scene_name, f"{scene_name}_vh_clean_2.labels.ply")

    scene_xyzrgb = PlyData.read(scene_rgb_path)
    scene_vertex_rgb = scene_xyzrgb["vertex"]
    scene_data = np.stack(
        (
            scene_vertex_rgb["x"],
            scene_vertex_rgb["y"],
            scene_vertex_rgb["z"],
            scene_vertex_rgb["red"],
            scene_vertex_rgb["green"],
            scene_vertex_rgb["blue"],
        ),
        axis=-1,
    ).astype(np.float32)

    scene_xyzlabel = PlyData.read(scene_label_path)
    scene_vertex = scene_xyzlabel["vertex"]
    scene_labels = np.array(scene_vertex["label"])

    # Map non-ScanNet20 labels to 0 (ignored)
    scene_labels[np.where(scene_labels > 40)] = 0
    scene_labels = label_map[scene_labels]

    return scene_data, scene_labels.astype(np.int32)


def calculate_miou(real_labels, predicted_labels):
    real_labels = real_labels.reshape(-1)
    predicted_labels = predicted_labels.reshape(-1)

    real_labels = real_labels - 1
    predicted_labels = predicted_labels - 1

    true_positive_classes = np.zeros(shape=[20])
    np.add.at(true_positive_classes, real_labels[real_labels == predicted_labels], 1)

    positive_classes = np.zeros(shape=[20])
    np.add.at(positive_classes, predicted_labels, 1)

    gt_classes = np.zeros(shape=[20])
    np.add.at(gt_classes, real_labels, 1)

    ious = true_positive_classes / (gt_classes + positive_classes - true_positive_classes + 1e-7)
    miou = np.mean(ious)

    return miou, ious


def calculate_acc(real_labels, predicted_labels):
    correct_predictions = np.sum(real_labels == predicted_labels)
    accuracy = correct_predictions / real_labels.size
    return accuracy


def get_predicted_labels(scene_data, y_hat_file, knn_k, confidence):
    features = []
    labels = []

    scene_points_num = scene_data.shape[0]
    print(f"scene_points_num: {scene_points_num}")

    points = scene_data[:, :3]
    probability_dict = {}

    with open(y_hat_file, "r", encoding="utf-8") as file:
        for line in file:
            elements = line.strip().split()
            pseudo_label_elements = [float(elem) for elem in elements[:21]]

            total = sum(pseudo_label_elements)
            if total == 0:
                total = 1.0
            normalized_data = [x / total for x in pseudo_label_elements]

            point_id = int(float(elements[-1]))
            probability_dict[point_id] = normalized_data

            if scene_points_num > point_id:
                point_xyz = scene_data[point_id][:3]
                features.append(point_xyz)
                labels.append(point_id)

    downsample_features = np.array(features)
    downsample_labels = np.array(labels)

    knn_classifier = KNeighborsClassifier(n_neighbors=knn_k)
    knn_classifier.fit(downsample_features, downsample_labels)
    predicted_point_ids = knn_classifier.predict(points)

    new_predicted_labels = []
    for point_id in predicted_point_ids:
        if point_id in probability_dict:
            normalized_data = probability_dict[point_id]
            max_value = max(normalized_data)
            if max_value < confidence:
                new_label = 0
            else:
                new_label = normalized_data.index(max_value)
        else:
            new_label = -2
        new_predicted_labels.append(new_label)

    return np.array(new_predicted_labels, dtype=np.int32)


def build_exp_name(setting):
    if setting == "20pt":
        return "Scannet_20pt"
    elif setting == "sparse":
        return "Scannet_sparse"
    else:
        raise ValueError(f"Unknown setting: {setting}")


def build_base_dir(output_root, exp_name, add_superpoint, add_sam_superpoint):
    if add_superpoint:
        return os.path.join(
            output_root,
            exp_name,
            "with_superpoint",
            f"AddSAMSuperpoint_{add_sam_superpoint}",
        )
    else:
        return os.path.join(
            output_root,
            exp_name,
            "without_superpoint",
            f"AddSAMSuperpoint_{add_sam_superpoint}",
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process ScanNet HLP predictions into full-scene pseudo labels."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of ScanNet data.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs/hlp",
        help="Root directory of HLP outputs.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["20pt", "sparse"],
        required=True,
        help="Label setting.",
    )
    parser.add_argument(
        "--add_superpoint",
        action="store_true",
        help="Use geometric superpoints.",
    )
    parser.add_argument(
        "--add_sam_superpoint",
        action="store_true",
        help="Use SAM mask hyperedges.",
    )
    parser.add_argument(
        "--enable_filtering",
        action="store_true",
        help="Enable confidence filtering.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence threshold used when filtering is enabled.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=1,
        help="Number of neighbors for KNN propagation.",
    )
    parser.add_argument(
        "--label_indices_root",
        type=str,
        default=None,
        help="Directory containing labeled point index files. "
             "Default: <output_root>/labeled_points/<setting>",
    )
    parser.add_argument(
        "--label_tag",
        type=str,
        default=None,
        help="Optional tag for labeled_points subdirectory. Default: same as setting.",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        help="Save per-scene and global metrics as a JSON summary.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    confidence = args.confidence if args.enable_filtering else 0.0
    exp_name = build_exp_name(args.setting)

    base_dir = build_base_dir(
        output_root=args.output_root,
        exp_name=exp_name,
        add_superpoint=args.add_superpoint,
        add_sam_superpoint=args.add_sam_superpoint,
    )

    hlp_result_dir = os.path.join(base_dir, "HLP_results")

    if confidence == 0.0:
        predicted_label_dir = os.path.join(base_dir, "predicted_labels_without_filtering")
    else:
        predicted_label_dir = os.path.join(base_dir, f"predicted_labels_confidence{confidence}")

    os.makedirs(predicted_label_dir, exist_ok=True)

    label_tag = args.label_tag if args.label_tag is not None else args.setting
    label_indices_root = (
        args.label_indices_root
        if args.label_indices_root is not None
        else os.path.join(args.output_root, "labeled_points", label_tag)
    )

    print("Arguments:")
    print(f"  data_root          : {args.data_root}")
    print(f"  output_root        : {args.output_root}")
    print(f"  setting            : {args.setting}")
    print(f"  add_superpoint     : {args.add_superpoint}")
    print(f"  add_sam_superpoint : {args.add_sam_superpoint}")
    print(f"  enable_filtering   : {args.enable_filtering}")
    print(f"  confidence         : {confidence}")
    print(f"  knn_k              : {args.knn_k}")
    print(f"  hlp_result_dir     : {hlp_result_dir}")
    print(f"  predicted_label_dir: {predicted_label_dir}")
    print(f"  label_indices_root : {label_indices_root}")

    scans_dir = os.path.join(args.data_root, "scans")
    scene_names = sorted(
        [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]
    )

    processed_num = 0
    scene_metrics_list = []
    all_gt = []
    all_pred = []
    not_equal_scene = []

    for i, scene_name in enumerate(scene_names, start=1):
        print(f"\nprocess: {i} {scene_name}")

        output_name = f"{scene_name}_predicted_labels.npy"
        output_path = os.path.join(predicted_label_dir, output_name)

        if os.path.exists(output_path):
            print("Predicted labels already exist. Skip.")
            processed_num += 1
            continue

        y_hat_file = os.path.join(hlp_result_dir, f"{scene_name}_Y_hat_with_pointID.txt")
        if not os.path.exists(y_hat_file):
            raise FileNotFoundError(f"HLP result file not found: {y_hat_file}")

        scene_data, scene_real_label = get_scene_data_and_labels(args.data_root, scene_name)

        predicted_labels = get_predicted_labels(
            scene_data=scene_data,
            y_hat_file=y_hat_file,
            knn_k=args.knn_k,
            confidence=confidence,
        )

        # Load labeled point indices
        label_indices_path = os.path.join(label_indices_root, f"{scene_name}_labeled_points.npy")
        if not os.path.exists(label_indices_path):
            raise FileNotFoundError(f"Labeled point index file not found: {label_indices_path}")

        selected_indices = np.load(label_indices_path).astype(np.int32)

        assert predicted_labels.shape == scene_real_label.shape, (
            f"Shape mismatch: predicted {predicted_labels.shape}, "
            f"real {scene_real_label.shape}"
        )

        # Keep the original labels on the initially labeled points
        predicted_labels[selected_indices] = scene_real_label[selected_indices]

        selected_pred = predicted_labels[selected_indices]
        selected_gt = scene_real_label[selected_indices]

        is_equal = selected_pred == selected_gt
        num_labeled_points_total = len(selected_indices)
        num_equal = int(np.sum(is_equal))
        num_not_equal = num_labeled_points_total - num_equal

        if num_not_equal > 0:
            print(f"Labeled points total: {num_labeled_points_total}")
            print(f"Consistent labels   : {num_equal}")
            print(f"Inconsistent labels : {num_not_equal}")
            not_equal_scene.append(scene_name)

        # Mask ignored points defined by the ScanNet20 label mapping
        predicted_labels[scene_real_label == 0] = 0

        np.save(output_path, predicted_labels)

        gt = scene_real_label
        pred = predicted_labels

        if args.enable_filtering:
            final_valid_mask = pred != 0
            gt = gt[final_valid_mask]
            pred = pred[final_valid_mask]

        miou, ious = calculate_miou(gt, pred)
        acc = calculate_acc(gt, pred)

        print(f"mIoU: {miou}, Acc: {acc}")

        scene_metrics_list.append({
            "scene": scene_name,
            "mIoU": float(miou),
            "ious": ious.tolist(),
            "acc": float(acc),
        })

        all_gt.append(gt)
        all_pred.append(pred)

        processed_num += 1

    print(f"\nFinished. Processed {processed_num} scenes.")

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    total_miou, total_ious = calculate_miou(all_gt, all_pred)
    total_acc = calculate_acc(all_gt, all_pred)

    print("Global mIoU:", total_miou)
    print("Global IoUs:", total_ious)
    print("Global Acc:", total_acc)

    print("\nScenes with inconsistent labeled points:")
    if len(not_equal_scene) == 0:
        print("None")
    else:
        for scene in not_equal_scene:
            print(f" - {scene}")

    if args.save_summary:
        if confidence == 0.0:
            summary_name = f"{exp_name}_summary_addSAMSuperpoint_{args.add_sam_superpoint}.json"
        else:
            summary_name = (
                f"{exp_name}_conf{confidence}_summary_addSAMSuperpoint_{args.add_sam_superpoint}.json"
            )

        summary_path = os.path.join(predicted_label_dir, summary_name)

        summary = {
            "global_mIoU": float(total_miou),
            "global_ious": total_ious.tolist(),
            "global_acc": float(total_acc),
            "scene_metrics": scene_metrics_list,
            "not_equal_scene": not_equal_scene,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
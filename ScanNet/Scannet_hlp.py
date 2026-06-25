#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Unified ScanNet HLP generation script with two settings:
1) 20pt
2) sparse

Usage examples:
    python Scannet_hlp.py --setting 20pt --add_superpoint --add_sam_superpoint
    python Scannet_hlp.py --setting sparse --add_superpoint --add_sam_superpoint --sparse_ratio 0.001
"""

import os
import re
import argparse
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cp_sparse

from plyfile import PlyData
from HyperGraph_gpu import HyperGraph


class HLPScannet:
    def __init__(
        self,
        scene_name,
        root,
        setting,
        add_superpoint,
        add_sam_superpoint,
        sparse_ratio=None,
        label_indices_root=None,
        check_saved_indices=False,
        random_seed=3,
        max_points_20pt=20,
    ):
        self.scene_name = scene_name
        self.root = root
        self.setting = setting
        self.add_superpoint = add_superpoint
        self.add_sam_superpoint = add_sam_superpoint
        self.sparse_ratio = sparse_ratio
        self.label_indices_root = label_indices_root
        self.check_saved_indices = check_saved_indices
        self.random_seed = random_seed
        self.max_points_20pt = max_points_20pt

        self.k_num = 21
        self.test_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

        self.scans_dir = os.path.join(self.root, "scans")
        self.scene_masks_folder_path = os.path.join(self.root, "SAM_output", self.scene_name)

        self.sparse_H_all_frames = []

        self.scene_data = None
        self.point_num = None

        self._set_point_num()

    def _set_point_num(self):
        scene_rgb_path = os.path.join(
            self.scans_dir, self.scene_name, f"{self.scene_name}_vh_clean_2.ply"
        )
        scene_xyzrgb = PlyData.read(scene_rgb_path)
        scene_vertex_rgb = scene_xyzrgb["vertex"]

        self.scene_data = np.stack(
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

        self.point_num = self.scene_data.shape[0]
        print(f"[{self.scene_name}] scene_points_num: {self.point_num}")

    @staticmethod
    def extract_last_number(filename):
        numbers = re.findall(r"\d+", filename)
        return int(numbers[-1]) if numbers else float("inf")

    def create_ybar(self):
        return np.zeros((self.point_num, self.k_num), dtype=np.float32)

    def gen_label_map(self):
        label_map = np.zeros(41, dtype=np.int32)
        for i in range(41):
            if i in self.test_class:
                label_map[i] = self.test_class.index(i)
            else:
                label_map[i] = 0
        return label_map

    def remove_unanno(self, scene_data, scene_label, scene_data_id):
        keep_idx = np.where((scene_label > 0) & (scene_label < 41))
        scene_data_clean = scene_data[keep_idx]
        scene_label_clean = scene_label[keep_idx]
        scene_data_id_clean = scene_data_id[keep_idx]
        return scene_data_clean, scene_label_clean, scene_data_id_clean

    def get_3d_info(self, split="train", keep_unanno=True):
        label_map = self.gen_label_map()
        scene_point_id = np.arange(self.point_num)

        label_ply_path = os.path.join(
            self.scans_dir, self.scene_name, f"{self.scene_name}_vh_clean_2.labels.ply"
        )
        scene_xyzlabel = PlyData.read(label_ply_path)
        scene_vertex = scene_xyzlabel["vertex"]
        scene_data_label_tmp = np.array(scene_vertex["label"])

        if not keep_unanno:
            _, scene_data_label_tmp, _ = self.remove_unanno(
                self.scene_data, scene_data_label_tmp, scene_point_id
            )
            scene_data_label_tmp = label_map[scene_data_label_tmp]
        elif split != "test":
            scene_data_label_tmp[np.where(scene_data_label_tmp > 40)] = 0
            scene_data_label_tmp = label_map[scene_data_label_tmp]

        return scene_data_label_tmp

    def read_mask_propagation_info(self, mask_propagation_npy):
        sam_output = np.load(mask_propagation_npy)

        group_id_max_val = np.max(sam_output)
        mask_num = int(group_id_max_val) + 1

        valid_mask = sam_output != -1
        row_indices = np.where(valid_mask)[0]
        col_indices = sam_output[valid_mask]

        row_cp = cp.asarray(row_indices)
        col_cp = cp.asarray(col_indices)
        data_cp = cp.ones_like(row_cp, dtype=cp.float32)

        incidence_matrix = cp_sparse.csr_matrix(
            (data_cp, (row_cp, col_cp)),
            shape=(self.point_num, mask_num),
            dtype=cp.float32,
        )
        self.sparse_H_all_frames.append(incidence_matrix)

    def set_specific_room_h_matrix(self):
        if self.add_superpoint:
            superpoint_npy = os.path.join(
                self.root,
                "Superpoint",
                f"{self.scene_name}_supervoxel_labels.npy",
            )
            self.read_mask_propagation_info(superpoint_npy)

        if self.add_sam_superpoint:
            all_items = os.listdir(self.scene_masks_folder_path)
            sam_output_files = [
                item
                for item in all_items
                if os.path.isfile(os.path.join(self.scene_masks_folder_path, item))
                and item.endswith(".npy")
            ]
            sam_output_files.sort(key=self.extract_last_number)

            for sam_file in sam_output_files:
                sam_file_path = os.path.join(self.scene_masks_folder_path, sam_file)
                self.read_mask_propagation_info(sam_file_path)

    def _downsample_subset_per_class(self, scene_data_label):
        if self.point_num > 15000:
            downsampling_total_num = 15000
        else:
            downsampling_total_num = 8000

        downsampling_ratio = downsampling_total_num / self.point_num

        subset_idx = []
        unique_classes = np.unique(scene_data_label)

        np.random.seed(self.random_seed)
        for labeled_class in unique_classes:
            labeled_indices_one_class = np.where(scene_data_label == labeled_class)[0]

            downsample_num = max(1, int(len(labeled_indices_one_class) * downsampling_ratio))
            downsample_num = min(downsample_num, len(labeled_indices_one_class))

            if downsample_num > 0:
                downsampled = np.random.choice(
                    labeled_indices_one_class, size=downsample_num, replace=False
                )
                subset_idx.extend(downsampled.tolist())

        return subset_idx

    def _sample_labeled_points_20pt(self, scene_data_label):
        unique_classes = np.unique(scene_data_label)
        class_to_indices = {
            cls: np.where(scene_data_label == cls)[0].tolist()
            for cls in unique_classes
        }

        np.random.seed(self.random_seed)
        for cls in class_to_indices:
            np.random.shuffle(class_to_indices[cls])

        labeled_indices = []

        while len(labeled_indices) < self.max_points_20pt:
            sorted_classes = sorted(unique_classes, key=lambda c: len(class_to_indices[c]))
            progressed = False

            for cls in sorted_classes:
                if len(labeled_indices) >= self.max_points_20pt:
                    break
                if len(class_to_indices[cls]) > 0:
                    idx = class_to_indices[cls].pop()
                    labeled_indices.append(idx)
                    progressed = True

            if not progressed:
                break

        labeled_indices = np.array(labeled_indices, dtype=np.int64)
        return labeled_indices

    def _sample_labeled_points_sparse(self, scene_data_label):
        if self.sparse_ratio is None:
            raise ValueError("sparse_ratio must be provided when setting='sparse'.")

        unique_classes = np.unique(scene_data_label)
        labeled_indices = []

        np.random.seed(self.random_seed)
        for labeled_class in unique_classes:
            labeled_indices_one_class = np.where(scene_data_label == labeled_class)[0]
            labeled_num = max(1, int(len(labeled_indices_one_class) * self.sparse_ratio))

            if len(labeled_indices_one_class) > 0:
                sampled = np.random.choice(
                    labeled_indices_one_class,
                    size=labeled_num,
                    replace=False,
                )
                labeled_indices.extend(sampled.tolist())

        labeled_indices = np.array(labeled_indices, dtype=np.int64)
        return labeled_indices

    def _check_saved_label_indices_if_needed(self, labeled_indices, inconsistent_scenes):
        if not self.check_saved_indices:
            return inconsistent_scenes

        if not self.label_indices_root:
            raise ValueError("label_indices_root must be provided when check_saved_indices=True")

        label_indices_path = os.path.join(
            self.label_indices_root, f"{self.scene_name}_labeled_points.npy"
        )

        if not os.path.exists(label_indices_path):
            raise FileNotFoundError(f"Label index file not found: {label_indices_path}")

        selected_indices = np.load(label_indices_path).astype(np.int32)

        if not np.array_equal(labeled_indices.astype(np.int32), selected_indices):
            print(f"❌ {self.scene_name}: sampled labeled indices are inconsistent with saved file.")
            inconsistent_scenes.append(self.scene_name)

        return inconsistent_scenes

    def update_subset_idx(self, ybar, inconsistent_scenes=None):
        if inconsistent_scenes is None:
            inconsistent_scenes = []

        scene_data_label = self.get_3d_info()

        subset_idx = self._downsample_subset_per_class(scene_data_label)

        if self.setting == "20pt":
            labeled_indices = self._sample_labeled_points_20pt(scene_data_label)
        elif self.setting == "sparse":
            labeled_indices = self._sample_labeled_points_sparse(scene_data_label)
            inconsistent_scenes = self._check_saved_label_indices_if_needed(
                labeled_indices, inconsistent_scenes
            )
        else:
            raise ValueError(f"Unknown setting: {self.setting}")

        labeled_labels = scene_data_label[labeled_indices].astype(np.int32)

        subset_set = set(subset_idx)
        labeled_points_exist_in_downsampled_num = 0
        for idx in labeled_indices:
            if int(idx) not in subset_set:
                subset_idx.append(int(idx))
                subset_set.add(int(idx))
            else:
                labeled_points_exist_in_downsampled_num += 1

        print(f"[{self.scene_name}] labeled_indices num: {len(labeled_indices)}")
        print(
            f"[{self.scene_name}] labeled points already in subset: "
            f"{labeled_points_exist_in_downsampled_num}"
        )

        ybar[labeled_indices, :] = 0.0
        ybar[labeled_indices, labeled_labels] = 1.0

        subset_idx = np.array(subset_idx, dtype=np.int32)
        return subset_idx, ybar, inconsistent_scenes

    def do_hlp(self, inconsistent_scenes=None):
        if inconsistent_scenes is None:
            inconsistent_scenes = []

        self.set_specific_room_h_matrix()

        if len(self.sparse_H_all_frames) == 0:
            raise RuntimeError(
                f"No hyperedges were constructed for scene {self.scene_name}. "
                f"Check add_superpoint/add_sam_superpoint settings."
            )

        sparse_H_combined = cp.sparse.hstack(self.sparse_H_all_frames, format="csr")
        graph = HyperGraph(sparse_H_combined)

        ybar = self.create_ybar()
        subset_idx, ybar, inconsistent_scenes = self.update_subset_idx(ybar, inconsistent_scenes)

        labeled_points_num = int(np.sum(ybar == 1))
        labeled_points_ratio = labeled_points_num / self.point_num * 100.0

        print(
            f"[{self.scene_name}] labeled_points_num: {labeled_points_num}, "
            f"ratio in full scene: {labeled_points_ratio:.6f}%"
        )
        print(f"[{self.scene_name}] subset_idx num: {len(subset_idx)}")

        y_hat = graph.LabelPropagation_Subset(subset_idx, ybar)

        subset_idx_cp = cp.asarray(subset_idx)
        y_hat_with_pointID = cp.hstack((y_hat, subset_idx_cp[:, cp.newaxis]))

        return labeled_points_num, y_hat_with_pointID, inconsistent_scenes


def build_output_dir(output_root, exp_name, add_superpoint, add_sam_superpoint):
    if add_superpoint:
        return os.path.join(
            output_root,
            exp_name,
            "with_superpoint",
            f"AddSAMSuperpoint_{add_sam_superpoint}",
            "HLP_results",
        )
    else:
        return os.path.join(
            output_root,
            exp_name,
            "without_superpoint",
            f"AddSAMSuperpoint_{add_sam_superpoint}",
            "HLP_results",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ScanNet HLP pseudo-labels")

    parser.add_argument(
        "--root",
        type=str,
        default="./data/ScanNet",
        help="Root directory of ScanNet data.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/HLP_data_Scannet",
        help="Root directory for saving HLP outputs.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["20pt", "sparse"],
        required=True,
        help="Label setting for ScanNet.",
    )
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=0.001,
        help="Sparse ratio used only when setting='sparse'.",
    )
    parser.add_argument(
        "--add_superpoint",
        action="store_true",
        help="Use geometric superpoints as hyperedges.",
    )
    parser.add_argument(
        "--add_sam_superpoint",
        action="store_true",
        help="Use SAM mask hyperedges.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=3,
        help="Random seed.",
    )
    parser.add_argument(
        "--check_saved_indices",
        action="store_true",
        help="Whether to check sampled labeled indices against saved index files.",
    )
    parser.add_argument(
        "--label_indices_root",
        type=str,
        default=None,
        help="Directory containing saved labeled point indices. "
             "Required only when --check_saved_indices is enabled.",
    )
    parser.add_argument(
        "--max_points_20pt",
        type=int,
        default=20,
        help="Maximum labeled points for 20pt setting.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    scans_dir = os.path.join(args.root, "scans")
    scene_names = sorted(
        [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]
    )

    print(f"Total scenes: {len(scene_names)}")

    if args.setting == "20pt":
        exp_name = "Scannet_20pt"
    elif args.setting == "sparse":
        exp_name = "Scannet_sparse"
    else:
        raise ValueError(f"Unknown setting: {args.setting}")

    output_dir = build_output_dir(
        args.output_root,
        exp_name,
        args.add_superpoint,
        args.add_sam_superpoint,
    )
    os.makedirs(output_dir, exist_ok=True)

    inconsistent_scenes = []
    labeled_points_total_num = 0
    processed_num = 0

    for i, scene_full_name in enumerate(scene_names, start=1):
        print(f"\nprocess: {i} {scene_full_name}")

        output_file_name = f"{scene_full_name}_Y_hat_with_pointID.txt"
        output_file_path = os.path.join(output_dir, output_file_name)

        if os.path.exists(output_file_path):
            print("Already processed. Skip.")
            processed_num += 1
            continue

        hlp_runner = HLPScannet(
            scene_name=scene_full_name,
            root=args.root,
            setting=args.setting,
            add_superpoint=args.add_superpoint,
            add_sam_superpoint=args.add_sam_superpoint,
            sparse_ratio=args.sparse_ratio if args.setting == "sparse" else None,
            label_indices_root=args.label_indices_root,
            check_saved_indices=args.check_saved_indices,
            random_seed=args.random_seed,
            max_points_20pt=args.max_points_20pt,
        )

        labeled_points_num, y_hat_with_pointID, inconsistent_scenes = hlp_runner.do_hlp(
            inconsistent_scenes=inconsistent_scenes
        )

        labeled_points_total_num += labeled_points_num
        np.savetxt(output_file_path, y_hat_with_pointID, fmt="%f")
        processed_num += 1

    print("\n================ Summary ================")
    print(f"Setting: {args.setting}")
    print(f"Processed scenes: {processed_num}")
    print(f"Total labeled points: {labeled_points_total_num}")
    print(f"Output dir: {output_dir}")

    if args.setting == "sparse":
        print(f"Sparse ratio (internal): {args.sparse_ratio}")

    if args.check_saved_indices:
        if len(inconsistent_scenes) > 0:
            print("\n❌ Inconsistent scenes:")
            for bad_scene in inconsistent_scenes:
                print(f" - {bad_scene}")
        else:
            print("\n✅ All scenes are consistent with saved labeled indices.")


if __name__ == "__main__":
    main()
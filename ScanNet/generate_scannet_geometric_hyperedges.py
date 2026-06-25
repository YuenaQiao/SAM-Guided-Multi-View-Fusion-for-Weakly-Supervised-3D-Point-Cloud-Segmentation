#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def ensure_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def list_scene_ids(scans_dir):
    return sorted(
        [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]
    )


def remap_labels_to_contiguous(labels, invalid_label=None):
    """
    Remap labels to contiguous ids: 0, 1, 2, ...
    If invalid_label is provided (e.g., -1), keep it unchanged.
    """
    labels = np.asarray(labels).copy()

    if invalid_label is None:
        valid_ids = np.unique(labels)
        id_remap = {old_id: new_id for new_id, old_id in enumerate(valid_ids)}
        remapped = np.array([id_remap[x] for x in labels], dtype=np.int32)
        return remapped

    valid_mask = labels != invalid_label
    valid_ids = np.unique(labels[valid_mask])
    id_remap = {old_id: new_id for new_id, old_id in enumerate(valid_ids)}

    remapped = np.full_like(labels, fill_value=invalid_label, dtype=np.int32)
    remapped[valid_mask] = np.array([id_remap[x] for x in labels[valid_mask]], dtype=np.int32)
    return remapped


def load_scene_superpoint(scene_id, scans_dir, distance_threshold=None):
    """
    For a given ScanNet scene:
      1) Load segIndices from vh_clean.segs.json
      2) Load dense points from vh_clean.ply
      3) Load sparse points from vh_clean_2.ply
      4) Project dense supervoxel ids to sparse points via KDTree nearest neighbor
      5) Optionally mark far matches as invalid
      6) Remap valid labels to contiguous ids
    """
    scene_dir = os.path.join(scans_dir, scene_id)

    seg_json_path = os.path.join(scene_dir, f"{scene_id}_vh_clean.segs.json")
    dense_ply_path = os.path.join(scene_dir, f"{scene_id}_vh_clean.ply")
    sparse_ply_path = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")

    if not os.path.exists(seg_json_path):
        raise FileNotFoundError(f"Missing seg json: {seg_json_path}")
    if not os.path.exists(dense_ply_path):
        raise FileNotFoundError(f"Missing dense ply: {dense_ply_path}")
    if not os.path.exists(sparse_ply_path):
        raise FileNotFoundError(f"Missing sparse ply: {sparse_ply_path}")

    # Load seg ids
    with open(seg_json_path, "r", encoding="utf-8") as f:
        seg_data = json.load(f)
    seg_ids = np.asarray(seg_data["segIndices"], dtype=np.int32)

    # Load dense point cloud
    pcd_dense = o3d.io.read_point_cloud(dense_ply_path)
    points_dense = np.asarray(pcd_dense.points, dtype=np.float32)

    # Load sparse point cloud
    pcd_sparse = o3d.io.read_point_cloud(sparse_ply_path)
    points_sparse = np.asarray(pcd_sparse.points, dtype=np.float32)

    if len(seg_ids) != points_dense.shape[0]:
        raise ValueError(
            f"{scene_id}: segIndices length ({len(seg_ids)}) != dense point count ({points_dense.shape[0]})"
        )

    if points_dense.shape[0] == 0:
        raise ValueError(f"{scene_id}: dense point cloud is empty.")
    if points_sparse.shape[0] == 0:
        raise ValueError(f"{scene_id}: sparse point cloud is empty.")

    # KDTree nearest neighbor mapping
    tree = KDTree(points_dense)
    distances, indices = tree.query(points_sparse, k=1)

    mapped_labels = seg_ids[indices[:, 0]].astype(np.int32)

    invalid_count = 0
    if distance_threshold is not None:
        invalid_mask = distances[:, 0] > distance_threshold
        invalid_count = int(np.sum(invalid_mask))
        mapped_labels[invalid_mask] = -1
        mapped_labels = remap_labels_to_contiguous(mapped_labels, invalid_label=-1)
    else:
        mapped_labels = remap_labels_to_contiguous(mapped_labels, invalid_label=None)

    if len(mapped_labels) != points_sparse.shape[0]:
        raise ValueError(
            f"{scene_id}: remapped label count ({len(mapped_labels)}) != sparse point count ({points_sparse.shape[0]})"
        )

    valid_mask = mapped_labels != -1
    unique_valid = np.unique(mapped_labels[valid_mask]) if np.any(valid_mask) else np.array([], dtype=np.int32)

    stats = {
        "dense_points": int(points_dense.shape[0]),
        "sparse_points": int(points_sparse.shape[0]),
        "valid_superpoints": int(len(unique_valid)),
        "invalid_points": int(invalid_count),
    }

    return mapped_labels.astype(np.int32), stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ScanNet superpoint labels for vh_clean_2.ply"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/ScanNet",
        help="Root directory of ScanNet data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for superpoint npy files. "
             "Default: <root>/Superpoint",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Process only one scene. Default: process all scenes.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=None,
        help="Optional max NN distance. Sparse points with larger distance "
             "will be marked as -1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scans_dir = os.path.join(args.root, "scans")
    output_dir = args.output_dir if args.output_dir is not None else os.path.join(args.root, "Superpoint")
    ensure_dir(output_dir)

    if args.scene_id is not None:
        scene_ids = [args.scene_id]
    else:
        scene_ids = list_scene_ids(scans_dir)

    print(f"Total scenes to process: {len(scene_ids)}")
    print(f"Output dir: {output_dir}")

    processed_num = 0
    skipped_num = 0

    for i, scene_id in enumerate(scene_ids, start=1):
        print(f"\nprocess: {i} {scene_id}")

        output_path = os.path.join(output_dir, f"{scene_id}_supervoxel_labels.npy")

        if os.path.exists(output_path) and not args.overwrite:
            print("Output already exists. Skip.")
            skipped_num += 1
            continue

        scene_superpoint, stats = load_scene_superpoint(
            scene_id=scene_id,
            scans_dir=scans_dir,
            distance_threshold=args.distance_threshold,
        )

        np.save(output_path, scene_superpoint)

        print(
            f"Saved: {output_path}\n"
            f"  dense_points      = {stats['dense_points']}\n"
            f"  sparse_points     = {stats['sparse_points']}\n"
            f"  valid_superpoints = {stats['valid_superpoints']}\n"
            f"  invalid_points    = {stats['invalid_points']}"
        )

        processed_num += 1

    print("\n================ Summary ================")
    print(f"Processed scenes: {processed_num}")
    print(f"Skipped scenes:   {skipped_num}")
    print(f"Output dir:       {output_dir}")


if __name__ == "__main__":
    main()
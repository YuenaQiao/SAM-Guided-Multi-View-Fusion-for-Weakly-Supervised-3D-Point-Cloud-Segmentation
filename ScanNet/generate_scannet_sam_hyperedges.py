#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


INVALID_GROUP_ID = -1  # Point/pixel not assigned to any valid SAM mask.


def ensure_dir(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def list_scene_ids(scans_dir: str) -> List[str]:
    return sorted(
        [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]
    )


def numeric_stem_sort_key(filename: str):
    """Sort ScanNet frame files whose names are usually numeric, e.g. 0.jpg."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def remap_group_ids_to_contiguous(labels: np.ndarray, invalid_group_id: int = INVALID_GROUP_ID) -> np.ndarray:
    """
    Remap valid labels to contiguous ids: 0, 1, 2, ...
    Invalid group ids, e.g. -1, are kept unchanged.
    """
    labels = np.asarray(labels).copy()
    valid_mask = labels != invalid_group_id

    if not np.any(valid_mask):
        return labels.astype(np.int32)

    valid_ids = np.unique(labels[valid_mask])
    id_remap = {old_id: new_id for new_id, old_id in enumerate(valid_ids)}

    remapped = np.full(labels.shape, fill_value=invalid_group_id, dtype=np.int32)
    remapped[valid_mask] = np.array(
        [id_remap[x] for x in labels[valid_mask]], dtype=np.int32
    )
    return remapped


def setup_logger(log_path: str) -> None:
    ensure_dir(os.path.dirname(log_path))
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_sam_mask_generator(
    sam_checkpoint: str,
    sam_model_type: str = "vit_h",
    device: str = "cuda",
):
    """Build the Segment Anything automatic mask generator."""
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if sam_model_type not in sam_model_registry:
        valid_types = ", ".join(sorted(sam_model_registry.keys()))
        raise ValueError(
            f"Unsupported SAM model type: {sam_model_type}. Available types: {valid_types}"
        )

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)


def generate_sam_group_ids(image: np.ndarray, mask_generator) -> np.ndarray:
    """
    Generate a 2D group-id map from an image using SAM.

    This mirrors the original script: SAM masks are assigned raw ids in reverse
    order. The 2D mask is remapped only when saving to PNG, and the projected
    valid-depth pixels are remapped after depth masking.
    """
    masks = mask_generator.generate(image)
    group_ids = np.full(image.shape[:2], INVALID_GROUP_ID, dtype=int)

    group_counter = 0
    for i in reversed(range(len(masks))):
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1

    return group_ids


def load_2d_group_ids(mask_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Load a saved 2D group-id PNG and resize it to the depth image size if needed.

    16-bit PNG does not store negative values reliably, so this script saves
    invalid pixels (-1) as 65535 and restores them to -1 when loading.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing saved 2D SAM mask: {mask_path}")

    group_ids = np.asarray(Image.open(mask_path), dtype=np.int32)
    group_ids[group_ids == 65535] = INVALID_GROUP_ID
    target_h, target_w = target_hw

    if group_ids.shape[:2] != (target_h, target_w):
        group_ids = cv2.resize(
            group_ids,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return group_ids


def save_2d_group_ids(group_ids: np.ndarray, mask_path: str) -> None:
    """Save group ids in the same way as the original script."""
    ensure_dir(os.path.dirname(mask_path))
    save_array = remap_group_ids_to_contiguous(
        group_ids, invalid_group_id=INVALID_GROUP_ID
    ).astype(np.int16)
    Image.fromarray(save_array, mode="I;16").save(mask_path)


def load_scene_points(scene_id: str, scans_dir: str) -> np.ndarray:
    """Load ScanNet sparse point coordinates from `<scene_id>_vh_clean_2.ply`."""
    sparse_ply_path = os.path.join(scans_dir, scene_id, f"{scene_id}_vh_clean_2.ply")

    if not os.path.exists(sparse_ply_path):
        raise FileNotFoundError(f"Missing sparse ply: {sparse_ply_path}")

    ply_data = PlyData.read(sparse_ply_path)
    vertex = ply_data["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    if points.shape[0] == 0:
        raise ValueError(f"{scene_id}: sparse point cloud is empty.")

    return points


def get_frame_paths(scene_id: str, frame_name: str, rgb_path: str) -> Dict[str, str]:
    frame_id = os.path.splitext(frame_name)[0]
    scene_rgbd_dir = os.path.join(rgb_path, scene_id)
    return {
        "intrinsic": os.path.join(scene_rgbd_dir, "intrinsics", "intrinsic_depth.txt"),
        "pose": os.path.join(scene_rgbd_dir, "pose", f"{frame_id}.txt"),
        "depth": os.path.join(scene_rgbd_dir, "depth", f"{frame_id}.png"),
        "color": os.path.join(scene_rgbd_dir, "color", frame_name),
    }


def back_project_frame(
    scene_id: str,
    frame_name: str,
    rgb_path: str,
    mask_2d_dir: str,
    mask_generator=None,
    use_saved_2d_masks: bool = False,
    save_2d_masks: bool = True,
    image_size: Tuple[int, int] = (640, 480),
    depth_shift: float = 1000.0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Back-project one RGB-D frame to 3D and attach the corresponding 2D SAM group id.

    Returns None if the camera pose is invalid.
    """
    frame_id = os.path.splitext(frame_name)[0]
    paths = get_frame_paths(scene_id, frame_name, rgb_path)

    for key, path in paths.items():
        if key == "color" and use_saved_2d_masks:
            # Color image is unnecessary when masks are loaded from disk.
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {key} file: {path}")

    depth_intrinsic = np.loadtxt(paths["intrinsic"])
    pose = np.loadtxt(paths["pose"])

    if not np.all(np.isfinite(pose)) or pose.shape != (4, 4):
        logging.warning(f"Invalid pose. Skip frame: {scene_id} {frame_name}")
        return None

    depth_img = cv2.imread(paths["depth"], cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"Failed to read depth image: {paths['depth']}")

    target_h, target_w = depth_img.shape[:2]
    mask_path = os.path.join(mask_2d_dir, scene_id, "image", f"{frame_id}.png")

    if use_saved_2d_masks:
        group_ids = load_2d_group_ids(mask_path, target_hw=(target_h, target_w))
    else:
        if mask_generator is None:
            raise ValueError("mask_generator is required when use_saved_2d_masks=False.")

        color_bgr = cv2.imread(paths["color"], cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Failed to read color image: {paths['color']}")

        # Keep the same preprocessing as the original script: cv2.imread returns BGR,
        # and the BGR image is directly passed to SAM.
        width, height = image_size
        color_bgr = cv2.resize(color_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        group_ids = generate_sam_group_ids(color_bgr, mask_generator)

        if group_ids.shape[:2] != (target_h, target_w):
            group_ids = cv2.resize(
                group_ids,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        if save_2d_masks:
            save_2d_group_ids(group_ids, mask_path)

    valid_depth_mask = depth_img != 0
    if not np.any(valid_depth_mask):
        logging.warning(f"Empty depth. Skip frame: {scene_id} {frame_name}")
        return None

    # Keep the original script's projection order and dtype behavior:
    # build a full uv-depth grid, flatten it in row-major order, then keep
    # non-zero depth pixels.
    projected_groups = group_ids[valid_depth_mask]
    projected_groups = remap_group_ids_to_contiguous(
        projected_groups, invalid_group_id=INVALID_GROUP_ID
    )

    x_grid, y_grid = np.meshgrid(
        np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
        np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]),
    )
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x_grid
    uv_depth[:, :, 1] = y_grid
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()

    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]

    n = uv_depth.shape[0]
    points = np.ones((n, 4))
    points[:, 0] = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    points[:, 1] = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 2] = uv_depth[:, 2]

    points_world = np.dot(points, np.transpose(pose))

    return {
        "points": points_world[:, :3],
        "groups": projected_groups,
    }


def majority_vote_group_ids(
    point_indices: np.ndarray,
    group_ids: np.ndarray,
    num_points: int,
    invalid_group_id: int = INVALID_GROUP_ID,
    ignore_invalid_groups: bool = False,
) -> np.ndarray:
    """
    Assign each scene point a group id by majority voting among projected pixels.
    """
    point_indices = np.asarray(point_indices, dtype=np.int64)
    group_ids = np.asarray(group_ids, dtype=int)

    if ignore_invalid_groups:
        valid_mask = group_ids != invalid_group_id
        point_indices = point_indices[valid_mask]
        group_ids = group_ids[valid_mask]

    output = np.full(num_points, invalid_group_id, dtype=int)
    if len(point_indices) == 0:
        return output

    order = np.argsort(point_indices)
    point_indices = point_indices[order]
    group_ids = group_ids[order]

    boundaries = np.concatenate(
        ([0], np.where(np.diff(point_indices) != 0)[0] + 1, [len(point_indices)])
    )

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        point_id = point_indices[start]
        labels, counts = np.unique(group_ids[start:end], return_counts=True)
        output[point_id] = labels[np.argmax(counts)]

    return output


def process_scene(
    scene_id: str,
    scans_dir: str,
    rgb_path: str,
    output_dir: str,
    mask_2d_dir: str,
    mask_generator=None,
    use_saved_2d_masks: bool = False,
    save_2d_masks: bool = True,
    image_size: Tuple[int, int] = (640, 480),
    depth_shift: float = 1000.0,
    projection_distance_threshold: Optional[float] = None,
    avg_dist_warning: float = 0.5,
    max_dist_warning: float = 1.5,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
    remap_output: bool = False,
) -> Dict[str, int]:
    """Generate all per-frame SAM hyperedges for one ScanNet scene."""
    scene_points = load_scene_points(scene_id, scans_dir)
    scene_tree = cKDTree(scene_points)

    color_dir = os.path.join(rgb_path, scene_id, "color")
    if not os.path.exists(color_dir):
        raise FileNotFoundError(f"Missing color directory: {color_dir}")

    frame_names = [f for f in os.listdir(color_dir) if f.lower().endswith(".jpg")]
    frame_names = sorted(frame_names, key=numeric_stem_sort_key)
    frame_names = frame_names[::frame_stride]

    if max_frames is not None:
        frame_names = frame_names[:max_frames]

    scene_output_dir = os.path.join(output_dir, scene_id)
    ensure_dir(scene_output_dir)

    processed_frames = 0
    skipped_frames = 0
    invalid_pose_frames = 0

    print(f"  sparse_points = {scene_points.shape[0]}")
    print(f"  frames        = {len(frame_names)}")

    for frame_index, frame_name in enumerate(frame_names, start=1):
        frame_id = os.path.splitext(frame_name)[0]
        output_path = os.path.join(scene_output_dir, f"{scene_id}_{frame_id}_group_ids.npy")

        if os.path.exists(output_path) and not overwrite:
            skipped_frames += 1
            continue

        print(f"  frame {frame_index:04d}/{len(frame_names):04d}: {frame_name}")

        projected_data = back_project_frame(
            scene_id=scene_id,
            frame_name=frame_name,
            rgb_path=rgb_path,
            mask_2d_dir=mask_2d_dir,
            mask_generator=mask_generator,
            use_saved_2d_masks=use_saved_2d_masks,
            save_2d_masks=save_2d_masks,
            image_size=image_size,
            depth_shift=depth_shift,
        )

        if projected_data is None:
            invalid_pose_frames += 1
            continue

        projected_points = projected_data["points"]
        projected_groups = projected_data["groups"]

        distances, indices = scene_tree.query(projected_points, k=1)
        avg_dist = float(np.mean(distances))
        max_dist = float(np.max(distances))

        if avg_dist > avg_dist_warning or max_dist > max_dist_warning:
            logging.warning(
                f"Large projection distance: scene={scene_id}, frame={frame_name}, "
                f"avg_dist={avg_dist:.4f}, max_dist={max_dist:.4f}"
            )

        if projection_distance_threshold is not None:
            valid_projection_mask = distances <= projection_distance_threshold
            indices = indices[valid_projection_mask]
            projected_groups = projected_groups[valid_projection_mask]

        scene_group_ids = majority_vote_group_ids(
            point_indices=indices,
            group_ids=projected_groups,
            num_points=scene_points.shape[0],
            invalid_group_id=INVALID_GROUP_ID,
            ignore_invalid_groups=False,
        )

        if remap_output:
            scene_group_ids = remap_group_ids_to_contiguous(
                scene_group_ids, invalid_group_id=INVALID_GROUP_ID
            )

        np.save(output_path, scene_group_ids)

        valid_points = int(np.sum(scene_group_ids != INVALID_GROUP_ID))
        valid_masks = int(len(np.unique(scene_group_ids[scene_group_ids != INVALID_GROUP_ID])))
        print(
            f"    saved: {output_path}\n"
            f"    valid_points = {valid_points}, valid_masks = {valid_masks}, "
            f"avg_dist = {avg_dist:.4f}, max_dist = {max_dist:.4f}"
        )

        processed_frames += 1

    return {
        "processed_frames": processed_frames,
        "skipped_frames": skipped_frames,
        "invalid_pose_frames": invalid_pose_frames,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ScanNet SAM hyperedges by back-projecting 2D SAM masks to 3D."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/ScanNet",
        help="Root directory of ScanNet data. It should contain the `scans` folder.",
    )
    parser.add_argument(
        "--rgb_path",
        type=str,
        default="./data/scannetv2_images",
        help="Root directory of extracted ScanNet RGB-D frames.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for per-frame SAM hyperedge npy files. "
        "Default: <root>/SAM_output",
    )
    parser.add_argument(
        "--mask_2d_dir",
        type=str,
        default=None,
        help="Directory for saved 2D SAM masks. Default: same as output_dir.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint. If omitted, --use_saved_2d_masks is enabled automatically.",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for SAM inference, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--use_saved_2d_masks",
        action="store_true",
        help="Load saved 2D SAM masks instead of running SAM online.",
    )
    parser.add_argument(
        "--no_save_2d_masks",
        action="store_false",
        dest="save_2d_masks",
        help="Do not save generated 2D SAM masks.",
    )
    parser.set_defaults(save_2d_masks=True)
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Process only one scene. Default: process all scenes.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index of the sorted scene list, useful for splitting jobs.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index of the sorted scene list, useful for splitting jobs.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every N-th frame. Default: 1.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames per scene. Useful for debugging.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=640,
        help="Width used when resizing color images before SAM inference.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=480,
        help="Height used when resizing color images before SAM inference.",
    )
    parser.add_argument(
        "--depth_shift",
        type=float,
        default=1000.0,
        help="Depth scale factor. ScanNet depth is usually stored in millimeters.",
    )
    parser.add_argument(
        "--projection_distance_threshold",
        type=float,
        default=None,
        help="Optional max NN distance for projected points. Larger distances are ignored.",
    )
    parser.add_argument(
        "--avg_dist_warning",
        type=float,
        default=0.5,
        help="Log a warning if average projection distance is larger than this value.",
    )
    parser.add_argument(
        "--max_dist_warning",
        type=float,
        default=1.5,
        help="Log a warning if maximum projection distance is larger than this value.",
    )
    parser.add_argument(
        "--remap_output",
        action="store_true",
        help="Optionally remap output group ids to contiguous ids after majority voting. "
             "The original script does not do this final remapping.",
    )
    parser.set_defaults(remap_output=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-frame output npy files.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to the warning log file. Default: <output_dir>/back_project_warning_log.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scans_dir = os.path.join(args.root, "scans")
    output_dir = args.output_dir if args.output_dir is not None else os.path.join(args.root, "SAM_output")
    mask_2d_dir = args.mask_2d_dir if args.mask_2d_dir is not None else output_dir
    log_path = args.log_path if args.log_path is not None else os.path.join(output_dir, "back_project_warning_log.txt")

    ensure_dir(output_dir)
    setup_logger(log_path)

    if args.scene_id is not None:
        scene_ids = [args.scene_id]
    else:
        scene_ids = list_scene_ids(scans_dir)
        scene_ids = scene_ids[args.start_idx : args.end_idx]

    use_saved_2d_masks = args.use_saved_2d_masks or args.sam_checkpoint is None

    if use_saved_2d_masks:
        mask_generator = None
        print("Use saved 2D SAM masks. SAM inference will be skipped.")
    else:
        if not os.path.exists(args.sam_checkpoint):
            raise FileNotFoundError(f"Missing SAM checkpoint: {args.sam_checkpoint}")
        print(f"Build SAM mask generator: model={args.sam_model_type}, device={args.device}")
        mask_generator = build_sam_mask_generator(
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            device=args.device,
        )

    print(f"Total scenes to process: {len(scene_ids)}")
    print(f"Output dir: {output_dir}")
    print(f"2D mask dir: {mask_2d_dir}")
    print(f"Log file:    {log_path}")

    processed_scenes = 0
    failed_scenes = 0
    total_processed_frames = 0
    total_skipped_frames = 0
    total_invalid_pose_frames = 0

    image_size = (args.image_width, args.image_height)

    for i, scene_id in enumerate(scene_ids, start=1):
        print(f"\nprocess: {i} {scene_id}")
        try:
            stats = process_scene(
                scene_id=scene_id,
                scans_dir=scans_dir,
                rgb_path=args.rgb_path,
                output_dir=output_dir,
                mask_2d_dir=mask_2d_dir,
                mask_generator=mask_generator,
                use_saved_2d_masks=use_saved_2d_masks,
                save_2d_masks=args.save_2d_masks,
                image_size=image_size,
                depth_shift=args.depth_shift,
                projection_distance_threshold=args.projection_distance_threshold,
                avg_dist_warning=args.avg_dist_warning,
                max_dist_warning=args.max_dist_warning,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
                remap_output=args.remap_output,
            )
        except Exception as exc:  # Keep batch processing robust for GitHub/data preprocessing use.
            logging.exception(f"Failed scene: {scene_id}. Error: {exc}")
            print(f"  [error] failed scene: {scene_id}. See log for details.")
            failed_scenes += 1
            continue

        processed_scenes += 1
        total_processed_frames += stats["processed_frames"]
        total_skipped_frames += stats["skipped_frames"]
        total_invalid_pose_frames += stats["invalid_pose_frames"]

    print("\n================ Summary ================")
    print(f"Processed scenes:      {processed_scenes}")
    print(f"Failed scenes:         {failed_scenes}")
    print(f"Processed frames:      {total_processed_frames}")
    print(f"Skipped frames:        {total_skipped_frames}")
    print(f"Invalid-pose frames:   {total_invalid_pose_frames}")
    print(f"Output dir:            {output_dir}")
    print(f"Log file:              {log_path}")


if __name__ == "__main__":
    main()

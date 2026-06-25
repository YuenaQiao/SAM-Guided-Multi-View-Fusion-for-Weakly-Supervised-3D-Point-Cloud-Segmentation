"""
Generate a .pth label assignment file from initial labeled point indices.

The output file is used as `la_file` in the training config to indicate which
points are originally annotated in each scene.
"""
import os
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a label assignment file from labeled point index files."
    )
    parser.add_argument(
        "--labeled_root",
        type=str,
        required=True,
        help="Folder containing initial labeled point index files (*_labeled_points.npy), e.g., ../labeled_points/20pt.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="la_file.pth",
        help="Output path for the generated .pth label assignment file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    la_dict = {}

    for fname in sorted(os.listdir(args.labeled_root)):
        if not fname.endswith("_labeled_points.npy"):
            continue

        # Get scene name by removing the suffix "_labeled_points.npy".
        scene_name = fname.replace("_labeled_points.npy", "")

        file_path = os.path.join(args.labeled_root, fname)

        # Load labeled point indices.
        indices = np.load(file_path)

        la_dict[scene_name] = indices

    # Save the label assignment dictionary as a .pth file.
    torch.save(la_dict, args.output_path)
    print(f"Saved label assignment file to: {args.output_path}")


if __name__ == "__main__":
    main()
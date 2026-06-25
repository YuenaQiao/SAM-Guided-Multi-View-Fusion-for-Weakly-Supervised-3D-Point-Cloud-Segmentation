import argparse
import os

import numpy as np
import open3d as o3d
import torch


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' has been created.")
    else:
        print(f"Folder '{folder_path}' already exists. No need to create it.")


def process_area_folder(area_folder_path, save_path, area_name):
    pth_files = [file for file in os.listdir(area_folder_path) if file.endswith(".pth")]
    for pth_file in pth_files:
        pth_file_name = os.path.splitext(pth_file)[0]
        pth_file_path = os.path.join(area_folder_path, pth_file)
        process_point_cloud(pth_file_path, save_path, area_name, pth_file_name)


def _safe_up_vector(camera_direction):
    """Return an up vector that is not parallel to the camera direction."""
    z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    camera_direction = camera_direction / (np.linalg.norm(camera_direction) + 1e-12)
    if abs(np.dot(camera_direction, z_up)) > 0.95:
        return y_up.tolist()
    return z_up.tolist()


def _save_camera_parameters(param_name, renderer, width, height, fov):
    focal = 0.5 * height / np.tan(np.deg2rad(fov) * 0.5)
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        focal,
        focal,
        (width - 1) * 0.5,
        (height - 1) * 0.5,
    )
    param.extrinsic = np.asarray(renderer.scene.camera.get_view_matrix(), dtype=np.float64)
    o3d.io.write_pinhole_camera_parameters(param_name, param)


def process_point_cloud(pth_path, save_path, area_name, specific_area_name):
    saved_data = torch.load(pth_path)
    points = saved_data["coord"].astype(np.float64)
    colors = saved_data["color"].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create output folders
    image_folder_path = os.path.join(save_path, area_name, specific_area_name, "image")
    depth_folder_path = os.path.join(save_path, area_name, specific_area_name, "depth")
    json_folder_path = os.path.join(save_path, area_name, specific_area_name, "json")
    create_folder_if_not_exists(image_folder_path)
    create_folder_if_not_exists(depth_folder_path)
    create_folder_if_not_exists(json_folder_path)

    # Create an offscreen renderer
    width, height = 1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("pcd", pcd, mat)

    # Camera settings
    fov = 60.0
    center = pcd.get_center()
    cam_distance = max(np.linalg.norm(points, axis=1))

    def save_view(view_idx, cam_pos, up):
        renderer.setup_camera(fov, center, cam_pos, up)

        # Save RGB image
        image_name = os.path.join(
            image_folder_path, f"{area_name}_{specific_area_name}_Rotate_{view_idx}.png"
        )
        image = renderer.render_to_image()
        o3d.io.write_image(image_name, image)

        # Save depth image
        depth_name = os.path.join(
            depth_folder_path, f"{area_name}_{specific_area_name}_Rotate_Depth_{view_idx}.png"
        )
        depth = renderer.render_to_depth_image(True)
        o3d.io.write_image(depth_name, depth)

        # Save camera parameters
        param_name = os.path.join(
            json_folder_path, f"{area_name}_{specific_area_name}_Rotate_{view_idx}.json"
        )
        _save_camera_parameters(param_name, renderer, width, height, fov)

    # 1) Generate 36 horizontal views: Rotate_0 ~ Rotate_35
    num_horizontal_views = 36
    horizontal_step = 360.0 / num_horizontal_views
    for i in range(num_horizontal_views):
        angle = i * horizontal_step
        cam_pos = center + np.array(
            [
                np.cos(np.deg2rad(angle)) * cam_distance,
                np.sin(np.deg2rad(angle)) * cam_distance,
                0.0,
            ],
            dtype=np.float64,
        )
        save_view(i, cam_pos, [0, 0, 1])

    # 2) Generate 10 vertical views: Rotate_36 ~ Rotate_45
    # The camera moves on a vertical circle. A half-step offset avoids duplicating
    # the horizontal front/back views exactly.
    num_vertical_views = 10
    vertical_step = 360.0 / num_vertical_views
    fixed_y = 0.0
    for j in range(num_vertical_views):
        view_idx = num_horizontal_views + j
        angle = (j + 0.5) * vertical_step
        cam_pos = center + np.array(
            [
                np.cos(np.deg2rad(angle)) * cam_distance,
                fixed_y,
                np.sin(np.deg2rad(angle)) * cam_distance,
            ],
            dtype=np.float64,
        )
        camera_direction = center - cam_pos
        up = _safe_up_vector(camera_direction)
        save_view(view_idx, cam_pos, up)

    renderer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render 46 2D views from S3DIS point cloud pth files, including 36 horizontal views and 10 vertical views."
    )
    parser.add_argument(
        "--root_folder",
        type=str,
        default="./data/s3dis",
        help="Root folder of S3DIS pth files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/3Dto2D",
        help="Output folder for rendered images, depths and camera json files.",
    )

    args = parser.parse_args()

    for area_folder in os.listdir(args.root_folder):
        area_folder_path = os.path.join(args.root_folder, area_folder)
        if os.path.isdir(area_folder_path):
            process_area_folder(area_folder_path, args.save_path, area_folder)

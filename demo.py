import numpy as np
from PIL import Image
import open3d as o3d
import depth_pro
import ipdb
import torch
import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i',"--image")
    opt=parser.parse_args()

    """
    Main function to perform depth estimation on a single RGB image and display its 3D reconstruction.
    """
    # Load the pre-trained model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device = torch.device("cuda"))
    model.eval()

    # Load and preprocess the RGB image
    image_path=opt.image
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_transformed = transform(image)

    # Run inference to get the depth map
    prediction = model.infer(image_transformed, f_px=f_px)
    depth_map = prediction["depth"]  # Depth in meters
    focallength_px = prediction["focallength_px"] 
    #ipdb.set_trace()
    depth_map=depth_map.to('cpu').numpy()
    focallength_px = focallength_px.to('cpu').numpy()

    # Generate the point cloud from the depth map and RGB image
    image_rgb = np.array(Image.open(image_path).convert('RGB'))
    point_cloud = create_point_cloud(depth_map, image_rgb, focallength_px)

    # Visualize the point cloud using Open3D
    visualize_point_cloud(point_cloud)

def create_point_cloud(depth_map, image_rgb, focal_length_px):
    """
    Create a 3D point cloud from the depth map and RGB image.

    Parameters:
        depth_map (np.ndarray): 2D array of depth values.
        image_rgb (np.ndarray): 3D array of RGB values.
        focal_length_px (float): Focal length in pixels.

    Returns:
        o3d.geometry.PointCloud: The generated point cloud.
    """
    # Image dimensions
    height, width = depth_map.shape

    # Intrinsic camera parameters
    fx = fy = focal_length_px  # Focal lengths
    cx = width / 2.0           # Principal point x-coordinate
    cy = height / 2.0          # Principal point y-coordinate

    # Create a grid of (x, y) coordinates
    x_indices = np.linspace(0, width - 1, width)
    y_indices = np.linspace(0, height - 1, height)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

    # Flatten the grids and depth map for vectorized computation
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth_map.flatten()

    # Filter out invalid depth values (e.g., zeros)
    valid_mask = z_flat > 0
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]

    valid_mask = z_flat < 70
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]

    # Compute the 3D coordinates
    x_3d = (x_valid - cx) * z_valid / fx
    y_3d = (y_valid - cy) * z_valid / fy
    z_3d = z_valid

    # Stack the coordinates into a single array
    points_3d = np.vstack((x_3d, y_3d, z_3d)).transpose()

    # Normalize the RGB values and filter with the valid mask
    colors = image_rgb.reshape(-1, 3)[valid_mask] / 255.0

    # Create the Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

def visualize_point_cloud(point_cloud):
    """
    Visualize the point cloud using Open3D.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The point cloud to visualize.
    """
    # Set up the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Reconstruction', width=1280, height=720)
    vis.add_geometry(point_cloud)

    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # Black background
    render_option.point_size = 2.0

    # Add coordinate axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axes)

    # Run the visualization
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()

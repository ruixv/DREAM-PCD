"""
RadarEyes Dataset Processing

This Python script is dedicated to the processing of the RadarEyes dataset, a comprehensive collection for the DREAM-PCD project focused on Deep Reconstruction and Enhancement of mmWave Radar Pointcloud data. The script encompasses a variety of functions essential for manipulating millimeter-wave radar point cloud data. Key operations include the conversion of point clouds between numerous formats (e.g., ndarray to Open3D, Cartesian voxel to PCD, polar voxel to PCD), the application of transformations on point clouds with given position and orientation data, as well as the translation of quaternions into Euler angles.

The RadarEyes dataset on GitHub is at https://github.com/ruixv/RadarEyes.

Author: USTC IP_LAB millimeter wave point cloud group
Main Contributors: Ruixu Geng (gengruixu@mail.ustc.edu.cn), Yadong Li, Jincheng Wu, Yating Gao
Copyright (C) USTC IP_LAB, 2023.
Initial Release Date: March 2023
"""

import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
try:
    import torch
except:
    Warning("Warning: PyTorch is not installed")
import pdb



def ndarrayPCD_to_o3dPCD(ndarray_pcd):
    """
    Converts an ndarray format point cloud to an Open3D format point cloud.

    :param ndarray_pcd: Input point cloud as an ndarray.
    :return: A tuple containing the color map and the Open3D format point cloud.
    """

    # Create an Open3D PointCloud object
    o3d_pcd = o3d.geometry.PointCloud()

    # Set the point coordinates in the Open3D PointCloud object
    o3d_pcd.points = o3d.utility.Vector3dVector(ndarray_pcd[:, :3])

    # Check if the input point cloud has color information
    if ndarray_pcd.shape[1] > 3:
        # Normalize the color values and create a color map using the 'jet' colormap
        cmap = plt.cm.get_cmap('jet')
        # pdb.set_trace()
        ndarray_pcd[:, 3] = np.log10(np.log10(ndarray_pcd[:, 3] + 1))
        colors = cmap(ndarray_pcd[:, 3] / ndarray_pcd[:, 3].max())

        # Set the colors in the Open3D PointCloud object
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return colors, o3d_pcd


def pcd2CartVoxel(pcd, range_fftsize, angle_fftsize, x_bin, y_bin, z_bin):
    # Initialize a 3D tensor (cartvoxel) with all zeros
    cartvoxel = torch.zeros(range_fftsize, angle_fftsize, angle_fftsize)

    # Coordinate system:
    # x: azimuth (facing the back of the Lidar, right-hand side is positive)
    # y: distance (towards the front of the Lidar is positive)
    # z: elevation (upwards is positive)

    # TODO: Modify coloradar
    if type(pcd) is torch.Tensor:
        # Compute indices for x, y, and z 
        x_index = ((pcd[:,0] + 15)/x_bin).long()
        y_index = ((pcd[:,1])/y_bin).long()
        z_index = ((pcd[:,2]+3)/z_bin).long()

        # Cap the indices to be between 0 and 127
        # For x_index
        x_index[x_index > 127] = 0  # Objects outside of the 15m range
        x_index[x_index < 0] = 0    # Objects beyond -15m
        y_index[x_index > 127] = 0
        y_index[x_index < 0] = 0
        z_index[x_index > 127] = 0
        z_index[x_index < 0] = 0

        # For y_index
        x_index[y_index > 127] = 0  # Objects beyond 20m
        y_index[y_index > 127] = 0
        x_index[y_index < 0] = 0  # Theoretically non-existent
        y_index[y_index < 0] = 0
        z_index[y_index > 127] = 0
        z_index[y_index < 0] = 0

        # For z_index
        x_index[z_index > 127] = 0  # Objects above 3m
        y_index[z_index > 127] = 0
        x_index[z_index < 0] = 0  # Objects below -3m
        y_index[z_index < 0] = 0
        z_index[z_index > 127] = 0
        z_index[z_index < 0] = 0
    else:
        # Similar operation as above but for numpy arrays instead of tensors
        # Compute indices for x, y, and z 
        x_index = ((pcd[:,0] + 15)/x_bin).astype(int)
        y_index = ((pcd[:,1])/y_bin).astype(int)
        z_index = ((pcd[:,2]+3)/z_bin).astype(int)

        # Cap the indices to be between 0 and 127
        # Similar operations for x_index, y_index, and z_index as above

    # Mark the position of points in the cartvoxel tensor
    cartvoxel[x_index, y_index, z_index] = 1
    # Make sure the origin (0,0,0) is not marked
    cartvoxel[0, 0, 0] = 0

    return cartvoxel

def pcd2PolarVoxel(pcd, range_fftsize, angle_fftsize, range_bin, theta_bin, phi_bin):
    """
    Converts the input point cloud (pcd) to a polar coordinate voxel representation.
    This function supports both PyTorch tensor and numpy ndarray inputs.

    :param pcd: Input point cloud as a PyTorch tensor or numpy ndarray.
    :param range_fftsize: Range FFT size.
    :param angle_fftsize: Angle FFT size.
    :param range_bin: Range bin size.
    :param theta_bin: Theta bin size.
    :param phi_bin: Phi bin size.
    :return: Polar voxel representation of the input point cloud.
    """

    polarvoxel = torch.zeros(range_fftsize, angle_fftsize, angle_fftsize)

    # Convert the input point cloud to a PyTorch tensor if it is a numpy ndarray
    if isinstance(pcd, np.ndarray):
        pcd = torch.from_numpy(pcd)

    r = torch.sqrt(pcd[:, 0] ** 2 + pcd[:, 1] ** 2 + pcd[:, 2] ** 2)
    r_index = (r / range_bin).long()

    theta = torch.atan(pcd[:, 0] / (pcd[:, 1] + 1e-8))  # -90 ~ 90
    sin_theta = torch.sin(theta) # -1 ~ 1
    sin_theta_index = ((sin_theta + 1) / theta_bin).long()  # 0 ~ 2

    phi = torch.asin(pcd[:, 2] / (r + 1e-8))
    sin_phi = torch.sin(phi)
    sin_phi_index = ((sin_phi + 1) / phi_bin).long()

    valid_index = (r_index < 128) & (sin_theta_index < 128) & (sin_phi_index < 128)
    r_index = r_index[valid_index]
    sin_theta_index = sin_theta_index[valid_index]
    sin_phi_index = sin_phi_index[valid_index]

    polarvoxel[r_index, sin_theta_index, sin_phi_index] = 1
    return polarvoxel


def CartVoxel2PCD(outputPCD, range_fftsize=128, angle_fftsize=128, x_bin=30/128, y_bin=20/128, z_bin=6/128):
    """
    This function converts Cartesian voxel output to point cloud data (PCD).
    
    Args:
    outputPCD (torch.Tensor): Input Cartesian voxel data.
    range_fftsize (int, optional): Range FFT size. Default is 128.
    angle_fftsize (int, optional): Angle FFT size. Default is 128.
    x_bin (float, optional): X-axis bin size. Default is 24/128.
    y_bin (float, optional): Y-axis bin size. Default is 40/128.
    z_bin (float, optional): Z-axis bin size. Default is 12/128.
    
    Returns:
    torch.Tensor: Converted point cloud data.
    """
    if len(outputPCD.shape) == 5:
        # [bs, ch, r_index, sin_theta_index, sin_phi_index]
        outputPCD = outputPCD[0,0,:,:,:].squeeze()

    point_index = torch.nonzero(outputPCD > 0)
    pcd = torch.zeros(len(point_index), 3)
    
    for point_i in range(len(point_index)):
        [x_index, y_index, z_index] = point_index[point_i]
        x = x_index * x_bin - 15
        y = (y_index * y_bin)
        z = (z_index * z_bin) - 3
        pcd[point_i, 0] = x
        pcd[point_i, 1] = y
        pcd[point_i, 2] = z
        
    return pcd


def PolarVoxel2PCD(outputPCD, range_fftsize, angle_fftsize, range_bin, theta_bin, phi_bin):
    """
    This function converts polar voxel output to point cloud data (PCD).
    
    Args:
    outputPCD (torch.Tensor): Input polar voxel data.
    range_fftsize (int): Range FFT size.
    angle_fftsize (int): Angle FFT size.
    range_bin (float): Range bin size.
    theta_bin (float): Theta bin size.
    phi_bin (float): Phi bin size.
    
    Returns:
    torch.Tensor: Converted point cloud data.
    """
    if len(outputPCD.shape) == 5:
        # [bs, ch, r_index, sin_theta_index, sin_phi_index]
        outputPCD = outputPCD[0,0,:,:,:].squeeze()

    point_index = torch.nonzero(outputPCD > 0)
    pcd = torch.zeros(len(point_index), 3)
    
    for point_i in range(len(point_index)):
        [r_index, sin_theta_index, sin_phi_index] = point_index[point_i]
        r = r_index * range_bin

        sin_theta = sin_theta_index * theta_bin - 1
        theta = torch.asin(sin_theta)

        sin_phi = sin_phi_index * phi_bin - 1
        phi = torch.asin(sin_phi)

        x = r * torch.cos(phi) * sin_theta
        y = r * torch.cos(phi) * torch.cos(theta)
        z = r * sin_phi

        pcd[point_i, 0] = x
        pcd[point_i, 1] = y
        pcd[point_i, 2] = z
        
    return pcd

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)"""
    roll_rad, pitch_rad, yaw_rad = np.radians([roll, pitch, yaw])

    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return x, y, z, w

def quaternion_to_euler(x, y, z, w):
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)"""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    roll, pitch, yaw = np.degrees([roll, pitch, yaw])

    return roll, pitch, yaw


def transform_point_cloud(points, position, quaternion, coor_transform=True):
    """
    Transforms the point cloud from its local coordinate system to the global coordinate system 
    (standard -x-y-z coordinate system at the initial position of the camera).

    Args:
        points (np.ndarray): Input point cloud, an ndarray of shape (N, 4).
        position (list): The position of the sensor in the global coordinate system.
        quaternion (list): The quaternion representing the sensor's orientation.

    Returns:
        np.ndarray: Transformed point cloud, an ndarray of shape (N, 4).
    """
    if points.shape[0] == 0:
        return points
    if coor_transform:
        # Convert the quaternion to a rotation matrix
        rotation = R.from_quat(quaternion)

        # Convert the LiDAR point cloud to the camera coordinate system
        points_in_camera = copy.deepcopy(points)
        transformed_points_with_color = copy.deepcopy(points)
        points_in_camera[:, 1] = points[:, 2]
        points_in_camera[:, 2] = -points[:, 1]

        # Apply rotation and translation to the point cloud
        rotated_points = rotation.apply(points_in_camera[:, :3])
        translated_points = rotated_points + np.array([position[0], position[2], -position[1]])

        # Convert back to the LiDAR coordinate system
        transformed_points_with_color[:, 0] = translated_points[:, 0]
        transformed_points_with_color[:, 1] = -translated_points[:, 2]
        transformed_points_with_color[:, 2] = translated_points[:, 1]
    else:
        rotation = R.from_quat(quaternion)
        points_in_camera = copy.deepcopy(points)
        transformed_points_with_color = copy.deepcopy(points)
        rotated_points = rotation.apply(points_in_camera[:, :3])
        translated_points = rotated_points + np.array([position[0], position[1], position[2]])
        transformed_points_with_color[:, 0] = translated_points[:, 0]
        transformed_points_with_color[:, 1] = translated_points[:, 1]
        transformed_points_with_color[:, 2] = translated_points[:, 2]

    return transformed_points_with_color


def transform_point_cloud_inverse(points, position, quaternion, coor_transform=True):
    """
    Transforms the point cloud from the global coordinate system to the local coordinate system
    of the radar.

    Args:
        points (np.ndarray): Input point cloud, an ndarray of shape (N, 4).
        position (list): The position of the sensor in the global coordinate system.
        quaternion (list): The quaternion representing the sensor's orientation.
        coor_transform (bool): Flag to indicate whether to perform the coordinate transformation.

    Returns:
        np.ndarray: Transformed point cloud, an ndarray of shape (N, 4).
    """
    # Convert the quaternion to a rotation matrix
    rotation = R.from_quat(quaternion).inv()

    if coor_transform:
        # Apply inverse translation and rotation to the point cloud
        translated_points = points[:, :3] - np.array([position[0], position[2], -position[1]])
        rotated_points = rotation.apply(translated_points)

        # Convert the point cloud from the camera coordinate system to the LiDAR coordinate system
        points_in_lidar = copy.deepcopy(points)
        points_in_lidar[:, 1] = rotated_points[:, 2]
        points_in_lidar[:, 2] = -rotated_points[:, 1]
    else:
        translated_points = points[:, :3] - np.array([position[0], position[1], position[2]])
        rotated_points = rotation.apply(translated_points)

        points_in_lidar = copy.deepcopy(points)
        points_in_lidar[:, 0] = rotated_points[:, 0]
        points_in_lidar[:, 1] = rotated_points[:, 1]
        points_in_lidar[:, 2] = rotated_points[:, 2]

    return points_in_lidar


def convert_quaternion(q1):
    """
    Converts a quaternion representing a rotation in the x-y-z coordinate system to a quaternion
    representing the same rotation in the x-z-(-y) coordinate system.

    Args:
    q1: (x, y, z, w) quaternion representing a rotation in the x-y-z coordinate system.

    Returns:
    q2: (x, y, z, w) quaternion representing the same rotation in the x-z-(-y) coordinate system.
    """

    # Define a rotation that swaps the y and z axes and inverts the y axis
    swap_yz = R.from_euler('yxz', [0, 90, 0], degrees=True)

    # Convert q1 to a rotation object
    r1 = R.from_quat(q1)

    # Apply the coordinate system transformation and convert back to a quaternion
    # pdb.set_trace()
    r2 = swap_yz * r1 * swap_yz.inv()
    q2 = r2.as_quat()

    return q2

def transform_point_cloud_new(points, position, quaternion):
    """
    Transforms the point cloud from its local coordinate system to the global coordinate system 
    (standard -x-y-z coordinate system at the initial position of the camera).

    Args:
        points (np.ndarray): Input point cloud, an ndarray of shape (N, 4).
        position (list): The position of the sensor in the global coordinate system.
        quaternion (list): The quaternion representing the sensor's orientation.

    Returns:
        np.ndarray: Transformed point cloud, an ndarray of shape (N, 4).
    """
    # Convert the quaternion to a rotation matrix
    quaternion_new = convert_quaternion(quaternion)
    rotation = R.from_quat(quaternion_new)

    # Convert the LiDAR point cloud to the camera coordinate system
    points_in_camera = copy.deepcopy(points)
    transformed_points_with_color = copy.deepcopy(points)
    # points_in_camera[:, 1] = points[:, 2]
    # points_in_camera[:, 2] = -points[:, 1]

    # Apply rotation and translation to the point cloud
    rotated_points = rotation.apply(points_in_camera[:, :3])
    translated_points = rotated_points + np.array([position[0], position[1], position[2]])

    # Convert back to the LiDAR coordinate system
    transformed_points_with_color[:, 0] = translated_points[:, 0]
    transformed_points_with_color[:, 1] = translated_points[:, 1]
    transformed_points_with_color[:, 2] = translated_points[:, 2]

    return transformed_points_with_color

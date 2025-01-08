"""
LIDAR Point Cloud Processing

This Python script processes LIDAR data files and generates point clouds with
instance information. It contains functions for calculating the vertical angle,
concatenating and converting signed integers, and reading LIDAR frames. The script
utilizes libraries such as NumPy, Open3D, and Matplotlib for efficient processing
and visualization of point clouds.

Author: USTC IP_LAB Millimeter Wave Point Cloud Group
Main Members: Ruixu Geng (gengruixu@mail.ustc.edu.cn), Yadong Li, Jincheng Wu, Yating Gao
Copyright: Copyright (C) 2023 USTC IP_LAB Millimeter Wave Point Cloud Group
Date: March 2023
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ..point_cloud_processing import downsample_denoise_pointcloud


def cal_vertical(byte3, byte4):
    """
    Calculate the vertical angle, sin(vertical angle), and cos(vertical angle)
    from two input bytes.

    Args:
        byte3 (int): The first byte.
        byte4 (int): The second byte.

    Returns:
        vertical_angle (float): The calculated vertical angle.
        fSinV_angle (float): The sin(vertical angle).
        fCosV_angle (float): The cos(vertical angle).
    """
    iTempAngle = byte3
    iChannelNumber = iTempAngle >> 6
    iSymmbol = (iTempAngle >> 5) & 0x01

    if iSymmbol == 1:
        iAngle_v = byte4 + byte3 * 256
        fAngle_v = iAngle_v | 0xc000
        if fAngle_v > 32767:
            fAngle_v = fAngle_v - 65536
    else:
        iAngle_Height = iTempAngle & 0x3f
        fAngle_v = byte4 + iAngle_Height * 256

    fAngle_V = fAngle_v * 0.0025

    if iChannelNumber == 0:
        fPutTheMirrorOffAngle = 0
    elif iChannelNumber == 1:
        fPutTheMirrorOffAngle = -2
    elif iChannelNumber == 2:
        fPutTheMirrorOffAngle = -1
    elif iChannelNumber == 3:
        fPutTheMirrorOffAngle = -3
    else:
        fPutTheMirrorOffAngle = -1.4

    fGalvanometrtAngle = fAngle_V + 7.26
    fAngle_R0 = (
        np.cos(30 * np.pi / 180) * np.cos(fPutTheMirrorOffAngle * np.pi / 180) * np.cos(fGalvanometrtAngle * np.pi / 180)
        - np.sin(fGalvanometrtAngle * np.pi / 180) * np.sin(fPutTheMirrorOffAngle * np.pi / 180)
    )
    fSinV_angle = 2 * fAngle_R0 * np.sin(fGalvanometrtAngle * np.pi / 180) + np.sin(fPutTheMirrorOffAngle * np.pi / 180)
    fCosV_angle = np.sqrt(1 - fSinV_angle ** 2)
    vertical_angle = np.arcsin(fSinV_angle) * 180 / np.pi

    return vertical_angle, fSinV_angle, fCosV_angle


def concat_and_convert_to_signed_int(x: int, y: int) -> int:
    """
    Concatenate two integers as hexadecimal values and convert the result
    to a signed integer.

    Args:
        x (int): The first integer.
        y (int): The second integer.

    Returns:
        result (int): The signed integer obtained from the concatenation and conversion.
    """
    # Convert x and y to hexadecimal strings
    hex_x = hex(x)[2:].zfill(2)
    hex_y = hex(y)[2:].zfill(2)

    # Concatenate the two hexadecimal strings into one
    hex_result = hex_x + hex_y

    # Convert the concatenated hexadecimal string to a signed integer
    result = int(hex_result, 16)
    if result > 0x7fff:
        result = result - 0x10000

    return result




def read_lidar_frame(lidar_file_name):
    """
    Read and process a LIDAR file and generate a point cloud with instance information.

    Args:
        lidar_file_name (str): Path to the LIDAR file.

    Returns:
        colors (np.ndarray): An array of color values associated with each point in the point cloud.
        point_cloud_points (np.ndarray): An array of points (x, y, z, instance) in the point cloud.
    """
    point_cloud_points_string = np.fromfile(lidar_file_name, dtype="|S8")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud_points = []

    for point_index in range(len(point_cloud_points_string)):
        point = point_cloud_points_string[point_index]
        if len(point) != 8:
            continue

        byte1, byte2, byte3, byte4, byte5, byte6, byte7, byte8 = point

        distance = (((byte5 << 8) | byte6) + byte7 / 256) * 0.01 * 25.6

        if (((byte5 << 8) | byte6) > 3):
            instance = byte8
            horizontal = concat_and_convert_to_signed_int(byte1, byte2) / 100 - 0.363
            vertical, sin_v, cos_v = cal_vertical(byte3, byte4)

            y = distance * cos_v * np.cos(horizontal * np.pi / 180)
            x = distance * cos_v * np.sin(horizontal * np.pi / 180)
            z = distance * sin_v
            point_cloud_points.append([x, y, z, instance])

    point_cloud_points = np.array(point_cloud_points)
    cmap = plt.cm.get_cmap('jet')
    colors = cmap(point_cloud_points[:, 3] / point_cloud_points.max())
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_points[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return colors[:, :3], point_cloud_points


def read_process_save_lidar_frame(file_path):
    """
    Read a LiDAR frame, process it by downsampling and denoising, and save the result to a new file.

    Args:
        file_path (str): Path to the input LiDAR frame file.

    Note:
        To load the saved data, use the following code:
            cloud_vals = np.fromfile(save_file_name, dtype='float32')
            cloud = cloud_vals.reshape((-1, 4))
    """
    # Read the LiDAR frame
    point_cloud = read_lidar_frame(file_path)[1]

    # Downsample and denoise the point cloud
    point_cloud_new = downsample_denoise_pointcloud(point_cloud).astype(np.float32)

    # Create the save file name by replacing 'Lidar' with 'Lidar_pcd' in the original file path
    save_file_name = file_path.replace('Lidar', 'Lidar_pcd', 1)
    # Save the processed point cloud to the new file
    point_cloud_new.tofile(save_file_name)
    print(save_file_name + " saved!")

    # Use the following code to load the saved data:
    # cloud_vals = np.fromfile(save_file_name,dtype = 'float32')
    # cloud = cloud_vals.reshape((-1,4))


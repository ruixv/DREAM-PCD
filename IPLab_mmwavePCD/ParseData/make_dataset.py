"""
This file contains functions for processing and aligning millimeter wave point cloud data collected from radar, lidar, and camera sensors.
The functions facilitate the extraction of ADC files, parsing of timestamps, and alignment of sensor data based on timestamps.

Author: USTC IP_LAB Millimeter Wave Point Cloud Group
Main Members: Ruixu Geng (gengruixu@mail.ustc.edu.cn), Yadong Li, Jincheng Wu, Yating Gao
Copyright: Copyright (C) 2023 USTC IP_LAB Millimeter Wave Point Cloud Group
Date: March 2023
"""

import re
import os
import numpy as np
import pdb
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
from ..Visualize import visData

ADC_EXTENSIONS = ['.mat', '.MAT', 'bin', 'BIN', "jpg", "JPG","png","PNG", "npy"]

def is_adc_file(filename):
    """
    Check if the given filename has a valid ADC file extension.

    Args:
        filename (str): The name of the file to be checked.

    Returns:
        bool: True if the file has a valid ADC extension, False otherwise.
    """
    return any(filename.endswith(extension) for extension in ADC_EXTENSIONS)

def make_pcd_dataset(dir):
    """
    Create a list of paths to ADC files in the given directory.
    For frame i, if both i.bin and i(n).bin exist, use the one with largest n.
    If no (n) version exists, use the regular .bin file.
    Files without numbers in the name will be skipped.

    Args:
        dir (str): The path to the directory containing ADC files.

    Returns:
        list: A list of sorted paths to the ADC files in the directory.
    """
    # Dictionary to store all versions of each frame
    frame_versions = {}

    # Ensure the given directory is valid
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # Traverse the directory and find all ADC files
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_adc_file(fname):
                path = os.path.join(root, fname)
                
                # Check if filename contains any numbers before split
                numbers = re.findall(r"\d+", fname.split('(')[0])
                if not numbers:
                    continue
                    
                # Extract frame number
                frame_num = int(numbers[-1])
                
                # Extract version number if exists (number in parentheses)
                version = 0  # default version for files without parentheses
                if '(' in fname:
                    version = int(re.findall(r"\d+", fname.split('(')[1])[0])
                
                # Store all versions of each frame
                if frame_num not in frame_versions:
                    frame_versions[frame_num] = {}
                frame_versions[frame_num][version] = path

    # Create final list by selecting highest version number for each frame
    final_adcs = []
    for frame_num in sorted(frame_versions.keys()):
        versions = frame_versions[frame_num]
        # Get the highest version number (could be 0 if only regular .bin exists)
        highest_version = max(versions.keys())
        final_adcs.append(versions[highest_version])

    return final_adcs


def make_selfmade_pcd_dataset(radar_dir, lidar_dir, camera_dir, radar_config):
    """
    Aligns Lidar, Camera, and Radar data based on their timestamps.

    Args:
        radar_dir (str): The path to the directory containing radar data files.
        lidar_dir (str): The path to the directory containing lidar data files.
        camera_dir (str): The path to the directory containing camera data files.

    Returns:
        tuple: A tuple containing the final aligned paths for radar, lidar, and
               camera data, along with their respective positions, angles,
               velocities, and timestamps.
    """
    lidar_path_final = []
    radar_path_final = []
    camera_path_final = []
    positions_final = []
    angles_final = []
    velocitys_final = []
    timestamps_radar_final = []
    timestamps_lidar_final = []
    timestamps_camera_final = []

    # Read timestamps from the corresponding files
    with open(os.path.join(lidar_dir, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        lidar_timestamps = [float(timestamp) for timestamp in timestamps]
    with open(os.path.join(radar_dir, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        radar_timestamps = [float(timestamp) for timestamp in timestamps]
    with open(os.path.join(camera_dir, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        camera_timestamps = [float(timestamp) for timestamp in timestamps]

    # Read positions, angles, and velocities from corresponding files
    with open(os.path.join(camera_dir, 'pose.txt')) as f:
        positions_lines = f.read().splitlines()[1:-1]
        positions = []
        angles = []
        for line in range(len(positions_lines)):
            pose = [float(i) for i in positions_lines[line][1:-1].split(",")]
            position = [pose[0], -pose[2], pose[1]]
            angle = [pose[3], pose[4], pose[5], pose[6]]
            positions.append(position)
            angles.append(angle)
        positions = np.array(positions)
        angles = np.array(angles)

    with open(os.path.join(camera_dir, 'velocity.txt')) as f:
        velocity_lines = f.read().splitlines()[1:-1]
        velocitys = []
        for line in range(len(velocity_lines)):
            vel = [float(i) for i in velocity_lines[line][1:-1].split(",")]
            velocity = [vel[0], -vel[2], vel[1]]
            velocitys.append(velocity)
        velocitys = np.array(velocitys)

    # Determine the valid timestamp range
    start_time = max(min(lidar_timestamps), min(radar_timestamps), min(camera_timestamps))
    end_time = min(max(lidar_timestamps), max(radar_timestamps), max(camera_timestamps))

    # Create lists of file paths for each data type
    radar_path = make_pcd_dataset(radar_dir)
    lidar_path = make_pcd_dataset(lidar_dir)
    camera_path = make_pcd_dataset(camera_dir)

    # Ensure that radar_path and radar_timestamps have the same length
    if len(radar_path) != len(radar_timestamps):
        print('%s and %s have different lenght, meaning radar collection ERROR!' % (radar_path[0], radar_timestamps[0]))
        radar_path = radar_path[:len(radar_timestamps)]


    radar_timestamps_interpolated_count = radar_config.framesPerFile
    
    if radar_timestamps_interpolated_count == 1:
        radar_timestamps_upsampled = radar_timestamps
        radar_path_upsamplesd = radar_path
    else:
        radar_timestamps_upsampled = upsample_timestamps(radar_timestamps, upsample_factor=radar_timestamps_interpolated_count+1)
        radar_path_upsamplesd = interpolate_filenames(radar_path, upsample_factor=radar_timestamps_interpolated_count+1)


    radar_frame_number = len(radar_path) * radar_timestamps_interpolated_count
    lidar_frame_number = len(lidar_path)

    if radar_frame_number/lidar_frame_number > 0.8 and radar_frame_number/lidar_frame_number < 1.2:
        print("radar and lidar and camera has the same frame rate")
        # Align radar, lidar, and camera data based on their timestamps
        for index_i in range(len(radar_path_upsamplesd)):
            i_time = radar_timestamps_upsampled[index_i]

            # Check if the current timestamp is within the valid range
            if (i_time > start_time) and (i_time < end_time):
                # Find the closest lidar and camera frames to the current radar frame
                lidar_diff_list = abs(np.array(lidar_timestamps) - i_time).tolist()
                lidar_index = lidar_diff_list.index(min(lidar_diff_list))
                camera_diff_list = abs(np.array(camera_timestamps) - i_time).tolist()
                camera_index = camera_diff_list.index(min(camera_diff_list))

                # Append the closest frames and their corresponding data to the final lists
                lidar_path_final.append(lidar_path[lidar_index])
                radar_path_final.append(radar_path_upsamplesd[index_i])
                camera_path_final.append(camera_path[camera_index])
                positions_final.append(positions[camera_index])
                angles_final.append(angles[camera_index])
                velocitys_final.append(velocitys[camera_index])
                timestamps_radar_final.append(radar_timestamps_upsampled[index_i])
                timestamps_lidar_final.append(lidar_timestamps[lidar_index])
                timestamps_camera_final.append(camera_timestamps[camera_index])

        positions_final_upsampled = positions_final
        angles_final_upsampled = angles_final
        velocitys_final_upsampled = velocitys_final
        radar_path_final_upsampled = radar_path_final
        timestamps_radar_final_upsampled = timestamps_radar_final
    

    else:
        print("radar and lidar and camera has different frame rate")
        # radar_timestamps_upsampled
        # radar_path_upsamplesd
        # lidar_timestamps
        # lidar_path
        # camera_path
        # camera_timestamps
        radar_path_final, lidar_path_final, camera_path_final, positions_final, angles_final, velocitys_final, timestamps_radar_final, timestamps_lidar_final, timestamps_camera_final, (timestamps_radar_final_upsampled, radar_path_final_upsampled, positions_final_upsampled, angles_final_upsampled, velocitys_final_upsampled) = align_timestamps(radar_timestamps_upsampled, lidar_timestamps, camera_timestamps, radar_path_upsamplesd, lidar_path, camera_path, positions, angles, velocitys)
    
    return radar_path_final, lidar_path_final, camera_path_final, positions_final, angles_final, velocitys_final, timestamps_radar_final, timestamps_lidar_final, timestamps_camera_final, (timestamps_radar_final_upsampled, radar_path_final_upsampled, positions_final_upsampled, angles_final_upsampled, velocitys_final_upsampled)


import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

def align_timestamps_positions_angles(camera_timestamps, positions, angles, radar_timestamps_ele, radar_PCDpaths_ele):

    camera_timestamps = np.array(camera_timestamps)
    radar_timestamps_ele = np.array(radar_timestamps_ele)


    unique_indices = np.unique(camera_timestamps, return_index=True)[1]
    camera_timestamps = camera_timestamps[unique_indices]
    positions = positions[unique_indices]
    angles = angles[unique_indices]


    positions_interpolator = interp1d(camera_timestamps, positions, axis=0, kind='linear', fill_value='extrapolate')


    valid_radar_ele_index = (radar_timestamps_ele >= camera_timestamps[0]) & (radar_timestamps_ele <= camera_timestamps[-1])
    radar_timestamps_ele = radar_timestamps_ele[valid_radar_ele_index]
    # radar_path_final_ele = np.array(radar_PCDpaths_ele)[valid_index].tolist()


    positions_radar_final_ele = positions_interpolator(radar_timestamps_ele)


    rotations = Rotation.from_quat(angles)


    slerp_interpolator = Slerp(camera_timestamps, rotations)


    rotations_radar_final_ele = slerp_interpolator(radar_timestamps_ele)


    angles_radar_final_ele = rotations_radar_final_ele.as_quat()



    # return radar_timestamps_ele, radar_path_final_ele, positions_radar_final_ele, angles_radar_final_ele
    return radar_timestamps_ele, valid_radar_ele_index, positions_radar_final_ele, angles_radar_final_ele



def correct_positions(camera_timestamps, positions, reference_timestamps, reference_positions):

    interp_function = interp1d(reference_timestamps, reference_positions, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')


    interpolated_reference_positions = interp_function(camera_timestamps)


    position_deltas = interpolated_reference_positions - positions


    smoothing_factor = 0.8
    smoothed_position_deltas = smoothing_factor * position_deltas + (1 - smoothing_factor) * position_deltas.mean(axis=0)


    positions_corrected = positions + smoothed_position_deltas

    return interpolated_reference_positions

def make_selfmade_pcd_dataset_tworadar(radar_path_azi, radar_path_ele, lidar_path, camera_path, radar_config_azi, radarObj_azi, radar_config_ele, radarObj_ele, T265_path = None):
    """
    Aligns Lidar, Camera, and Radar data based on their timestamps.

    Args:
        radar_dir (str): The path to the directory containing radar data files.
        lidar_dir (str): The path to the directory containing lidar data files.
        camera_dir (str): The path to the directory containing camera data files.

    Returns:
        tuple: A tuple containing the final aligned paths for radar, lidar, and
               camera data, along with their respective positions, angles,
               velocities, and timestamps.
    """
    radar_path_final_azi = []
    positions_radar_final_azi = []
    angles_radar_final_azi = []
    timestamps_radar_final_azi = []

    radar_path_final_ele = []
    positions_radar_final_ele = []
    angles_radar_final_ele = []
    timestamps_radar_final_ele = []

    lidar_path_final = []
    positions_lidar_final = []
    angles_lidar_final = []
    timestamps_lidar_final = []

    camera_path_final = []
    positions_camera_final = []
    angles_camera_final = []
    timestamps_camera_final = []

    # Read timestamps from the corresponding files
    # pdb.set_trace()
    with open(os.path.join(lidar_path, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        lidar_timestamps = [float(timestamp) for timestamp in timestamps]
    with open(os.path.join(radar_path_azi, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        radar_timestamps_azi = [float(timestamp) for timestamp in timestamps]
    try:
        with open(os.path.join(radar_path_ele, 'timestamp.txt')) as f:
            timestamps = f.read().splitlines()[1:-1]
            radar_timestamps_ele = [float(timestamp) for timestamp in timestamps]
    except:
        radar_timestamps_ele = []
    # pdb.set_trace()
    with open(os.path.join(camera_path, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        camera_timestamps = [float(timestamp) for timestamp in timestamps]

    # Read positions, angles, and velocities from corresponding files
    # pdb.set_trace()
    if "ZED" in camera_path:
        with open(os.path.join(camera_path, 'pose.txt')) as f:
            positions_lines = f.read().splitlines()[1:-1]
            positions = []
            angles = []
            for line in range(len(positions_lines)):
                pose = [float(i) for i in positions_lines[line][1:-1].split(",")]
                position = [pose[0], pose[1], pose[2]]
                angle = [pose[3], pose[4], pose[5], pose[6]]
                positions.append(position)
                angles.append(angle)
            positions = np.array(positions)
            angles = np.array(angles)
    else:
        with open(os.path.join(camera_path, 'pose.txt')) as f:
            positions_lines = f.read().splitlines()[1:-1]
            positions = []
            angles = []
            for line in range(len(positions_lines)):
                pose = [float(i) for i in positions_lines[line][1:-1].split(",")]
                position = [pose[0], -pose[2], pose[1]]
                angle = [pose[3], pose[4], pose[5], pose[6]]
                positions.append(position)
                angles.append(angle)
            positions = np.array(positions)
            angles = np.array(angles)
        with open(os.path.join(camera_path, 'velocity.txt')) as f:
            velocity_lines = f.read().splitlines()[1:-1]
            velocitys = []
            for line in range(len(velocity_lines)):
                vel = [float(i) for i in velocity_lines[line][1:-1].split(",")]
                velocity = [vel[0], -vel[2], vel[1]]
                velocitys.append(velocity)
            velocitys = np.array(velocitys)

    if T265_path is not None:
        print("Start use T265 to perform correction")
        
        with open(os.path.join(T265_path, 'timestamp.txt')) as f:
            timestamps = f.read().splitlines()[1:-1]
            T265_timestamps = [float(timestamp) for timestamp in timestamps]
        with open(os.path.join(T265_path, 'pose.txt')) as f:
            positions_lines = f.read().splitlines()[1:-1]
            T265_positions = []
            T265_angles = []
            for line in range(len(positions_lines)):
                pose = [float(i) for i in positions_lines[line][1:-1].split(",")]
                position = [pose[2], pose[0], pose[1]]
                angle = [pose[3], pose[4], pose[5], pose[6]]
                T265_positions.append(position)
                T265_angles.append(angle)
            T265_positions = np.array(T265_positions)
            T265_angles = np.array(T265_angles)
        with open(os.path.join(T265_path, 'velocity.txt')) as f:
            velocity_lines = f.read().splitlines()[1:-1]
            T265_velocitys = []
            for line in range(len(velocity_lines)):
                vel = [float(i) for i in velocity_lines[line][1:-1].split(",")]
                velocity = [vel[0], -vel[2], vel[1]]
                T265_velocitys.append(velocity)
            T265_velocitys = np.array(T265_velocitys)
        
        reference_positions = np.array(T265_positions)  # 1820,3
        reference_timestamps = np.array(T265_timestamps)    # 1820
        camera_timestamps = np.array(camera_timestamps) # 1801
        
        positions = correct_positions(camera_timestamps, positions, reference_timestamps, reference_positions)
    else:
        print("No T265 available, unable to perform correction")
    # pdb.set_trace()
    # visData.plotPcd(np.array(positions))
    # visData.plotPcd(np.array(positions_new))
    # Create lists of file paths for each data type
    radar_path_azi_pcd = os.path.join(radar_path_azi, 'PCD_SamePaddingUDPERROR')
    radar_path_ele_adc = os.path.join(radar_path_ele, 'ADC')
    radar_PCDpaths_azi = make_pcd_dataset(radar_path_azi_pcd)
    # pdb.set_trace()
    try:
        radar_PCDpaths_ele = make_pcd_dataset(radar_path_ele_adc) # 5671
    except:
        radar_PCDpaths_ele = []
    # pdb.set_trace()
    lidar_paths = make_pcd_dataset(lidar_path)
    camera_paths = make_pcd_dataset(camera_path)

    
    try:
        del_frame_index_from_file = []
        del_file_path = os.path.join(radar_path_azi, "del_frame.txt")
        with open(del_file_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            del_frame_index_from_file.append(int(line.strip()))

        deleted_frames_count = 0
        for index in del_frame_index_from_file:
            if index - deleted_frames_count < len(radar_PCDpaths_azi):
                del radar_PCDpaths_azi[index - deleted_frames_count]
                del radar_timestamps_azi[index - deleted_frames_count]
                deleted_frames_count += 1
                print("Deleted missing frame:", index)
    except:
        print("Warning: Missing frames were not deleted")



    # Determine the valid timestamp range
    # pdb.set_trace()
    try:
        start_time = max(min(lidar_timestamps), min(radar_timestamps_azi), min(radar_timestamps_ele), min(camera_timestamps))
        end_time = min(max(lidar_timestamps), max(radar_timestamps_azi), max(radar_timestamps_ele), max(camera_timestamps))
    except:
        start_time = max(min(lidar_timestamps), min(radar_timestamps_azi),  min(camera_timestamps))
        end_time = min(max(lidar_timestamps), max(radar_timestamps_azi),  max(camera_timestamps))



    # Ensure that radar_path and radar_timestamps have the same length
    # pdb.set_trace()
    if (len(radar_PCDpaths_azi) != len(radar_timestamps_azi)):
        # Assert Error
        print('Warning: radar_path and radar_timestamps have different lenght')
        # raise Exception('radar_path and radar_timestamps have different lenght')

    # STEP 1: Perform timestamp matching and interpolation with the radar set horizontally
    radar_azi_frame_number = len(radar_timestamps_azi)
    lidar_frame_number = len(lidar_paths)

    if radar_azi_frame_number/lidar_frame_number > 0.8 and radar_azi_frame_number/lidar_frame_number < 1.2 and ("static" not in radar_path_azi) or True:
        print("radar_azi and lidar and camera has the same frame rate")
        # Align radar, lidar, and camera data based on their timestamps
        for index_i in range(len(radar_PCDpaths_azi)):
            i_time = radar_timestamps_azi[index_i]  # 283

            # Check if the current timestamp is within the valid range
            if (i_time > start_time) and (i_time < end_time):
                # Find the closest lidar and camera frames to the current radar frame
                lidar_diff_list = abs(np.array(lidar_timestamps) - i_time).tolist()
                lidar_index = lidar_diff_list.index(min(lidar_diff_list))
                lidar_time = lidar_timestamps[lidar_index]
                camera_diff_list = abs(np.array(camera_timestamps) - i_time).tolist()
                camera_diff_list_lidar = abs(np.array(camera_timestamps) - lidar_time).tolist()
                camera_index_lidar = camera_diff_list_lidar.index(min(camera_diff_list_lidar))
                camera_index = camera_diff_list.index(min(camera_diff_list))

                # Append the closest frames and their corresponding data to the final lists
                lidar_path_final.append(lidar_paths[lidar_index])
                
                radar_path_final_azi.append(radar_PCDpaths_azi[index_i])
                # pdb.set_trace()
                # if "ZED" in camera_path:
                #     camera_path_final.append(camera_paths[camera_index // 1000])
                # else:
                #     camera_path_final.append(camera_paths[camera_index // 6])

                positions_radar_final_azi.append(positions[camera_index])
                angles_radar_final_azi.append(angles[camera_index])

                timestamps_radar_final_azi.append(radar_timestamps_azi[index_i])
                timestamps_lidar_final.append(lidar_timestamps[lidar_index])
                positions_lidar_final.append(positions[camera_index_lidar])
                angles_lidar_final.append(angles[camera_index_lidar])
                timestamps_camera_final.append(camera_timestamps[camera_index])
    
    else:
        raise Exception('radar_azi and lidar should has different frame rate')

    # STEP 2: Perform timestamp matching and interpolation with the radar set vertically
    radar_ele_frame_number = len(radar_timestamps_ele)
    lidar_frame_number = len(lidar_paths)

    try:
        # pdb.set_trace()
        timestamps_radar_final_ele, valid_radar_ele_index, positions_radar_final_ele, angles_radar_final_ele = align_timestamps_positions_angles(camera_timestamps, positions, angles, radar_timestamps_ele, radar_PCDpaths_ele)
        # visData.plotPcd(positions_radar_final_ele)
    except:
        print("Warning: no ele data")
    
    return radar_path_final_azi, positions_radar_final_azi, angles_radar_final_azi,timestamps_radar_final_azi,valid_radar_ele_index, positions_radar_final_ele, angles_radar_final_ele,timestamps_radar_final_ele,lidar_path_final, positions_lidar_final, angles_lidar_final, timestamps_lidar_final, camera_path_final, positions_camera_final, angles_camera_final, timestamps_camera_final



def make_selfmade_pcd_dataset_newcoherent(radar_path_azi, radar_path_ele, lidar_path, camera_path, radar_config):
    """
    Aligns Lidar, Camera, and Radar data based on their timestamps.

    Args:
        radar_dir (str): The path to the directory containing radar data files.
        lidar_dir (str): The path to the directory containing lidar data files.
        camera_dir (str): The path to the directory containing camera data files.

    Returns:
        tuple: A tuple containing the final aligned paths for radar, lidar, and
               camera data, along with their respective positions, angles,
               velocities, and timestamps.
    """
    radar_path_final_azi = []
    positions_radar_final_azi = []
    angles_radar_final_azi = []
    timestamps_radar_final_azi = []

    radar_path_final_ele = []
    positions_radar_final_ele = []
    angles_radar_final_ele = []
    timestamps_radar_final_ele = []

    lidar_path_final = []
    positions_lidar_final = []
    angles_lidar_final = []
    timestamps_lidar_final = []

    camera_path_final = []
    positions_camera_final = []
    angles_camera_final = []
    timestamps_camera_final = []



    # Read timestamps from the corresponding files
    with open(os.path.join(lidar_dir, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        lidar_timestamps = [float(timestamp) for timestamp in timestamps]
    with open(os.path.join(radar_dir, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        radar_timestamps = [float(timestamp) for timestamp in timestamps]
    with open(os.path.join(camera_dir, 'timestamp.txt')) as f:
        timestamps = f.read().splitlines()[1:-1]
        camera_timestamps = [float(timestamp) for timestamp in timestamps]

    # Read positions, angles, and velocities from corresponding files
    with open(os.path.join(camera_dir, 'pose.txt')) as f:
        positions_lines = f.read().splitlines()[1:-1]
        positions = []
        angles = []
        for line in range(len(positions_lines)):
            pose = [float(i) for i in positions_lines[line][1:-1].split(",")]
            position = [pose[0], -pose[2], pose[1]]
            angle = [pose[3], pose[4], pose[5], pose[6]]
            positions.append(position)
            angles.append(angle)
        positions = np.array(positions)
        angles = np.array(angles)

    with open(os.path.join(camera_dir, 'velocity.txt')) as f:
        velocity_lines = f.read().splitlines()[1:-1]
        velocitys = []
        for line in range(len(velocity_lines)):
            vel = [float(i) for i in velocity_lines[line][1:-1].split(",")]
            velocity = [vel[0], -vel[2], vel[1]]
            velocitys.append(velocity)
        velocitys = np.array(velocitys)

    # Determine the valid timestamp range
    start_time = max(min(lidar_timestamps), min(radar_timestamps), min(camera_timestamps))
    end_time = min(max(lidar_timestamps), max(radar_timestamps), max(camera_timestamps))

    # Create lists of file paths for each data type
    radar_path = make_pcd_dataset(radar_dir)
    lidar_path = make_pcd_dataset(lidar_dir)
    camera_path = make_pcd_dataset(camera_dir)

    # Ensure that radar_path and radar_timestamps have the same length
    if len(radar_path) != len(radar_timestamps):
        print('%s and %s have different lenght, meaning radar collection ERROR!' % (radar_path[0], radar_timestamps[0]))
        radar_path = radar_path[:len(radar_timestamps)]


    radar_timestamps_interpolated_count = radar_config.framesPerFile
    if radar_timestamps_interpolated_count == 1:
        radar_timestamps_upsampled = radar_timestamps
        radar_path_upsamplesd = radar_path
    else:
        radar_timestamps_upsampled = upsample_timestamps(radar_timestamps, upsample_factor=radar_timestamps_interpolated_count*16+1)
        radar_path_upsamplesd = interpolate_filenames(radar_path, upsample_factor=radar_timestamps_interpolated_count*16+1)
        # pdb.set_trace()


    radar_frame_number = len(radar_path) * radar_timestamps_interpolated_count
    lidar_frame_number = len(lidar_path)

    if radar_frame_number/lidar_frame_number > 0.8 and radar_frame_number/lidar_frame_number < 1.2:
        print("radar and lidar and camera has the same frame rate")
        # Align radar, lidar, and camera data based on their timestamps
        for index_i in range(len(radar_path_upsamplesd)):
            i_time = radar_timestamps_upsampled[index_i]

            # Check if the current timestamp is within the valid range
            if (i_time > start_time) and (i_time < end_time):
                # Find the closest lidar and camera frames to the current radar frame
                lidar_diff_list = abs(np.array(lidar_timestamps) - i_time).tolist()
                lidar_index = lidar_diff_list.index(min(lidar_diff_list))
                camera_diff_list = abs(np.array(camera_timestamps) - i_time).tolist()
                camera_index = camera_diff_list.index(min(camera_diff_list))

                # Append the closest frames and their corresponding data to the final lists
                lidar_path_final.append(lidar_path[lidar_index])
                radar_path_final.append(radar_path_upsamplesd[index_i])
                camera_path_final.append(camera_path[camera_index])
                positions_final.append(positions[camera_index])
                angles_final.append(angles[camera_index])
                velocitys_final.append(velocitys[camera_index])
                timestamps_radar_final.append(radar_timestamps_upsampled[index_i])
                timestamps_lidar_final.append(lidar_timestamps[lidar_index])
                timestamps_camera_final.append(camera_timestamps[camera_index])

        positions_final_upsampled = positions_final
        angles_final_upsampled = angles_final
        velocitys_final_upsampled = velocitys_final
        radar_path_final_upsampled = radar_path_final
        timestamps_radar_final_upsampled = timestamps_radar_final
    

    else:
        print("radar and lidar and camera has different frame rate")
        # radar_timestamps_upsampled
        # radar_path_upsamplesd
        # lidar_timestamps
        # lidar_path
        # camera_path
        # camera_timestamps
        radar_path_final, lidar_path_final, camera_path_final, positions_final, angles_final, velocitys_final, timestamps_radar_final, timestamps_lidar_final, timestamps_camera_final, (timestamps_radar_final_upsampled, radar_path_final_upsampled, positions_final_upsampled, angles_final_upsampled, velocitys_final_upsampled) = align_timestamps(radar_timestamps_upsampled, lidar_timestamps, camera_timestamps, radar_path_upsamplesd, lidar_path, camera_path, positions, angles, velocitys)
    
    return radar_path_final_azi, radar_path_final_ele, lidar_path_final, camera_path_final, positions_final, angles_final, velocitys_final, timestamps_radar_final_azi, timestamps_radar_final_ele, timestamps_lidar_final, timestamps_camera_final

def find_closest(timestamp, timestamps):
    closest_diff = float("inf")
    closest_idx = None
    for i, t in enumerate(timestamps):
        diff = abs(timestamp - t)
        if diff < closest_diff:
            closest_diff = diff
            closest_idx = i
    return closest_idx

def align_timestamps(radar_timestamps_upsampled, lidar_timestamps, camera_timestamps, radar_path_upsamplesd, lidar_path, camera_path, positions, angles, velocitys):
    start_time = max(radar_timestamps_upsampled[0], lidar_timestamps[0], camera_timestamps[0])
    end_time = min(radar_timestamps_upsampled[-1], lidar_timestamps[-1], camera_timestamps[-1])


    camera_start_idx = find_closest(start_time, camera_timestamps) + 1
    start_time = camera_timestamps[camera_start_idx]
    radar_start_idx = find_closest(start_time, radar_timestamps_upsampled)
    lidar_start_idx = find_closest(start_time, lidar_timestamps)
    
    camera_end_idx = find_closest(end_time, camera_timestamps)
    end_time = camera_timestamps[camera_end_idx]
    radar_end_idx = find_closest(end_time, radar_timestamps_upsampled)
    lidar_end_idx = find_closest(end_time, lidar_timestamps)
    

    aligned_radar_timestamps = radar_timestamps_upsampled[radar_start_idx:radar_end_idx+1]
    aligned_lidar_timestamps = lidar_timestamps[lidar_start_idx:lidar_end_idx+1]
    aligned_lidar_path = lidar_path[lidar_start_idx:lidar_end_idx+1]
    aligned_camera_timestamps = camera_timestamps[camera_start_idx:camera_end_idx+1]
    timestamps_radar_final_upsampled = aligned_radar_timestamps
    timestamps_lidar_final = aligned_lidar_timestamps
    timestamps_camera_final = aligned_camera_timestamps

    radar_path_final_upsampled = radar_path_upsamplesd[radar_start_idx:radar_end_idx+1]
    lidar_path_final = lidar_path[lidar_start_idx:lidar_end_idx+1]
    camera_path_final = camera_path[camera_start_idx:camera_end_idx+1]
    positions_final = positions[camera_start_idx:camera_end_idx+1]
    angles_final = angles[camera_start_idx:camera_end_idx+1]
    velocitys_final = velocitys[camera_start_idx:camera_end_idx+1]


    radar_path_final = []
    lidar_path_final = []
    timestamps_radar_final = []
    timestamps_lidar_final = []
    for i, camera_timestamp in enumerate(aligned_camera_timestamps):
        radar_idx = find_closest(camera_timestamp, aligned_radar_timestamps)
        lidar_idx = find_closest(camera_timestamp, aligned_lidar_timestamps)
        radar_path_final.append(radar_path_final_upsampled[radar_idx])
        lidar_path_final.append(aligned_lidar_path[lidar_idx])
        timestamps_radar_final.append(aligned_radar_timestamps[radar_idx])
        timestamps_lidar_final.append(aligned_lidar_timestamps[lidar_idx])
    

    positions_final_upsampled = []
    angles_final_upsampled = []
    velocitys_final_upsampled = []

    for i, radar_timestamp in enumerate(aligned_radar_timestamps):
        camera_idx = find_closest(radar_timestamp, aligned_camera_timestamps)
        camera_timestamp = aligned_camera_timestamps[camera_idx]
        if camera_timestamp < radar_timestamp:
            camera_idx_left = camera_idx
            camera_idx_right = camera_idx + 1
            if camera_idx_right > len(aligned_camera_timestamps) - 1:
                continue
        else:
            camera_idx_left = camera_idx - 1
            camera_idx_right = camera_idx
    
        camera_left_timestamp = aligned_camera_timestamps[camera_idx_left]
        camera_right_timestamp = aligned_camera_timestamps[camera_idx_right]
        camera_left_position = positions_final[camera_idx_left]
        camera_right_position = positions_final[camera_idx_right]
        camera_left_angle = angles_final[camera_idx_left]
        camera_right_angle = angles_final[camera_idx_right]
        # np.linalg.norm(camera_right_angle)
        camera_left_velocity = velocitys_final[camera_idx_left]
        camera_right_velocity = velocitys_final[camera_idx_right]
        delta_timestamp = (radar_timestamp - camera_left_timestamp) / (camera_right_timestamp - camera_left_timestamp)
        position = camera_left_position + (camera_right_position - camera_left_position) * delta_timestamp
        angle = slerp(camera_left_angle, camera_right_angle, delta_timestamp)
        velocity = camera_left_velocity + (camera_right_velocity - camera_left_velocity) * delta_timestamp
        positions_final_upsampled.append(position)
        angles_final_upsampled.append(angle)
        velocitys_final_upsampled.append(velocity)
    timestamps_radar_final_upsampled = timestamps_radar_final_upsampled[:len(positions_final_upsampled)]
    radar_path_final_upsampled = radar_path_final_upsampled[:len(positions_final_upsampled)]

    # pdb.set_trace()
    return radar_path_final, lidar_path_final, camera_path_final, positions_final, angles_final, velocitys_final, timestamps_radar_final, timestamps_lidar_final, timestamps_camera_final, (timestamps_radar_final_upsampled, radar_path_final_upsampled, positions_final_upsampled, angles_final_upsampled, velocitys_final_upsampled)


def slerp(q1, q2, t):
    # # Example
    # q1 = np.array([0.1, 0.2, 0.3, 0.4])
    # q2 = np.array([0.5, 0.6, 0.7, 0.8])
    # t1 = 0
    # t2 = 1
    # t3 = 0.5 

    # q1 = q1 / np.linalg.norm(q1)
    # q2 = q2 / np.linalg.norm(q2)

    # q3 = slerp(q1, q2, t3)

    # print("Quaternion at t3:", q3)

    dot_product = np.dot(q1, q2)

    # If the dot product is negative, reverse one of the input
    # quaternions to take the shorter path on the quaternion sphere.
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    # If the dot product is very close to 1, the quaternions are very close
    # and linear interpolation can be used to avoid numerical instability.
    if dot_product > 0.9995:
        return (1 - t) * q1 + t * q2

    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q1 + s1 * q2


def upsample_timestamps(radar_timestamps, upsample_factor=11):
    upsampled_timestamps = []

    for i in range(len(radar_timestamps) - 1):
        start_timestamp = radar_timestamps[i]
        end_timestamp = radar_timestamps[i + 1]
        interval = (end_timestamp - start_timestamp) / upsample_factor

        new_timestamps = [start_timestamp + j * interval for j in range(upsample_factor - 1)]
        upsampled_timestamps.extend(new_timestamps)

    # Add the last timestamp from the original list to the upsampled list
    upsampled_timestamps.append(radar_timestamps[-1])

    return upsampled_timestamps

def interpolate_filenames(radar_path, upsample_factor=11):
    interpolated_filenames = []

    for path in radar_path[:-1]:
        # Get the directory, filename, and extension
        directory, filename = os.path.split(path)
        file_base, file_ext = os.path.splitext(filename)
        
        # Interpolate K times
        for i in range(upsample_factor - 1):
            new_filename = f"{file_base}_{i}{file_ext}"
            new_path = os.path.join(directory, new_filename)
            interpolated_filenames.append(new_path)

    # Add the last filename from the original list to the interpolated list
    interpolated_filenames.append(radar_path[-1])

    return interpolated_filenames

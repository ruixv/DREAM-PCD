"""
RadarEyes Dataset Processing

This Python script is dedicated to the processing of the RadarEyes dataset, a comprehensive collection for the DREAM-PCD project focused on Deep Reconstruction and Enhancement of mmWave Radar Pointcloud data. The script encompasses a variety of functions essential for manipulating millimeter-wave radar point cloud data. Key operations include the conversion of point clouds between numerous formats (e.g., ndarray to Open3D, Cartesian voxel to PCD, polar voxel to PCD), the application of transformations on point clouds with given position and orientation data, as well as the translation of quaternions into Euler angles.

The RadarEyes dataset on GitHub is at https://github.com/ruixv/RadarEyes.

Author: USTC IP_LAB millimeter wave point cloud group
Main Contributors: Ruixu Geng (gengruixu@mail.ustc.edu.cn), Yadong Li, Jincheng Wu, Yating Gao
Copyright (C) USTC IP_LAB, 2023.
Initial Release Date: March 2023
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mmwavePCD_util import transform_point_cloud, ndarrayPCD_to_o3dPCD
from radarPcdProcessing import adc2pcd, adc2pcd_peakdetection
from ParseData import parsingRadarData
from ParseData import make_dataset
import radarPcdProcessing
import os
import numpy as np
import open3d as o3d
import pdb
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from Visualize import open3d_extend, visData
from point_cloud_processing import load_point_cloud
import copy 


if __name__ == "__main__":
  
    Scene_names = ["2023_06_04_22_48_54_A409_hall_4_50s"]

    dataset_path = "../../"
    # dataset_path = "D://RadarEyes//"
    for scene_i in range(len(Scene_names)):

        # Set parameters
        scene_name = Scene_names[scene_i]
        Scene_path = os.path.join(dataset_path, scene_name)
        dataset_config = scene_name.split("_")[-1]
        
        radar_config_azi, radarObj_azi = parsingRadarData.parse_radar_config(sensor="1843", parameter="azi")
        radar_config_ele, radarObj_ele = parsingRadarData.parse_radar_config(sensor="1843", parameter="coherentEle")

        # Given a scene, get the directories
        radar_path_azi = os.path.join(Scene_path, "1843_azi")
        radar_path_ele = os.path.join(Scene_path, "1843_ele")
        camera_path = os.path.join(Scene_path, "Camera_ZED")
        lidar_path = os.path.join(Scene_path, "Lidar")

        # set the output video filename
        output_filename = os.path.join(Scene_path, '1843.mp4')
        radar_path_final_azi, positions_radar_final_azi, angles_radar_final_azi,timestamps_radar_final_azi,radar_path_final_ele, positions_radar_final_ele, angles_radar_final_ele,timestamps_radar_final_ele,lidar_path_final, positions_lidar_final, angles_lidar_final, timestamps_lidar_final, camera_path_final, positions_camera_final, angles_camera_final, timestamps_camera_final  = make_dataset.make_selfmade_pcd_dataset_tworadar(radar_path_azi, radar_path_ele, lidar_path, camera_path, radar_config_azi, radarObj_azi, radar_config_ele, radarObj_ele)
        
        # ===================================================================================
        # Example: How to load ADC files and generate point cloud
        # This section of code can be commented out when necessary
        # radar_file_10 = radar_path_final_azi[10]
        # radar_azi_file_10 = radar_file_10.replace("PCD_SamePaddingUDPERROR", "ADC", 1)
        # with open(radar_azi_file_10, "rb") as f:
        #     loaded_data = np.frombuffer(f.read(), dtype='complex128')
        # # or
        # loaded_data = np.fromfile(radar_azi_file_10,dtype = "complex128")

        # loaded_data_adc = loaded_data.reshape((radar_config_azi.numAdcSamples, radar_config_azi.numChirpsPerFrame, radar_config_azi.numRxChan, radar_config_azi.numTxChan))
        # rangeFFTOut, DopplerFFTOut, xyz_ticode, xyz_ticode_nomultipath = adc2pcd(loaded_data_adc,radar_config_azi)
        # print("Test code for a single radar frmae")
        # visData.plotPcd(xyz_ticode, cmp=None, with_grid=False, view_init_angles=(100, 10), point_size=3)
        # ====================================================================================
        
        print("radar_frame_number", len(radar_path_final_azi))
        print("lidar_frame_number", len(lidar_path_final))
        # print("camera_frame_number", len(camera_path_final))
        # 
        # if "PCD_SamePaddingUDPERROR" not in radar_path_final_azi[0]:
        #     point_clouds = [np.fromfile(f.replace("PCD","PCD_SamePaddingUDPERROR"),dtype = 'float32').reshape((-1,4)) for f in radar_path_final_azi]
        # else:
        #     point_clouds = [np.fromfile(f,dtype = 'float32').reshape((-1,4)) for f in radar_path_final_azi]
        # pdb.set_trace()

        point_clouds = [load_point_cloud(f) for f in radar_path_final_azi]
        print("success", scene_name)
        

        
        if "vertical" in scene_name:
            for pcd_index in range(len(point_clouds)):
                # pdb.set_trace()
                pcd = point_clouds[pcd_index]
                temp_x = copy.deepcopy(pcd[:,0])
                temp_z = copy.deepcopy(pcd[:,2])
                point_clouds[pcd_index][:,0] = temp_z
                point_clouds[pcd_index][:,2] = -temp_x
                point_clouds[pcd_index] = point_clouds[pcd_index][(point_clouds[pcd_index][:, 2] >= -0.0) & (point_clouds[pcd_index][:, 2] <= 1)]
                # pdb.set_trace()

        positions_radar_final_azi = positions_radar_final_azi + [-0.086, -0.01, 0.102]

        if "ZED" in camera_path:
            coor_transform_flag = False
        else:
            coor_transform_flag = True

        transformed_point_clouds = [
            transform_point_cloud(point_clouds[i], positions_radar_final_azi[i], angles_radar_final_azi[i],coor_transform_flag)
            for i in range(len(point_clouds))
        ]

        # Merge all point clouds
        merged_point_cloud = np.vstack(transformed_point_clouds)
        
        method_pcd = merged_point_cloud[:,:3]
        min_height = -1
        max_height = 3
        mask = (method_pcd[:, 2] >= min_height-1) & (method_pcd[:, 2] <= max_height+1)
        method_pcd = method_pcd[mask]
        # Convert NumPy array to Open3D point cloud format
        visData.plotPcd(method_pcd, cmp=None, with_grid=True, view_init_angles=(100, 10), point_size=3)

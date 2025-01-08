from mmwavePCD_util import transform_point_cloud, ndarrayPCD_to_o3dPCD
from ParseData import parsingRadarData
from ParseData import make_dataset
import os
import numpy as np
import open3d as o3d
import pdb
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from Visualize import open3d_extend, visData
import copy 


if __name__ == "__main__":
    Scene_names = ["2023_06_30_19_54_23"]
    # Set the dataset path
    # dataset_path = "E://Dataset//RadarEyes//"
    dataset_path = "../../"

    for scene_i in range(len(Scene_names)):
        # Setting parameters
        scene_name = Scene_names[scene_i]
        Scene_path = os.path.join(dataset_path, scene_name)
        dataset_config = scene_name.split("_")[-1]
        radar_config_azi, radarObj_azi = parsingRadarData.parse_radar_config(sensor="1843", parameter="azi") # coloradar
        radar_config_ele, radarObj_ele = parsingRadarData.parse_radar_config(sensor="1843", parameter="coherentEle")

        # Given the scene, get the directory
        radar_path_azi = os.path.join(Scene_path, "1843_azi")
        radar_path_ele = os.path.join(Scene_path, "1843_ele")
        camera_path = os.path.join(Scene_path, "Camera_ZED")
        lidar_path = os.path.join(Scene_path, "Lidar")
        
        # Set the output video filename
        output_filename = os.path.join(Scene_path, '1843.mp4')
        
        radar_path_final_azi, positions_radar_final_azi, angles_radar_final_azi, timestamps_radar_final_azi, radar_path_final_ele, positions_radar_final_ele, angles_radar_final_ele, timestamps_radar_final_ele, lidar_path_final, positions_lidar_final, angles_lidar_final, timestamps_lidar_final, camera_path_final, positions_camera_final, angles_camera_final, timestamps_camera_final  = make_dataset.make_selfmade_pcd_dataset_tworadar(radar_path_azi, radar_path_ele, lidar_path, camera_path, radar_config_azi, radarObj_azi, radar_config_ele, radarObj_ele)
        
        print("radar_frame_number", len(radar_path_final_azi))
        print("lidar_frame_number", len(lidar_path_final))
        print("camera_frame_number", len(camera_path_final))

        point_clouds = []
        for f in lidar_path_final:
            try:
                point_cloud_this_file = np.fromfile(f.replace('Lidar', 'Lidar_pcd', 1),dtype = 'float32').reshape((-1,4))
                point_cloud_this_file = point_cloud_this_file[point_cloud_this_file[:,2]>-0.3]
            except:
                point_cloud_this_file = np.zeros((1,4))
            point_clouds.append(point_cloud_this_file)
        
        # Apply transformations
        if "ZED" in camera_path:
            coor_transform_flag = False
        else:
            coor_transform_flag = True

        # Uncomment the line below to use pdb debugger
        # pdb.set_trace()

        # Uncomment the line below to visualize the lidar position data
        # visData.plotPcd(np.array(positions_lidar_final))

        # Transform point clouds using the lidar position and angle data
        transformed_point_clouds = [
            transform_point_cloud(point_clouds[i], positions_lidar_final[i], angles_lidar_final[i],coor_transform_flag)
            for i in range(len(point_clouds)-5)
        ]
        
        # Merge all point clouds
        merged_point_cloud = np.vstack(transformed_point_clouds)

        method_pcd = merged_point_cloud[:,:3]
        min_height = -1
        max_height = 3
        mask = (method_pcd[:, 2] >= min_height-1) & (method_pcd[:, 2] <= max_height+1)
        method_pcd = method_pcd[mask]

        visData.plotPcd(method_pcd,cmp=None, with_grid=False, view_init_angles=(100, 10),point_size=3)

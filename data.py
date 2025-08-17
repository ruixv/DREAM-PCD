#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RadarEyes Dataset Handler for Point Cloud Classification/Segmentation with Dual Input Support
"""


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pdb

def load_dual_data_semseg_RadarEyes(partition, scenes_list, data_type1, data_type2, num_points):
    """
    Load two types of RadarEyes datasets for dual-input processing
    
    Parameters:
    -----------
    partition : str
        'train' or 'test' to specify the dataset partition
    scenes_list : str
        Comma-separated scene identifiers
    data_type1 : str
        First type of radar data to use
    data_type2 : str
        Second type of radar data to use
    num_points : int
        Number of points to sample from each point cloud
        
    Returns:
    --------
    all_data1 : numpy.ndarray
        First type point cloud data
    all_data2 : numpy.ndarray
        Second type point cloud data
    all_seg : numpy.ndarray
        Segmentation labels
    """
    # Determine the data directory based on the operating system
    # Change to your dataset path
    if os.name == 'nt':  # Windows
        dataset_path = "./Datasets/"
        if os.path.exists(dataset_path):
            DATA_DIR = dataset_path
        else:
            DATA_DIR = "./Datasets/"
    else:  # Unix-like systems (Linux, macOS, etc.)
        DATA_DIR = "./Datasets/"
    
    # Determine the folder name and feature count based on data_type
    folder_dict = {
        "azi_pcd_normalthr_lq_accumulated": ("azi_pcd_normalthr_lq_accumulated", 10),
        "azi_pcd_normalthr_lq_single": ("azi_pcd_normalthr_lq_singleframe", 10),
        "azi_pcd_normalthr_hq_accumulated": ("azi_pcd_normalthr_hq_accumulated", 10),
        "azi_pcd_normalthr_hq_single": ("azi_pcd_normalthr_hq_singleframe", 10),
        "azi_pcd_lowthr_lq_accumulated": ("azi_pcd_lowthr_lq_accumulated", 10),
        "azi_pcd_lowthr_lq_single": ("azi_pcd_lowthr_lq_singleframe", 10),
        "azi_pcd_lowthr_hq_accumulated": ("azi_pcd_lowthr_hq_accumulated", 10),
        "azi_pcd_lowthr_hq_single": ("azi_pcd_lowthr_hq_singleframe", 10),
        "ele_coherent": ("ele_coherent_accumulated", 5),
        "ele_noncoherent_accumulated": ("lowthr_noncoherent_ele_accumulated", 10),
        "ele_noncoherent_single": ("ele_singleframe", 10)
    }
    
    if data_type1 not in folder_dict:
        raise ValueError(f"Unsupported data_type1: {data_type1}")
    if data_type2 not in folder_dict:
        raise ValueError(f"Unsupported data_type2: {data_type2}")
    
    folder_name1, feature_nums1 = folder_dict[data_type1]
    folder_name2, feature_nums2 = folder_dict[data_type2]

    # Parse scenes list
    scenes_lists = scenes_list.replace("'", "").split(",")
    
    # ==== adjust according to the dataset ====
    train_lists = scenes_lists[:3]
    test_lists = scenes_lists[-2:]
    
    # Function to normalize features
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    # Load appropriate dataset based on partition
    if partition == 'train':
        from IPLab_mmwavePCD.ParseData import make_dataset
        
        all_pcds_train1 = []
        all_pcds_train2 = []
        
        for folder_list in train_lists:
            # Load first data type
            pcd_dir1 = os.path.join(DATA_DIR, folder_list, folder_name1)
            pcd_file_lists1 = make_dataset.make_pcd_dataset(pcd_dir1)
            
            # Load second data type
            pcd_dir2 = os.path.join(DATA_DIR, folder_list, folder_name2)
            pcd_file_lists2 = make_dataset.make_pcd_dataset(pcd_dir2)
            
            # Make sure both lists have the same file names
            common_files = list(set([os.path.basename(f) for f in pcd_file_lists1]).intersection(
                            set([os.path.basename(f) for f in pcd_file_lists2])))
            
            for file_name in common_files:
                # Get full path for both data types
                file1 = next(f for f in pcd_file_lists1 if os.path.basename(f) == file_name)
                file2 = next(f for f in pcd_file_lists2 if os.path.basename(f) == file_name)
                
                # Load point clouds
                point_cloud1 = np.fromfile(file1, dtype='float32').reshape((-1, feature_nums1))
                point_cloud2 = np.fromfile(file2, dtype='float32').reshape((-1, feature_nums2))
                
                # Normalize features if needed
                if feature_nums1 > 5:
                    point_cloud1[:, 4] = norm(point_cloud1[:, 4])
                    point_cloud1[:, 5] = norm(point_cloud1[:, 5])
                
                if feature_nums2 > 5:
                    point_cloud2[:, 4] = norm(point_cloud2[:, 4])
                    point_cloud2[:, 5] = norm(point_cloud2[:, 5])

                # Skip if either point cloud doesn't have enough points
                if point_cloud1.shape[0] < num_points or point_cloud2.shape[0] < num_points:
                    continue
                
                # Ensure both point clouds have same number of points
                if point_cloud1.shape[0] > num_points:
                    # Use same indices for both point clouds to maintain correspondence
                    choice_indices = np.random.choice(point_cloud1.shape[0], num_points, replace=False)
                    point_cloud1 = point_cloud1[choice_indices]
                    
                if point_cloud2.shape[0] > num_points:
                    choice_indices = np.random.choice(point_cloud2.shape[0], num_points, replace=False)
                    point_cloud2 = point_cloud2[choice_indices]
                    
                all_pcds_train1.append(point_cloud1)
                all_pcds_train2.append(point_cloud2)
                
            print(f"Finished: {folder_list} in {partition} set; data size: {len(train_lists)} in {partition} set")
            
        all_data1 = np.stack(all_pcds_train1, axis=0)
        all_data2 = np.stack(all_pcds_train2, axis=0)
        
    else:  # test partition
        from IPLab_mmwavePCD.ParseData import make_dataset
        
        all_pcds_test1 = []
        all_pcds_test2 = []
        
        for folder_list in test_lists:
            # Load first data type
            pcd_dir1 = os.path.join(DATA_DIR, folder_list, folder_name1)
            pcd_file_lists1 = make_dataset.make_pcd_dataset(pcd_dir1)
            
            # Load second data type
            pcd_dir2 = os.path.join(DATA_DIR, folder_list, folder_name2)
            pcd_file_lists2 = make_dataset.make_pcd_dataset(pcd_dir2)
            
            # Make sure both lists have the same file names
            common_files = list(set([os.path.basename(f) for f in pcd_file_lists1]).intersection(
                            set([os.path.basename(f) for f in pcd_file_lists2])))
            
            for file_name in common_files:
                # Get full path for both data types
                file1 = next(f for f in pcd_file_lists1 if os.path.basename(f) == file_name)
                file2 = next(f for f in pcd_file_lists2 if os.path.basename(f) == file_name)
                
                # Load point clouds
                point_cloud1 = np.fromfile(file1, dtype='float32').reshape((-1, feature_nums1))
                point_cloud2 = np.fromfile(file2, dtype='float32').reshape((-1, feature_nums2))
                
                # Normalize features if needed
                if feature_nums1 > 5:
                    point_cloud1[:, 4] = norm(point_cloud1[:, 4])
                    point_cloud1[:, 5] = norm(point_cloud1[:, 5])
                
                if feature_nums2 > 5:
                    point_cloud2[:, 4] = norm(point_cloud2[:, 4])
                    point_cloud2[:, 5] = norm(point_cloud2[:, 5])

                # Skip if either point cloud doesn't have enough points
                if point_cloud1.shape[0] < num_points or point_cloud2.shape[0] < num_points:
                    continue
                
                # Ensure both point clouds have same number of points
                if point_cloud1.shape[0] > num_points:
                    # Use same indices for both point clouds to maintain correspondence
                    choice_indices = np.random.choice(point_cloud1.shape[0], num_points, replace=False)
                    point_cloud1 = point_cloud1[choice_indices]
                    
                if point_cloud2.shape[0] > num_points:
                    choice_indices = np.random.choice(point_cloud2.shape[0], num_points, replace=False)
                    point_cloud2 = point_cloud2[choice_indices]
                    
                all_pcds_test1.append(point_cloud1)
                all_pcds_test2.append(point_cloud2)
                
            print(f"Finished: {folder_list} in {partition} set; data size: {len(test_lists)} in {partition} set")
            
        all_data1 = np.stack(all_pcds_test1, axis=0)
        all_data2 = np.stack(all_pcds_test2, axis=0)

    # Extract segmentation labels and prepare data (using labels from first data type)
    all_seg = all_data1[:, :, -1]
    input_num1 = int(feature_nums1 - 1)
    input_num2 = int(feature_nums2 - 1)
    all_data1 = all_data1[:, :, :input_num1]
    all_data2 = all_data2[:, :, :input_num2]
    
    print(f"{partition} set's data size: {all_data1.shape[0]}")
    
    return all_data1, all_data2, all_seg


class DualRadarEyes(Dataset):
    """
    PyTorch Dataset for Dual-Input RadarEyes point cloud segmentation
    """
    def __init__(self, num_points, partition, scenes_list, data_type1, data_type2):
        """
        Initialize the Dual RadarEyes dataset
        
        Parameters:
        -----------
        num_points : int
            Number of points to sample from each point cloud
        partition : str
            'train' or 'test' to specify the dataset partition
        scenes_list : str
            Comma-separated scene identifiers
        data_type1 : str
            First type of radar data to use
        data_type2 : str
            Second type of radar data to use
        """
        # Load the data and segmentation labels
        self.data1, self.data2, self.seg = load_dual_data_semseg_RadarEyes(
            partition, scenes_list, data_type1, data_type2, num_points)
        self.num_points = num_points
        self.partition = partition
        

    def __getitem__(self, item):
        """
        Get two point clouds and their segmentation labels
        
        Parameters:
        -----------
        item : int
            Index of the item to retrieve
            
        Returns:
        --------
        tuple
            (point_cloud1, point_cloud2, segmentation_labels)
        """
        pointcloud1 = self.data1[item][:self.num_points]
        pointcloud2 = self.data2[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        
        # Shuffle points during training for better generalization
        # Use same shuffle indices for both point clouds to maintain correspondence
        if self.partition == 'train':
            indices = list(range(pointcloud1.shape[0]))
            np.random.shuffle(indices)
            pointcloud1 = pointcloud1[indices]
            pointcloud2 = pointcloud2[indices]
            seg = seg[indices]
            
        seg = torch.LongTensor(seg)
        return pointcloud1, pointcloud2, seg

    def __len__(self):
        """
        Get the length of the dataset
        
        Returns:
        --------
        int
            Number of point clouds in the dataset
        """
        return self.data1.shape[0]


# Keep the original dataset class for backward compatibility
class RadarEyes(Dataset):
    """
    PyTorch Dataset for RadarEyes point cloud segmentation
    """
    def __init__(self, num_points, partition, scenes_list, data_type):
        """
        Initialize the RadarEyes dataset
        
        Parameters:
        -----------
        num_points : int
            Number of points to sample from each point cloud
        partition : str
            'train' or 'test' to specify the dataset partition
        scenes_list : str
            Comma-separated scene identifiers
        data_type : str
            Type of radar data to use
        """
        # Determine the data directory based on the operating system
        if os.name == 'nt':  # Windows
            dataset_path = "E://Dataset//selfmade_Coloradar//"
            if os.path.exists(dataset_path):
                DATA_DIR = dataset_path
            else:
                DATA_DIR = "D://SelfColoradar//DataCaptured//"
        else:  # Unix-like systems (Linux, macOS, etc.)
            DATA_DIR = "/share2/data/ruixu/RadarEyes/"
        
        # Determine the folder name and feature count based on data_type
        folder_dict = {
            "azi_pcd_normalthr_lq_accumulated": ("azi_pcd_normalthr_lq_accumulated", 10),
            "azi_pcd_normalthr_lq_single": ("azi_pcd_normalthr_lq_singleframe", 10),
            "azi_pcd_normalthr_hq_accumulated": ("azi_pcd_normalthr_hq_accumulated", 10),
            "azi_pcd_normalthr_hq_single": ("azi_pcd_normalthr_hq_singleframe", 10),
            "azi_pcd_lowthr_lq_accumulated": ("azi_pcd_lowthr_lq_accumulated", 10),
            "azi_pcd_lowthr_lq_single": ("azi_pcd_lowthr_lq_singleframe", 10),
            "azi_pcd_lowthr_hq_accumulated": ("azi_pcd_lowthr_hq_accumulated", 10),
            "azi_pcd_lowthr_hq_single": ("azi_pcd_lowthr_hq_singleframe", 10),
            "ele_coherent": ("ele_coherent_accumulated", 5),
            "ele_noncoherent_accumulated": ("lowthr_noncoherent_ele_accumulated", 10),
            "ele_noncoherent_single": ("ele_singleframe", 10)
        }
        
        if data_type not in folder_dict:
            raise ValueError(f"Unsupported data_type: {data_type}")
        
        folder_name, feature_nums = folder_dict[data_type]

        # Parse scenes list
        scenes_lists = scenes_list.replace("'", "").split(",")

        # ==== adjust according to the dataset ====
        train_lists = scenes_lists[:3]
        test_lists = scenes_lists[-2:]
        
        # Function to normalize features
        norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Load appropriate dataset based on partition
        if partition == 'train':
            from IPLab_mmwavePCD.ParseData import make_dataset
            
            all_pcds_train = []
            for folder_list in train_lists:
                pcd_dir = os.path.join(DATA_DIR, folder_list, folder_name)
                pcd_file_lists = make_dataset.make_pcd_dataset(pcd_dir)
                
                for pcd_file in pcd_file_lists:
                    point_cloud_this_file = np.fromfile(pcd_file, dtype='float32').reshape((-1, feature_nums))
                    
                    # Normalize features if needed
                    if feature_nums > 5:
                        point_cloud_this_file[:, 4] = norm(point_cloud_this_file[:, 4])
                        point_cloud_this_file[:, 5] = norm(point_cloud_this_file[:, 5])

                    # Skip if not enough points
                    if point_cloud_this_file.shape[0] < num_points:
                        continue
                        
                    # Random sampling if more than required points
                    if point_cloud_this_file.shape[0] > num_points:
                        point_cloud_this_file = point_cloud_this_file[np.random.choice(
                            point_cloud_this_file.shape[0], num_points, replace=False)]
                        
                    all_pcds_train.append(point_cloud_this_file)
                    
                print(f"Finished: {folder_list} in {partition} set; data size: {len(train_lists)} in {partition} set")
                
            self.data = np.stack(all_pcds_train, axis=0)
            
        else:  # test partition
            from IPLab_mmwavePCD.ParseData import make_dataset
            
            all_pcds_test = []
            for folder_list in test_lists:
                pcd_dir = os.path.join(DATA_DIR, folder_list, folder_name)
                pcd_file_lists = make_dataset.make_pcd_dataset(pcd_dir)
                
                for pcd_file in pcd_file_lists:
                    point_cloud_this_file = np.fromfile(pcd_file, dtype='float32').reshape((-1, feature_nums))
                    
                    # Normalize features if needed
                    if feature_nums > 5:
                        point_cloud_this_file[:, 4] = norm(point_cloud_this_file[:, 4])
                        point_cloud_this_file[:, 5] = norm(point_cloud_this_file[:, 5])

                    # Skip if not enough points
                    if point_cloud_this_file.shape[0] < num_points:
                        continue
                        
                    # Random sampling if more than required points
                    if point_cloud_this_file.shape[0] > num_points:
                        point_cloud_this_file = point_cloud_this_file[np.random.choice(
                            point_cloud_this_file.shape[0], num_points, replace=False)]
                        
                    all_pcds_test.append(point_cloud_this_file)
                    
                print(f"Finished: {folder_list} in {partition} set; data size: {len(test_lists)} in {partition} set")
                
            self.data = np.stack(all_pcds_test, axis=0)

        # Extract segmentation labels and prepare data
        self.seg = self.data[:, :, -1]
        input_num = int(feature_nums - 1)
        self.data = self.data[:, :, :input_num]
        # pdb.set_trace()
        print(f"{partition} set's data size: {self.data.shape[0]}")
        
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        """
        Get a point cloud and its segmentation labels
        
        Parameters:
        -----------
        item : int
            Index of the item to retrieve
            
        Returns:
        --------
        tuple
            (point_cloud, segmentation_labels)
        """
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        
        # Shuffle points during training for better generalization
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
            
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        """
        Get the length of the dataset
        
        Returns:
        --------
        int
            Number of point clouds in the dataset
        """
        return self.data.shape[0]
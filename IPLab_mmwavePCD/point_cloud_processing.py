import open3d as o3d
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

def save_point_cloud(file_path, point_cloud, dim_description):
    """
    Save point cloud data to a binary file.
    
    Args:
        file_path: Path to save the binary file
        point_cloud: Point cloud array of shape (N, X), where N is the number of points
                    and X is the feature dimension
        dim_description: String describing the meaning of each dimension in the point cloud
    """
    # Ensure point cloud is a float32 numpy array
    point_cloud = np.array(point_cloud, dtype=np.float32)
    
    # Get the dimensionality of point cloud
    point_cloud_dims = point_cloud.shape[1]
    
    # Convert dimension description to bytes
    dim_description_bytes = dim_description.encode('utf-8')
    
    # Open file in binary write mode
    with open(file_path, 'wb') as file:
        # Write dimensionality information first
        file.write(np.array([point_cloud_dims], dtype=np.int32).tobytes())
        
        # Write length and content of dimension description
        file.write(np.array([len(dim_description_bytes)], dtype=np.int32).tobytes())
        file.write(dim_description_bytes)
        
        # Write point cloud data
        file.write(point_cloud.tobytes())
        
    print(f"Point cloud saved to {file_path}. Dimensionality: {point_cloud_dims}")

def load_point_cloud(file_path, expected_dims=None, printDescription=False):
    """
    Load point cloud data from a binary file.
    Must be used in conjunction with save_point_cloud function.
    
    Args:
        file_path: Path to the binary file containing point cloud data
        expected_dims: Optional. Expected number of dimensions in the point cloud.
                      Raises ValueError if dimensions don't match.
    
    Returns:
        numpy.ndarray: Point cloud array of shape (N, X)
    """
    # Open file in binary read mode
    with open(file_path, 'rb') as file:
        # Read dimensionality information
        point_cloud_dims = np.frombuffer(file.read(4), dtype=np.int32)[0]
        
        # Validate dimensions if expected_dims is provided
        if expected_dims is not None and point_cloud_dims != expected_dims:
            raise ValueError(f"Expected point cloud dimensions to be {expected_dims}, but found {point_cloud_dims}")
        
        # Read dimension description length and content
        dim_description_length = np.frombuffer(file.read(4), dtype=np.int32)[0]
        dim_description_bytes = file.read(dim_description_length)
        dim_description = dim_description_bytes.decode('utf-8')
        
        # Read remaining data as point cloud
        point_cloud_data = np.frombuffer(file.read(), dtype=np.float32)
        
    # Reshape array to recover original dimensions
    point_cloud = point_cloud_data.reshape(-1, point_cloud_dims)
    if printDescription:
        print(f"Loaded point cloud from {file_path}.")
        print(f"Dimension description: {dim_description}")
    
    return point_cloud

def downsample_pointcloud(in_pcl, vox_size=1, num_points=3000):
    """
    Downsamples a pointcloud for faster plotting using a voxel grid.
    The output pointcloud will have at most one point in a given voxel.

    :param in_pcl: Input pointcloud to be downsampled.
    :param vox_size: Voxel size for downsampling.
    :param num_points: Number of points in the output pointcloud.
    :return: Downsampled pointcloud.
    """

    # Create a unique index for each point in the input pointcloud based on voxel coordinates
    _, idx = np.unique((in_pcl[:, :3] / vox_size).round(), return_index=True, axis=0)

    # If the number of unique indices is greater than the desired number of points,
    # randomly select a subset of indices
    if idx.size > num_points:
        sample_list = list(range(idx.size))
        sample_list = random.sample(sample_list, num_points)
        idx = idx[sample_list]

    # If the number of unique indices is less than the desired number of points,
    # create additional samples to fill the gap
    elif idx.size < num_points:
        sample_list = list(range(idx.size))

        # Try different sampling factors to achieve the desired number of points
        for factor in [10, 100, 500]:
            try:
                sample_list = random.sample(sample_list * factor, num_points - idx.size)
                break
            except ValueError:
                continue

        idx_new = idx[sample_list]
        idx = np.concatenate((idx, idx_new))

    # Create the output pointcloud using the selected indices
    out_pcl = in_pcl[idx, :]
    return out_pcl



def downsample_denoise_pointcloud(points: np.ndarray, num_points_target: int = 4000, voxel_size: float = 0.02, num_neighbors: int = 20, std_dev_factor: float = 2.0) -> np.ndarray:
    """
    Downsample and denoise a point cloud.

    Args:
        points (np.ndarray): Input point cloud as an (N, 4) ndarray.
        num_points_target (int): Target number of downsampled points, default is 4000.
        voxel_size (float): Voxel size, default is 0.02.
        num_neighbors (int): Number of neighboring points, default is 20.
        std_dev_factor (float): Standard deviation multiplier for determining the threshold for removing points, default is 2.0.

    Returns:
        np.ndarray: Downsampled and denoised point cloud as an (M, 4) ndarray, where M <= num_points_target.
    """
    # Convert ndarray to Open3D PointCloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Only use the first 3 columns (x, y, z coordinates)

    # Use the 4th column as color information and duplicate it three times to create a pseudo RGB color
    colors = np.tile(points[:, 3].reshape(-1, 1), (1, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Downsample using VoxelGrid
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    # Denoise using statistical outlier removal
    denoised_pcd, _ = downsampled_pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_dev_factor)

    # Further downsample if the number of points after downsampling is greater than 4000
    if len(denoised_pcd.points) > num_points_target:
        ratio = num_points_target / len(denoised_pcd.points)
        final_pcd = denoised_pcd.uniform_down_sample(every_k_points=int(1 / ratio))
    else:
        final_pcd = denoised_pcd

    # Convert the downsampled and denoised point cloud back to an ndarray
    final_points = np.hstack((np.asarray(final_pcd.points), np.mean(np.asarray(final_pcd.colors), axis=1).reshape(-1, 1)))
    
    return final_points

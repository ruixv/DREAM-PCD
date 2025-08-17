"""
Title: Millimeter Wave Point Cloud Visualization
Author: USTC IP_LAB Millimeter Wave Point Cloud Group
Main Members: Ruixu Geng (gengruixu@mail.ustc.edu.cn), Yadong Li, Jincheng Wu, Yating Gao
Copyright: Copyright (C) 2023 USTC IP_LAB Millimeter Wave Point Cloud Group
Date: March 2023

This Python script provides various functions for visualizing point clouds and data related to millimeter wave imaging. The functions support numpy arrays and PyTorch tensors for visualization, and make use of Open3D, Matplotlib, and other libraries to display and save the visualizations.

The main functions provided in this script are:

plotPcd: Visualize a single point cloud using Open3D
plotTwoPcds: Visualize two point clouds side by side using Open3D
plotHeatmap: Visualize heatmap data using Matplotlib
plotVector: Plot a vector with optional x-axis and bin size
plotVectorNetwork: Plot three vectors for neural network output, ground truth, and input
plotVectorHist: Plot a histogram of the input vector
plotThreeView: Plot three views of four input images for neural network output and ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
try:
    import torch
except:
    print("Warning: PyTorch is not installed")
from .. import mmwavePCD_util
from ..Visualize import open3d_extend
from pylab import xticks,yticks,np
import pdb
from matplotlib.colors import LinearSegmentedColormap

def plotPcd(pcd, cmp=None, with_grid=True, view_init_angles=None, point_size=5.0):
    """
    Visualize point clouds using Open3D.

    Args:
        pcd (np.ndarray, torch.Tensor, or o3d.geometry.PointCloud): Point cloud data with shape (N, 3) or (N, 4).
        view_init_angles (tuple): A tuple containing two angles (azimuth, elevation) for the initial view.
        point_size (float): The size of the points in the point cloud visualization.
    """

    # Convert input point cloud to o3d.geometry.PointCloud format
    if isinstance(pcd, np.ndarray):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    elif torch.is_tensor(pcd):
        pcd = pcd.cpu().numpy()
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    elif isinstance(pcd, o3d.geometry.PointCloud):
        pcd_o3d = pcd
    else:
        raise TypeError("Only Numpy/Tensor/Open3D visualization is supported")

    if cmp is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(cmp)
    else: 

        cdict = {'red': [(0.0, 0.0, 0.0),
                            (0.5, 0.0, 0.0),
                            (1.0, 0.5, 0.5)],
                    'green': [(0.0, 0.0, 0.0),
                            (0.5, 0.0, 0.0),
                            (1.0, 0.0, 0.0)],
                    'blue': [(0.0, 0.0, 0.0),
                            (0.5, 0.5, 0.5),
                            (1.0, 0.0, 0.0)]}
        custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
        cmap = plt.cm.get_cmap('turbo')
        pcd_np = np.asarray(pcd_o3d.points)
        normalized_heights = (pcd_np[:, 2] - pcd_np[:, 2].min()) / (pcd_np[:, 2].max() - pcd_np[:, 2].min())
        colors = cmap(normalized_heights)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3])

    def custom_draw_geometry(pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()

        if view_init_angles is not None:
            azimuth, elevation = view_init_angles
            ctr.set_lookat([0, 0, 0])  # Set the lookat point
            ctr.set_up([0, 1, 0])  # Set the up vector
            ctr.set_front([np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)), 
                           np.sin(np.radians(elevation)), 
                           np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))])  # Set the front vector

        if with_grid:
            FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
            grid = open3d_extend.create_grid_thick()
            vis.add_geometry(FOR1)
            vis.add_geometry(grid)

        # Adjust point size
        render_option = vis.get_render_option()
        render_option.point_size = point_size

        vis.run()
        vis.destroy_window()

    custom_draw_geometry(pcd_o3d)

def plotTwoPcds(pcd1, pcd2):
    """
    Visualize two point clouds using Open3D.

    Args:
        pcd1 (np.ndarray or torch.Tensor): First point cloud data with shape (N, 3).
        pcd2 (np.ndarray or torch.Tensor): Second point cloud data with shape (N, 3).
    """
    if isinstance(pcd1, np.ndarray):
        pass
    elif torch.is_tensor(pcd1):
        pcd1 = pcd1.cpu().numpy()
    else:
        raise TypeError("Only Numpy/Tensor visualization is supported")

    if isinstance(pcd2, np.ndarray):
        pass
    elif torch.is_tensor(pcd2):
        pcd2 = pcd2.cpu().numpy()
    else:
        raise TypeError("Only Numpy/Tensor visualization is supported")

    pcd_o3d_1 = o3d.geometry.PointCloud()
    pcd_o3d_1.points = o3d.utility.Vector3dVector(pcd1[:, :3])
    pcd_o3d_2 = o3d.geometry.PointCloud()
    pcd_o3d_2.points = o3d.utility.Vector3dVector(pcd2[:, :3])

    pcd_o3d_1.paint_uniform_color([1, 0.706, 0])  # Yellow
    pcd_o3d_2.paint_uniform_color([0, 0.706, 1])  # Blue

    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    grid = open3d_extend.create_grid_thick()
    o3d.visualization.draw_geometries([FOR1, pcd_o3d_1, pcd_o3d_2, grid], zoom=0.455, front=[-0.4999, -0.1659, -0.8499], lookat=[2.1813, 2.0619, 2.0999], up=[0.1204, -0.9852, 0.1215])

def visualize_point_cloud_on_server(point_cloud, colors=None, filename='pcd_debug.png'):
    """
    Visualize a 3D point cloud from four different angles and save the plots as PNG files.

    :param point_cloud: A numpy array with shape (n, 3) or (n, 4), where n is the number of points and columns represent x, y, z coordinates and optionally color.
    :param colors: Optional, a numpy array with shape (n, 4) representing RGBA colors.
    :param output_prefix: The prefix for the output PNG file names.
    """
    if point_cloud.shape[1] == 4 and colors is None:
        cmap = plt.cm.get_cmap('jet')
        colors = cmap(point_cloud[:, 3] / point_cloud[:, 3].max())
        point_cloud = point_cloud[:, :3]

    assert colors is None or colors.shape[1] == 4, "Colors should be in RGBA format"

    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]
    zs = point_cloud[:, 2]

    # Define the four viewpoints (azimuth, elevation)
    viewpoints = [(0, 90), (0, 0), (90, 0), (45, 30)]
    fig = plt.figure()
    for i, (azim, elev) in enumerate(viewpoints):
        
        # Add a 3D subplot to the figure with a 2x2 grid layout and assign it to 'ax'
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Set the title for each subplot based on the index 'i'
        # The title represents the viewpoint for each subplot
        if i == 0:
            ax.set_title('Top view') 
        elif i == 1:
            ax.set_title('Side view')     
        elif i == 2:
            ax.set_title('Front view')
        elif i == 3:
            ax.set_title('(45, 30)')       # Set title for the (45, 30) viewpoint
        ax.view_init(azim=azim, elev=elev)

        if colors is not None:
            ax.scatter(xs, ys, zs, marker='o', c=colors, s=0.02)
        else:
            ax.scatter(xs, ys, zs, marker='o', s=0.02)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.savefig(filename)
    plt.close(fig)


def plotHeatmap(heatmap, x_min=0, x_bin=1, y_min=0, y_bin=1):
    """
    Visualize heatmap data using Matplotlib.

    Args:
        heatmap (np.ndarray or torch.Tensor): Heatmap data.
    """
    if isinstance(heatmap, np.ndarray):
        pass
    elif torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()
    else:
        raise TypeError("Only Numpy/Tensor visualization is supported")
    
    # Get the dimensions of the heatmap
    num_x_cols, num_y_rows = heatmap.shape

    plt.matshow(heatmap, cmap=plt.cm.Reds)

    x_tick_step = int(np.ceil(num_x_cols / 50))
    y_tick_step = int(np.ceil(num_y_rows / 50))

    x_ticks = np.arange(0, num_x_cols, x_tick_step)
    x_labels = x_min + x_ticks * x_bin

    y_ticks = np.arange(0, num_y_rows, y_tick_step)
    y_labels = y_min + y_ticks * y_bin
    
    plt.yticks(x_ticks, x_labels)
    plt.xticks(y_ticks, y_labels)

    plt.show()
    plt.close()



def plotCFAR(heatmap1, Ind_obj_Rag1, heatmap2=None, Ind_obj_Rag2=None):
    """
    This function plots the absolute and angle values of one or two input heatmaps.
    
    :param heatmap1: First input heatmap as a 2D array (MxN) or tensor.
    :param Ind_obj_Rag1: Indices of objects in the first heatmap.
    :param heatmap2: Second input heatmap as a 2D array (MxN) or tensor, optional.
    :param Ind_obj_Rag2: Indices of objects in the second heatmap, optional.
    """

    # Convert input heatmaps to numpy arrays if they are tensors
    if isinstance(heatmap1, np.ndarray):
        pass
    elif torch.is_tensor(heatmap1):
        heatmap1 = heatmap1.cpu().detach().numpy()
    else:
        raise TypeError("Visualization only supports Numpy/Tensor")

    if heatmap2 is None:
        # Plot the absolute and angle values for the first heatmap
        ax1 = plt.subplot(121)
        ax1.matshow(np.abs(heatmap1), cmap=plt.cm.Blues)
        ax1.scatter(Ind_obj_Rag1[:, 1], Ind_obj_Rag1[:, 0])
        plt.title("abs")
        ax2 = plt.subplot(122)
        ax2.matshow(np.angle(heatmap1), cmap=plt.cm.Blues)
        plt.title("angle")
        plt.show()
        plt.savefig('./debug_vis.png')
        plt.close()

    else:
        # Convert the second heatmap to a numpy array if it is a tensor
        if isinstance(heatmap2, np.ndarray):
            pass
        elif torch.is_tensor(heatmap2):
            heatmap2 = heatmap2.cpu().detach().numpy()
        else:
            raise TypeError("Visualization only supports Numpy/Tensor")

        # Plot the absolute and angle values for both heatmaps
        ax1 = plt.subplot(221)
        ax1.matshow(np.abs(heatmap1), cmap=plt.cm.Blues)
        ax1.scatter(Ind_obj_Rag1[:, 1], Ind_obj_Rag1[:, 0])
        plt.title("abs")
        ax2 = plt.subplot(222)
        ax2.matshow(np.angle(heatmap1), cmap=plt.cm.Blues)
        plt.title("angle")

        ax3 = plt.subplot(223)
        ax3.matshow(np.abs(heatmap2), cmap=plt.cm.Blues)
        ax3.scatter(Ind_obj_Rag2[:, 1], Ind_obj_Rag2[:, 0])
        plt.title("abs")
        ax4 = plt.subplot(224)
        ax4.matshow(np.angle(heatmap2), cmap=plt.cm.Blues)
        plt.title("angle")
        plt.show()
        plt.savefig('./debug_vis.png')
        plt.close()



def plotVector(y, x=0, x_bin=1):
    """
    Plot a vector with optional x-axis and bin size.

    Args:
    y (numpy.ndarray or torch.Tensor): Input vector to plot.
    x (float, optional): Starting value of x-axis. Default is 0.
    x_bin (float, optional): Bin size of x-axis. Default is 1.

    Raises:
    TypeError: If the input vector is neither a Numpy array nor a PyTorch tensor.

    Returns:
    None
    """
    # Generate an array for the x-axis using the length of the input vector and the given bin size
    if x is 0:
        x = np.arange(0, len(y), 1) * x_bin
    else:
        pass
    
    # Check if the input vector is a Numpy array
    if type(y) is np.ndarray:
        pass
    
    # If the input vector is a PyTorch tensor, convert it to a Numpy array
    elif torch.is_tensor(y):
        y = y.cpu().numpy()
    
    # If the input vector is neither a Numpy array nor a PyTorch tensor, raise a TypeError
    else:
        raise TypeError("Visualization only supports Numpy/Tensor")
    
    # Plot the absolute values of the input vector against the x-axis
    plt.subplot(211)
    plt.plot(x, abs(y), 'r')
    plt.subplot(212)
    plt.plot(x, np.angle(y), 'r')
    # Display the plot
    plt.show()
    
    # Save the plot as an image named "debug_vis.png"
    plt.savefig("debug_vis.png")
    
    # Close the plot to free up memory
    plt.close()

def plotTwoVector(y1, y2, x_bin=1):
    """
    Plot two vectors with optional x-axis and bin size.

    Args:
    y1, y2 (numpy.ndarray or torch.Tensor): Input vectors to plot.
    x_bin (float, optional): Bin size of x-axis. Default is 1.

    Raises:
    TypeError: If the input vectors are neither Numpy arrays nor PyTorch tensors.

    Returns:
    None
    """
    # Determine the longer vector and generate an array for the x-axis using its length and the given bin size
    max_len = max(len(y1), len(y2))
    x = np.arange(0, max_len, 1) * x_bin
    
    # Check if the input vectors are Numpy arrays
    if type(y1) is np.ndarray and type(y2) is np.ndarray:
        pass
    
    # If the input vectors are PyTorch tensors, convert them to Numpy arrays
    elif torch.is_tensor(y1) and torch.is_tensor(y2):
        y1 = y1.cpu().numpy()
        y2 = y2.cpu().numpy()
    
    # If the input vectors are neither Numpy arrays nor PyTorch tensors, raise a TypeError
    else:
        raise TypeError("Visualization only supports Numpy/Tensor")
    
    # Extend the vectors if necessary to match the maximum length
    y1 = np.concatenate((y1, y1.min() * np.ones(max_len - len(y1))))
    y2 = np.concatenate((y2, y2.min() * np.ones(max_len - len(y2))))

    # Plot the absolute values of the input vectors against the x-axis
    plt.subplot(211)
    plt.plot(x, abs(y1), 'r', label='Vector 1')
    plt.plot(x, abs(y2), 'b', label='Vector 2')
    plt.legend(loc='upper right')
    plt.title('Magnitude of Vectors')

    # Plot the angles of the input vectors against the x-axis
    plt.subplot(212)
    plt.plot(x, np.angle(y1), 'r', label='Vector 1')
    plt.plot(x, np.angle(y2), 'b', label='Vector 2')
    plt.legend(loc='upper right')
    plt.title('Angle of Vectors')

    # Display the plot
    plt.show()

    # Save the plot as an image named "debug_vis.png"
    plt.savefig("debug_vis.png")
    
    # Close the plot to free up memory
    plt.close()


def plotVectorNetwork(y, y2, y3, epoch, total_steps, name, x=0, x_bin=1):
    """
    Plot three vectors (typically for neural network output, ground truth, and input) for visualization during training/validation.

    Args:
    y (numpy.ndarray or torch.Tensor): First input vector to plot.
    y2 (numpy.ndarray or torch.Tensor): Second input vector to plot.
    y3 (numpy.ndarray or torch.Tensor): Third input vector to plot.
    epoch (int): Current epoch number.
    total_steps (int): Total number of steps.
    name (str): Name of the model being trained.
    x (float, optional): Starting value of x-axis. Default is 0.
    x_bin (float, optional): Bin size of x-axis. Default is 1.

    Returns:
    None
    """
    # Generate an array for the x-axis using the length of the input vector and the given bin size
    x = np.arange(0, len(y), 1) * x_bin
    
    # Plot the first vector in the top subplot
    plt.subplot(311)
    plt.plot(x, abs(y), 'r')
    plt.title("output")
    
    # Plot the second vector in the middle subplot
    plt.subplot(312)
    plt.plot(x, abs(y2), 'r')
    plt.title("gt")
    
    # Plot the third vector in the bottom subplot
    plt.subplot(313)
    plt.plot(x, abs(y3), 'r')
    plt.title("input")
    
    # Save the plot as an image with the given filename and directory path
    plt.savefig(f'{epoch}_{total_steps}.jpg')
    
    # Close the plot to free up memory
    plt.close()

def plotVectorHist(y, x=0, filename="hist.jpg"):
    """
    Plot a histogram of the input vector.

    Args:
    y (torch.Tensor): Input vector to plot.
    x (float or int, optional): Starting value of x-axis. Default is 0.
    filename (str, optional): Name of the output image file. Default is "hist.jpg".

    Returns:
    None
    """
    # If x is not specified, generate an array for the x-axis using the length of the input vector
    if x == 0:
        x = np.arange(0, len(y), 1)
    
    # Plot the histogram of the input vector using 128 bins, blue facecolor, black edgecolor, and alpha of 0.7
    try:
        plt.hist(y.cpu(), bins=128, facecolor="blue", range=(0, 14.9887), edgecolor="black", alpha=0.7)
    except:
        plt.hist(y, bins=128, facecolor="blue", edgecolor="black", alpha=0.7)
    
    # Save the plot as an image with the given filename and directory path
    plt.savefig(filename)
    
    # Close the plot to free up memory
    plt.close()


def plotThreeView(img0, img1, img2, img3, x=0, x_bin=1, filename="filename", save_dir="./results/"):
    """
    Plot three views of four input images (typically for neural network output and ground truth) and save the plot as an image.

    Args:
    img0 (numpy.ndarray): First input image to plot.
    img1 (numpy.ndarray): Second input image to plot.
    img2 (numpy.ndarray): Third input image to plot.
    img3 (numpy.ndarray): Fourth input image to plot.
    x (float or int, optional): Starting value of x-axis. Default is 0.
    x_bin (float or int, optional): Bin size of x-axis. Default is 1.
    filename (str, optional): Name of the output image file. Default is "filename".
    save_dir (str, optional): Directory path to save the output image file. Default is "./results/".

    Returns:
    None
    """
    # Extract the maximum intensity projection of the top, front, and left views of the first input image
    img0_top = img0[0, 0, :, :, :].squeeze().max(axis=2)
    img0_front = img0[0, 0, :, :, :].squeeze().max(axis=1)
    img0_left = img0[0, 0, :, :, :].squeeze().max(axis=0)
    
    # Plot the three views of the first input image in the top row of subplots
    ax1 = plt.subplot(431)
    ax1.matshow(img0_top, cmap=plt.cm.Blues)
    plt.title("outputAOA_top")
    ax2 = plt.subplot(432)
    ax2.matshow(img0_front, cmap=plt.cm.Blues)
    plt.title("outputAOA_front")
    ax3 = plt.subplot(433)
    ax3.matshow(img0_left, cmap=plt.cm.Blues)
    plt.title("outputAOA_left")
    
    # Extract the maximum intensity projection of the top, front, and left views of the second input image
    img1_top = img1[0, :, :, :].squeeze().max(axis=2)
    img1_front = img1[0, :, :, :].squeeze().max(axis=1)
    img1_left = img1[0, :, :, :].squeeze().max(axis=0)
    
    # Plot the three views of the second input image in the middle row of subplots
    ax1 = plt.subplot(434)
    ax1.matshow(img1_top, cmap=plt.cm.Blues)
    plt.title("output_top")
    ax2 = plt.subplot(435)
    ax2.matshow(img1_front, cmap=plt.cm.Blues)
    plt.title("output_front")
    ax3 = plt.subplot(436)
    ax3.matshow(img1_left, cmap=plt.cm.Blues)
    plt.title("output_left")
    
    # Extract the maximum intensity projection of the top, front, and left views of the third input image
    img2_top = img2[0, :, :, :].squeeze().max(axis=2)
    img2_front = img2[0, :, :, :].squeeze().max(axis=1)
    img2_left = img2[0, :, :, :].squeeze().max(axis=0)
    
    # Plot the three views of the third input image in the bottom row of subplots
    ax1 = plt.subplot(437)

    ax1.matshow(img2_top, cmap=plt.cm.Blues)
    plt.title("outputPCD_top")
    ax2 = plt.subplot(438)
    ax2.matshow(img2_front, cmap=plt.cm.Blues)
    plt.title("outputPCD_front")
    ax3 = plt.subplot(439)
    ax3.matshow(img2_left, cmap=plt.cm.Blues)
    plt.title("outputPCD_left")
    
    # Extract the maximum intensity projection of the top, front, and left views of the fourth input image
    img3_top = img3[:, :, :].squeeze().max(axis=2)
    img3_front = img3[:, :, :].squeeze().max(axis=1)
    img3_left = img3[:, :, :].squeeze().max(axis=0)
    
    # Plot the three views of the fourth input image in the bottom row of subplots
    ax1 = plt.subplot(4, 3, 10)
    ax1.matshow(img3_top, cmap=plt.cm.Blues)
    plt.title("GT_top")
    ax2 = plt.subplot(4, 3, 11)
    ax2.matshow(img3_front, cmap=plt.cm.Blues)
    plt.title("GT_front")
    ax3 = plt.subplot(4, 3, 12)
    ax3.matshow(img3_left, cmap=plt.cm.Blues)
    plt.title("GT_left")
    
    # Save the plot as an image with the given filename and directory path
    plt.savefig(filename)
    
    # Close the plot to free up memory
    plt.close()


def plotThreeView_in_getitem(input_B_voxel, input_B_voxel_polar, filename="voxel_debug.png"):
    """Visualize and save the maximum intensity projection of top, front, and left views
    of two input images.

    Args:
        input_B_voxel (numpy.ndarray or torch.tensor): First input image with 3D voxel data.
        input_B_voxel_polar (numpy.ndarray or torch.tensor): Second input image with 3D voxel data.
        filename (str): The output filename for the saved image. Defaults to "voxel_debug.png".
    """
    if isinstance(input_B_voxel, torch.Tensor):
        input_B_voxel = input_B_voxel.numpy()
    if isinstance(input_B_voxel_polar, torch.Tensor):
        input_B_voxel_polar = input_B_voxel_polar.numpy()

    # Extract the maximum intensity projection of the top, front, and left views of the first input image
    input_B_voxel_top = input_B_voxel.squeeze().max(axis=2)
    input_B_voxel_front = input_B_voxel.squeeze().max(axis=1)
    input_B_voxel_left = input_B_voxel.squeeze().max(axis=0)

    # Plot the three views of the first input image in the top row of subplots
    ax1 = plt.subplot(231)
    ax1.matshow(input_B_voxel_top, cmap=plt.cm.Blues)
    ax1.set_xlabel('Y') 
    ax1.set_ylabel('X') 
    plt.title("input_B_voxel_top")
    ax2 = plt.subplot(232)
    ax2.matshow(input_B_voxel_front, cmap=plt.cm.Blues)

    ax2.set_xlabel('Z')
    ax2.set_ylabel('X')
    plt.title("input_B_voxel_front")
    ax3 = plt.subplot(233)
    ax3.matshow(input_B_voxel_left, cmap=plt.cm.Blues)
    ax3.set_xlabel('Z')
    ax3.set_ylabel('Y')
    plt.title("input_B_voxel_left")

    # Extract the maximum intensity projection of the top, front, and left views of the second input image
    input_B_voxel_polar_top = input_B_voxel_polar.squeeze().max(axis=2)
    input_B_voxel_polar_front = input_B_voxel_polar.squeeze().max(axis=1)
    input_B_voxel_polar_left = input_B_voxel_polar.squeeze().max(axis=0)

    # Plot the three views of the second input image in the middle row of subplots
    ax1 = plt.subplot(234)
    ax1.matshow(input_B_voxel_polar_top, cmap=plt.cm.Blues)
    plt.title("input_B_voxel_polar_top")
    ax2 = plt.subplot(235)
    ax2.matshow(input_B_voxel_polar_front, cmap=plt.cm.Blues)
    plt.title("input_B_voxel_polar_front")
    ax3 = plt.subplot(236)
    ax3.matshow(input_B_voxel_polar_left, cmap=plt.cm.Blues)
    plt.title("input_B_voxel_polar_left")

    # Save the plot as an image with the given filename and directory path
    plt.savefig(filename)

    # Close the plot to free up memory
    plt.close()




def scatterPcds(point_cloud_1, point_cloud_2, colors=None, filename='pcd_debug.png'):
    """
    Visualize a 3D point cloud from four different angles and save the plots as PNG files.

    :param point_cloud: A numpy array with shape (n, 3) or (n, 4), where n is the number of points and columns represent x, y, z coordinates and optionally color.
    :param colors: Optional, a numpy array with shape (n, 4) representing RGBA colors.
    :param output_prefix: The prefix for the output PNG file names.
    """
    if point_cloud_1.shape[1] == 4 and colors is None:
        cmap = plt.cm.get_cmap('jet')
        colors = cmap(point_cloud_1[:, 3] / point_cloud_1[:, 3].max())
        point_cloud_1 = point_cloud_1[:, :3]

    assert colors is None or colors.shape[1] == 4, "Colors should be in RGBA format"

    xs = point_cloud_1[:, 0]
    ys = point_cloud_1[:, 1]
    zs = point_cloud_1[:, 2]

    xs_2 = point_cloud_2[:, 0]
    ys_2 = point_cloud_2[:, 1]
    zs_2 = point_cloud_2[:, 2]

    # Define the four viewpoints (azimuth, elevation)
    viewpoints = [(0, 90), (0, 0), (90, 0), (45, 30)]
    fig = plt.figure()
    for i, (azim, elev) in enumerate(viewpoints):
        
        # Add a 3D subplot to the figure with a 2x2 grid layout and assign it to 'ax'
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Set the title for each subplot based on the index 'i'
        # The title represents the viewpoint for each subplot
        if i == 0:
            ax.set_title('Top view') 
        elif i == 1:
            ax.set_title('Side view') 
        elif i == 2:
            ax.set_title('Front view') 
        elif i == 3:
            ax.set_title('(45, 30)')       # Set title for the (45, 30) viewpoint
        ax.view_init(azim=azim, elev=elev)

        if colors is not None:
            ax.scatter(xs, ys, zs, marker='o', c=colors, s=0.02)
        else:
            ax.scatter(xs, ys, zs, marker='o', c = 'b',s=0.02)
        ax.scatter(xs_2, ys_2, zs_2, marker='o', c='r', s=0.02)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.savefig(filename)
    plt.close(fig)




def plot_lidar_bin(lidar_bin_path = "E://Dataset//selfmade_Coloradar//2023-03-26-15-58-48_GTC_backword//Lidar_pcd//frame_822.bin"):
    f = lidar_bin_path
    pcd = np.fromfile(f, dtype = 'float32').reshape((-1,4))
    colors, o3d_merged_point_cloud = mmwavePCD_util.ndarrayPCD_to_o3dPCD(pcd)
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    grid = open3d_extend.create_grid_thick()
    open3d_extend.custom_draw_geometries([FOR1, o3d_merged_point_cloud, grid], point_size=0.8)

def imshow(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be an ndarray.")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image should have 2 or 3 dimensions (M, N) or (M, N, C).")
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError("Image data type should be one of the following: uint8, float32, float64.")
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        raise ValueError("Image channels should be 1 (grayscale), 3 (RGB), or 4 (RGBA).")
    
    if image.dtype in [np.float32, np.float64]:
        if np.min(image) < 0 or np.max(image) > 1:
            raise ValueError("Float images should have pixel values in the range [0, 1].")
    
    plt.imshow(image)
    plt.show()



# plot_lidar_bin(lidar_bin_path = "E://Dataset//selfmade_Coloradar//2023-03-26-15-58-48_GTC_backword//Lidar_pcd//frame_822.bin")

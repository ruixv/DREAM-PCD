"""
This file contains a custom visualization toolset for visualizing 3D point clouds, bounding boxes, and grids using Open3D library.
The main functions included are:

custom_draw_geometries: A function for drawing geometries with a custom point size.
create_bounding_box: A function for creating a bounding box LineSet.
create_grid: A function for creating a simple grid LineSet.
create_line_mesh: A function for creating line meshes for visualizing thick lines.
create_cuboid_mesh: A function for creating a cuboid mesh to represent line segments.
create_grid_thick: A function for creating a thick grid.
Usage: import this file in your Python script and use the provided functions to visualize your 3D data.

Author: USTC IP_LAB Millimeter Wave Point Cloud Group
Main Members: Ruixu Geng (gengruixu@mail.ustc.edu.cn), Yadong Li, Jincheng Wu, Yating Gao
Copyright: 2023, USTC IP_LAB
Creation Date: March 2023
"""

import open3d as o3d
import numpy as np

def custom_draw_geometries(geometries, point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    vis.run()
    vis.destroy_window()


def create_bounding_box(center, width):
    # Define the 8 corners of the bounding box
    corners = np.array([
        [0, 0, 0],
        [width[0], 0, 0],
        [0, width[1], 0],
        [0, 0, width[2]],
        [width[0], width[1], 0],
        [width[0], 0, width[2]],
        [0, width[1], width[2]],
        [width[0], width[1], width[2]]
    ])

    # Shift the corners to the correct position around the center point
    corners = corners - width / 2
    corners = corners + center

    # Define the 12 edges of the bounding box
    edges = [
        [0, 1], [0, 2], [0, 3],
        [1, 4], [1, 5], [2, 4],
        [2, 6], [3, 5], [3, 6],
        [4, 7], [5, 7], [6, 7]
    ]

    # Create the bounding box lineset
    box_lineset = o3d.geometry.LineSet()
    box_lineset.points = o3d.utility.Vector3dVector(corners)
    box_lineset.lines = o3d.utility.Vector2iVector(edges)

    return box_lineset

def create_grid():

    points = [
        [0,0,0],
        [0,-10,0],
        [0,10,0],
        [10,0,0],
        [0,0,-5],
        [0,0,5],]

        # y
    for y in range(-10,10):
        points.append([0.1,y,0])
        points.append([-0.1,y,0])
        

        # x
    for x in range(0,10):
        points.append([x,0.1,0])
        points.append([x,-0.1,0])

        # z
    for z in range(-5,5):
        points.append([0,0.1,z])
        points.append([0,-0.1,z])


    lines = [
        [1,2],
        [0,3],
        [4,5],
    ]
    

    for kedu in range(6,len(points)+1,2):
        lines.append([kedu,kedu+1])


    colors = []
    for i in range(len(lines)):
        if i<3:
            colors.append([0,0,0])
        else:
            colors.append([1,0,0])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set



def create_line_mesh(lines, colors, thickness=0.1):
    all_meshes = []

    for end_points, color in zip(lines, colors):
        pt1 = end_points[0]
        pt2 = end_points[1]
        cuboid_mesh = create_cuboid_mesh(pt1, pt2, color, thickness)
        all_meshes.append(cuboid_mesh)

    combined_mesh = all_meshes[0]
    for mesh in all_meshes[1:]:
        combined_mesh += mesh

    return combined_mesh

def create_cuboid_mesh(pt1, pt2, color, thickness=0.1):
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    direction = direction / length

    rotation_axis = np.cross(direction, np.array([0, 0, 1]))
    if np.linalg.norm(rotation_axis) < 1e-6:
        rotation_axis = np.array([1, 0, 0])
    else:
        rotation_axis /= np.linalg.norm(rotation_axis)

    angle = np.arccos(np.dot(direction, np.array([0, 0, 1])))

    cylinder_radius = thickness / 2
    cylinder_height = length


    cylinder = o3d.geometry.TriangleMesh.create_cylinder(cylinder_radius, cylinder_height)
    cylinder.compute_vertex_normals()


    translation = (pt1 + pt2) / 2
    cylinder.translate(translation, relative=False)


    cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle), center=translation)


    cylinder.paint_uniform_color(color)

    return cylinder


def create_grid_thick(thickness=0.05):

    points = [
        [-10,0,0],
        [0,-10,0],
        [0,10,0],
        [10,0,0],
        [0,0,-5],
        [0,0,5],]


    for y in range(-10,10):
        points.append([0.1,y,0])
        points.append([-0.1,y,0])
        


    for x in range(-10,10):
        points.append([x,0.1,0])
        points.append([x,-0.1,0])


    for z in range(-5,5):
        points.append([0,0.1,z])
        points.append([0,-0.1,z])


    lines = [
        (np.array(points[1]),np.array(points[2])),
        (np.array(points[0]),np.array(points[3])),
        (np.array(points[4]),np.array(points[5])),
    ]
    

    for kedu in range(6,len(points)+1,2):
        try:
            lines.append((np.array(points[kedu]),np.array(points[kedu+1])))
        except:
            pass
        # lines.append([kedu,kedu+1])


    colors = []
    for i in range(len(lines)):
        if i<3:
            colors.append([0,0,0])
        else:
            colors.append([1,0,0])
    
    line_mesh = create_line_mesh(lines, colors, thickness=0.05)

    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(points),
    #     lines=o3d.utility.Vector2iVector(lines),
    # )
    # line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_mesh






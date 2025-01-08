from ..mmwavePCD_util import quaternion_to_euler
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import os

def load_camera_data(thread_name, Scene_name, camera_path_final, positions_final, angles_final, velocitys_final, timestamps_camera_final):
    """
    This function generates a video from a set of images with overlayed text displaying
    timestamp, position, and angle information for each frame.
    Parameters:
    - thread_name (str): Sensor type
    - Scene_name (str): Name of the scene
    - camera_path_final (list): List of file paths to camera images
    - positions_final (list): List of position data for each camera image
    - angles_final (list): List of angle data (in quaternion format) for each camera image
    - velocitys_final (list): List of velocity data for each camera image
    - timestamps_camera_final (list): List of timestamp data for each camera image
    """

    print(f"Process {os.getpid()} loading dataset data, sensor type: {thread_name}, scene name: {Scene_name}")

    # Set the path to the folder with images
    Scene_path = os.path.join("E://Dataset//selfmade_Coloradar//", Scene_name)
    folder_path = os.path.join(Scene_path, "Camera")

    # Set the frame rate for the output video
    frame_rate = 10

    # Set the output video filename
    output_filename = os.path.join(Scene_path, 'camera.mp4')

    # Convert lists to numpy arrays
    timestamps = np.array(timestamps_camera_final)
    positions = np.array(positions_final)
    angles = np.array(angles_final)
    velocitys = np.array(velocitys_final)

    # Create a list of all image filenames in the folder
    image_filenames = camera_path_final

    # Read the first image to get the frame size
    frame = cv2.imread(image_filenames[0])
    height, width, channels = frame.shape

    # Create the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, frame_rate, (width, height))

    # Loop through all images and add them to the video
    for i, image_filename in enumerate(image_filenames):
        # Read the image
        frame = cv2.imread(image_filename)

        # Add the timestamp and position information to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'timestamp: {:.2f}; x: {:.1f}; y: {:.1f}; z: {:.1f}'.format(timestamps[i], positions[i, 0], positions[i, 1], positions[i, 2])
        bottomLeftCornerOfText = (0, height - 10)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(frame, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        # Convert quaternion angles to Euler angles and add them to the image
        roll, pitch, yaw = quaternion_to_euler(angles[i, 0], angles[i, 1], angles[i, 2], angles[i, 3])
        text2 = 'q: {:.1f};{:.1f};{:.1f};{:.1f}||angle: x {:.1f} y {:.1f} z {:.1f}'.format(angles[i, 0], angles[i, 1], angles[i, 2], angles[i, 3], roll, pitch, yaw)
        bottomLeftCornerOfText = (0, height - 40)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(frame, text2, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        # Add the frame to the video
        video_writer.write(frame)

        # Print the progress every 100 frames
        if i % 100 == 0:
            print(i)

    # Release the video writer and close the file
    video_writer.release()

def auto_crop_white(frame, threshold=200):
    """
    Automatically crop the input frame by removing white space.
    :param frame: Input image/frame.
    :param threshold: Threshold value for white pixel (default: 200).
    :return: Cropped frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find non-white rows and columns
    mask = gray < threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find boundaries
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return frame[y1:y2 + 1, x1:x2 + 1]

def add_margin(frame, margin_size, color=(0, 0, 0)):
    """
    Add a margin of specified color and size to the input frame.
    :param frame: Input image/frame.
    :param margin_size: The size of the margin to add.
    :param color: Margin color (default: black).
    :return: Frame with added margin.
    """
    h, w, _ = frame.shape
    return cv2.copyMakeBorder(frame, margin_size, margin_size, margin_size, margin_size, cv2.BORDER_CONSTANT, value=color)


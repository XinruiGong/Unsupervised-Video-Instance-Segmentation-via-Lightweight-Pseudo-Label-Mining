import time
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os


# # For every 5th entry in the DataFrame
# for i in range(0, len(df_image)):
#     # 获取原始图像
#     image_data = df_image['[CameraImageComponent].image'].iloc[i]
#     image_np_array = np.frombuffer(image_data, dtype=np.uint8)
#     image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

#     # print(df_image['key.frame_timestamp_micros'])
#     # # 获取分割图
#     # segmentation_data = df_segmentation['[CameraSegmentationLabelComponent].panoptic_label'].iloc[i]
#     # segmentation_np_array = np.frombuffer(segmentation_data, dtype=np.uint8)
#     # segmentation = cv2.imdecode(segmentation_np_array, cv2.IMREAD_COLOR)
#     # 显示原始图像
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title(f'Image from iloc[{i}]')

#     # # 显示分割图
#     # plt.subplot(1, 2, 2)
#     # plt.imshow(segmentation, cmap='gray')
#     # plt.title(f'Segmentation from iloc[{i}]')
#     speeds = []

#     vx = df_image['[CameraImageComponent].velocity.linear_velocity.x'].iloc[i]
#     vy = df_image['[CameraImageComponent].velocity.linear_velocity.y'].iloc[i]
#     vz = df_image['[CameraImageComponent].velocity.linear_velocity.z'].iloc[i]

#     # Calculate the magnitude of the velocity
#     speed = np.sqrt(vx**2 + vy**2 + vz**2)
#     # print(f"For iloc[{i}], vx Camera Car Speed: {vx:.2f} m/s")
#     # print(f"For iloc[{i}], vy Camera Car Speed: {vy:.2f} m/s")
#     # print(f"For iloc[{i}], vz Camera Car Speed: {vz:.2f} m/s")
#     print(f"For iloc[{i}], Camera Car Speed: {speed:.2f} m/s")
#     plt.show()

def plot_camera_car_speed(parquet_path):
    """
    Reads image data from a Parquet file, calculates the camera car's speed, and plots a speed chart.

    Args:
    parquet_path (str): Path to the Parquet file.
    """
    # Read the Parquet file
    df_image = pd.read_parquet(parquet_path)

    # Calculate speeds
    speeds = []
    for i in range(len(df_image)):
        vx = df_image['[CameraImageComponent].velocity.linear_velocity.x'].iloc[i]
        vy = df_image['[CameraImageComponent].velocity.linear_velocity.y'].iloc[i]
        vz = df_image['[CameraImageComponent].velocity.linear_velocity.z'].iloc[i]

        # Calculate the magnitude of the speed
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        speeds.append(speed)

    # Extract indices (here we are simply using every index)
    indices = list(range(len(df_image)))

    # Plot the speed chart
    plt.figure(figsize=(12, 7))
    plt.plot(indices, speeds, label='Original Speed', color='red', alpha=0.3)
    plt.xlabel('DataFrame Index')
    plt.ylabel('Camera Car Speed (m/s)')
    plt.title('Camera Car Speed vs. DataFrame Index')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_camera_images_as_video(parquet_path):
    """
    Saves images from the front camera in a Parquet file as a video.

    Args:
    parquet_path (str): Path to the Parquet file.
    """
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)
    # Filter for images from the front camera
    df = df[df['key.camera_name'] == 1]

    # Define video parameters
    frame_width = 1280  # Example width
    frame_height = 720  # Example height
    frame_rate = 30     # Example frame rate
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]
    video_dir = 'waymo_subset/videos'
    video_path = os.path.join(video_dir, base_name + '.avi')

    # Create the directory for saving videos
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Process each row of image data
    for _, row in df.iterrows():
        image_data = row['[CameraImageComponent].image']
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    # Release resources
    out.release()

# parquet_path = 'waymo_subset\camera_image/1022527355599519580_4866_960_4886_960_seg0.parquet'
# plot_camera_car_speed(parquet_path)
# save_camera_images_as_video(parquet_path)

file_path='waymo_subset\camera_image'

for filename in os.listdir(file_path):
    if filename.endswith('.parquet'):
        parquet_path = os.path.join(file_path, filename)
        save_camera_images_as_video(parquet_path)
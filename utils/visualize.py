import time 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取原始图像的 Parquet 文件
df_image = pd.read_parquet('camera_image/10017090168044687777_6380_000_6400_000.parquet')

# 读取分割数据的 Parquet 文件
df_segmentation = pd.read_parquet('camera_segmentation/10017090168044687777_6380_000_6400_000.parquet')
print(df_segmentation['[CameraSegmentationLabelComponent].panoptic_label'])
print(df_segmentation.head())
print(df_segmentation.columns)
print(len(df_segmentation))
print(len(df_image))
# For every 5th entry in the DataFrame
for i in range(0, len(df_image), 5):
    # 获取原始图像
    image_data = df_image['[CameraImageComponent].image'].iloc[i]
    image_np_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

    # # 获取分割图
    # segmentation_data = df_segmentation['[CameraSegmentationLabelComponent].panoptic_label'].iloc[i]
    # segmentation_np_array = np.frombuffer(segmentation_data, dtype=np.uint8)
    # segmentation = cv2.imdecode(segmentation_np_array, cv2.IMREAD_COLOR)

    # 显示原始图像
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1) 
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Image from iloc[{i}]')
    
    # # 显示分割图
    # plt.subplot(1, 2, 2)
    # plt.imshow(segmentation, cmap='gray') 
    # plt.title(f'Segmentation from iloc[{i}]')

    plt.show()
   
import pandas as pd
import io
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=100)
# 设置pandas的打印选项
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.max_colwidth', None) # 显示所有数据，不截断
pd.set_option('display.width', None)        # 确保没有换行
np.set_printoptions(threshold=np.inf)

# # 读取 Parquet 文件
# df = pd.read_parquet('waymo_open_dataset_v_2_0_0/training/camera_segmentation/1005081002024129653_5313_150_5333_150.parquet')

# # 过滤特定的时间戳和 key.camera_name
# specific_timestamp = "1005081002024129653_5313_150_5333_150;1510593602540538"
# filtered_df = df[(df.index == specific_timestamp) & (df['key.camera_name'] == 1)]

# # 打印与 segmentation 相关的列
# segmentation_columns = [
#     '[CameraSegmentationLabelComponent].panoptic_label_divisor',
#     # '[CameraSegmentationLabelComponent].panoptic_label',
#     '[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.local_instance_ids',
#     '[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.global_instance_ids',
#     '[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.is_tracked',
#     '[CameraSegmentationLabelComponent].sequence_id',
#     '[CameraSegmentationLabelComponent].num_cameras_covered'
# ]

# for column in segmentation_columns:
#     print(f"Column: {column}")
#     print("-------------")
#     print(filtered_df[column])
#     print("\n\n")

# # 从过滤后的数据框中提取 PNG 数据
# png_data = filtered_df['[CameraSegmentationLabelComponent].panoptic_label'].iloc[0]

# # 使用 io.BytesIO 将 PNG 数据转换为文件类对象
# image_data = io.BytesIO(png_data)

# # 使用 PIL 打开 PNG 数据并将其转换为图像对象
# image = Image.open(image_data)

# # 将图像对象的像素数据转换为 numpy 数组
# pixel_array = np.array(image)

# # Mock pixel_array for demonstration purposes

# # Calculate the decimal part of pixel_array/1000
# decimal_part = pixel_array / 1000 - pixel_array // 1000

# # Find the rows and columns where the decimal part is not 0
# non_integer_rows, non_integer_cols = np.where(decimal_part != 0)

# non_integer_values = pixel_array[non_integer_rows, non_integer_cols]
# df_non_integer_values = pd.DataFrame(non_integer_values)

# # Combine positions and their corresponding values
# print(non_integer_values)
# csv_filename = "non_integer_values.csv"
# df_non_integer_values.to_csv(csv_filename, index=False)


# import pandas as pd

# 文件路径
# file_path = 'waymo_open_dataset_v_2_0_0/training\camera_image/1005081002024129653_5313_150_5333_150.parquet'

# # 读取 Parquet 文件
# df = pd.read_parquet(file_path)
# len1=len(df)
# print(df.columns)
# # 打印数据集的信息
# print(len1)


# 如果你想查看特定列的内容（例如 'image_data' 列）
# print(df['image_data'].head())
# import os
# import pandas as pd

# def calculate_subset_lengths(original_dir, subset_dir):
#     def list_parquet_files(directory):
#         """列出指定目录下的所有Parquet文件"""
#         return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

#     results = []

#     # 遍历原始目录中的所有Parquet文件
#     for original_file in list_parquet_files(original_dir):
#         original_length = len(pd.read_parquet(original_file)) // 5
#         base_name = os.path.splitext(os.path.basename(original_file))[0]

#         # 初始化找到的subset文件列表
#         found_subset_files = False

#         # 遍历subset目录中的所有文件
#         for subset_file in list_parquet_files(subset_dir):
#             if base_name in subset_file:
#                 found_subset_files = True
#                 subset_length = len(pd.read_parquet(subset_file))
#                 percentage = int((subset_length / original_length) * 100 )if original_length else 0
#                 results.append({
#                     'original_file': os.path.basename(original_file),
#                     'subset_file': os.path.basename(subset_file),
#                     'subset_length': subset_length,
#                     'percentage': percentage
#                 })

#         # 如果没有找到对应的subset文件，记录长度为0
#         if not found_subset_files:
#             results.append({
#                 'original_file': os.path.basename(original_file),
#                 'subset_file': 'No corresponding file',
#                 'subset_length': 0,
#                 'percentage': 0
#             })

#     return results

# # 示例使用
# original_dir = 'waymo_open_dataset_v_2_0_0/training/camera_image'
# subset_dir = 'waymo_subset/camera_image'
# results = calculate_subset_lengths(original_dir, subset_dir)

# # 打印结果
# for result in results:
#     print(f"Original File: {result['original_file']}, Subset File: {result['subset_file']}, "
#           f"Subset Length: {result['subset_length']}, Percentage: {result['percentage']}%")
# import pandas as pd
# import os
# from PIL import Image
# import io



# def save_images_from_parquet(parquet_path,png_output_dir, image_column_name='[CameraImageComponent].image'):
#     """
#     Extracts and saves images from a Parquet file as PNG files.

#     Parameters:
#     parquet_path (str): Path to the Parquet file.
#     image_column_name (str): Column name with image data.
#     """
#     df = pd.read_parquet(parquet_path)

#     # Get the base name of the Parquet file (without extension)
#     base_name = os.path.splitext(os.path.basename(parquet_path))[0]

#     # Create a subdirectory within the output directory using the base name of the Parquet file
#     specific_output_dir = os.path.join(png_output_dir, base_name)
#     os.makedirs(specific_output_dir, exist_ok=True)

#     # Iterate through each row in the DataFrame
#     for index, row in df.iterrows():
#         # Extract the image data
#         image_data = row[image_column_name]

#         # Convert and save as PNG
#         image = Image.open(io.BytesIO(image_data))
#         image_path = os.path.join(specific_output_dir, f'{index}.png')
#         image.save(image_path, 'PNG')

# # 设置 Parquet 文件路径和输出目录
# parquet_file_path = 'waymo_open_dataset_v_2_0_0\\training\\camera_image\\1022527355599519580_4866_960_4886_960.parquet'
# output_directory = 'waymo_open_dataset_v_2_0_0\\training\\png\\1022527355599519580_4866_960_4886_960'

# # 调用函数
# save_images_from_parquet(parquet_file_path, output_directory)

# import os
# import json
# import shutil

# def filter_and_save_images_by_continuous_segment(base_path, new_base_path, threshold=1.5, sequence_length=5):
#     optical_flow_path = os.path.join(base_path, 'optical_flow')
#     png_image_path = os.path.join(base_path, 'png_image')

#     new_optical_flow_path = os.path.join(new_base_path, 'optical_flow')
#     new_png_image_path = os.path.join(new_base_path, 'pn231321312g_image')

#     os.makedirs(new_optical_flow_path, exist_ok=True)
#     os.makedirs(new_png_image_path, exist_ok=True)

#     for folder in os.listdir(optical_flow_path):
#         folder_path = os.path.join(optical_flow_path, folder)
#         json_file_path = os.path.join(folder_path, folder + '.json')
#         if os.path.isfile(json_file_path):
#             with open(json_file_path, 'r') as json_file:
#                 data = json.load(json_file)
#                 keys = list(data.keys())

#                 segment_index = 0
#                 i = 0
#                 while i < len(keys) - sequence_length + 1:
#                     if all(float(data[keys[j]]) < threshold for j in range(i, i + sequence_length)):
#                         segment_index += 1
#                         filtered_dict = {}
#                         segment_start = i
#                         while i < len(keys) and float(data[keys[i]]) < threshold:
#                             key = keys[i]
#                             filtered_dict[key] = data[key]
#                             flow_img = key + '_flow.png'
#                             png_img = key + '.png'
#                             source_flow_img = os.path.join(folder_path, flow_img)
#                             source_png_img = os.path.join(png_image_path, folder, png_img)
#                             target_flow_folder = os.path.join(new_optical_flow_path, folder, f'{folder}_seg{segment_index}')
#                             target_png_folder = os.path.join(new_png_image_path, folder, f'{folder}_seg{segment_index}')
#                             os.makedirs(target_flow_folder, exist_ok=True)
#                             os.makedirs(target_png_folder, exist_ok=True)
#                             shutil.copy(source_flow_img, target_flow_folder)
#                             shutil.copy(source_png_img, target_png_folder)
#                             i += 1

#                         if filtered_dict:
#                             filtered_json_path = os.path.join(target_flow_folder, f'{folder}_seg{segment_index}.json')
#                             with open(filtered_json_path, 'w') as f:
#                                 json.dump(filtered_dict, f, indent=4)
#                     else:
#                         i += 1

# base_path = 'waymo_open_dataset_v_2_0_0'
# new_base_path = 'waymo_subset_of'
# filter_and_save_images_by_continuous_segment(base_path, new_base_path)

import os
import json


# def merge_all_json_files(base_dir):
#     merged_data = {}

#     # Walk through all subdirectories and files in base_dir
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             # Check if the file is a JSON file
#             if file.endswith('.json'):
#                 json_file_path = os.path.join(root, file)
                
#                 # Read and merge the JSON file content
#                 with open(json_file_path, 'r') as json_file:
#                     data = json.load(json_file)
#                     merged_data.update(data)

#     # Write the merged data to a new JSON file in the base directory
#     with open(os.path.join(base_dir, 'merged_data.json'), 'w') as merged_file:
#         json.dump(merged_data, merged_file, indent=4)

# # Usage
# base_dir = 'waymo_subset_of_2.0/optical_flow'
# merge_all_json_files(base_dir)


# def load_keys_from_json_files(directory):
#     keys = set()
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.json'):
#                 json_file_path = os.path.join(root, file)
#                 with open(json_file_path, 'r') as json_file:
#                     data = json.load(json_file)
#                     keys.update(data.keys())
#     return keys
# def calculate_accuracy(directory1, directory2):
#     keys1 = load_keys_from_json_files(directory1)
#     keys2 = load_keys_from_json_files(directory2)

#     correct_predictions = len(keys1.intersection(keys2))
#     total_predictions = len(keys1)

#     if total_predictions == 0:
#         return 0  # Avoid division by zero

#     return correct_predictions / total_predictions

# directory1 = 'waymo_subset_of_1.5'
# directory2 = 'waymo_subset_0.05\optical_flow'
# accu = calculate_accuracy(directory1, directory2)
# print("Accuracy:", accu)


# import os
# import io
# import pandas as pd
# from PIL import Image

# def save_images_from_original_parquet(parquet_dir, png_output_dir, image_column_name='[CameraImageComponent].image', camera_count=5):
#     """
#     Extracts and saves images from Parquet files in a directory as PNG files, processing images from the first camera only.

#     Parameters:
#     parquet_dir (str): Path to the directory containing Parquet files.
#     image_column_name (str): Column name with image data.
#     camera_count (int): Total number of cameras in the Parquet files.
#     """
#     # Get a list of all Parquet files in the directory
#     parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

#     # Iterate through each Parquet file in the directory
#     for parquet_file in parquet_files:
#         parquet_path = os.path.join(parquet_dir, parquet_file)

#         # Read the Parquet file into a DataFrame
#         df = pd.read_parquet(parquet_path)

#         # Get the base name of the Parquet file (without extension)
#         base_name = os.path.splitext(parquet_file)[0]

#         # Create a subdirectory within the output directory using the base name of the Parquet file
#         specific_output_dir = os.path.join(png_output_dir, base_name)
#         os.makedirs(specific_output_dir, exist_ok=True)

#         # Initialize a counter to determine when to process the first camera image
#         camera_counter = 0

#         # Iterate through each row in the DataFrame
#         for index, row in df.iterrows():
#             # Only process images from the first camera
#             if camera_counter % camera_count == 0:
#                 # Extract the image data
#                 image_data = row[image_column_name]

#                 # Convert and save as PNG
#                 image = Image.open(io.BytesIO(image_data))
#                 image_path = os.path.join(specific_output_dir, f'{index}.png')
#                 image.save(image_path, 'PNG')

#             # Increment the camera counter
#             camera_counter += 1

#         print(f"Processed {camera_counter // camera_count} images from the first camera in {parquet_file}.")

# # 定义Parquet文件所在的目录和PNG输出文件夹路径
# parquet_directory = 'waymo_open_dataset_v_2_0_0/training/camera_image'
# png_output_directory = 'waymo_open_dataset_v_2_0_0/png_image'

# # 调用函数来处理Parquet文件并保存PNG图像
# save_images_from_original_parquet(parquet_directory, png_output_directory

# 用JSON文件的路径替换这里的 'path_to_your_json_file.json'
# import json

# def count_key_value_pairs_in_json(file_path):
#     """
#     Counts the number of key-value pairs in a dictionary in a JSON file.

#     Args:
#     file_path (str): The path to the JSON file.

#     Returns:
#     int: The number of key-value pairs in the dictionary, or None if the file content is not a dictionary.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             data = json.load(file)

#         if isinstance(data, dict):
#             return len(data)
#         else:
#             print("The file content is not a dictionary.")
#             return None
#     except Exception as e:
#         print(f"Error reading or parsing the JSON file: {e}")
#         return None

# # Example usage
# file_path = 'path_to_your_json_file.json'  # Replace with your JSON file path
# num_key_value_pairs = count_key_value_pairs_in_json(file_path)
# if num_key_value_pairs is not None:
#     print(f"There are {num_key_value_pairs} key-value pairs in the dictionary.")

# file_path = 'waymo_subset_0.05\optical_flow\merged_data.json'
import os
import json
import shutil

# import os
# import json
# import shutil

# def filter_and_save_images_by_corner_flow(
#     base_path, new_base_path, threshold=0.1, sequence_length=5
# ):
#     optical_flow_path = os.path.join(base_path, "optical_flow")
#     png_image_path = os.path.join(base_path, "png_image")

#     new_optical_flow_path = os.path.join(new_base_path, "optical_flow")
#     new_png_image_path = os.path.join(new_base_path, "png_image")

#     os.makedirs(new_optical_flow_path, exist_ok=True)
#     os.makedirs(new_png_image_path, exist_ok=True)

#     # Iterate through each folder in the optical flow directory.
#     for folder in os.listdir(optical_flow_path):
#         folder_path = os.path.join(optical_flow_path, folder)
#         json_file_path = os.path.join(folder_path, f"{folder}_corner_flows.json")

#         # Check if the corresponding JSON file exists.
#         if os.path.isfile(json_file_path):
#             with open(json_file_path, "r") as json_file:
#                 data = json.load(json_file)
#                 keys = list(data.keys())

#                 segment_index = -1
#                 i = 0

#                 # Iterate through the keys in the JSON file.
#                 while i < len(keys) - sequence_length + 1:
#                     # Check for a sequence where 'top_left' and 'top_right' are below the threshold.
#                     if all(
#                         data[keys[j]]["top_left"] < threshold and data[keys[j]]["top_right"] < threshold
#                         for j in range(i, i + sequence_length)
#                     ):
#                         segment_index += 1
#                         segment_start = i

#                         # Create directories for the new segment.
#                         target_flow_folder = os.path.join(
#                             new_optical_flow_path, folder, f"{folder}_seg{segment_index}"
#                         )
#                         target_png_folder = os.path.join(
#                             new_png_image_path, folder, f"{folder}_seg{segment_index}"
#                         )
#                         os.makedirs(target_flow_folder, exist_ok=True)
#                         os.makedirs(target_png_folder, exist_ok=True)

#                         # Initialize the dictionary to store filtered data.
#                         filtered_dict = {}

#                         # Copy images to the new directories while the condition holds.
#                         while i < len(keys) and data[keys[i]]["top_left"] < threshold and data[keys[i]]["top_right"] < threshold:
#                             key = keys[i]
#                             filtered_dict[key] = data[key]

#                             flow_img = key + "_flow.png"
#                             png_img = key + ".png"
#                             source_flow_img = os.path.join(folder_path, flow_img)
#                             source_png_img = os.path.join(png_image_path, folder, png_img)

#                             shutil.copy(source_flow_img, target_flow_folder)
#                             shutil.copy(source_png_img, target_png_folder)
#                             i += 1

#                         # Save the filtered data to a JSON file in the filtered folder.
#                         if filtered_dict:
#                             filtered_json_path = os.path.join(
#                                 target_flow_folder, f"{folder}_seg{segment_index}_filtered.json"
#                             )
#                             with open(filtered_json_path, "w") as f:
#                                 json.dump(filtered_dict, f, indent=4)

#                     else:
#                         i += 1




# base_path = "waymo_open_dataset_v_2_0_0"
# # base_path = "waymo_subset_100"
# new_base_path = "waymo_subset_of_0.1"
# filter_and_save_images_by_corner_flow(base_path, new_base_path)

# def merge_json_files(base_output_folder):
#     merged_data = {}

#     # 遍历文件夹
#     for root, dirs, files in os.walk(base_output_folder):
#         for file in files:
#             # 检查是否为JSON文件
#             if file.endswith('.json'):
#                 file_path = os.path.join(root, file)

#                 # 打开并读取JSON文件
#                 with open(file_path, 'r') as json_file:
#                     data = json.load(json_file)
#                     # 合并数据
#                     merged_data.update(data)
    
#     # 保存合并后的 JSON 文件
#     output_file_path = os.path.join(base_output_folder, 'merged_data.json')
#     with open(output_file_path, 'w') as output_file:
#         json.dump(merged_data, output_file, indent=4)

#     return output_file_path

# merge_json_files('waymo_subset_0.05\optical_flow')

# import os
# import json

# def load_keys_from_json_files(directory):
#     keys = set()
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".json"):
#                 json_file_path = os.path.join(root, file)
#                 with open(json_file_path, "r") as json_file:
#                     data = json.load(json_file)
#                     keys.update(data.keys())
#     return keys

# def calculate_metrics(all_data_dir, true_positive_dir, prediction_dir):
#     # 从目录加载键
#     all_data_keys = load_keys_from_json_files(all_data_dir)
#     true_positive_keys = load_keys_from_json_files(true_positive_dir)
#     prediction_keys = load_keys_from_json_files(prediction_dir)

#     # 打印键集合的大小以进行调试
#     print(f"Total keys in all data: {len(all_data_keys)}")
#     print(f"Total keys in true positives: {len(true_positive_keys)}")
#     print(f"Total keys in predictions: {len(prediction_keys)}")

#     # 计算TP, FP, TN, FN
#     TP = len(true_positive_keys.intersection(prediction_keys))
#     FP = len(prediction_keys - true_positive_keys)
#     FN = len(true_positive_keys - prediction_keys)
#     TN = len(all_data_keys - true_positive_keys - prediction_keys)

#     # 打印中间结果以进行调试
#     print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

#     # 计算accuracy, precision, 和recall
#     accuracy = (TP + TN) / len(all_data_keys) if len(all_data_keys) > 0 else 0
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0

#     # 打印最终计算结果
#     print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

#     return accuracy, precision, recall
# # 使用示例
# accuracy, precision, recall = calculate_metrics('waymo_open_dataset_v_2_0_0\optical_flow', 'waymo_subset_0.05\optical_flow', 'waymo_subset_of_0.1\optical_flow')
# print(accuracy)
# print(precision)
# print(recall)

# import pandas as pd
# import os
# dir='waymo_subset_of_0.1\camera_image\\1022527355599519580_4866_960_4886_960\\1022527355599519580_4866_960_4886_960_seg0.parquet'

# df=pd.read_parquet(dir)
# print(df['key.camera_name'])

# import os
# import pandas as pd
# import shutil


# # import os
# # import pandas as pd

# # 基本路径
# base_png_path = 'waymo_subset_of_0.1/png_image'
# base_parquet_path = 'waymo_open_dataset_v_2_0_0/training/camera_image'
# output_path = 'waymo_subset_of_0.1/camera_image'

# # 确保输出目录存在
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# # 遍历 PNG 图像文件夹
# for scene_dir in os.listdir(base_png_path):
#     scene_path = os.path.join(base_png_path, scene_dir)
#     if os.path.isdir(scene_path):
#         # 处理每个场景下的 segment
#         for segment_dir in os.listdir(scene_path):
#             segment_path = os.path.join(scene_path, segment_dir)
#             if os.path.isdir(segment_path):
#                 timestamps = []

#                 # 收集 segment 中所有 PNG 图像的时间戳
#                 for png_file in os.listdir(segment_path):
#                     if png_file.endswith('.png'):
#                         timestamp = png_file.split(';')[-1].split('.')[0]
#                         timestamps.append(int(timestamp))

#                 # 构建对应的 Parquet 文件路径
#                 parquet_file = f"{scene_dir}.parquet"
#                 parquet_path = os.path.join(base_parquet_path, parquet_file)

#                 if os.path.exists(parquet_path):
#                     # 读取 Parquet 文件并找到匹配的时间戳
#                     df = pd.read_parquet(parquet_path)
#                     matching_rows = df[(df['key.frame_timestamp_micros'].isin(timestamps))&(df['key.camera_name']==1)]

#                     if not matching_rows.empty:
#                         # 保存匹配的 Parquet 文件段
#                         output_segment_path = os.path.join(output_path, scene_dir, f"{segment_dir}.parquet")
#                         os.makedirs(os.path.dirname(output_segment_path), exist_ok=True)
#                         matching_rows.to_parquet(output_segment_path)
#                 else:
#                     print(f"Parquet file not found for: {parquet_path}")
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_camera_car_speed(parquet_path, output_folder):
#     """
#     Reads image data from a Parquet file, calculates the camera car's speed, and plots and saves a speed chart.

#     Args:
#     parquet_path (str): Path to the Parquet file.
#     output_folder (str): Folder to save the speed chart.
#     """
#     # Read the Parquet file
#     df_image = pd.read_parquet(parquet_path)

#     # Calculate speeds
#     speeds = np.sqrt(df_image['[CameraImageComponent].velocity.linear_velocity.x']**2 + 
#                      df_image['[CameraImageComponent].velocity.linear_velocity.y']**2 + 
#                      df_image['[CameraImageComponent].velocity.linear_velocity.z']**2)

#     # Plot the speed chart
#     plt.figure(figsize=(12, 7))
#     plt.plot(speeds, label='Camera Car Speed', color='blue')
#     plt.ylabel('Speed (m/s)')
#     plt.title(f'Camera Car Speed Chart for {os.path.basename(parquet_path)}')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()

#     # Save the plot
#     output_path = os.path.join(output_folder, os.path.basename(parquet_path).replace('.parquet', '.png'))
#     plt.savefig(output_path)
#     plt.close()

# # Directory containing parquet files
# parquet_dir = 'waymo_subset_of_0.1/camera_image'

# # Iterate over each parquet file in the directory
# for root, dirs, files in os.walk(parquet_dir):
#     for file in files:
#         if file.endswith('.parquet'):
#             parquet_path = os.path.join(root, file)
#             output_folder = root
# #             plot_camera_car_speed(parquet_path, output_folder)
# import os
# import shutil
# import json

# def copy_optical_flows_and_update_json(src_optical_flow, src_png_images, dest_optical_flow):
#     if not os.path.exists(dest_optical_flow):
#         os.makedirs(dest_optical_flow)

#     for seg_dir in os.listdir(src_png_images):
#         seg_path = os.path.join(src_png_images, seg_dir)
#         if os.path.isdir(seg_path):
#             dest_seg_path = os.path.join(dest_optical_flow, seg_dir)
#             if not os.path.exists(dest_seg_path):
#                 os.makedirs(dest_seg_path)

#             dest_json_data = {}

#             for file in os.listdir(seg_path):
#                 if file.endswith('.png'):
#                     timestamp = file.split(';')[1].split('.')[0]
#                     scene = file.split(';')[0]
#                     src_flow_file = os.path.join(src_optical_flow, scene, f"{scene};{timestamp}_flow.png")
#                     dest_flow_file = os.path.join(dest_seg_path, f"{scene};{timestamp}_flow.png")

#                     if os.path.exists(src_flow_file):
#                         shutil.copy(src_flow_file, dest_flow_file)

#                         src_json_file = os.path.join(src_optical_flow, scene, f"{scene}_corner_flows.json")
#                         dest_json_file = os.path.join(dest_seg_path, f"{seg_dir}_corner_flows.json")

#                         if os.path.exists(src_json_file):
#                             with open(src_json_file, 'r') as f:
#                                 src_json_data = json.load(f)
                            
#                             # Update the destination JSON data with the source JSON data
#                             dest_json_data.update(src_json_data)

#             # Save the updated JSON data to the destination file
#             with open(dest_json_file, 'w') as f:
#                 json.dump(dest_json_data, f, indent=4)

# copy_optical_flows_and_update_json('waymo_open_dataset_v_2_0_0/optical_flow', 'waymo_subset_0.05/png_image', 'waymo_subset_0.05/optical_flow')
# import json

# # 打开并读取 JSON 文件
# with open('waymo_subset_vel_0.05/optical_flow/merged_data.json', 'r') as file:
#     data = json.load(file)

# # 如果 JSON 数据是一个字典，计算键值对的数量
# if isinstance(data, dict):
#     number_of_pairs = len(data)
#     print(f"字典中的键值对数量为: {number_of_pairs}")
# else:
#     print("JSON 数据不是一个字典。")


# import torch
# print(torch.cuda.is_available())

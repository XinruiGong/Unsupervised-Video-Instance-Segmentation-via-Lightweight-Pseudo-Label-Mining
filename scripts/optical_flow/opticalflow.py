import cv2
import numpy as np
from scripts.optical_flow.opti_utils import flow_to_color
import os
import glob
import json
# https://blog.csdn.net/qq_41685265/article/details/111391755  optical flow methods


# def estimate_optical_flow_gpu(image_file1, image_file2):
#     # Read images
#     img1 = cv2.imread(image_file1, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imread(image_file2, cv2.IMREAD_GRAYSCALE)

#     # Upload images to GPU
#     gpu_img1 = cv2.cuda_GpuMat()
#     gpu_img2 = cv2.cuda_GpuMat()
#     gpu_img1.upload(img1)
#     gpu_img2.upload(img2)

#     # Initialize DualTVL1 optical flow algorithm on GPU
#     tvl1 = cv2.cuda_OpticalFlowDual_TVL1.create()

#     # Compute optical flow
#     gpu_flow = tvl1.calc(gpu_img1, gpu_img2, None)

#     # Download flow from GPU to CPU
#     flow = gpu_flow.download()

#     return img1, img2, flow

# DualTVL1 method
def estimate_optical_flow(image_file1, image_file2):
    # Read images as color
    img1 = cv2.imread(image_file1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_file2, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded correctly
    if img1 is None or img2 is None:
        raise ValueError("Could not open or find the images!")

    # Initialize DualTVL1 optical flow algorithm
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    
    # Compute optical flow using DualTVL1 algorithm
    flow = tvl1.calc(img1, img2, None)

    return img1, img2, flow

def calculate_corner_flows(flow, img_shape, corner_size=0.15):
    """
    Calculate the average optical flow values for the four corners of the optical flow image.
    Args:
    flow: Optical flow values.
    img_shape: Shape of the image.
    corner_size: Fraction of width/height to be considered as corner (default is 15%).

    return: A dictionary containing average flows for each corner.
    """
    # Calculate corner sizes
    corner_height = int(img_shape[0] * corner_size)
    corner_width = int(img_shape[1] * corner_size)

    # Defining corners
    corners = {
        "top_left": flow[:corner_height, :corner_width],
        "top_right": flow[:corner_height, -corner_width:],
        "bottom_left": flow[-corner_height:, :corner_width],
        "bottom_right": flow[-corner_height:, -corner_width:]
    }

    # Calculating average flow for each corner
    corner_averages = {corner: np.mean(np.abs(corners[corner])) for corner in corners}

    return corner_averages

# def process_and_save_corner_flows(base_input_folder, base_output_folder, corner_size):
#     for folder_name in os.listdir(base_input_folder):
#         input_folder = os.path.join(base_input_folder, folder_name)
#         output_folder = os.path.join(base_output_folder, folder_name)

#         if not os.path.isdir(input_folder):
#             continue

#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)

#         print(f"Processing folder: {folder_name}")

#         image_files = sorted(glob.glob(os.path.join(input_folder, '*.png')))
#         json_filename = os.path.join(output_folder, folder_name + '_corner_flows.json')
#         corner_flows_dict = {}

#         # Load existing data if the JSON file already exists
#         if os.path.exists(json_filename):
#             with open(json_filename, 'r') as json_file:
#                 corner_flows_dict = json.load(json_file)

#         for i in range(len(image_files) - 1):
#             image_file1 = image_files[i]
#             image_file2 = image_files[i + 1]

#             flow_filename = os.path.basename(image_file1).replace('.png', '_flow.png')
#             flow_filepath = os.path.join(output_folder, flow_filename)

#             if os.path.exists(flow_filepath):
#                 print(f"Skipping already processed file: {flow_filename}")
#                 continue

#             img1, img2, flow = estimate_optical_flow(image_file1, image_file2)
#             flow_color = flow_to_color(flow, clip_flow=None)
#             cv2.imwrite(flow_filepath, flow_color)

#             average_corner_flows = calculate_corner_flows(flow, img1.shape, corner_size)
#             converted_flows = {k: float(v) for k, v in average_corner_flows.items()}

#             key = os.path.splitext(os.path.basename(image_file1))[0]
#             corner_flows_dict[key] = converted_flows

#             # Update the JSON file after processing each image pair
#             with open(json_filename, 'w') as json_file:
#                 json.dump(corner_flows_dict, json_file, indent=4)

#             print(f"Processed {flow_filename}, Average Corner Flows: {converted_flows}")

#         print(f"Saved corner flow data for folder: {folder_name}")

def process_and_save_corner_flows(base_input_folder, base_output_folder,corner_size):
    """
    Process image files to calculate corner flows and save them as JSON.

    This function processes pairs of consecutive images in each subfolder of the base input folder,
    estimates the optical flow, calculates the average corner flows, and saves the result
    in a JSON file in the corresponding subfolder of the base output folder.

    Args:
    base_input_folder (str): The base directory containing subfolders with image files.
    base_output_folder (str): The base directory where the output JSON files will be saved.
    """
    for folder_name in os.listdir(base_input_folder):
        input_folder = os.path.join(base_input_folder, folder_name)
        output_folder = os.path.join(base_output_folder, folder_name)

        # Skip if the input folder is not a directory
        if not os.path.isdir(input_folder):
            continue

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Processing folder: {folder_name}")

        # Get a sorted list of image files in the input folder
        image_files = sorted(glob.glob(os.path.join(input_folder, '*.png')))
        corner_flows_dict = {}

        for i in range(len(image_files) - 1):
            # Process each pair of consecutive images
            image_file1 = image_files[i]
            image_file2 = image_files[i + 1]

            # Estimate optical flow between the pair of images
            img1, img2, flow = estimate_optical_flow(image_file1, image_file2)
            flow_color = flow_to_color(flow, clip_flow=None)

            # Generate a filename for the flow image
            flow_filename = os.path.basename(image_file1).replace('.png', '_flow.png')
            flow_filepath = os.path.join(output_folder, flow_filename)
            cv2.imwrite(flow_filepath, flow_color)

            # Calculate average corner flows for the image pair
            average_corner_flows = calculate_corner_flows(flow, img1.shape,corner_size)
            # Convert numpy.float32 values to standard floats for JSON serialization
            converted_flows = {k: float(v) for k, v in average_corner_flows.items()}

            # Construct a key from the filename
            key = os.path.splitext(os.path.basename(image_file1))[0]
            corner_flows_dict[key] = converted_flows

            print(f"Processed {flow_filename}, Average Corner Flows: {converted_flows}")

        # Save the corner flow data as a JSON file
        json_filename = os.path.join(output_folder, folder_name + '_corner_flows.json')
        with open(json_filename, 'w') as json_file:
            json.dump(corner_flows_dict, json_file, indent=4)

        print(f"Saved corner flow data for folder: {folder_name}")

def merge_json_files(base_output_folder):
    """
    Merges all JSON files found in a specified directory and its subdirectories into a single JSON file.

    Args:
    base_output_folder (str): The path to the base directory where the JSON files to be merged are located.

    """    
    merged_data = {}

    # Traverse through the base output folder and its subdirectories
    for root, dirs, files in os.walk(base_output_folder):
        for file in files:
            # Check if the file is a JSON file
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                # Open and read the JSON file
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    # Merge the data
                    merged_data.update(data)
    
    # Save the merged data to a new JSON file
    output_file_path = os.path.join(base_output_folder, 'merged_data.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)


def main():
    base_input_folder = "waymo_subset_0.05\\png_image"
    base_output_folder = "waymo_subset_0.05\\optical_flow"
    process_and_save_corner_flows(base_input_folder, base_output_folder,corner_size=0.15)
    merge_json_files(base_output_folder)

if __name__=='__main__':
    main() 
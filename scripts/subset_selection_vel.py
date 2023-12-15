import argparse
import pandas as pd
import numpy as np
import os
from PIL import Image
import io
import shutil
import json
import subset_selection_of as sof

def save_images_from_original_parquet(parquet_dir, png_output_dir, image_column_name='[CameraImageComponent].image', camera_count=5):
    """
    Extracts and saves images from Parquet files in a directory as PNG files, processing images from the first camera only.

    Parameters:
    parquet_dir (str): Path to the directory containing Parquet files.
    image_column_name (str): Column name with image data.
    camera_count (int): Total number of cameras in the Parquet files.
    """
    # Get a list of all Parquet files in the directory
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

    # Iterate through each Parquet file in the directory
    for parquet_file in parquet_files:
        parquet_path = os.path.join(parquet_dir, parquet_file)

        # Read the Parquet file into a DataFrame
        df = pd.read_parquet(parquet_path)

        # Get the base name of the Parquet file (without extension)
        base_name = os.path.splitext(parquet_file)[0]

        # Create a subdirectory within the output directory using the base name of the Parquet file
        specific_output_dir = os.path.join(png_output_dir, base_name)
        os.makedirs(specific_output_dir, exist_ok=True)

        # Initialize a counter to determine when to process the first camera image
        camera_counter = 0

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Only process images from the first camera
            if camera_counter % camera_count == 0:
                # Extract the image data
                image_data = row[image_column_name]

                # Convert and save as PNG
                image = Image.open(io.BytesIO(image_data))
                image_path = os.path.join(specific_output_dir, f'{index}.png')
                image.save(image_path, 'PNG')

            # Increment the camera counter
            camera_counter += 1


def process_parquet_file(file_path, output_dir, speed_threshold=0.5):
    """
    Processes a Parquet file and extracts segments where speed is below a threshold. 
    The segments are saved as individual Parquet files in the output directory.

    Parameters:
    file_path (str): Path to the Parquet file with camera image data.
    output_dir (str): Directory to save extracted segments.
    speed_threshold (float): Speed threshold for segment extraction. Default is 0.5.
    """
    df_image = pd.read_parquet(file_path)

    # Calculating speeds and identifying segments below the threshold
    speeds, indices = [], []
    for i in range(0, len(df_image), 5):
        vx, vy, vz = [df_image[column].iloc[i] for column in ['[CameraImageComponent].velocity.linear_velocity.x', 
                                                              '[CameraImageComponent].velocity.linear_velocity.y', 
                                                              '[CameraImageComponent].velocity.linear_velocity.z']]
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        speeds.append(speed)
        indices.append(i)

    # Segmenting data
    segments, current_segment = [], []
    for i, speed in enumerate(speeds):
        if speed < speed_threshold:
            current_segment.append(indices[i])
        elif len(current_segment) >= 3:
            segments.append(current_segment)
            current_segment = []
    if len(current_segment) >= 3:
        segments.append(current_segment)

    # Saving segments
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    for i, segment in enumerate(segments):
        df_segment = df_image.iloc[segment]
        df_segment.to_parquet(os.path.join(output_dir, f"{base_filename}_seg{i}.parquet"))



def process_segmentation_file(image_subset_path, seg_input_dir, seg_output_dir):
    """
    Processes camera_image subset file and extracts corresponding segments from camera_segmentation dataset.
    Segments are saved with the same filenames as the input subset in the output directory.

    Parameters:
    image_subset_path (str): Path to the camera_image subset Parquet file.
    seg_input_dir (str): Directory containing the camera_segmentation dataset.
    seg_output_dir (str): Directory where the extracted segmentation segments will be saved.
    """
    df_image_subset = pd.read_parquet(image_subset_path)
    image_timestamps = df_image_subset['key.frame_timestamp_micros'].tolist()

    base_filename = os.path.basename(image_subset_path).split('_seg')[0] + '.parquet'
    seg_file_path = os.path.join(seg_input_dir, base_filename)
    df_segmentation = pd.read_parquet(seg_file_path)
    df_segmentation = df_segmentation[df_segmentation['key.camera_name'] == 1]

    df_segmentation_subset = df_segmentation[df_segmentation['key.frame_timestamp_micros'].isin(image_timestamps)]

    # Ensure the output directory exists before saving the file
    output_file_path = os.path.join(seg_output_dir, os.path.basename(image_subset_path))
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    df_segmentation_subset.to_parquet(output_file_path)


def save_images_from_parquet(parquet_path,png_output_dir, image_column_name='[CameraImageComponent].image'):
    """
    Extracts and saves images from a Parquet file as PNG files.

    Parameters:
    parquet_path (str): Path to the Parquet file.
    image_column_name (str): Column name with image data.
    """
    df = pd.read_parquet(parquet_path)

    # Get the base name of the Parquet file (without extension)
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]

    # Create a subdirectory within the output directory using the base name of the Parquet file
    specific_output_dir = os.path.join(png_output_dir, base_name)
    os.makedirs(specific_output_dir, exist_ok=True)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the image data
        image_data = row[image_column_name]

        # Convert and save as PNG
        image = Image.open(io.BytesIO(image_data))
        image_path = os.path.join(specific_output_dir, f'{index}.png')
        image.save(image_path, 'PNG')

def calculate_subset_lengths(original_dir, subset_dir):
    """
    Calculates and returns the lengths of Parquet files in a subset directory relative to an original directory.

    Parameters:
    original_dir (str): Directory with original Parquet files.
    subset_dir (str): Directory with subset Parquet files.

    Returns:
    List of dicts with details about original and subset files and their relative lengths.
    """
    def list_parquet_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

    results = []
    for original_file in list_parquet_files(original_dir):
        original_length = len(pd.read_parquet(original_file)) // 5
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        found_subset_files = False

        for subset_file in list_parquet_files(subset_dir):
            if base_name in subset_file:
                found_subset_files = True
                subset_length = len(pd.read_parquet(subset_file))
                percentage = int((subset_length / original_length) * 100) if original_length else 0
                results.append({
                    'original_file': os.path.basename(original_file),
                    'subset_file': os.path.basename(subset_file),
                    'subset_length': subset_length,
                    'percentage': percentage
                })

        if not found_subset_files:
            results.append({
                'original_file': os.path.basename(original_file),
                'subset_file': 'No corresponding file',
                'subset_length': 0,
                'percentage': 0
            })

    return results

# Main processing loop for Parquet files
def main_processing(image_input_dir, seg_input_dir, image_output_dir, seg_output_dir, png_output_dir, speed_threshold, base_output_dir):
    all_files = [os.path.join(image_input_dir, f) for f in os.listdir(image_input_dir) if f.endswith('.parquet')]

    # Process each Parquet file to extract segments and save them
    for file_path in all_files:
        process_parquet_file(file_path, image_output_dir, speed_threshold)

    # Process camera_image subsets for corresponding segmentation data
    subset_files = [os.path.join(image_output_dir, f) for f in os.listdir(image_output_dir) if f.endswith('.parquet')]
    for subset_file in subset_files:
        process_segmentation_file(subset_file, seg_input_dir, seg_output_dir)

    # Save images from Parquet files
    for filename in os.listdir(image_output_dir):
        if filename.endswith('.parquet'):
            parquet_path = os.path.join(image_output_dir, filename)
            save_images_from_parquet(parquet_path, png_output_dir, '[CameraImageComponent].image')

    # Calculate and print subset lengths, and save them to a log file
    results = calculate_subset_lengths(image_input_dir, image_output_dir)
    log_file_name = os.path.join(base_output_dir, f"processing_log_{speed_threshold}.txt")
    with open(log_file_name, 'w') as log_file:
        log_file.write(f"Speed Threshold: {speed_threshold}\n")
        log_file.write("Processing Results:\n")
        for result in results:
            log_entry = (f"Original File: {result['original_file']}, Subset File: {result['subset_file']}, "
                         f"Subset Length: {result['subset_length']}, Percentage: {result['percentage']}%\n")
            print(log_entry, end='')
            log_file.write(log_entry)



def copy_optical_flows_and_update_json(src_optical_flow, src_png_images, dest_optical_flow):
    """
    Copies optical flow images and updates corresponding JSON files with relevant data.

    This function iterates through segments in the 'src_png_images' directory, 
    copies matching optical flow images from 'src_optical_flow' to 'dest_optical_flow', 
    and updates the JSON files in the destination directory with relevant data from the source JSON files.

    Args:
        src_optical_flow (str): Path to the source optical flow directory.
        src_png_images (str): Path to the source directory containing PNG images.
        dest_optical_flow (str): Path to the destination directory where optical flows and JSONs will be stored.
    """
    if not os.path.exists(dest_optical_flow):
        os.makedirs(dest_optical_flow)

    for seg_dir in os.listdir(src_png_images):
        seg_path = os.path.join(src_png_images, seg_dir)
        if os.path.isdir(seg_path):
            dest_seg_path = os.path.join(dest_optical_flow, seg_dir)
            if not os.path.exists(dest_seg_path):
                os.makedirs(dest_seg_path)

            dest_json_data = {}

            for file in os.listdir(seg_path):
                if file.endswith('.png'):
                    timestamp = file.split(';')[1].split('.')[0]
                    scene = file.split(';')[0]
                    src_flow_file = os.path.join(src_optical_flow, scene, f"{scene};{timestamp}_flow.png")
                    dest_flow_file = os.path.join(dest_seg_path, f"{scene};{timestamp}_flow.png")
                    print(os.path.exists(src_flow_file))
                    if os.path.exists(src_flow_file):
                        os.makedirs(os.path.dirname(dest_flow_file), exist_ok=True)
                        shutil.copy(src_flow_file, dest_seg_path)
                        src_json_file = os.path.join(src_optical_flow, scene, f"{scene}_corner_flows.json")
                        dest_json_file = os.path.join(dest_seg_path, f"{seg_dir}_corner_flows.json")

                        if os.path.exists(src_json_file):
                            with open(src_json_file, 'r') as f:
                                src_json_data = json.load(f)
                            
                            # Add only the relevant key-value pair to the destination JSON data
                            key = f"{scene};{timestamp}"
                            if key in src_json_data:
                                dest_json_data[key] = src_json_data[key]
            # Save the updated JSON data to the destination file
            with open(dest_json_file, 'w') as f:
                json.dump(dest_json_data, f, indent=4)


def main(args):
    os.makedirs(args.image_output_dir, exist_ok=True)
    os.makedirs(args.seg_output_dir, exist_ok=True)
    os.makedirs(args.png_output_dir, exist_ok=True)
    os.makedirs(args.optical_flow_output_dir, exist_ok=True)

    # main_processing(args.image_input_dir, args.seg_input_dir, args.image_output_dir, args.seg_output_dir, args.png_output_dir, args.speed_threshold, args.base_output_dir)
    copy_optical_flows_and_update_json(args.optical_flow_input_dir, args.png_output_dir, args.optical_flow_output_dir)
    sof.merge_all_json_files(args.base_output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process Waymo Open Dataset.")

    parser.add_argument('--speed_threshold', type=float, default=0.05, help='Speed threshold for processing.')
    parser.add_argument('--image_input_dir', type=str, default='waymo_dataset_v2/training/camera_image', help='Directory for camera images.')
    parser.add_argument('--optical_flow_input_dir', type=str, default='waymo_dataset_v2/optical_flow', help='Directory for optical flow images.')
    parser.add_argument('--seg_input_dir', type=str, default='waymo_dataset_v2/training/camera_segmentation', help='Directory for camera segmentation.')
    parser.add_argument('--base_output_dir', type=str, default='waymo_subset_vel_0.05', help='Base directory for output files.')

    args = parser.parse_args()

    args.image_output_dir = os.path.join(args.base_output_dir, 'camera_image')
    args.seg_output_dir = os.path.join(args.base_output_dir, 'camera_segmentation')
    args.png_output_dir = os.path.join(args.base_output_dir, 'png_image')
    args.optical_flow_output_dir = 'waymo_subset_vel_0.05\\optical_flow'

    main(args)

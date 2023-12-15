import os
import shutil
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def filter_and_save_images_by_corner_flow(
    base_path, new_base_path, threshold=0.2, sequence_length=5, required_corners=2
):
    """
    Filters and saves images based on corner flow values from JSON files. A frame is considered
    if a minimum number of corners ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    have values below the threshold.

    Args:
    base_path (str): Path to the base directory containing 'optical_flow' and 'png_image' folders.
    new_base_path (str): Path to the directory where filtered images and JSON files will be saved.
    threshold (float): The threshold value for corner flow values. Default is 0.2.
    sequence_length (int): The minimum number of consecutive frames that should meet the threshold criteria. Default is 5.
    min_corners_below_threshold (int): The minimum number of corners that need to be below the threshold to consider a frame. Default is 2.

    Returns:
    None: The function does not return any value. It saves the filtered images and JSON data to the specified new base path.
    """
    optical_flow_path = os.path.join(base_path, "optical_flow")
    png_image_path = os.path.join(base_path, "png_image")

    new_optical_flow_path = os.path.join(new_base_path, "optical_flow")
    new_png_image_path = os.path.join(new_base_path, "png_image")

    os.makedirs(new_optical_flow_path, exist_ok=True)
    os.makedirs(new_png_image_path, exist_ok=True)

    # Iterate through each folder in the optical flow directory.
    for folder in os.listdir(optical_flow_path):
        folder_path = os.path.join(optical_flow_path, folder)
        json_file_path = os.path.join(folder_path, f"{folder}_corner_flows.json")

        # Check if the corresponding JSON file exists.
        if os.path.isfile(json_file_path):
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                keys = list(data.keys())

                segment_index = -1
                i = 0

                # Iterate through the keys in the JSON file.
                while i < len(keys) - sequence_length + 1:
                    if all(
                        sum(
                            data[keys[j]][corner] < threshold
                            for corner in [
                                "top_left",
                                "top_right",
                                "bottom_left",
                                "bottom_right",
                            ]
                        )
                        >= required_corners
                        for j in range(i, i + sequence_length)
                    ):
                        segment_index += 1
                        segment_start = i

                        # Create directories for the new segment.
                        target_flow_folder = os.path.join(
                            new_optical_flow_path,
                            folder,
                            f"{folder}_seg{segment_index}",
                        )
                        target_png_folder = os.path.join(
                            new_png_image_path, folder, f"{folder}_seg{segment_index}"
                        )
                        os.makedirs(target_flow_folder, exist_ok=True)
                        os.makedirs(target_png_folder, exist_ok=True)

                        # Initialize the dictionary to store filtered data.
                        filtered_dict = {}

                        # Copy images to the new directories while the condition holds.
                        while (
                            i < len(keys)
                            and sum(
                                data[keys[i]][corner] < threshold
                                for corner in [
                                    "top_left",
                                    "top_right",
                                    "bottom_left",
                                    "bottom_right",
                                ]
                            )
                            >= required_corners
                        ):
                            key = keys[i]
                            filtered_dict[key] = data[key]

                            flow_img = key + "_flow.png"
                            png_img = key + ".png"
                            source_flow_img = os.path.join(folder_path, flow_img)
                            source_png_img = os.path.join(
                                png_image_path, folder, png_img
                            )

                            shutil.copy(source_flow_img, target_flow_folder)
                            shutil.copy(source_png_img, target_png_folder)
                            i += 1

                        # Save the filtered data to a JSON file in the filtered folder.
                        if filtered_dict:
                            filtered_json_path = os.path.join(
                                target_flow_folder,
                                f"{folder}_seg{segment_index}_filtered.json",
                            )
                            with open(filtered_json_path, "w") as f:
                                json.dump(filtered_dict, f, indent=4)

                    else:
                        i += 1


def merge_all_json_files(base_dir):
    """
    Merges the contents of all JSON files found within a specified directory and its subdirectories into a single JSON file.

    Args:
    base_dir (str): The base directory path where the JSON files are located. The function will search this directory and all its subdirectories for JSON files to merge.

    Returns:
    None: The function does not return any value. It writes the merged data to a file in the specified location.
    """
    merged_data = {}

    # Traverse all subdirectories and files within base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file is a JSON file
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)

                # Read and merge the contents of the JSON file
                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)

                    # Iterate through each key-value pair in the current JSON data
                    for key, value in data.items():
                        # If the key already exists in the merged data, update the value
                        if key in merged_data:
                            merged_data[key].update(value)
                        else:
                            # Otherwise, add the new key-value pair to the merged data
                            merged_data[key] = value

    # Write the merged data to a new JSON file in the optical_flow directory
    with open(
        os.path.join(base_dir, "optical_flow", "merged_data.json"), "w"
    ) as merged_file:
        json.dump(merged_data, merged_file, indent=4)


def load_keys_from_json_files(directory):
    """
    Loads and aggregates keys from all JSON files within a specified directory and its subdirectories.

    Args:
    directory (str): Path to the directory where JSON files are located. The function will search this directory
    and all its subdirectories for JSON files.

    Returns:
    set: A set of keys aggregated from all the JSON files in the specified directory.
    """
    keys = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)
                    keys.update(data.keys())
    return keys


def calculate_metrics(all_data_dir, true_positive_dir, prediction_dir):
    """
    Calculates the metrics of accuracy, precision, and recall based on the keys extracted from JSON files
    in given directories.

    Args:
    all_data_dir (str): Path to the directory containing JSON files of all data.
    true_positive_dir (str): Path to the directory containing JSON files of true positive samples.
    prediction_dir (str): Path to the directory containing JSON files of predicted samples.

    Returns:
    tuple: A tuple containing accuracy, precision, and recall in that order.
    """

    # Load keys from the directories
    all_data_keys = load_keys_from_json_files(all_data_dir)
    true_positive_keys = load_keys_from_json_files(true_positive_dir)
    prediction_keys = load_keys_from_json_files(prediction_dir)

    # Calculate TP, FP, TN, FN
    TP = len(true_positive_keys.intersection(prediction_keys))
    FP = len(prediction_keys - true_positive_keys)
    FN = len(true_positive_keys - prediction_keys)
    TN = len(all_data_keys - true_positive_keys - prediction_keys)

    # Calculate accuracy, precision, and recall
    accuracy = (TP + TN) / len(all_data_keys) if len(all_data_keys) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Prepare the text to be saved
    results_text = f"""Total keys in all data: {len(all_data_keys)}
Total keys in true positives: {len(true_positive_keys)}
Total keys in predictions: {len(prediction_keys)}
TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}
Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}
"""

    # Determine the directory to save the file
    save_directory = os.path.dirname(prediction_dir)
    save_path = os.path.join(save_directory, "metrics_summary.txt")

    # Write the results to the file
    with open(save_path, "w") as file:
        file.write(results_text)
    print(results_text)

    return accuracy, precision, recall


def find_corresponding_parquet(base_png_path, base_parquet_path):
    """
    This function finds the corresponding Parquet files for PNG images in a Waymo dataset.
    It iterates through directories of PNG images, matches them with Parquet files based on
    timestamps, and saves the matching Parquet data.

    Args:
        base_png_path (str): Base path to the directory containing PNG images.
        base_parquet_path (str): Base path to the directory containing Parquet files.
    """
    # Construct specific paths
    png_path = os.path.join(base_png_path, "png_image")
    parquet_path = os.path.join(base_parquet_path, "training", "camera_image")
    output_path = os.path.join(base_png_path, "camera_image")

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate through the PNG image folders
    for scene_dir in os.listdir(png_path):
        scene_path = os.path.join(png_path, scene_dir)
        if os.path.isdir(scene_path):
            # Process each segment in a scene
            for segment_dir in os.listdir(scene_path):
                segment_path = os.path.join(scene_path, segment_dir)
                if os.path.isdir(segment_path):
                    timestamps = []

                    # Collect timestamps of all PNG images in the segment
                    for png_file in os.listdir(segment_path):
                        if png_file.endswith(".png"):
                            timestamp = png_file.split(";")[-1].split(".")[0]
                            timestamps.append(int(timestamp))

                    # Build the corresponding Parquet file path
                    parquet_file = f"{scene_dir}.parquet"
                    full_parquet_path = os.path.join(parquet_path, parquet_file)

                    if os.path.exists(full_parquet_path):
                        # Read the Parquet file and find matching timestamps
                        df = pd.read_parquet(full_parquet_path)
                        matching_rows = df[
                            (df["key.frame_timestamp_micros"].isin(timestamps))
                            & (df["key.camera_name"] == 1)
                        ]

                        if not matching_rows.empty:
                            # Save the matched Parquet file segment
                            output_segment_path = os.path.join(
                                output_path, scene_dir, f"{segment_dir}.parquet"
                            )
                            os.makedirs(
                                os.path.dirname(output_segment_path), exist_ok=True
                            )
                            matching_rows.to_parquet(output_segment_path)
                    else:
                        print(f"Parquet file not found for: {full_parquet_path}")


def plot_camera_car_speed(parquet_path, output_folder):
    """
    Reads image data from a Parquet file, calculates the camera car's speed, and plots and saves a speed chart.

    Args:
    parquet_path (str): Path to the Parquet file.
    output_folder (str): Folder to save the speed chart.
    """
    # Read the Parquet file
    df_image = pd.read_parquet(parquet_path)

    # Calculate speeds
    speeds = np.sqrt(
        df_image["[CameraImageComponent].velocity.linear_velocity.x"] ** 2
        + df_image["[CameraImageComponent].velocity.linear_velocity.y"] ** 2
        + df_image["[CameraImageComponent].velocity.linear_velocity.z"] ** 2
    )

    # Plot the speed chart
    plt.figure(figsize=(12, 7))
    plt.plot(speeds, label="Camera Car Speed", color="blue")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Camera Car Speed Chart for {os.path.basename(parquet_path)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks([])

    # Save the plot
    output_path = os.path.join(
        output_folder, os.path.basename(parquet_path).replace(".parquet", ".png")
    )
    plt.savefig(output_path)
    plt.close()


def process_and_plot_speeds(parquet_dir):
    """
    Processes and plots speeds from Parquet files in a given directory.

    Args:
    parquet_dir (str): Directory containing Parquet files.
    """
    # Iterate over each parquet file in the directory
    for root, dirs, files in os.walk(parquet_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_path = os.path.join(root, file)
                output_folder = root
                plot_camera_car_speed(parquet_path, output_folder)


def main(optical_threshold, required_corners, base_path):
    new_base_path = f"waymo_subset_of_{optical_threshold}_corner_{required_corners}"
    new_op = os.path.join(new_base_path, "optical_flow")
    filter_and_save_images_by_corner_flow(
        base_path,
        new_base_path,
        optical_threshold,
        sequence_length=5,
        required_corners=required_corners,
    )
    target_file = os.path.join("waymo_subset_0.05", "optical_flow")
    merge_all_json_files(new_base_path)
    print(
        f"For optical threshold of {optical_threshold} and required corners {required_corners}: "
    )
    calculate_metrics(os.path.join(base_path, "optical_flow"), target_file, new_op)
    find_corresponding_parquet(new_base_path, base_path)
    parquet_dir = os.path.join(new_base_path, "camera_image")
    process_and_plot_speeds(parquet_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and filter Waymo Open Dataset based on optical flow and corner detection."
    )

    parser.add_argument(
        "--optical_threshold",
        type=float,
        default=0.4,
        help="Optical flow threshold for filtering data",
    )
    parser.add_argument(
        "--required_corners",
        type=int,
        default=4,
        help="Number of corners required below threshold to consider a frame",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="waymo_open_dataset_v_2_0_0",
        help="Base path for the Waymo Open Dataset",
    )

    args = parser.parse_args()

    main(args.optical_threshold, args.required_corners, args.base_path)

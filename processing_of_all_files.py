import scripts.subset_selection_of as sof
import scripts.subset_selection_vel as svel
import scripts.optical_flow.opticalflow as of
import os
import argparse


def create_output_directories(base_output_dir):
    # Create output directories and subdirectories
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "camera_image"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "camera_segmentation"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "png_image"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "optical_flow"), exist_ok=True)


def main(args):
    image_input_dir = os.path.join(args.base_input_dir, "training/camera_image")
    seg_input_dir = os.path.join(args.base_input_dir, "training/camera_segmentation")
    png_input_dir = os.path.join(args.base_input_dir, "png_image")
    optical_flow_input_dir = os.path.join(args.base_input_dir, "optical_flow")

    base_vel_output_dir = f"waymo_subset_vel_{args.speed_threshold}"
    image_vel_output_dir = os.path.join(base_vel_output_dir, "camera_image")
    seg_vel_output_dir = os.path.join(base_vel_output_dir, "camera_segmentation")
    png_vel_output_dir = os.path.join(base_vel_output_dir, "png_image")

    create_output_directories(base_vel_output_dir)
    os.makedirs(png_input_dir, exist_ok=True)
    os.makedirs(optical_flow_input_dir, exist_ok=True)

    # Save PNG images generated from original Parquet files
    svel.save_images_from_original_parquet(image_input_dir, png_input_dir)
    of.process_and_save_corner_flows(png_input_dir, optical_flow_input_dir,args.corner_size)
    of.merge_json_files(optical_flow_input_dir)

    svel.main_processing(
        image_input_dir,
        seg_input_dir,
        image_vel_output_dir,
        seg_vel_output_dir,
        png_vel_output_dir,
        args.speed_threshold,
        base_vel_output_dir,
    )
    # Optical flow processing for subset selected by velocity threshold
    optical_flow_vel_output_dir = os.path.join(base_vel_output_dir, "optical_flow")
    svel.copy_optical_flows_and_update_json(optical_flow_input_dir,png_vel_output_dir,optical_flow_vel_output_dir)
    of.merge_json_files(optical_flow_vel_output_dir)

    base_of_output_dir = f"waymo_subset_of_{args.optical_threshold}_corner_{args.required_corners}"
    optical_flow_of_output_dir = os.path.join(base_of_output_dir, "optical_flow")
    sof.filter_and_save_images_by_corner_flow(
        args.base_input_dir, base_of_output_dir, args.optical_threshold,args.sequence_length,args.required_corners
    )
    sof.merge_all_json_files(base_of_output_dir)
    print(f"For optical threshold of {args.optical_threshold} and required corners {args.required_corners}: ")
    sof.calculate_metrics(
        optical_flow_input_dir, optical_flow_vel_output_dir, optical_flow_of_output_dir
    )
    sof.find_corresponding_parquet(base_of_output_dir, args.base_input_dir)
    parquet_dir = os.path.join(base_of_output_dir, "camera_image")
    sof.process_and_plot_speeds(parquet_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Waymo Open Dataset.")
    parser.add_argument(
        "--base_input_dir",type=str,default="waymo_dataset_v2",help="Base input directory for Waymo dataset")
    parser.add_argument(
        "--speed_threshold",type=float,default=0.05,help="Speed threshold for filtering data")
    parser.add_argument(
        "--optical_threshold",type=float,default=0.1,help="Optical flow threshold for filtering data")
    parser.add_argument(
        "--corner_size",type=float,default=0.15,help="Size of the corner for optical flow processing")
    parser.add_argument(
        "--sequence_length", type=int, default=5,help="Minimal number of consecutive frames")
    parser.add_argument(
        "--required_corners",type=int,default=2,help="Minimal number of corners to meet the optical flow threshold requirement")
    args = parser.parse_args()
    main(args)
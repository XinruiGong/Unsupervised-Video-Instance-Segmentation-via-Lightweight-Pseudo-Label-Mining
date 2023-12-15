import os
import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import dask.dataframe as dd
from waymo_open_dataset.utils import (range_image_utils, transform_utils,
                                      frame_utils, camera_segmentation_utils)
import io
from PIL import Image

# Enable eager execution if not already enabled
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2
from waymo_open_dataset.protos import (camera_segmentation_metrics_pb2 as metrics_pb2,
                                       camera_segmentation_submission_pb2 as submission_pb2)
from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics

# Data locations - these should be replaced by the user
FILE_NAME = '/content/waymo-open-dataset/tutorial/.../tfexample.tfrecord'
EVAL_DIR = '/content/drive/MyDrive/Colab Notebooks/waymo/waymo-open-dataset/waymo_open_dataset_v_2_0_0/training'
EVAL_RUNS = [
    'segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord',
    'segment-11048712972908676520_545_000_565_000_with_camera_labels.tfrecord'
]
TEST_SET_SOURCE = '/content/waymo-open-dataset/tutorial/2d_pvps_validation_frames.txt'
TEST_DIR = '/dataset_path/testing/'
SAVE_FOLDER = '/tmp/camera_segmentation_challenge/testing/'

context_name = '10075870402459732738_1060_000_1080_000'


def read(tag: str, dataset_dir: str = EVAL_DIR) -> dd.DataFrame:
    """Creates a Dask DataFrame for the specified component tag."""
    paths = f'{dataset_dir}/{tag}/{context_name}.parquet'
    return dd.read_parquet(paths)


# Read camera image and segmentation data
camera_image_df = read('camera_image')
cam_segmentation_df = read('camera_segmentation')

# Group segmentation labels by frame using context name and timestamp
frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
cam_segmentation_per_frame_df = cam_segmentation_df.groupby(
    frame_keys, group_keys=False).agg(list)
camera_image_per_frame_df = camera_image_df.groupby(
    frame_keys).apply(lambda x: x)


def ungroup_row(key_names: Sequence[str], key_values: Sequence[str], row: dd.DataFrame) -> Iterator[Dict[str, Any]]:
    """Splits a group of dataframes into individual dictionaries."""
    keys = dict(zip(key_names, key_values))
    cols, cells = list(zip(*[(col, cell) for col, cell in r.items()]))
    for values in zip(*cells):
        yield dict(zip(cols, values), **keys)


cam_segmentation_list = []
for i, (key_values, r) in enumerate(cam_segmentation_per_frame_df.iterrows()):
    # For this demo, only read three sequences of 5 camera images
    if i >= 20:
        break
    # Store a segmentation label component for each camera
    cam_segmentation_list.append(
        [v2.CameraSegmentationLabelComponent.from_dict(d)
         for d in ungroup_row(frame_keys, key_values, r)]
    )

timestamps = [key_values[1]
              for key_values, _ in cam_segmentation_per_frame_df.iterrows()]

# Order labels from left to right for visualization
camera_left_to_right_order = [
    open_dataset.CameraName.SIDE_LEFT,
    open_dataset.CameraName.FRONT_LEFT,
    open_dataset.CameraName.FRONT,
    open_dataset.CameraName.FRONT_RIGHT,
    open_dataset.CameraName.SIDE_RIGHT
]

segmentation_protos_ordered = []
for it, label_list in enumerate(cam_segmentation_list):
    segmentation_dict = {label.key.camera_name: label for label in label_list}
    segmentation_protos_ordered.append(
        [segmentation_dict[name] for name in camera_left_to_right_order])


# Retrieve camera images based on timestamps
front_images = []

for timestamp in timestamps:
    # Filter camera images based on the timestamp and camera name (FRONT)
    matching_images = camera_image_per_frame_df.loc[
        (camera_image_per_frame_df['key.frame_timestamp_micros'] == timestamp) &
        (camera_image_per_frame_df['key.camera_name']
         == open_dataset.CameraName.FRONT)
    ].compute()

    # If there's a matching image, decode it
    if not matching_images.empty:
        # Assuming the image data is stored in the '[CameraImageComponent].image' column
        image_data = matching_images.iloc[0]['[CameraImageComponent].image']

        # Check if the data represents a JPEG image and decode if true
        if isinstance(image_data, bytes) and image_data[:2] == b'\xff\xd8':
            decoded_image = np.array(Image.open(io.BytesIO(image_data)))
            front_images.append(decoded_image)


def _pad_to_common_shape(label):
    """Pad the label to a common shape."""
    return np.pad(label, [[1280 - label.shape[0], 0], [0, 0], [0, 0]])


# Keep the decoding of panoptic labels unchanged
# Extracting only the FRONT camera segmentation labels
segmentation_protos_flat = [
    segmentation[camera_left_to_right_order.index(
        open_dataset.CameraName.FRONT)]
    for segmentation in segmentation_protos_ordered
]

# Decode multi-frame panoptic labels from segmentation labels
panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor = \
    camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
        segmentation_protos_flat, remap_to_global=True
    )

# Extract semantic and instance labels for the "front image"
semantic_labels = []
instance_labels = []

# Decode semantic labels and instance labels for each frame
for i in range(len(segmentation_protos_flat)):
    semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
        panoptic_labels[i], panoptic_label_divisor
    )
    semantic_labels.append(semantic_label)
    instance_labels.append(instance_label)

# NUM_CAMERA_FRAMES = 1
# semantic_labels_multiframe = []
# instance_labels_multiframe = []
# for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
#     semantic_labels = []
#     instance_labels = []
#     for j in range(NUM_CAMERA_FRAMES):
#         semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
#             panoptic_labels[i + j], panoptic_label_divisor)
#         semantic_labels.append(semantic_label)
#         instance_labels.append(instance_label)
#     semantic_labels_multiframe.append(semantic_labels)
#     instance_labels_multiframe.append(instance_labels)

# Padding and concatenating the semantic and instance labels
semantic_labels = [_pad_to_common_shape(label) for label in semantic_labels]
instance_labels = [_pad_to_common_shape(label) for label in instance_labels]

instance_label_concat = np.concatenate(instance_labels, axis=0)
semantic_label_concat = np.concatenate(semantic_labels, axis=0)

# Convert the panoptic labels to RGB format for visualization
panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
    semantic_label_concat, instance_label_concat)

# # Code for displaying individual images with their segmentation
# frame_height = int(panoptic_label_rgb.shape[0]/len(front_images))
# for i, front_image in enumerate(front_images):
#     segmentation = panoptic_label_rgb[i*frame_height: (i+1)*frame_height, :, :]
#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#     axs[0].imshow(front_image)
#     axs[0].set_title('Front Image')
#     axs[0].axis('off')
#     axs[1].imshow(segmentation)
#     axs[1].set_title('Segmentation')
#     axs[1].axis('off')
#     plt.show()

# Determine the height and width for individual frames in the RGB image
frame_height = int(panoptic_label_rgb.shape[0]/len(front_images))
frame_width = panoptic_label_rgb.shape[1]

# Width for concatenated images (image + segmentation)
combined_width = 2 * frame_width

# Initialize an empty canvas for the combined images
combined_image = np.zeros(
    (panoptic_label_rgb.shape[0], combined_width, 3), dtype=np.uint8)

# Combine each front image with its corresponding segmentation side by side
for i, front_image in enumerate(front_images):
    segmentation = panoptic_label_rgb[i*frame_height: (i+1)*frame_height, :, :]
    combined_row = np.concatenate((front_image, segmentation), axis=1)
    combined_image[i*frame_height: (i+1)*frame_height, :, :] = combined_row

# Visualize the final combined image (image + segmentation)
plt.figure(figsize=(96, 90))
plt.imshow(combined_image)
plt.axis('off')
plt.show()

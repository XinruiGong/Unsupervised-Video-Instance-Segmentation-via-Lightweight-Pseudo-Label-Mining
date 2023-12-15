import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import dask.dataframe as dd

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2
from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics
from waymo_open_dataset.utils import camera_segmentation_utils                                                       
context_name = '10023947602400723454_1120_000_1140_000'
def read(tag: str, dataset_dir: str = EVAL_DIR) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = f'{dataset_dir}/{tag}/{context_name}.parquet'
  return dd.read_parquet(paths)

cam_segmentation_df = read('camera_segmentation')

# Group segmentation labels into frames by context name and timestamp.
frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
cam_segmentation_per_frame_df = cam_segmentation_df.groupby(
    frame_keys, group_keys=False).agg(list)

def ungroup_row(key_names: Sequence[str],
                key_values: Sequence[str],
                row: dd.DataFrame) -> Iterator[Dict[str, Any]]:
  """Splits a group of dataframes into individual dicts."""
  keys = dict(zip(key_names, key_values))
  cols, cells = list(zip(*[(col, cell) for col, cell in r.items()]))
  for values in zip(*cells):
    yield dict(zip(cols, values), **keys)

cam_segmentation_list = []
for i, (key_values, r) in enumerate(cam_segmentation_per_frame_df.iterrows()):
  # Read three sequences of 5 camera images for this demo.
  if i >= 20:
    break
  # Store a segmentation label component for each camera.
  cam_segmentation_list.append(
      [v2.CameraSegmentationLabelComponent.from_dict(d)
       for d in ungroup_row(frame_keys, key_values, r)])                                                                                       
# Order labels from left to right for visualization later.
# For each frame with segmentation labels, all cameras should have a label.
camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                              open_dataset.CameraName.FRONT_LEFT,
                              open_dataset.CameraName.FRONT,
                              open_dataset.CameraName.FRONT_RIGHT,
                              open_dataset.CameraName.SIDE_RIGHT]
segmentation_protos_ordered = []
for it, label_list in enumerate(cam_segmentation_list):
  segmentation_dict = {label.key.camera_name: label for label in label_list}
  segmentation_protos_ordered.append([segmentation_dict[name] for name in camera_left_to_right_order])   
  
  
  # Read a single panoptic label                                                                                                                                              
  # Decode a single panoptic label.
panoptic_label_front = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
    segmentation_protos_ordered[0][open_dataset.CameraName.FRONT]
)

# Separate the panoptic label into semantic and instance labels.
semantic_label_front, instance_label_front = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
    panoptic_label_front,
    segmentation_protos_ordered[0][open_dataset.CameraName.FRONT].panoptic_label_divisor
)








# The dataset provides tracking for instances between cameras and over time.
# By setting remap_to_global=True, this function will remap the instance IDs in
# each image so that instances for the same object will have the same ID between
# different cameras and over time.
segmentation_protos_flat = sum(segmentation_protos_ordered, [])
panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
    segmentation_protos_flat, remap_to_global=True
)

# We can further separate the semantic and instance labels from the panoptic
# labels.
NUM_CAMERA_FRAMES = 5
semantic_labels_multiframe = []
instance_labels_multiframe = []
for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
  semantic_labels = []
  instance_labels = []
  for j in range(NUM_CAMERA_FRAMES):
    semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
      panoptic_labels[i + j], panoptic_label_divisor)
    semantic_labels.append(semantic_label)
    instance_labels.append(instance_label)
  semantic_labels_multiframe.append(semantic_labels)
  instance_labels_multiframe.append(instance_labels)                                                                                             
def _pad_to_common_shape(label):
  return np.pad(label, [[1280 - label.shape[0], 0], [0, 0], [0, 0]])

# Pad labels to a common size so that they can be concatenated.
instance_labels = [[_pad_to_common_shape(label) for label in instance_labels] for instance_labels in instance_labels_multiframe]
semantic_labels = [[_pad_to_common_shape(label) for label in semantic_labels] for semantic_labels in semantic_labels_multiframe]
instance_labels = [np.concatenate(label, axis=1) for label in instance_labels]
semantic_labels = [np.concatenate(label, axis=1) for label in semantic_labels]

instance_label_concat = np.concatenate(instance_labels, axis=0)
semantic_label_concat = np.concatenate(semantic_labels, axis=0)
panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
    semantic_label_concat, instance_label_concat)                                                                                                    
plt.figure(figsize=(64, 60))
plt.imshow(panoptic_label_rgb)
plt.grid(False)
plt.axis('off')
plt.show() 
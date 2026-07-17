"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using stratified sampling, and custom collate functions for handling variable-length video 
sequences.
"""

import os
import glob
import re

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.
    
    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.
    
    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are 
                            integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): clip-level transform (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None, random_temporal_crop=False):  # pylint: disable=redefined-outer-name
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.random_temporal_crop = random_temporal_crop

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths from the video directory and select up to fr_per_vid frames
        video_dir, label = self.dataset[idx]
        frame_paths = sorted(
            glob.glob(os.path.join(video_dir, "*.jpg"))
        )

        # Uniformly sample frames from a video file if too many frames have been captured
        if len(frame_paths) > self.fpv:
            if self.random_temporal_crop:
                max_start = len(frame_paths) - self.fpv
                start = torch.randint(0, max_start+1, size=(1,)).item()
                frame_paths = frame_paths[
                    start:start+self.fpv
                ]
            else:
                frame_indices = np.linspace(
                    0, len(frame_paths)-1, num=self.fpv, dtype=np.int64
                )
                frame_paths = [
                    frame_paths[index] for index in frame_indices
                ]

        # Open frames as RGB using PIL
        frames = [
            Image.open(frame_path).convert("RGB") for frame_path in frame_paths
        ]

        if not frames:
            return [], label

        # Convert individual PIL frames and stack them into one video tensor
        # Shape: (T, C, H, W), dtype: uint8
        frames_tensor = torch.stack([
            pil_to_tensor(frame) for frame in frames
        ])

        # Apply same transforms to all frames from the same video
        if self.transforms is not None:
            frames_tensor = self.transforms(frames_tensor)

        return frames_tensor, label


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.
    
    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.
    
    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.
    
    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    category_names = sorted(
        category
        for category in os.listdir(frame_dir)
        if os.path.isdir(os.path.join(frame_dir, category))
    )

    label_dict = {
        category: index
        for index, category in enumerate(category_names)
    }

    vid_dataset = {}
    print("Loading video dataset....")

    for category in tqdm(category_names):
        category_path = os.path.join(frame_dir, category)

        for video_name in sorted(os.listdir(category_path)):
            video_path = os.path.join(
                category_path,
                video_name,
            )

            if os.path.isdir(video_path):
                vid_dataset[video_path] = label_dict[category]

    return vid_dataset, label_dict


def get_source_group(video_path):
    """
    Extract the original source-video identifier from an HMDB51 clip path.

    HMDB51 clip directory names follow a structure resembling:

        source_title_action_f_cm_np1_ba_med_0

    Multiple numbered clips may therefore originate from the same source video.
    These clips must remain in the same dataset split.

    Args:
        video_path (str): Path to one extracted video-clip directory.

    Returns:
        str: Original source-video identifier.

    Raises:
        ValueError: If the expected HMDB51 naming pattern is not found.
    """
    video_path = os.path.normpath(video_path)
    class_name = os.path.basename(os.path.dirname(video_path))
    video_name = os.path.basename(video_path)

    pattern = re.compile(
        rf"_{re.escape(class_name)}_[^_]+_[cn]m_np\d+_"
    )
    match = pattern.search(video_name)

    if match is None:
        raise ValueError(
            "Could not extract the source-video group from "
            f"clip directory: {video_path}"
        )

    return video_name[:match.start()]


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0, max_attempts=1000):  # pylint: disable=too-many-locals
    """
    Split clips by original source video.

    Every clip derived from the same source video is assigned entirely to
    training, validation, or testing. This prevents source-level leakage.

    GroupShuffleSplit preserves groups rather than individual clips, so the
    final sample proportions are approximate.

    Args:
        vid_dataset (dict): Mapping from clip-directory paths to labels.
        tr_ratio (float): Requested training proportion.
        ts_ratio (float): Requested testing proportion.
        seed (int): Initial random seed.
        max_attempts (int): Maximum attempts to find splits containing all
                            classes.

    Returns:
        tuple: Training, validation, and testing sample lists.

    Raises:
        ValueError: If ratios are invalid or a complete split cannot be found.
    """
    val_ratio = 1.0 - tr_ratio - ts_ratio

    if tr_ratio <= 0 or ts_ratio <= 0 or val_ratio <= 0:
        raise ValueError(
            "Training, validation, and testing ratios must all be positive."
        )

    vid_paths = np.array(sorted(vid_dataset.keys()))
    vid_labels = np.array([vid_dataset[path] for path in vid_paths])
    vid_groups = np.array([get_source_group(path) for path in vid_paths])

    all_classes = set(np.unique(vid_labels))
    val_weight = val_ratio / (tr_ratio + val_ratio)

    print("Splitting train/validation/test datasets by source video....")

    for attempt in range(max_attempts):
        current_seed = seed + attempt

        test_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=ts_ratio,
            random_state=current_seed,
        )

        train_val_idx, test_idx = next(
            test_splitter.split(
                vid_paths,
                vid_labels,
                groups=vid_groups,
            )
        )

        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_weight,
            random_state=current_seed + max_attempts,
        )

        train_relative_idx, val_relative_idx = next(
            val_splitter.split(
                vid_paths[train_val_idx],
                vid_labels[train_val_idx],
                groups=vid_groups[train_val_idx],
            )
        )

        train_idx = train_val_idx[train_relative_idx]
        val_idx = train_val_idx[val_relative_idx]

        split_indices = (train_idx, val_idx, test_idx)

        if all(
            set(np.unique(vid_labels[indices])) == all_classes
            for indices in split_indices
        ):
            break
    else:
        raise RuntimeError(
            "Unable to create grouped splits containing every class."
        )

    train_groups = set(vid_groups[train_idx])
    val_groups = set(vid_groups[val_idx])
    test_groups = set(vid_groups[test_idx])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    train_dataset = list(zip(vid_paths[train_idx], vid_labels[train_idx]))
    val_dataset = list(zip(vid_paths[val_idx], vid_labels[val_idx]))
    test_dataset = list(zip(vid_paths[test_idx], vid_labels[test_idx]))

    total_samples = len(vid_paths)

    print(
        "Grouped split sizes: "
        f"train={len(train_dataset)} "
        f"({len(train_dataset) / total_samples:.2%}), "
        f"val={len(val_dataset)} "
        f"({len(val_dataset) / total_samples:.2%}), "
        f"test={len(test_dataset)} "
        f"({len(test_dataset) / total_samples:.2%})"
    )

    return train_dataset, val_dataset, test_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in batch if len(imgs) > 0]
    if not valid_samples:
        return None, None

    imgs_batch, label_batch = zip(*valid_samples)

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    imgs_tensor = pad_sequence(imgs_batch, batch_first=True)

    # Swap T and C channels for Conv3d
    # Resulting shape: (batch_size, C, max_T, H, W)
    imgs_tensor = imgs_tensor.transpose(1, 2)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch, dtype=torch.long)

    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    
    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))

    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None, None
    imgs_batch, label_batch = zip(*valid_samples)

    # Record lengths before padding
    lengths = torch.tensor(
        [len(images) for images in imgs_batch],
        dtype=torch.long,
    )

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)

    return padded_imgs, labels_tensor, lengths

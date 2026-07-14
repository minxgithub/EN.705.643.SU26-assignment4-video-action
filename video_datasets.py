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

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):  # pylint: disable=redefined-outer-name
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

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
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):  # pylint: disable=too-many-locals
    """
    Split the dataset into training, validation, and test sets using stratified sampling.
    
    This function uses StratifiedShuffleSplit to ensure that each split has a representative
    distribution of classes.
    
    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.
    
    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    vid_paths = np.array(list(vid_dataset.keys()))
    vid_labels = np.array(list(vid_dataset.values()))
    print('Splitting train/validation/test datasets....')

    # Test split using StratifiedShuffleSplit
    ts_spliter = StratifiedShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
    for tr_val_idx, ts_idx in ts_spliter.split(vid_paths, vid_labels):
        ts_paths, ts_labels = vid_paths[ts_idx], vid_labels[ts_idx]
        tr_val_paths, tr_val_labels = vid_paths[tr_val_idx], vid_labels[tr_val_idx]
    ts_dataset = list(zip(ts_paths, ts_labels))

    # Train/validation split
    val_ratio = 1 - tr_ratio - ts_ratio
    val_wt = val_ratio / (tr_ratio + val_ratio)
    val_spliter = StratifiedShuffleSplit(n_splits=1, test_size=val_wt, random_state=seed)
    for tr_idx, val_idx in val_spliter.split(tr_val_paths, tr_val_labels):
        tr_paths, tr_labels = tr_val_paths[tr_idx], tr_val_labels[tr_idx]
        val_paths, val_labels = tr_val_paths[val_idx], tr_val_labels[val_idx]
    tr_dataset = list(zip(tr_paths, tr_labels))
    val_dataset = list(zip(val_paths, val_labels))

    return tr_dataset, val_dataset, ts_dataset


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

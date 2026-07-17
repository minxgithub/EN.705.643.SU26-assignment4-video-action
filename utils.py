"""
Module: utils.py

This module provides helper functions for video processing and data transformations 
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn

# Random Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.
            
    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []

    v_cap = cv2.VideoCapture(vid)  # pylint: disable=no-member
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0

    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable=no-member
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return frames, 0

    if n_frames <= 0:
        print(
            f"Requested n_frames={n_frames}. Please specify a positive number of frames for {vid}."
        )
        v_cap.release()
        return frames, v_len

    frame_idx = np.linspace(0, v_len-1, num=min(n_frames, v_len), dtype=np.int64)
    frame_idx = set(frame_idx.tolist())

    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            break
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
            frames.append(frame)
    v_cap.release()

    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    os.makedirs(store_path, exist_ok=True)

    for idx, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
        path_to_frame = os.path.join(store_path, f"frame{idx:04d}.jpg")
        cv2.imwrite(path_to_frame, frame_bgr)  # pylint: disable=no-member
        if not cv2.imwrite(path_to_frame, frame_bgr):  # pylint: disable=no-member
            print(f"Failed to save frame: {path_to_frame}")


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets.

    The training transforms include data augmentation such as random horizontal flipping and 
    random affine transformations, while the validation/test transforms consist solely of resizing, 
    converting to tensor, and normalizing.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(
            size=(height, width),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    val_test_transforms = v2.Compose([
        v2.Resize((height, width)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn'):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    if model == "lrcn":
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_rnn,
                              num_workers=4, pin_memory=torch.cuda.is_available())
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn,
                            num_workers=4, pin_memory=torch.cuda.is_available())
    elif model == "3dcnn":
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_r3d_18)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_r3d_18)
    else:
        raise ValueError(
            f"Unsupported model: {model}"
        )
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn'):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    if model == "lrcn":
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_rnn,
                             num_workers=4, pin_memory=torch.cuda.is_available())
    elif model == "3dcnn":
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_r3d_18,
                             num_workers=4, pin_memory=torch.cuda.is_available())
    else:
        raise ValueError(
            f"Unsupported model: {model}"
        )
    dataloaders = {'test': test_dl}
    return dataloaders

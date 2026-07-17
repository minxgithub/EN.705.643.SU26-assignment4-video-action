# Video Classification with HMDB51

This project implements a video classification pipeline using the HMDB51 dataset with two arcitectures: a Long-term Recurrent Convolutional Network (LRCN) model that extracts spatial features from individual video frames via a ResNet backbone and learns temporal dynamics through an LSTM, and a pretrained R3D-18 that learns spatial and temporal representations jointly using 3D residual convolutions. <br>
The project includes scripts for preprocessing, training, and testing the model.

---

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Project Structure](#project-structure)
- [Customization and Hyperparameters](#customization-and-hyperparameters)
- [Bugs Fixed and Code Improvements](#bugs-fixed-and-code-improvements)
---

## Dataset Preparation

### Step 0: Download and Unzip Dataset

1. **Download Dataset:**
   Download the HMDB51 dataset from [Kaggle](https://www.kaggle.com/datasets/easonlll/hmdb51). This dataset contains videos of 51 different human action classes.

2. **Unzip and Organize:**
   Unzip the downloaded dataset. The expected folder structure should be as follows:

        - HMDB51
            - Action_Class1
            - Action_Class2
            ... ... ... ...
            - Action_Class51

Each subdirectory represents a different action class.

---

## Environment Setup

1. **Python Version:**
This project requires Python 3.7 or higher.

2. **Dependencies:**
Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

# Key Libraries

- **PyTorch**
- **torchvision**
- **OpenCV**
- **scikit-learn**
- **tqdm**
- **numpy**
- **Pillow**
- **Weights & Biases**

## Hardware Requirements

A CUDA-enabled GPU is recommended for training. The code automatically detects GPU availability.

---

## Preprocessing and Frame Extraction

When using raw video clips, the raw video files must be converted into frame sequences. The preprocessing module includes functions for:

### Uniform Frame Sampling

- The `get_frames` function uses OpenCV to sample a fixed number of frames per video.

### Saving Frames to Disk

- The `store_frames` function writes the extracted frames as JPEG images.

The resulting frame directories should follow the same class and clip hierarchy expected by `VideoDataset`.

The supplied Kaggle archive already contains extracted JPEG frames organized into clip directories, so preprocessing is not required for normal training. The get_frames() and store_frames() utilities remain available for converting raw videos into frame folders if needed.

---

## Training the Model

### Step 1: Run Training

#### Configure Training Parameters

The training is managed via a bash script (e.g., `train.sh`) that calls the main training module.
**Important:** Update the `--frame_dir` argument in the script to point to the directory where your preprocessed frame data is stored. You can also adjust other parameters (e.g., number of frames per video, batch size, learning rate) to see how they affect the experiment.

#### Run the Training Script

Execute the training script from your terminal:

```bash
bash train.sh
```

## During Training, the Script Will:

- **Load the frame dataset.**
- **Split the dataset** into training, validation, and test sets using stratified sampling.
- **Apply data augmentation** techniques (resizing, random flips, affine transformations).
- **Create custom PyTorch Datasets and DataLoaders.**
- **Initialize the LRCN model** using a specified ResNet backbone.
- **Set up the loss function, optimizer, and learning rate scheduler.**
- **Run the training loop** while tracking loss and accuracy, saving the best model weights.

---

## Testing and Evaluation

### Step 2: Run Testing

- **Configure Testing Parameters:**
  Update the `--ckpt` argument in your testing script (e.g., `test.sh`) to point to the saved best model weights generated during training.

- **Run the Testing Script:**
  Execute the testing script from your terminal:

```bash
bash test.sh
```

## Testing Script Overview

The testing script will:

- **Load the dataset splits** (previously saved during training).
- **Create a DataLoader for the test set.**
- **Load the trained model checkpoint.**
- **Evaluate the model** on the test data by computing overall accuracy, generating classification reports, and optionally producing confusion matrices.

---

## Customization and Hyperparameters

You can modify several parameters to experiment with different settings:

### Data Parameters

- `--frame_dir`: Path to your preprocessed frames.
- `--fr_per_vid`: Number of frames to sample per video.

### Model Parameters

- `--model_type`: Choose between `'lrcn'` (default) or other supported models.
- `--cnn_backbone`: Options include `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152`.
- `--rnn_hidden_size` and `--rnn_n_layers`: Configure the LSTM network.

### Training Parameters

- `--batch_size`, `--learning_rate`, `--n_epochs`, and `--dropout` control the training dynamics.
- `--train_size` and `--test_size` determine dataset splits.

By tweaking these parameters, you can study their impact on model performance and experiment with different network configurations.

---

## Summary of Steps

- **Step 0: Dataset Preparation**
  Download, unzip, and organize the HMDB51 dataset into subdirectories by action class.

- **Step 1: Run Training**
  Execute `train.sh` after configuring the `--frame_dir` and other hyperparameters to train the model.

- **Step 2: Run Testing**
  Execute `test.sh` after updating the `--ckpt` argument to point to the best model checkpoint to evaluate the model.

Happy Training!

---

## Bugs Fixed and Code Improvements
1. `video_datasets.py`
- **Fixed the source-video leakage prevention in the `dataset_split()` function** <br>
The original implementation used StratifiedShuffleSplit on individual clip directories. Multiple clips derived from the same original video could therefore be distributed across training, validation, and test sets. This leaked actors, backgrounds, camera settings, and related temporal content across subsets and produced an overly optimistic evaluation.<br>
The revised implementation extracts a source-video identifier from each clip name and uses GroupShuffleSplit, ensuring that all clips derived from one source video remain in the same subset. It also asserts that train, validation, and test source groups are disjoint and retries the split until all 51 classes are represented.

- **Fixed the data transfomration in `class VideoDataset` `__getitem__()` method** <br>
The original implementation truncated longer clips to their first `fr_per_vid` frames. Because the supplied Kaggle dataset already contains extracted frames, the preprocessing `get_frames()` function was not invoked during training. Consequently, the model saw only the beginning of longer clips.<br>
The revised implementation uniformly selects `fr_per_vid` frame positions across the complete extracted clip, providing temporal coverage of the full action.<br>
The original imeplementation applies data augmentation (random geometric transforms) independently to each frame from the same video. <br>
The revised implementation applies the same image transformation to all the frames from the same video. <br>

- **Fixed the `collate_fn_r3d_18()` function** <br>
The original implementation filtered video tensors and labels independently, which could misalign videos with their corresponding class labels whenever empty samples occurred within a batch.<br>
The revised implementation filters '(video, label)' pairs together to preserve the correct label mapping and pads shorter video sequences along the temporal dimension before batching, ensuring robust handling of variable-length clips for 3D CNN models.

- **Improved the `collate_fn_rnn()` function** <br>
The original RNN collation logic padded short clips but discarded their genuine sequence lengths. The LRCN therefore could classify a short video using an artificial padded timestep. <br>
The revised RNN collate function returns each clip’s valid length. The LRCN uses a validity mask so padded images are not processed by the ResNet backbone and selects the LSTM output corresponding to each sample’s final genuine frame. <br>

- **Improved the `load_dataset()` function** <br>
The original dataset discovery logic treated every filesystem entry as a class or video directory, allowing metadata files such as .DS_Store to enter the dataset index. <br>
The revised loader includes directories only and sorts filesystem entries for deterministic label assignment and dataset ordering. <br>


2. `utils.py`
- **Improved the `compose_data_transforms()` function** <br>
The original implementation uses `torchvision.transforms` that performs transformation individually.<br>
The revised implementation uses `torchvision.transforms v2` and applies the composed transform once to the complete [T,C,H,W] clip tensor. Consequently, one set of random spatial parameters is shared across all frames. RandomResizedCrop replaces RandomAffine to provide consistent variation in framing and scale.<br>
Training clips containing more than `fr_per_vid` frames additionally use random temporal cropping, allowing different 16-frame windows to be observed across epochs. Validation and testing retain deterministic uniform temporal sampling.<br>

- **Improved the `store_frames()` function** <br>
The original implementation store frames with filename 'frame{}.jpg', which would destroys temporal order when frame paths are sorted lexicographically.<br>
The revised implementation used zero-padded filenames 'frame{idx:04d}.jpg' to preserve the temporal order when network reads them.<br>

- **Improved the `train_val_dloaders()` and `test_dloaders()` functions** <br>
Added 'num_workers' and 'pin_memory' arguments to both functions.


3. `models.py`
- **Removed `class Identity` and replaced it with `nn.Identity()`** <br>

- **Improved `class LRCN`, `__init__()`** <br>
Updated `elif cnn_model == 'resnet152':` to the correct corresponding model. <br>
When pretrained ImageNet weights are enabled, the LRCN ResNet backbone is frozen so that training updates only the LSTM and final classifier. The frozen backbone is kept in evaluation mode during training to prevent BatchNorm running statistics from changing. <br>
Added 'batch_first=True' in 'self.rnn'. <br>

- **Improved `class LRCN`, `forward()`** <br>
The original implementation process one frame at a time, iteratively use CNN backbone to extract its feature, then pass to RNN until the last frame. <br>
The revised implementation use CNN backbone to extract all the features from all the frames in one batched operation, then use RNN to process the feature sequence in the temporal order for final output. <br>


4. `run.py`
- **Added pretrained R3D-18 as the 3dcnn model option.** <br>

- **Introduced Weights & Biases loggings:** <br>
Per epoch: train loss, train accuracy, validation loss, validation accuracy, learning rate <br>
Evaluation summary: test loss, test accuracy <br>
Configuration: model type, CNN backbone, pretrained setting, batch sizes, learning rate, epochs, frames per video, split proportions, device <br>

- **Changed the loss function from `nn.CrossEntropyLoss(reduction='sum')` to `nn.CrossEntropyLoss()`, which uses the default `reduction='mean'`.**


5. `train.py` and `test.py`
- **Updated the training and testing loops to support both LRCN batches (videos, labels, lengths) and R3D batches (videos, labels).** <br>
LRCN receives valid sequence lengths, while R3D receives the fixed video tensor directly. <br>

- **Updated the statistics calculation in the `get_epoch_loss()` and `test()` function** <br>
The original implementation used `len(dataloader.dataset)` as the total number of samples to compute statistics, which does not count the case when there are empty samples in a batch. <br>
The revised implementation counts the actual samples that survive collation to calculate sample-weighted train and validation loss and accuracy. <br>

- **Updated the `test()` function to track both loss and accuracy.** <br>

- **Fixed the learning rate update in the `train()` function** <br>
The original implementation restores the current best weights when updating the learning rate. However, it only restored the best model weights without restoring the optimizer state, so the optimizer would continue running estimates of gradients correspond to the last model trajectory, not the older restored best weights. <br>
The revised implementation simply update learning rate without restoring the previous best model state, let the model to continue refining from its current state. <br>

- **Improved the best weights update in the `train()` function** <br>
The original implementation use the checkpoint corresponding to the highest accuracy as the best model state. <br>
The revised implementation changed to use checkpoint corresponding to the lowest validation loss as the best model state. <br>


6. `test.sh`
- **Removed duplicated '--model_type lrcn' argument** <br>

- **Added '--fr_per_vid 16' to the argument** <br>


7. `run_training.py`.
- Its functionality was duplicated by <run.py>, which already supports both training and evaluation modes. Its script also depended on a nonexistent `compose_dataloaders()` function. Hence it is obsoleted. <br>
The repository now uses <run.py> as its single command-line entry point. <br>


8. All Python modules were linted using Pylint
- Each module achieved a score of 10/10 individually. The combined scored showed 9.98/10 due to a cross-file "duplicate-code" warning. It was reviewed as intentional similarity between training and evaluation batch processing.

---

## Major Model Improvements

### 1. Temporally consistent spatial and temporal augmentation

The original pipeline invoked random transforms independently for every frame, allowing adjacent frames to receive different flips and translations. This introduced artificial motion and disrupted temporal coherence.

The revised pipeline stacks frames into a `[T,C,H,W]` clip tensor and invokes torchvision transforms v2 once per clip. RandomResizedCrop and RandomHorizontalFlip therefore use consistent spatial parameters across all frames. During training, clips longer than 16 frames use random temporal cropping so different temporal windows may be observed across epochs. Validation and testing use deterministic uniform sampling.

### 2. Pretrained R3D-18 architecture

The original repository included 3D-CNN normalization statistics and an R3D-specific collate function but did not instantiate a 3D model. The revised implementation adds a Kinetics-400-pretrained R3D-18 and replaces its classifier with a 51-class output layer.

Unlike LRCN, which extracts spatial features independently and delegates temporal learning to an LSTM, R3D-18 applies residual 3D convolutions over time, height and width, allowing motion and appearance to be learned jointly.

### 3. Vectorized LRCN feature extraction and sequence processing

The original LRCN forward pass processed one timestep at a time. For every frame, it separately called the ResNet backbone and then invoked the LSTM for a sequence of length one while manually passing the hidden and cell states to the next iteration. Although this recurrence was executable, it caused repeated model calls and did not efficiently use PyTorch's batched operations.

The revised implementation processes all genuine frames from the batch through the ResNet backbone in one vectorized operation. The resulting feature vectors are reconstructed into a tensor of shape `[B,T,feature_dim]` and passed through the LSTM in one call with `batch_first=True`.

For an input batch shaped `[B,T,C,H,W]`, valid frames are temporarily combined into:

`[total_valid_frames,C,H,W]`

The CNN produces:

`[total_valid_frames,feature_dim]`

The features are then restored to:

`[B,T,feature_dim]`

before being processed by the LSTM. This reduces repeated Python loops and model calls, improves GPU utilization, supports the `T=1` edge case, and makes the implementation more closely reflect the LRCN architecture.

### 4. Valid-length-aware LRCN processing

The revised LRCN receives the genuine length of each clip. Padded frames are excluded from ResNet feature extraction, and classification uses the LSTM output at each sample’s final genuine timestep rather than the end of the padded batch tensor.

---

## Experimental Results

| Model | Split policy | Best validation accuracy | Test accuracy |
|---|---|---:|---:|
| LRCN | Source-grouped | 48-49% | Not evaluated |
| R3D-18 | Source-grouped | 67–68% | approximately 60.48% |

R3D-18 substantially outperformed the LRCN baseline on the source-grouped split. Training accuracy approached 100%, while validation and test performance remained lower, indicating overfitting and a generalization gap rather than insufficient training capacity.
"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification 
report, and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def test(model, dataloader, criterion, device):  # pylint: disable=too-many-locals
    """
    Evaluate the model on the test dataset and compute overall accuracy.
    
    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
    
    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    total_loss, total_correct_preds = 0.0, 0
    processed_samples = 0
    targets, outputs = [], []

    with torch.no_grad():
        for x_batch, y_batch, lengths in tqdm(dataloader):
            if x_batch is None or y_batch is None or lengths is None:
                continue

            x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
            output = model(x_batch, lengths)
            loss = criterion(output, y_batch)
            pred = output.argmax(dim=1, keepdim=True)
            batch_size = y_batch.size(0)

            # Compute loss
            total_loss += loss.item() * batch_size

            # Compute correct predictions
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds

            # Compute processed samples
            processed_samples += batch_size

            outputs.extend(pred.view(-1).detach().cpu().numpy().tolist())
            targets.extend(y_batch.detach().cpu().numpy().tolist())

        if processed_samples == 0:
            raise RuntimeError(
                "No valid video samples were processed."
            )

        avg_loss = total_loss / processed_samples
        avg_accuracy = total_correct_preds / processed_samples

    return targets, outputs, avg_loss, avg_accuracy

def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding 
              confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return dict(zip(all_cats, confusion_mat))

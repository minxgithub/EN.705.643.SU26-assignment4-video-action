"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, 
          and a final fully-connected layer to produce class logits.
"""


import torch
from torch import nn
from torchvision import models


class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video 
    frame. An LSTM network is then used to model the temporal dynamics across the sequence of 
    frame features. Dropout is applied before the final fully-connected layer that produces 
    class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. 
                                     Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
                                   'resnet152'.
                                   Default is 'resnet34'.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes,  # pylint: disable=too-many-arguments, too-many-positional-arguments
                 pretrained=True, cnn_model='resnet34'):
        super().__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if cnn_model == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(pretrained=pretrained)
        elif cnn_model == 'resnet50':
            base_cnn = models.resnet50(pretrained=pretrained)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model == 'resnet152':
            base_cnn = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(
                'The input CNN backbone is not supported, please choose a valid ResNet variant.'
            )

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features

        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = nn.Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)

        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)

        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x, lengths):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        Use the CNN backbone to extract features from all time step first, then use LSTM to process
        the feature sequence (in temporal order) for final output 

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        # x: (B, T, C, H, W)
        batch_size, time_steps, channels, height, width = x.shape

        # Combine batch and time so the CNN processes all the frames together
        x = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        # CNN output: (B*T, feature_dim)
        features = self.base_model(x)

        # Restore the temporal structure for LSTM: (B, T, feature_dim)
        features = features.reshape(
            batch_size,
            time_steps,
            -1,
        )

        # Process the complete feature sequence in LSTM
        rnn_output, _ = self.rnn(features)

        # Use the rnn output (B, T, hidden_size) from the final valid timestep
        last_valid_indices = lengths-1
        batch_indices = torch.arange(batch_size, device=rnn_output.device)

        final_features = rnn_output[batch_indices, last_valid_indices, :]

        final_features = self.dropout(final_features)
        logits = self.fc(final_features)

        return logits

"""
GAN Model for Transaction Behavior Analysis
===========================================

This module implements a Generative Adversarial Network (GAN) for financial transaction behavior modeling.
It consists of a Generator and a Discriminator, designed to generate and evaluate synthetic financial transactions.

Dependencies:
-------------
- torch
- torch.nn as nn

Device Configuration:
---------------------
- The model automatically selects GPU (CUDA) if available; otherwise, it runs on CPU.

Classes:
--------
1. Generator:
    - Generates synthetic transactions using a combination of noise and transaction types.
    - Uses fully connected (FC) layers with ReLU and Tanh activations.

2. Discriminator:
    - Evaluates whether a given transaction is real or synthetic.
    - Uses LeakyReLU activation and Sigmoid output for classification.

"""

import torch
import torch.nn as nn

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    """
    Generator Model
    ----------------
    The Generator takes random noise and transaction types as input and produces synthetic transaction data.

    Parameters:
    -----------
    - noise_dim (int): Dimensionality of the random noise vector.
    - transaction_type_dim (int): Dimensionality of transaction type input.
    - output_dim (int): Dimensionality of the generated transaction data.

    Architecture:
    -------------
    - Input: Concatenation of noise and transaction type.
    - Three fully connected layers with ReLU activation.
    - Final output layer uses Tanh activation to scale output between -1 and 1.

    Forward Pass:
    -------------
    - Accepts noise and transaction types.
    - Passes through three layers.
    - Outputs generated transaction data.

    """
    def __init__(self, noise_dim, transaction_type_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim + transaction_type_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, noise, transaction_types):
        """
        Forward pass of the Generator.

        Inputs:
        -------
        - noise (torch.Tensor): Random noise vector of shape (batch_size, noise_dim).
        - transaction_types (torch.Tensor): Encoded transaction types of shape (batch_size, transaction_type_dim).

        Returns:
        --------
        - Generated synthetic transactions (torch.Tensor) of shape (batch_size, output_dim).
        """
        x = torch.cat((noise, transaction_types), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output scaled between -1 and 1
        return x


class Discriminator(nn.Module):
    """
    Discriminator Model
    -------------------
    The Discriminator evaluates whether a transaction is real or synthetic.

    Parameters:
    -----------
    - input_dim (int): Dimensionality of transaction data.
    - transaction_type_dim (int): Dimensionality of transaction type input.

    Architecture:
    -------------
    - Input: Concatenation of transaction data and transaction type.
    - Three fully connected layers with LeakyReLU activation.
    - Sigmoid activation at the output for binary classification.

    Forward Pass:
    -------------
    - Accepts transaction data and transaction types.
    - Passes through three layers.
    - Outputs probability score indicating whether the transaction is real or fake.

    """
    def __init__(self, input_dim, transaction_type_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim + transaction_type_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, transaction_types):
        """
        Forward pass of the Discriminator.

        Inputs:
        -------
        - data (torch.Tensor): Transaction data of shape (batch_size, input_dim).
        - transaction_types (torch.Tensor): Encoded transaction types of shape (batch_size, transaction_type_dim).

        Returns:
        --------
        - Probability score (torch.Tensor) indicating real (1) or fake (0) transaction.
        """
        x = torch.cat((data, transaction_types), dim=1)  # Ensure correct concatenation
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Probability score (0 to 1)
        return x

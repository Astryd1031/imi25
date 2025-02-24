import torch
import torch.nn as nn

# Generator Class
class Generator(nn.Module):
    """
    A Generator network for a Generative Adversarial Network (GAN).
    It generates synthetic transaction data conditioned on transaction types.

    Parameters:
    -----------
    noise_dim : int
        The size of the random noise vector used for data generation.
    transaction_type_dim : int
        The size of the one-hot encoded transaction type vector.
    output_dim : int
        The size of the generated transaction data (same as real transaction data).

    Layers:
    -------
    - Input: Concatenation of noise and transaction type vector.
    - Hidden Layers: Fully connected layers with ReLU activation.
    - Output Layer: Tanh activation to scale outputs between -1 and 1.

    Methods:
    --------
    forward(noise, transaction_types):
        Takes in a noise vector and transaction type vector, concatenates them,
        and passes them through the network to generate synthetic data.

    """

    def __init__(self, noise_dim, transaction_type_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + transaction_type_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()  # Output scaled between -1 and 1
        )

    def forward(self, noise, transaction_types):
        """
        Forward pass for the Generator.

        Parameters:
        -----------
        noise : torch.Tensor
            A batch of random noise vectors (batch_size, noise_dim).
        transaction_types : torch.Tensor
            A batch of one-hot encoded transaction type vectors (batch_size, transaction_type_dim).

        Returns:
        --------
        torch.Tensor
            A batch of generated synthetic transaction data (batch_size, output_dim).
        """
        x = torch.cat((noise, transaction_types), dim=1)  # Concatenate noise and transaction types
        return self.model(x)



class Discriminator(nn.Module):
    """
    A Discriminator network for a Generative Adversarial Network (GAN).
    It distinguishes between real and fake transaction data while conditioning on transaction types.

    Parameters:
    -----------
    input_dim : int
        The size of the transaction data vector.
    transaction_type_dim : int
        The size of the one-hot encoded transaction type vector.

    Layers:
    -------
    - Input: Concatenation of transaction data and transaction type vector.
    - Hidden Layers: Fully connected layers with LeakyReLU activation.
    - Embedding Layer: A hidden layer for extracting embeddings (128 dimensions).
    - Output Layer: Sigmoid activation to output probability of data being real.

    Methods:
    --------
    forward(data, transaction_types, return_embedding=False):
        Takes in transaction data and transaction type vector, concatenates them,
        and passes them through the network to determine real/fake probability.
        Optionally, returns the extracted embeddings.
    """

    def __init__(self, input_dim, transaction_type_dim, embedding_dim=128):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim + transaction_type_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embedding_dim)  # Embedding layer
        self.fc4 = nn.Linear(embedding_dim, 1)  # Output layer for classification
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, transaction_types, return_embedding=False):
        """
        Forward pass for the Discriminator.

        Parameters:
        -----------
        data : torch.Tensor
            A batch of transaction data (real or fake) (batch_size, input_dim).
        transaction_types : torch.Tensor
            A batch of one-hot encoded transaction type vectors (batch_size, transaction_type_dim).
        return_embedding : bool, optional
            If True, return both classification output and embeddings.

        Returns:
        --------
        torch.Tensor
            A batch of probabilities indicating whether the data is real (1) or fake (0).
        torch.Tensor (optional)
            A batch of extracted embeddings (batch_size, embedding_dim).
        """
        x = torch.cat((data, transaction_types), dim=1)  # Concatenate transaction data + type
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        embeddings = self.leaky_relu(self.fc3(x))  # Extract embeddings
        out = self.sigmoid(self.fc4(embeddings))  # Classification output

        return (out, embeddings) if return_embedding else out

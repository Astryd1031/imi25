import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from gan_class.gan_preprocess import preprocess_customer_data
from gan_class.small_gan_class import Generator, Discriminator
import numpy as np


def small_training(df):
    """
    Train a small GAN model for anomaly detection and calculate customer risk scores based on transaction data.

    Parameters:
    - df (pandas.DataFrame): The dataset containing customer transaction information. It is assumed to include numeric and boolean columns
      related to transaction details such as `Transaction_channel_Abm`, `Transaction_channel_Card`, etc., and customer-specific information.

    Returns:
    - customer_risk_data_small (list of dicts): A list containing customer IDs and their calculated AML risk scores based on anomaly detection.
    """

    # Preprocess customer transaction data
    transactions = preprocess_customer_data(df)

    # Select numeric columns (features for GAN model training)
    numeric_columns = transactions.select_dtypes(include=[np.number]).columns.tolist()

    # Set the device for model (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    noise_dim = 100  # Dimension of the noise vector input for the generator
    transaction_type_dim = 6  # Number of different transaction types (boolean features)
    output_dim = len(numeric_columns)  # Output dimensions based on the number of numeric features
    learning_rate = 0.002  # Learning rate for optimizer
    beta1 = 0.5  # Beta1 parameter for Adam optimizer
    input_dim = noise_dim + transaction_type_dim  # Combined input dimension for the generator

    # Optimizers for Generator and Discriminator
    generator = Generator(noise_dim, transaction_type_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim, transaction_type_dim).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Loss function for binary classification (real/fake)
    criterion = nn.BCELoss()

    # Check dataset size to determine the training strategy (single batch or mini-batch)
    dataset_size = len(transactions)

    # List of boolean transaction columns to determine transaction type
    boolean_columns = ['Transaction_channel_Abm', 'Transaction_channel_Card', 'Transaction_channel_Cheque',
                       'Transaction_channel_EFT', 'Transaction_channel_EMT', 'Transaction_channel_Wire']

    if dataset_size < 64:
        # If the dataset is small (less than 64 rows), train on the entire dataset at once
        real_data = torch.tensor(transactions[numeric_columns].values, dtype=torch.float32).to(device)
        transaction_types = torch.tensor(transactions[boolean_columns].values, dtype=torch.float32).to(device)

        # Generate noise input for the generator
        z = torch.randn(dataset_size, noise_dim).to(device)

        # Generate fake data from the generator
        fake_data = generator(z, transaction_types)

        # Train the Discriminator
        optimizer_d.zero_grad()

        # Real data loss (Discriminator classifies real data as "1")
        real_output = discriminator(real_data, transaction_types)
        real_loss = criterion(real_output, torch.ones(dataset_size, 1).to(device))

        # Fake data loss (Discriminator classifies fake data as "0")
        fake_output = discriminator(fake_data.detach(), transaction_types)
        fake_loss = criterion(fake_output, torch.zeros(dataset_size, 1).to(device))

        # Total loss for the Discriminator
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # Train the Generator
        optimizer_g.zero_grad()
        fake_output = discriminator(fake_data, transaction_types)
        g_loss = criterion(fake_output, torch.ones(dataset_size, 1).to(device))

        # Generator's goal is to make fake data look like real data (classify as "1")
        g_loss.backward()
        optimizer_g.step()

        # Anomaly detection: Use the Discriminator's output probabilities for anomaly detection
        real_probabilities = torch.sigmoid(real_output).cpu().detach().numpy()

        # Set an anomaly threshold based on the real data's output probabilities
        anomaly_threshold = real_probabilities.mean() - 2 * real_probabilities.std()
        anomalous_transactions = real_probabilities < anomaly_threshold

        # Aggregate anomaly counts per customer
        customer_anomalies = defaultdict(lambda: {'anomalous': 0, 'total': 0})

        for i in range(dataset_size):
            customer_id = transactions.iloc[i]['customer_id']  # Retrieve the customer ID
            customer_anomalies[customer_id]['total'] += 1
            if anomalous_transactions[i]:
                customer_anomalies[customer_id]['anomalous'] += 1

        # Calculate and store the final risk score for each customer
        customer_risk_data_small = [
            {'Customer_ID': customer_id, 'AML_Risk_Score': data['anomalous'] / data['total']}
            for customer_id, data in customer_anomalies.items()
        ]
    else:
        print("Dataset size is sufficient for batch training.")

    # Return the customer risk data (AML Risk Score for each customer)
    return customer_risk_data_small

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from gan_class.big_gan_class import Generator, Discriminator
from gan_class.gan_preprocess import preprocess_customer_data
import numpy as np

def big_training(df):
    """
    Trains a GAN-based anomaly detection model on preprocessed customer transaction data
    to assess behavioral patterns and detect potential AML (Anti-Money Laundering) risks.

    Parameters:
    df (pd.DataFrame): DataFrame containing raw transaction data before preprocessing.

    Returns:
    list[dict]: A list of dictionaries, each containing a customer's ID and their calculated AML risk score.
    """
    # Load preprocessed transactions data
    transactions = preprocess_customer_data(df)

    # Define boolean columns representing transaction channels
    boolean_columns = ['Transaction_channel_Abm', 'Transaction_channel_Card', 'Transaction_channel_Cheque',
                       'Transaction_channel_EFT', 'Transaction_channel_EMT', 'Transaction_channel_Wire']

    # Extract numeric columns from the dataset
    numeric_columns = transactions.select_dtypes(include=[np.number]).columns.tolist()

    # Set up the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    noise_dim = 100  # Dimensionality of the random noise vector
    transaction_type_dim = 6  # Number of transaction channel types
    output_dim = len(numeric_columns)  # Number of features in the dataset
    num_epochs = 100  # Number of training epochs
    batch_size = 64  # Size of each batch during training
    learning_rate = 0.0002  # Learning rate for optimizers
    beta1 = 0.5  # Beta1 parameter for Adam optimizer
    input_dim = noise_dim + transaction_type_dim  # Generator input dimension

    # Initialize the Generator and Discriminator models
    generator = Generator(noise_dim, transaction_type_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim, transaction_type_dim).to(device)

    # Set up Adam optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Define the Binary Cross-Entropy Loss function
    criterion = nn.BCELoss()

    # Dictionary to track anomalies per customer
    customer_anomalies = defaultdict(lambda: {'anomalous': 0, 'total': 0})

    # Training loop
    for epoch in range(num_epochs):
        d_loss, g_loss = 0.0, 0.0  # Track losses
        total_anomalous_transactions, total_transactions = 0, 0  # Track anomalies

        # Iterate over batches of transactions
        for i in range(0, len(transactions), batch_size):
            end_idx = min(i + batch_size, len(transactions))
            current_batch_size = end_idx - i

            # Skip batches that do not match the expected batch size
            if current_batch_size != batch_size:
                continue

            # Prepare real transaction data and transaction types
            real_data = torch.tensor(transactions.iloc[i:end_idx][numeric_columns].values, dtype=torch.float32).to(device)
            transaction_types = torch.tensor(transactions.iloc[i:end_idx][boolean_columns].values, dtype=torch.float32).to(device)

            # Generate synthetic data
            z = torch.randn(current_batch_size, noise_dim).to(device)
            fake_data = generator(z, transaction_types)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_output = discriminator(real_data, transaction_types)
            fake_output = discriminator(fake_data.detach(), transaction_types)
            real_loss = criterion(real_output, torch.ones(current_batch_size, 1).to(device))
            fake_loss = criterion(fake_output, torch.zeros(current_batch_size, 1).to(device))
            d_loss_batch = real_loss + fake_loss
            d_loss_batch.backward()
            optimizer_d.step()
            d_loss += d_loss_batch.item()

            # Train Generator
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_data, transaction_types)
            g_loss_batch = criterion(fake_output, torch.ones(current_batch_size, 1).to(device))
            g_loss_batch.backward()
            optimizer_g.step()
            g_loss += g_loss_batch.item()

            # Anomaly detection using discriminator scores
            real_probabilities = torch.sigmoid(real_output)
            anomaly_threshold = real_probabilities.mean() - 2 * real_probabilities.std()
            anomalous_transactions = real_probabilities < anomaly_threshold

            # Update anomaly statistics
            total_anomalous_transactions += anomalous_transactions.sum().item()
            total_transactions += real_data.size(0)

            # Aggregate anomalies per customer
            for j in range(current_batch_size):
                customer_id = transactions.iloc[i + j]['customer_id']
                if anomalous_transactions[j]:
                    customer_anomalies[customer_id]['anomalous'] += 1
                customer_anomalies[customer_id]['total'] += 1

        # Compute average losses for the epoch
        d_loss /= (len(transactions) // batch_size)
        g_loss /= (len(transactions) // batch_size)

        # Compute overall anomaly rate
        anomalous_percentage = total_anomalous_transactions / total_transactions
        aml_risk_score = anomalous_percentage  # Assign AML risk score based on anomaly percentage

    # Aggregate customer risk scores
    customer_risk_data = [{'Customer_ID': cid, 'AML_Risk_Score': data['anomalous'] / data['total']}
                           for cid, data in customer_anomalies.items()]

    return customer_risk_data

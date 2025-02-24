from GAN_advanced_task_class.big_gan_class import Generator
from GAN_advanced_task_class.gan_preprocess import preprocess_customer_data

"""
Implementations for neural discriminator training
def kbig_training(df, csv_path="customer_foundation_model.csv"):
    
    Trains a GAN-based anomaly detection model and appends 128D customer embeddings to a CSV file.

    Parameters:
    df (pd.DataFrame): DataFrame containing raw transaction data before preprocessing.
    csv_path (str): Path to the CSV file where embeddings will be appended.

    Returns:
    pd.DataFrame: DataFrame with appended customer embeddings.
    
    transactions = preprocess_customer_data(df)

    boolean_columns = ['Transaction_channel_Abm', 'Transaction_channel_Card', 'Transaction_channel_Cheque',
                       'Transaction_channel_EFT', 'Transaction_channel_EMT', 'Transaction_channel_Wire']
    numeric_columns = transactions.select_dtypes(include=[np.number]).columns.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_dim = 100
    transaction_type_dim = 6
    output_dim = len(numeric_columns)
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.0002
    beta1 = 0.5
    input_dim = noise_dim + transaction_type_dim

    generator = Generator(noise_dim, transaction_type_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim, transaction_type_dim).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    criterion = nn.BCELoss()

    customer_embeddings = defaultdict(list)

    for epoch in range(num_epochs):
        for i in range(0, len(transactions), batch_size):
            end_idx = min(i + batch_size, len(transactions))
            current_batch_size = end_idx - i
            if current_batch_size != batch_size:
                continue

            real_data = torch.tensor(transactions.iloc[i:end_idx][numeric_columns].values, dtype=torch.float32).to(device)
            transaction_types = torch.tensor(transactions.iloc[i:end_idx][boolean_columns].values, dtype=torch.float32).to(device)

            z = torch.randn(current_batch_size, noise_dim).to(device)
            fake_data = generator(z, transaction_types)

            optimizer_d.zero_grad()
            real_output = discriminator(real_data, transaction_types, return_embedding=True)
            fake_output = discriminator(fake_data.detach(), transaction_types)

            real_loss = criterion(real_output["score"], torch.ones(current_batch_size, 1).to(device))
            fake_loss = criterion(fake_output["score"], torch.zeros(current_batch_size, 1).to(device))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_output = discriminator(fake_data, transaction_types)
            g_loss = criterion(fake_output["score"], torch.ones(current_batch_size, 1).to(device))
            g_loss.backward()
            optimizer_g.step()

            embeddings = real_output["embedding"].detach().cpu().numpy()
            for j in range(current_batch_size):
                customer_id = transactions.iloc[i + j]['customer_id']
                customer_embeddings[customer_id].append(embeddings[j])

    final_embeddings = {cid: np.mean(np.vstack(embeds), axis=0) for cid, embeds in customer_embeddings.items()}
    embeddings_df = pd.DataFrame.from_dict(final_embeddings, orient='index')

    embeddings_df.index.name = 'Customer_ID'

    # Check if file exists to decide on writing header
    file_exists = os.path.isfile(csv_path)

    # Append new data to existing CSV file
    embeddings_df.to_csv(csv_path, mode='a', header=not file_exists)

    return embeddings_df
"""

import os
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict, deque
from scipy.sparse import csr_matrix
import torch.optim as optim
import torch.nn as nn

def big_training(df, csv_path="customer_foundation_model.csv"):
    transactions = preprocess_customer_data(df)

    boolean_columns = ['Transaction_channel_Abm', 'Transaction_channel_Card', 'Transaction_channel_Cheque',
                       'Transaction_channel_EFT', 'Transaction_channel_EMT', 'Transaction_channel_Wire']
    numeric_columns = transactions.select_dtypes(include=[np.number]).columns.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_dim = 100
    transaction_type_dim = len(boolean_columns)
    output_dim = len(numeric_columns)
    num_epochs = 100
    batch_size = 128
    learning_rate = 0.0002
    beta1 = 0.5
    input_dim = noise_dim + transaction_type_dim

    generator = Generator(noise_dim, transaction_type_dim, output_dim).to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    customer_embeddings = defaultdict(list)

    real_features = deque(maxlen=3000)
    fake_features = deque(maxlen=3000)
    labels = deque(maxlen=3000)

    xgb_model = None
    xgb_update_threshold = 2000  # Train when at least 2000 labeled samples accumulate

    for epoch in range(num_epochs):
        for i in range(0, len(transactions), batch_size):
            end_idx = min(i + batch_size, len(transactions))
            current_batch_size = end_idx - i
            if current_batch_size != batch_size:
                continue

            real_data = torch.tensor(transactions.iloc[i:end_idx][numeric_columns].values, dtype=torch.float32).to(device)
            transaction_types = csr_matrix(transactions.iloc[i:end_idx][boolean_columns].values)

            z = torch.randn(current_batch_size, noise_dim).to(device)
            fake_data = generator(z, torch.tensor(transaction_types.toarray(), dtype=torch.float32).to(device))

            real_features.append(real_data.cpu().numpy())
            fake_features.append(fake_data.detach().cpu().numpy())
            labels.extend([1] * current_batch_size + [0] * current_batch_size)

            # Train XGBoost dynamically when at least xgb_update_threshold new samples are available
            if len(labels) >= xgb_update_threshold:
                X = np.vstack(real_features + fake_features)
                y = np.array(labels)
                min_length = min(X.shape[0], y.shape[0])
                X = X[:min_length]
                y = y[:min_length]
                dtrain = xgb.DMatrix(X, label=y)
                xgb_model = xgb.train(
                    {"objective": "binary:logistic", "eval_metric": "logloss"},
                    dtrain,
                    num_boost_round=50
                )
                labels.clear()  # Reset after training

            optimizer_g.zero_grad()

            if xgb_model:
                fake_data_np = fake_data.detach().cpu().numpy()
                dtest = xgb.DMatrix(fake_data_np)
                xgb_predictions = xgb_model.predict(dtest)
                xgb_predictions = torch.tensor(xgb_predictions, dtype=torch.float32, requires_grad=True).to(device).view(-1, 1)

                g_loss = criterion(xgb_predictions, torch.ones(current_batch_size, 1).to(device))
                g_loss.backward()
                optimizer_g.step()

            embeddings = real_data.detach().cpu().numpy()
            for j in range(current_batch_size):
                customer_id = transactions.iloc[i + j]['customer_id']
                customer_embeddings[customer_id].append(embeddings[j])

    # Compute mean embeddings per customer
    final_embeddings = {cid: np.mean(np.vstack(embeds), axis=0) for cid, embeds in customer_embeddings.items()}
    embeddings_df = pd.DataFrame.from_dict(final_embeddings, orient='index')
    embeddings_df.index.name = 'Customer_ID'

    # Save or update existing CSV
    if os.path.isfile(csv_path):
        existing_df = pd.read_csv(csv_path, index_col=0)
        if existing_df.shape[0] != embeddings_df.shape[0]:
            reduced_embeddings = embeddings_df.values  # Use raw embeddings when customer count changes
        else:
            reduced_embeddings = embeddings_df.values
    else:
        reduced_embeddings = embeddings_df.values  # No PCA, direct saving

    reduced_df = pd.DataFrame(reduced_embeddings, index=embeddings_df.index)

    # Save reduced embeddings
    file_exists = os.path.isfile(csv_path)
    reduced_df.to_csv(csv_path, mode='a', header=not file_exists)
    return reduced_df








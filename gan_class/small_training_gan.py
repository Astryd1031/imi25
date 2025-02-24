import torch.optim as optim
from gan_class.gan_preprocess import preprocess_customer_data
from gan_class.small_gan_class import Generator

""""
def ksmall_training(df, save_path="customer_foundation_model.csv"):
    
    Train a small GAN model for anomaly detection and extract customer embeddings.
    Appends embeddings per unique customer ID (one row per customer) to an existing CSV.

    Parameters:
    - df (pd.DataFrame): The dataset containing customer transaction information.
    - save_path (str): File path to save the customer embeddings CSV.

    Returns:
    - customer_risk_data_small (list of dicts): A list containing customer IDs and their calculated AML risk scores.
    - customer_embeddings (pandas.DataFrame): DataFrame of unique customer IDs and their aggregated embeddings.
    

    transactions = preprocess_customer_data(df)
    numeric_columns = transactions.select_dtypes(include=[np.number]).columns.tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_dim = 100
    transaction_type_dim = 6
    output_dim = len(numeric_columns)
    learning_rate = 0.002
    beta1 = 0.5

    generator = Generator(noise_dim, transaction_type_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim, transaction_type_dim, embedding_dim=128).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    criterion = nn.BCELoss()
    dataset_size = len(transactions)

    boolean_columns = ['Transaction_channel_Abm', 'Transaction_channel_Card', 'Transaction_channel_Cheque',
                       'Transaction_channel_EFT', 'Transaction_channel_EMT', 'Transaction_channel_Wire']

    updated_df = None

    if dataset_size < 64:
        real_data = torch.tensor(transactions[numeric_columns].values, dtype=torch.float32).to(device)
        transaction_types = torch.tensor(transactions[boolean_columns].values, dtype=torch.float32).to(device)

        z = torch.randn(dataset_size, noise_dim).to(device)
        fake_data = generator(z, transaction_types)

        optimizer_d.zero_grad()
        real_output, real_embeddings = discriminator(real_data, transaction_types, return_embedding=True)
        real_loss = criterion(real_output, torch.ones(dataset_size, 1).to(device))

        fake_output, _ = discriminator(fake_data.detach(), transaction_types, return_embedding=True)
        fake_loss = criterion(fake_output, torch.zeros(dataset_size, 1).to(device))

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        fake_output, _ = discriminator(fake_data, transaction_types, return_embedding=True)
        g_loss = criterion(fake_output, torch.ones(dataset_size, 1).to(device))
        g_loss.backward()
        optimizer_g.step()

        customer_embeddings = defaultdict(list)
        for i in range(dataset_size):
            customer_id = transactions.iloc[i]['customer_id']
            customer_embeddings[customer_id].append(real_embeddings[i].cpu().detach().numpy())

        aggregated_embeddings = {cid: np.mean(embeds, axis=0) for cid, embeds in customer_embeddings.items()}
        new_embeddings_df = pd.DataFrame.from_dict(aggregated_embeddings, orient='index').reset_index()
        new_embeddings_df.columns = ["Customer_ID"] + [f"dim_{i}" for i in range(128)]

        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            existing_df.set_index("Customer_ID", inplace=True)
            existing_df.update(new_embeddings_df.set_index("Customer_ID"))
            updated_df = pd.concat([existing_df, new_embeddings_df.set_index("Customer_ID")], axis=0).reset_index()
            updated_df.drop_duplicates(subset=["Customer_ID"], keep="last", inplace=True)
        else:
            updated_df = new_embeddings_df

        updated_df.to_csv(save_path, index=False)
        print(f"Updated embeddings saved to {save_path}")

    return updated_df
"""
import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from collections import defaultdict

def small_training(df, save_path="customer_foundation_model.csv"):
    """
    Train a small GAN model with an XGBoost discriminator for anomaly detection.
    Omit PCA for dimensionality reduction.
    """
    transactions = preprocess_customer_data(df)
    numeric_columns = transactions.select_dtypes(include=[np.number]).columns.tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_dim = 100
    transaction_type_dim = 6
    output_dim = len(numeric_columns)

    generator = Generator(noise_dim, transaction_type_dim, output_dim).to(device)
    boolean_columns = ['Transaction_channel_Abm', 'Transaction_channel_Card',
                       'Transaction_channel_Cheque', 'Transaction_channel_EFT',
                       'Transaction_channel_EMT', 'Transaction_channel_Wire']

    dataset_size = len(transactions)
    updated_df = None

    if dataset_size < 64:
        # Prepare Real Data
        real_data = transactions[numeric_columns].values
        transaction_types = transactions[boolean_columns].values

        # Generate Fake Data
        z = torch.randn(dataset_size, noise_dim).to(device)
        transaction_types_tensor = torch.tensor(transaction_types, dtype=torch.float32).to(device)
        fake_data = generator(z, transaction_types_tensor).cpu().detach().numpy()

        # Prepare Labels (1 for real, 0 for fake)
        X_train = np.vstack((real_data, fake_data))
        y_train = np.hstack((np.ones(len(real_data)), np.zeros(len(fake_data))))

        # Train XGBoost Discriminator with Dynamic Update Frequency
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        xgb_model.fit(X_train, y_train)

        # Extract Feature Importance as Embeddings
        real_embeddings = xgb_model.apply(real_data)

        # Ensure real_embeddings is 2D
        if real_embeddings.ndim == 1:
            real_embeddings = real_embeddings.reshape(-1, 1)

        # Aggregate Embeddings per Customer
        customer_embeddings = defaultdict(list)
        for i, customer_id in enumerate(transactions['customer_id']):
            customer_embeddings[customer_id].append(real_embeddings[i])

        aggregated_embeddings = {cid: np.mean(embeds, axis=0) for cid, embeds in customer_embeddings.items()}
        new_embeddings_df = pd.DataFrame.from_dict(aggregated_embeddings, orient='index').reset_index()
        new_embeddings_df.columns = ["Customer_ID"] + [f"dim_{i}" for i in range(real_embeddings.shape[1])]

        reduced_df = new_embeddings_df.copy()

        # Merge with Existing Embeddings Efficiently
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path).set_index("Customer_ID")
            new_indexed_df = reduced_df.set_index("Customer_ID")

            # Update existing rows without unnecessary modifications
            existing_df.update(new_indexed_df)

            # Combine unique rows and remove duplicates
            updated_df = pd.concat([existing_df, new_indexed_df]).reset_index().drop_duplicates(subset=["Customer_ID"], keep="last")
        else:
            updated_df = reduced_df

        updated_df.to_csv(save_path, index=False)
        print(f"Updated embeddings saved to {save_path}")

    return updated_df

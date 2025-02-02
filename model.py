from data import userdf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = userdf.copy()
data['customer_id'] = userdf['customer_id']  # Preserve customer_id


# Preprocessing Function
def preprocess_data(data):
    # Exclude non-numeric columns
    exclude_cols = ['transaction_id', 'customer_id']
    data = data.drop(columns=[col for col in exclude_cols if col in data.columns], errors='ignore')

    # Handle missing values
    if 'amount_cad' in data.columns:
        data['amount_cad'].fillna(data['amount_cad'].median(), inplace=True)

    for col in ['Transaction_channel', 'trans_type']:
        if col in data.columns:
            data[col].fillna('Unknown', inplace=True)

    # Encode categorical features
    categorical_cols = ['debit_credit', 'cash_indicator', 'country_x', 'province_x', 'city_x',
                        'Transaction_channel', 'trans_type', 'ecommerce_ind', 'country_y', 'province_y', 'city_y']
    categorical_cols = [col for col in categorical_cols if col in data.columns]

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Normalize numerical features
    numerical_cols = ['amount_cad']
    numerical_cols = [col for col in numerical_cols if col in data.columns]

    if numerical_cols:
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Convert temporal features
    temporal_cols = ['transaction_date', 'established_date', 'onboard_date']
    for col in temporal_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            data[col] = data[col].view('int64') // 10 ** 9  # Convert to UNIX timestamp

    if 'transaction_time' in data.columns:
        data['transaction_time'] = pd.to_timedelta(data['transaction_time'], errors='coerce')
        data['transaction_time'] = data['transaction_time'].dt.total_seconds().fillna(0)

    # Ensure all columns are numeric
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    return data


preprocessed_data = preprocess_data(data)
preprocessed_data = preprocessed_data.astype(np.float32)

# Define GAN parameters
latent_dim = 100
data_dim = preprocessed_data.shape[1]


# Generator
def build_generator(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation="relu", input_dim=input_dim),
        Dense(256, activation="relu"),
        Dense(512, activation="relu"),
        Dense(output_dim, activation="tanh")
    ])
    return model


# Discriminator
def build_discriminator(data_dim):
    model = Sequential([
        Dense(256, input_dim=data_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model


# Compile models
generator = build_generator(latent_dim, data_dim)
discriminator = build_discriminator(data_dim)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')


# Training function
def train_gan(generator, discriminator, gan, data, epochs=5000, batch_size=32):
    batch_size = min(batch_size, len(data))
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))

    for epoch in range(epochs):
        real_data = data.sample(batch_size).values
        noise = tf.random.normal([batch_size, latent_dim])
        fake_data = generator(noise)

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        noise = tf.random.normal([batch_size, latent_dim])
        g_loss = gan.train_on_batch(noise, real_labels)

    return generator


# Anomaly Detection
def detect_anomalies(generator, transactions, threshold=None, batch_size=32):
    num_transactions = len(transactions)
    anomalies = []
    scores = []

    for i in range(0, num_transactions, batch_size):
        batch = transactions[i:i + batch_size]
        noise = tf.random.normal([len(batch), latent_dim])
        generated_transactions = generator.predict(noise, verbose=0)

        for txn, gen_txn in zip(batch, generated_transactions):
            score = np.mean(np.square(txn - gen_txn))
            scores.append(score)

            if threshold is None:
                threshold = np.percentile(scores, 95)  # Dynamic threshold at 95th percentile

            if score > threshold:
                anomalies.append(txn)

    return anomalies, scores, threshold


# Train GAN
trained_generator = train_gan(generator, discriminator, gan, preprocessed_data, epochs=5000, batch_size=32)

# Run anomaly detection
anomalous_transactions, anomaly_scores, threshold = detect_anomalies(trained_generator, preprocessed_data.values)

print(f"There are {len(anomalous_transactions)} anomalous transactions detected.")

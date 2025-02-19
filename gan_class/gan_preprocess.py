import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_customer_data(df):
    """
    Preprocesses the customer data for machine learning or analysis.

    Steps:
    1. Drops duplicate rows from the dataframe.
    2. Converts date columns to datetime format.
    3. Extracts additional time-based features (hour, day of the week, month).
    4. Applies Label Encoding to high-cardinality categorical columns.
    5. Ensures expected values exist in 'Transaction_channel' and encodes them as categories.
    6. One-Hot Encodes low-cardinality categorical columns.
    7. Converts numeric columns to optimal types (float32).
    8. Fills missing values for numeric columns with default values.
    9. Normalizes numeric columns using Min-Max scaling.
    10. Applies Label Encoding to additional categorical columns.
    11. Extracts hour and day of the week from datetime columns for feature extraction.
    12. Drops unnecessary datetime columns.
    13. Converts boolean columns to 0/1 format.

    Parameters:
    - df: pandas DataFrame containing the raw customer transaction data.

    Returns:
    - df: pandas DataFrame with preprocessed data, ready for analysis or modeling.
    """

    # Step 1: Drop duplicate rows (if any exist)
    df = df.drop_duplicates()

    # Step 2: Convert date columns to datetime format
    date_columns = ['transaction_date', 'onboard_date', 'established_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Step 3: Extract additional time-based features (hour, day_of_week, month)
    if 'transaction_time' in df.columns:
        df['hour'] = df['transaction_time'].apply(lambda x: pd.to_datetime(x).hour)

    if 'transaction_date' in df.columns:
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['month'] = df['transaction_date'].dt.month

    # Step 4: Convert categorical features with high cardinality using Label Encoding
    high_cardinality_cols = ['merchant_category', 'industry_code', 'industry']
    for col in high_cardinality_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")  # Fill missing values
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Step 5: Ensure all possible 'Transaction_channel' values exist in the dataset
    expected_channels = ['Abm','Card', 'Cheque', 'EFT', 'EMT', 'Wire']
    df['Transaction_channel'] = df['Transaction_channel'].astype(pd.CategoricalDtype(categories=expected_channels))

    # Step 6: One-Hot Encode Low-Cardinality Categorical Variables
    low_cardinality_cols = ['debit_credit_debit', 'cash_indicator_True', 'ecommerce_ind_True','Transaction_channel_Abm',
                            'Transaction_channel_Card', 'Transaction_channel_Cheque', 'Transaction_channel_EFT',
                            'Transaction_channel_EMT', 'Transaction_channel_Wire']
    df = pd.get_dummies(df, columns=[col for col in low_cardinality_cols if col in df.columns], drop_first=True)

    # One-Hot Encode 'Transaction_channel'
    if 'Transaction_channel' in df.columns:
        df = pd.get_dummies(df, columns=['Transaction_channel'], drop_first=False)

    # Step 7: Convert numeric columns to optimal types (float32)
    numeric_cols = ['amount_cad', 'employee_count', 'sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype(np.float32)  # Downcast to save memory

    # Step 8: Fill missing values for numeric columns
    df.fillna({'employee_count': 0, 'sales': 0}, inplace=True)

    # Step 9: Normalize numerical values using Min-Max Scaling
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Step 10: Apply label encoding to categorical columns
    categorical_columns = ['debit_credit', 'cash_indicator', 'country_x', 'province_x', 'city_x',
                           'country_y', 'province_y', 'city_y', 'ecommerce_ind']
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure they are strings
        label_encoders[col] = le

    # Step 11: Handle datetime columns (extract hour and day of week)
    df['hour'] = df['transaction_time'].apply(
        lambda x: int(x.split(':')[0]) if isinstance(x, str) else -1)
    df['day_of_week'] = df['transaction_date'].dt.dayofweek

    # Step 12: Drop unnecessary datetime columns
    df = df.drop(
        columns=['transaction_date', 'transaction_time', 'established_date', 'onboard_date'])

    # Step 13: Convert boolean columns to 0/1 format
    boolean_columns = ['Transaction_channel_Abm', 'Transaction_channel_Card', 'Transaction_channel_Cheque',
                       'Transaction_channel_EFT', 'Transaction_channel_EMT', 'Transaction_channel_Wire']

    df[boolean_columns] = df[boolean_columns].astype(int)

    return df

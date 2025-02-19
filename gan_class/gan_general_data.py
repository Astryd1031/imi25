"""
Customer Transaction Data Consolidation Script
===============================================

This script consolidates transaction data from multiple sources, enriches it with customer KYC data,
and generates a final dataset for analysis.

Features:
---------
- Loads transaction data from different channels (ABM, Card, Cheque, EFT, EMT, Wire).
- Merges KYC data with industry classifications.
- Standardizes transaction data fields for consistency.
- Outputs a consolidated dataset to a CSV file.
- Provides a function `get_customer_data(customer_id)` to fetch transactions for a specific customer.

"""

import os
import pandas as pd

# Define the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load datasets (limited to 10,000 rows for demonstration, except wire which is 100,000)
datasets = {
    "abm": "abm.csv",
    "card": "card.csv",
    "cheque": "cheque.csv",
    "eft": "eft.csv",
    "emt": "emt.csv",
    "kyc": "kyc.csv",
    "kycd": "kyc_industry_codes.csv",
    "wire": "wire.csv",
}

# Read CSV files into Pandas DataFrames
dataframes = {key: pd.read_csv(os.path.join(PROJECT_ROOT, file)).iloc[:10000] for key, file in datasets.items()}
dataframes["wire"] = pd.read_csv(os.path.join(PROJECT_ROOT, "wire.csv")).iloc[:100000]  # More rows for Wire dataset

# Ensure 'industry_code' in KYC and KYCD is treated as a string for consistency
dataframes["kyc"]["industry_code"] = dataframes["kyc"]["industry_code"].astype(str)
dataframes["kycd"]["industry_code"] = dataframes["kycd"]["industry_code"].astype(str)

# Merge KYC data with industry classification
kyc = dataframes["kyc"].merge(dataframes["kycd"], on="industry_code", how="left")

# Debugging: Verify if 'industry' column exists after merging
if "industry" not in kyc.columns:
    print("Error: 'industry' column not found in KYC data after merge.")
    print("Columns in KYC data:", kyc.columns.tolist())
    print("Columns in KYCD data:", dataframes["kycd"].columns.tolist())
    exit()

# Add transaction channel and ID mapping to each transaction dataset
channel_mappings = {
    "abm": "ABM",
    "card": "Card",
    "cheque": "Cheque",
    "eft": "EFT",
    "emt": "EMT",
    "wire": "Wire",
}

id_mappings = {
    "abm": "abm_id",
    "card": "card_trxn_id",
    "cheque": "cheque_id",
    "eft": "eft_id",
    "emt": "emt_id",
    "wire": "wire_id",
}

for key, df in dataframes.items():
    if key in channel_mappings:
        df["Transaction_channel"] = channel_mappings[key]
        df["chanel_id"] = df[id_mappings[key]]  # Assign corresponding transaction ID column

# Add missing 'merchant_category' column to datasets that lack it
for key in ["abm", "cheque", "eft", "emt", "wire"]:
    dataframes[key]["merchant_category"] = None

# Standardize 'debit_credit' column in EMT dataset
dataframes["emt"]["debit_credit"] = dataframes["emt"]["debit_credit"].map({"D": "debit", "C": "credit"})

# Combine all transaction datasets
transactions = pd.concat(
    [dataframes["abm"], dataframes["card"], dataframes["cheque"], dataframes["eft"], dataframes["emt"], dataframes["wire"]],
    ignore_index=True,
)

# Merge transactions with KYC data
consolidated_data = transactions.merge(kyc, on="customer_id", how="left")

# Drop redundant transaction ID columns
columns_to_drop = list(id_mappings.values())  # List of transaction ID columns
consolidated_data = consolidated_data.drop(columns=columns_to_drop)

# Define final columns for output
final_columns = [
    "customer_id", "amount_cad", "debit_credit", "cash_indicator", "country_x",
    "province_x", "city_x", "transaction_date", "transaction_time",
    "Transaction_channel", "chanel_id", "merchant_category",
    "ecommerce_ind", "country_y", "province_y", "city_y", "industry_code",
    "employee_count", "sales", "established_date", "onboard_date", "industry"
]

# Check for missing columns before finalizing
missing_columns = [col for col in final_columns if col not in consolidated_data.columns]
if missing_columns:
    print("Error: The following columns are missing in consolidated_data:", missing_columns)
    exit()

# Select the final columns
final_data = consolidated_data[final_columns]

# Save the consolidated data to a CSV file
output_file = "consolidated_customer_transactions.csv"
final_data.to_csv(output_file, index=False)


def get_customer_data(customer_id):
    """
    Retrieves transaction data for a given customer.

    Parameters:
    -----------
    - customer_id (str): Unique identifier for the customer.

    Returns:
    --------
    - pd.DataFrame: A filtered dataset containing transactions associated with the given customer ID.

    Notes:
    ------
    - If the customer ID does not exist in the dataset, an empty DataFrame is returned.
    - The dataset includes transaction history and KYC details.
    """
    return final_data[final_data["customer_id"] == customer_id]

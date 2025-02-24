import os
import time
import pandas as pd
from gan_class.big_training_gan import big_training
from gan_class.small_training_gan import small_training
"""
This is the code for customer foundation model representation task. 
Please note that the csv file will be created
within the class structure.
"""
# Define project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load datasets with limits
DATASETS = {
    "abm": ("abm.csv", None),
    "card": ("card.csv",None),
    "cheque": ("cheque.csv",None),
    "eft": ("eft.csv", None),
    "emt": ("emt.csv",None),
    "kyc": ("kyc.csv", None),  # No limit for KYC
    "kycd": ("kyc_industry_codes.csv", None),
    "wire": ("wire.csv", None),  # Higher limit for wire transactions
}

# Read CSV files into Pandas DataFrames
dataframes = {
    key: pd.read_csv(os.path.join(PROJECT_ROOT, file)).iloc[:limit] if limit else pd.read_csv(
        os.path.join(PROJECT_ROOT, file))
    for key, (file, limit) in DATASETS.items()
}

# Convert industry_code to string for consistency
dataframes["kyc"]["industry_code"] = dataframes["kyc"]["industry_code"].astype(str)
dataframes["kycd"]["industry_code"] = dataframes["kycd"]["industry_code"].astype(str)

# Merge KYC with industry classification
dataframes["kyc"] = dataframes["kyc"].merge(dataframes["kycd"], on="industry_code", how="left")

# Define mappings for transaction datasets
CHANNEL_MAPPINGS = {
    "abm": "ABM", "card": "Card", "cheque": "Cheque",
    "eft": "EFT", "emt": "EMT", "wire": "Wire"
}
ID_MAPPINGS = {
    "abm": "abm_id", "card": "card_trxn_id", "cheque": "cheque_id",
    "eft": "eft_id", "emt": "emt_id", "wire": "wire_id"
}

# Process transaction datasets
for key, df in dataframes.items():
    if key in CHANNEL_MAPPINGS:
        df["Transaction_channel"] = CHANNEL_MAPPINGS[key]
        df["chanel_id"] = df[ID_MAPPINGS[key]]
        if key in ["abm", "cheque", "eft", "emt", "wire"]:
            df["merchant_category"] = None  # Add missing column

# Standardize 'debit_credit' column in EMT dataset
dataframes["emt"]["debit_credit"] = dataframes["emt"]["debit_credit"].map({"D": "debit", "C": "credit"})

# Consolidate transaction data
transactions = pd.concat([dataframes[key] for key in CHANNEL_MAPPINGS], ignore_index=True)

# Merge transactions with KYC data
consolidated_data = transactions.merge(dataframes["kyc"], on="customer_id", how="left")

transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], format="%Y-%m-%d", errors="coerce")

# Drop redundant transaction ID columns
consolidated_data.drop(columns=ID_MAPPINGS.values(), inplace=True)

# Define final columns
FINAL_COLUMNS = [
    "customer_id", "amount_cad", "debit_credit", "cash_indicator", "country_x",
    "province_x", "city_x", "transaction_date", "transaction_time",
    "Transaction_channel", "chanel_id", "merchant_category", "ecommerce_ind",
    "country_y", "province_y", "city_y", "industry_code", "employee_count",
    "sales", "established_date", "onboard_date", "industry"
]

# Validate missing columns
missing_columns = [col for col in FINAL_COLUMNS if col not in consolidated_data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in consolidated_data: {missing_columns}")



# Select final columns
final_data = consolidated_data[FINAL_COLUMNS]

final_data = final_data[final_data["transaction_date"] >= "2020-01-01"]

# Save to CSV
final_data.to_csv("consolidated_customer_transactions.csv", index=False)

def compute_gan_scores():
    """
    Computes GAN-based risk scores for all customers in the KYC dataset.
    """
    # Create a dictionary to cache data for each customer
    customer_data = {customer_id: final_data[final_data["customer_id"] == customer_id]
                     for customer_id in final_data["customer_id"].unique()}

    results = []

    for customer_id, df in customer_data.items():
        if df.empty:
            score = None
        else:
            score = small_training(df) if df.shape[0] < 129 else big_training(df)

        results.append({"customer_id": customer_id, "gan_score": score})

    # Convert results to DataFrame
    scores_df = pd.DataFrame(results)

    # Optionally, return or save scores_df
    return scores_df


if __name__ == "__main__":
    start_time = time.time()
    compute_gan_scores()
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    with open('customer_foundation_model.csv', 'a') as file:
        file.write('\n')
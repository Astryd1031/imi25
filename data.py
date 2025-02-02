import pandas as pd


# Load datasets (limiting rows for demonstration)
abm = pd.read_csv("abm.csv").iloc[:10000]
card = pd.read_csv("card.csv").iloc[:10000]
cheque = pd.read_csv("cheque.csv").iloc[:10000]
eft = pd.read_csv("eft.csv").iloc[:10000]
emt = pd.read_csv("emt.csv").iloc[:10000]
kyc = pd.read_csv("kyc.csv").iloc[:10000]
kycd = pd.read_csv("kyc_industry_codes.csv").iloc[:10000]
wire = pd.read_csv("wire.csv").iloc[:100000]

# Ensure 'industry_code' has the same data type in both DataFrames
kyc['industry_code'] = kyc['industry_code'].astype(str)
kycd['industry_code'] = kycd['industry_code'].astype(str)

# Merge KYC data with industry codes
kyc = kyc.merge(kycd, on="industry_code", how="left")

# Debugging: Check if 'industry' column exists after merge
if 'industry' not in kyc.columns:
    print("Error: 'industry' column not found in KYC data after merge.")
    print("Columns in KYC data:", kyc.columns.tolist())
    print("Columns in KYCD data:", kycd.columns.tolist())
    exit()

# Add a 'Transaction_channel' column to each transaction dataset
abm['Transaction_channel'] = 'ABM'
card['Transaction_channel'] = 'Card'
cheque['Transaction_channel'] = 'Cheque'
eft['Transaction_channel'] = 'EFT'
emt['Transaction_channel'] = 'EMT'
wire['Transaction_channel'] = 'Wire'

# Add a 'chanel_id' column to each transaction dataset
abm['chanel_id'] = abm['abm_id']
card['chanel_id'] = card['card_trxn_id']
cheque['chanel_id'] = cheque['cheque_id']
eft['chanel_id'] = eft['eft_id']
emt['chanel_id'] = emt['emt_id']
wire['chanel_id'] = wire['wire_id']

# Add a 'trans_type' column to each transaction dataset
abm['trans_type'] = abm['debit_credit']
card['trans_type'] = card['debit_credit']
cheque['trans_type'] = cheque['debit_credit']
eft['trans_type'] = eft['debit_credit']
emt['trans_type'] = emt['debit_credit']
wire['trans_type'] = wire['debit_credit']

# Add a 'merchant_category' column to datasets that don't have it
abm['merchant_category'] = None
cheque['merchant_category'] = None
eft['merchant_category'] = None
emt['merchant_category'] = None
wire['merchant_category'] = None

# Combine all transaction datasets
transactions = pd.concat([abm, card, cheque, eft, emt, wire], ignore_index=True)

# Merge transactions with KYC data
consolidated_data = transactions.merge(kyc, on="customer_id", how="left")

# Drop redundant transaction ID columns
columns_to_drop = ['abm_id', 'card_trxn_id', 'cheque_id', 'eft_id', 'emt_id', 'wire_id']
consolidated_data = consolidated_data.drop(columns=columns_to_drop)

# Select and rename columns for the final output
final_columns = [
    'customer_id', 'amount_cad', 'debit_credit', 'cash_indicator', 'country_x',
    'province_x', 'city_x', 'transaction_date', 'transaction_time',
    'Transaction_channel', 'chanel_id', 'trans_type', 'merchant_category',
    'ecommerce_ind', 'country_y', 'province_y', 'city_y', 'industry_code',
    'employee_count', 'sales', 'established_date', 'onboard_date', 'industry'
]

# Check if all final columns exist in consolidated_data
missing_columns = [col for col in final_columns if col not in consolidated_data.columns]
if missing_columns:
    print("Error: The following columns are missing in consolidated_data:", missing_columns)
    exit()

# Select the final columns
final_data = consolidated_data[final_columns]

# Save the consolidated data to a new CSV file
final_data.to_csv("consolidated_customer_transactions.csv", index=False)

# Separate data into user and business based on employee_count
user_data = final_data[final_data['employee_count'] == 0.0]  # Users have employee_count = 0.0
business_data = final_data[final_data['employee_count'] > 0.0]  # Businesses have employee_count > 0.0

# Save user and business data to separate CSV files
user_data.to_csv("user_transactions.csv", index=False)
business_data.to_csv("business_transactions.csv", index=False)

# drop unnecessary lines from user data
columns_to_drop = [ 'employee_count','sales', 'industry', 'merchant_category', 'industry_code']
userdf = user_data.drop(columns=columns_to_drop)
businessdf = business_data

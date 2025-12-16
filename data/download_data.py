"""
Download Telco Customer Churn dataset.
Source: IBM Sample Data
"""
import urllib.request
import os

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "telco_churn.csv")

if __name__ == "__main__":
    print(f"Downloading dataset from IBM...")
    urllib.request.urlretrieve(URL, OUTPUT_PATH)
    print(f"Saved to: {OUTPUT_PATH}")
    
    # Quick validation
    with open(OUTPUT_PATH, 'r') as f:
        lines = f.readlines()
    print(f"Rows: {len(lines) - 1} (excluding header)")
    print(f"Columns: {lines[0].count(',') + 1}")

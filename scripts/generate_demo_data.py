"""
Generate synthetic fraud detection data for demo/showcase when real data is unavailable.
Creates a realistic imbalanced dataset matching the expected schema.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "Fraud_Transaction_Detection.xlsx"

# Typical fraud dataset: ~1-3% fraud rate (imbalanced)
N_SAMPLES = 5000
FRAUD_RATIO = 0.02  # 2% fraud

MERCHANT_TYPES = ["Retail", "Online", "ATM", "Restaurant", "Gas", "Travel", "Grocery", "Healthcare"]
TRANSACTION_TYPES = ["Purchase", "Withdrawal", "Transfer", "Payment"]
REGIONS = ["North", "South", "East", "West", "Central"]


def generate_demo_data(n_samples: int = N_SAMPLES, output_path: Path | None = None) -> pd.DataFrame:
    """Generate synthetic fraud detection dataset. Saves to output_path if provided."""
    path = output_path or OUTPUT_PATH
    n_fraud = int(n_samples * FRAUD_RATIO)

    indices = list(range(n_samples))
    random.shuffle(indices)
    fraud_indices = set(indices[:n_fraud])

    rows = []
    for i in range(n_samples):
        is_fraud = 1 if i in fraud_indices else 0

        if is_fraud:
            amount = np.random.lognormal(7, 2)
            hour = random.choice(list(range(0, 6)) + list(range(22, 24)))
            merchant = random.choice(["Online", "ATM", "Travel"])
        else:
            amount = np.random.lognormal(6, 1.2)
            hour = random.randint(0, 23)
            merchant = random.choice(MERCHANT_TYPES)

        rows.append({
            "Customer Name": f"Customer_{i % 500}",
            "Customer Email": f"user{i % 500}@example.com",
            "Customer Phone": f"+1-555-{i % 1000:04d}",
            "Transaction ID": f"TXN-{i:06d}",
            "Amount": min(amount, 50000),
            "Merchant_Type": merchant,
            "Transaction_Type": random.choice(TRANSACTION_TYPES),
            "Region": random.choice(REGIONS),
            "Hour_of_Day": hour,
            "Day_of_Week": random.randint(0, 6),
            "Is_Fraud": is_fraud,
        })

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return df


def main():
    df = generate_demo_data(output_path=OUTPUT_PATH)
    print(f"Generated demo data: {OUTPUT_PATH}")
    print(f"  Samples: {len(df)}, Fraud: {df['Is_Fraud'].sum()} ({100*df['Is_Fraud'].mean():.1f}%)")


if __name__ == "__main__":
    main()

# check_balance.py
import pandas as pd

print("Checking the balance of 'positive' vs 'negative' labels in your data...")

try:
    df = pd.read_csv('dataset.csv', encoding='latin1')
    
    # Check the counts of 1s (positive) and 0s (negative)
    balance_counts = df['Label'].value_counts()
    
    print("\n--- Data Balance Report ---")
    print(balance_counts)
    print("---------------------------\n")

    if 1 in balance_counts and 0 in balance_counts:
        print("As you can see, the number of '1's (positive) is likely higher than the number of '0's (negative).")
        print("This imbalance is causing the model to always guess positive.")
    else:
        print("Could not find both 1s and 0s in the 'Label' column.")

except Exception as e:
    print(f"An error occurred: {e}")

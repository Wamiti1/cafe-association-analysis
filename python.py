import pandas as pd
import os
import glob
import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


print(f"Current Working Directory: {os.getcwd()}")

# 1. Automatically find the Excel file
excel_files = glob.glob('*.xlsx')

if not excel_files:
    print("Error: Could not find any .xlsx files in this directory.")
    print("Make sure your dataset is in the same folder printed above.")
    exit()

# Grab the first Excel file it finds
file_path = excel_files[0]
print(f"--- SUCCESS: Found dataset -> {file_path} ---\n")

# 2. Load the dataset using read_excel
try:
    df = pd.read_excel(file_path)
    print("--- SUCCESS: Loaded Excel file ---")
except Exception as e:
    print(f"An unexpected error occurred while loading: {e}")
    exit()

# 3. General Information & Data Types
print("\n--- DATASET INFO ---")
df.info()
print(f"Original dataset size (Total rows): {len(df)}")

# 4. Check for Missing Values & Noise ('NONE')
print("\n--- MISSING VALUES (Standard) ---")
print(df.isnull().sum())

print("\n--- NOISE DATA DETECTED ---")
try:
    # Find the item column dynamically
    item_col = [col for col in df.columns if 'item' in col.lower()][0]
    
    # Count how many times 'NONE' appears (ignoring case sensitivity just to be safe)
    none_count = (df[item_col].astype(str).str.strip().str.upper() == 'NONE').sum()
    print(f"Number of 'NONE' entries in '{item_col}': {none_count}")
    
    # 5. Clean the Dataset
    if none_count > 0:
        # Keep only the rows where the item is NOT 'NONE'
        df = df[df[item_col].astype(str).str.strip().str.upper() != 'NONE']
        print(f"-> Action Taken: Successfully removed {none_count} 'NONE' rows from the dataset.")
        
except IndexError:
    print("Item column not found for noise checking.")

# 6. Updated Dataset Size
print(f"\nCleaned dataset size (Total valid transactions): {len(df)}")

# 7. Date Range
try:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    print(f"\nDate Range: From {df['Date'].dt.date.min()} to {df['Date'].dt.date.max()}")
except KeyError:
    print("\nDate Range: 'Date' column not found.")

# 8. Item Analysis (On cleaned data)
print("\n--- TOP 10 BEST-SELLING ITEMS ---")
try:
    print(df[item_col].value_counts().head(10))
    print(f"\nTotal Unique Items Sold: {df[item_col].nunique()}")
except NameError:
    print("\nItem column was not successfully identified earlier.")
except Exception as e:
    print(f"\nCould not generate item analysis: {e}")
    
    
    
print("\n---------------------------------------------------------")
print("--- STEP 2: DATA PREPROCESSING (ONE-HOT ENCODING) ---")
print("---------------------------------------------------------\n")

try:
    # 1. Group the data by Transaction and Item, then pivot it into a matrix
    # We use 'Time' just as a dummy column to count occurrences
    basket = (df.groupby(['Transaction', item_col])['Time']
              .count().unstack().reset_index().fillna(0)
              .set_index('Transaction'))

    # 2. Define a function to convert numerical counts into boolean values (True/False)
    # Market Basket algorithms care IF an item was bought, not HOW MANY times it was bought in one go.
    def encode_units(x):
        if x <= 0:
            return False
        if x >= 1:
            return True

    # 3. Apply the encoding to the entire matrix
    basket_encoded = basket.map(encode_units)

    print("-> Successfully transformed data into Market Basket format.")
    print(f"-> New Matrix Shape: {basket_encoded.shape[0]} Transactions x {basket_encoded.shape[1]} Unique Items\n")

    # 4. Generate the Snapshot required for the report
    print("--- SNAPSHOT OF PREPARED DATA (First 5 Transactions, First 10 Items) ---")
    # We slice the first 10 columns just so it fits nicely in the terminal output
    print(basket_encoded.iloc[:5, :10])

    # 5. Save a sample to a CSV file 
    snapshot_filename = 'Prepared_Data_Snapshot.csv'
    basket_encoded.head(20).to_csv(snapshot_filename)
    print(f"\n-> Saved a 20-row sample to '{os.getcwd()}/{snapshot_filename}'")
    print("-> Open this file in Excel/LibreOffice and take a screenshot for your report!")

except Exception as e:
    print(f"An error occurred during preprocessing: {e}")    
    


print("\n---------------------------------------------------------")
print("--- STEP 3 & 4: MODELING (APRIORI vs FP-GROWTH) ---")
print("---------------------------------------------------------\n")

# 1. Set our Association Rule Thresholds
# We only care about itemsets that appear in at least 2% of all 9465 transactions
MIN_SUPPORT = 0.02 
# We only care about rules where the consequent is bought at least 30% of the time the antecedent is bought
MIN_CONFIDENCE = 0.3 

# ALGORITHM 1: APRIORI
print("Running Algorithm 1: Apriori...")
start_time_ap = time.time()

# Generate frequent itemsets
frequent_itemsets_ap = apriori(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)

# Generate rules
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=MIN_CONFIDENCE)

ap_duration = time.time() - start_time_ap
print(f"-> Apriori Execution Time: {ap_duration:.4f} seconds")
print(f"-> Apriori Rules Generated: {len(rules_ap)}\n")


#ALGORITHM 2: FP-GROWTH
print("Running Algorithm 2: FP-Growth...")
start_time_fp = time.time()

# Generate frequent itemsets
frequent_itemsets_fp = fpgrowth(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)

# Generate rules
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=MIN_CONFIDENCE)

fp_duration = time.time() - start_time_fp
print(f"-> FP-Growth Execution Time: {fp_duration:.4f} seconds")
print(f"-> FP-Growth Rules Generated: {len(rules_fp)}\n")


#DISPLAY THE RESULTS
print("--- TOP ASSOCIATION RULES (Sorted by Lift) ---")
# Both algorithms produce the same rules, so we just sort and print the FP-Growth output
if not rules_fp.empty:
    top_rules = rules_fp.sort_values('lift', ascending=False).head(10)
    
    # Clean up the output formatting for the terminal
    top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))
else:
    print("No rules generated. Try lowering the MIN_SUPPORT or MIN_CONFIDENCE thresholds.")    
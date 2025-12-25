import pandas as pd
import csv

# 1. THE RECOMMENDED WAY: Using Pandas
# Best for data analysis, filtering, and large datasets.
def read_tsv_pandas(file_path):
    try:
        # 'sep=\t' tells pandas to look for tabs instead of commas
        df = pd.read_csv(file_path, sep='\t')
        print("--- Pandas Output (First 5 rows) ---")
        print(df.head())
        return df
    except FileNotFoundError:
        print("File not found. Please check the path.")

# 2. THE BUILT-IN WAY: Using the CSV module
# Best if you don't want to install extra libraries or need to iterate row-by-row.
def read_tsv_csv_module(file_path):
    print("\n--- CSV Module Output ---")
    with open(file_path, mode='r', encoding='utf-8') as f:
        # delimiter='\t' is the key here
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            print(f"Row {i}: {row}")

# 3. THE MANUAL WAY: Using pure Python
# Best for extremely simple files or quick scripts.
def read_tsv_manual(file_path):
    print("\n--- Manual Split Output ---")
    with open(file_path, mode='r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Strip removes the newline, split('\t') gets the columns
            columns = line.strip().split('\t')
            print(f"Line {i}: {columns}")
            if i >= 4: break

# Example Usage (Uncomment to test):
path = r"C:\Projects\fmri-algonauts-2025\fmri-algonauts-2025 data\algonauts_2025.competitors\stimuli\transcripts\friends\s1\friends_s01e01a.tsv"
df = read_tsv_pandas(path)
print(read_tsv_csv_module(path))
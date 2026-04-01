import joblib
import numpy as np

file_path = r"D:\fmri-algonauts-2025-data\preprocessing_pipeline\dataset_config.pkl"

try:
    data = joblib.load(file_path)
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\nKey: {key}")
            print(f"Type: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"Shape: {value.shape}")
            elif isinstance(value, list):
                print(f"Length: {len(value)}")
            elif isinstance(value, dict):
                print(f"Sub-keys: {list(value.keys())}")
            
            # Check for 'pca' inside nested dicts if it's not a top-level key
            if isinstance(value, dict) and 'pca' in value:
                print(f"  --> Contains 'pca' in {key}")
        
        print(f"\nTop-level 'pca' exists: {'pca' in data}")
        
    else:
        print("Data is not a dictionary.")
        
except Exception as e:
    print(f"Error loading or processing file: {e}")

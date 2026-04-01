import numpy as np

def inspect_npy(file_path):
    # Load the file
    data = np.load(file_path, allow_pickle=True)

    print("=== BASIC INFO ===")
    print(f"Type: {type(data)}")

    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Size (total elements): {data.size}")
        print(f"Memory (bytes): {data.nbytes}")

        print("\n=== SAMPLE VALUES ===")
        print(data[:5] if data.ndim > 0 else data)

        # Stats only for numeric types
        if np.issubdtype(data.dtype, np.number):
            print("\n=== STATISTICS ===")
            print(f"Min: {np.min(data)}")
            print(f"Max: {np.max(data)}")
            print(f"Mean: {np.mean(data)}")
            print(f"Std: {np.std(data)}")

    else:
        print("\n⚠️ Not a NumPy array. Likely pickled object.")
        print(data)


# Example usage
inspect_npy(r"C:\Projects\fmri-algonauts-2025\fmri-algonauts-2025 code\phase1_ridge_submission_updated\ridge_baseline_submission\predictions.npy")
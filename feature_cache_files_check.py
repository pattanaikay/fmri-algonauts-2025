import numpy as np

# Load the npz file
npz_file = r"C:\Projects\fmri-algonauts-2025\fmri-algonauts-2025 data\feature_cache_v2\bourne02_features.npz"
data = np.load(npz_file, allow_pickle=True)

# View all keys
print("Keys:", list(data.keys()))

# View details for each key
for key in data.keys():
    arr = data[key]
    print(f"\n{key}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Size: {arr.size}")



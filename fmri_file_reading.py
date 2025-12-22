import h5py

file_path = r"C:\Projects\fmri-algonauts-2025\fmri-algonauts-2025 data\algonauts_2025.competitors\fmri\sub-05\func\sub-05_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"

with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        print(name, obj)

    f.visititems(print_structure)

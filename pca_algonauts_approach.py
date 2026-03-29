
"""
Here, you will reduce the dimensionality of the extracted stimulus features using principal component analysis (PCA). By removing reduntant or uninformative dimensions, the stimulus features will decrease in size while still retaining the most important stimulus information, thus reducing computational cost when used to train fMRI encoding models. You will, as an example, apply PCA on the extracted stimulus features for the first episode of the first season of Friends, which can be found at ../algonauts_2025_challenge_tutorial_data/stimulus_features/raw/<modality>/friends_s01e01a_features_<modality>.h5, where:
modality: String indicating the stimulus modality of the extracted features. Options are: ["visual", "audio", "language", "all"].
This function loads the extracted stimulus features. Since PCA requires data to be in a (Samples × Features) format, the pooler_output and last_hidden_state language features are vectorized and appended to each other during loading.

"""

def load_features(root_data_dir, modality):
    """
    Load the extracted features from the HDF5 file.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        The modality of the features ('visual', 'audio', or 'language').

    Returns
    -------
    features : float
        Stimulus features.

    """

    ### Get the stimulus features file directory ###
    data_dir = os.path.join(root_data_dir, 'stimulus_features', 'raw', modality,
        'friends_s01e01a_features_'+modality+'.h5')

    ### Load the stimulus features ###
    with h5py.File(data_dir, 'r') as data:
        for episode in data.keys():
            if modality != 'language':
                features = np.asarray(data[episode][modality])
            else:
                # Vectorize and append pooler_output and last_hidden_state
                # language features
                pooler_output = np.asarray(
                    data[episode][modality+'_pooler_output'])
                last_hidden = np.asarray(np.reshape(
                    data[episode][modality+'_last_hidden_state'],
                    (len(pooler_output), -1)))
                features = np.append(pooler_output, last_hidden, axis=1)
    print(f"{modality} features original shape: {features.shape}")
    print('(Movie samples × Features)')

    ### Output ###
    return features

"""
This function replaces NaN values in the stimulus features with zeros, and then z-scores the features (to correctly identify the directions that maximize the variance in the data, PCA requires the data to be centered and scaled).

"""

def preprocess_features(features):
    """
    Rplaces NaN values in the stimulus features with zeros, and z-score the
    features.

    Parameters
    ----------
    features : float
        Stimulus features.

    Returns
    -------
    prepr_features : float
        Preprocessed stimulus features.

    """

    ### Convert NaN values to zeros ###
    features = np.nan_to_num(features)

    ### Z-score the features ###
    scaler = StandardScaler()
    prepr_features = scaler.fit_transform(features)

    ### Output ###
    return prepr_features

"""
This function runs a PCA on the stimulus features, reducing their dimensionality to the specified number of principal components (PCs) to retain. Note that the number of PCs cannot exceed the dimensionality of the stimulus features.

"""

def perform_pca(prepr_features, n_components):
    """
    Perform PCA on the standardized features.

    Parameters
    ----------
    prepr_features : float
        Preprocessed stimulus features.
    n_components : int
        Number of components to keep

    Returns
    -------
    features_pca : float
        PCA-downsampled stimulus features.

    """

    ### Set the number of principal components to keep ###
    # If number of PCs is larger than the number of features, set the PC number
    # to the number of features
    if n_components > prepr_features.shape[1]:
        n_components = prepr_features.shape[1]

    ### Perform PCA ###n_init=4, max_iter=300
    pca = PCA(n_components, random_state=20200220)
    features_pca = pca.fit_transform(prepr_features)
    print(f"\n{modality} features PCA shape: {features_pca.shape}")
    print('(Movie samples × Principal components)')

    ### Output ###
    return features_pca
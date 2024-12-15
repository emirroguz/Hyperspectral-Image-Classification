import numpy as np


def reshape_features(image, ground_truth):
    """
    Reshape the image and ground truth for ML models.

    Parameters:
        image (np.ndarray): Hyperspectral image data (3D array).
        ground_truth (np.ndarray): Ground truth labels (2D array).

    Returns:
        tuple: (X, y) where X is 2D feature matrix, and y is a 1D label vector.
    """
    X = image.reshape(-1, image.shape[-1])
    y = ground_truth.ravel()
    return X, y


def filter_valid_samples(X, y):
    """
    Filter out samples where the label is 'Undefined' (e.g., 0).

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.

    Returns:
        tuple: Filtered (X, y) where Undefined labels are removed.
    """
    valid_mask = y > 0
    return X[valid_mask], y[valid_mask]


# example usage
if __name__ == "__main__":
    from dataset_loader import load_dataset
    dataset_name = "Indian_Pines"
    image, ground_truth = load_dataset(dataset_name)
    X, y = reshape_features(image, ground_truth)
    print(f"\n- Reshaped features (X): {X.shape}")
    print(f"- Reshaped labels (y): {y.shape}\n")
    X_filtered, y_filtered = filter_valid_samples(X, y)
    print(f"- Filtered features (X): {X_filtered.shape}")
    print(f"- Filtered labels (y): {y_filtered.shape}\n")

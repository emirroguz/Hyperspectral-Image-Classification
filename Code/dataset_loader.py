import os
import scipy.io


DATASET_PATHS = {
    "Indian_Pines": {
        "image": "Indian_Pines\\indian_pines_corrected.mat",
        "ground_truth": "Indian_Pines\\indian_pines_gt.mat",
    },
    "Kennedy_Space_Center": {
        "image": "Kennedy_Space_Center\\ksc.mat",
        "ground_truth": "Kennedy_Space_Center\\ksc_gt.mat",
    },
    "Pavia_Centre": {
        "image": "Pavia_Centre\\pavia_centre.mat",
        "ground_truth": "Pavia_Centre\\pavia_centre_gt.mat",
    },
    "Pavia_University": {
        "image": "Pavia_University\\pavia_university.mat",
        "ground_truth": "Pavia_University\\pavia_university_gt.mat",
    },
    "Salinas": {
        "image": "Salinas\\salinas_corrected.mat",
        "ground_truth": "Salinas\\salinas_gt.mat",
    }
}


def get_base_path():
    """
    Determine the correct base path based on the current working directory.

    Parameters:
        None.

    Returns:
        None.
    """
    current_dir = os.getcwd()
    if current_dir.endswith("Code"):
        os.chdir("..")


def load_mat_file(file_path):
    """
    Load a .mat file and extract the content.

    Parameters:
        file_path (str): Path to the .mat file.

    Returns:
        The content of the loaded .mat file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        data = scipy.io.loadmat(file_path)
        key = next(iter(k for k in data.keys() if not k.startswith("__")))
        return data[key]
    except StopIteration:
        raise ValueError(f"No valid data found in file: {file_path}")


def load_dataset(dataset_name, base_path=".\\Dataset"):
    """
    Load the specified dataset and return its image and ground truth data.

    Parameters:
        dataset_name (str): Name of the dataset.
        base_path (str): Base directory where datasets are stored.

    Returns:
        tuple: (image_array, ground_truth_array) where both are numpy arrays.
    """
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Dataset '{dataset_name}' is not configured.")
    dataset_config = DATASET_PATHS[dataset_name]
    image_path = os.path.join(base_path, dataset_config["image"])
    ground_truth_path = os.path.join(base_path, dataset_config["ground_truth"])
    image_array = load_mat_file(image_path)
    ground_truth_array = load_mat_file(ground_truth_path)
    return image_array, ground_truth_array


# example usage
if __name__ == "__main__":
    try:
        dataset_name = "Indian_Pines"
        base_path = ".\\Dataset"
        image_data, ground_truth_data = load_dataset(dataset_name, base_path)
    except Exception as e:
        print(f"Error: {e}")

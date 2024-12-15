import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_dataset_labels(dataset_name):
    """
    Determine class labels based on dataset name.

    Parameters:
        dataset_name (str): The name of the dataset (e.g., 'Indian_Pines').

    Returns:
        list: A list of class labels for the specified dataset.
    """
    if dataset_name == "Indian_Pines":
        label_names = [
            "Undefined", "Alfalfa", "Corn-notill",
            "Corn-mintill", "Corn", "Grass-Pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
            "Oats", "Soybean-notill", "Soybean-mintill",
            "Soybean-clean", "Wheat", "Woods",
            "Buildings-grass-trees-drives", "Stone-steel-towers"
        ]
    elif dataset_name == "Kennedy_Space_Center":
        label_names = [
            "Undefined", "Scrub", "Willow swamp",
            "Cabbage palm hammock", "Slash pine", "Oak/broadleaf hammock",
            "Hardwood", "Swamp", "Graminoid marsh",
            "Spartina marsh", "Cattail marsh", "Salt marsh",
            "Mud flats", "Water"
        ]
    elif dataset_name == "Pavia_Centre":
        label_names = [
            "Undefined", "Water", "Trees",
            "Asphalt", "Self-Blocking Bricks", "Bitumen",
            "Tiles", "Shadows", "Meadows", "Bare Soil"
        ]
    elif dataset_name == "Pavia_University":
        label_names = [
            "Undefined", "Asphalt", "Meadows",
            "Gravel", "Trees", "Painted metal sheets",
            "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"
        ]
    elif dataset_name == "Salinas":
        label_names = [
            "Undefined", "Broccoli_green_weeds_1", "Broccoli_green_weeds_2",
            "Fallow", "Fallow_rough_plow", "Fallow_smooth",
            "Stubble", "Celery", "Grapes_untrained",
            "Soil_vineyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
            "Vineyard_untrained", "Vineyard_vertical_trellis"
        ]
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not yet supported.")
    return label_names


def dataset_summary(image, ground_truth):
    """
    Print a summary of the dataset including shape, data type, and range.

    Parameters:
        image (np.ndarray): Hyperspectral image data.
        ground_truth (np.ndarray): Ground truth labels.
    
    Returns:
        None.
    """
    print("Dataset Summary:")
    print(f"- Image Shape: {image.shape} (Width x Height x Bands)")
    print(f"- Image Data Type: {image.dtype}")
    print(f"- Number of Spectral Bands: {image.shape[2]}")
    print(f"- Image Value Range: {np.min(image)} - {np.max(image)}")
    unique_labels, counts = np.unique(ground_truth, return_counts=True)
    print("\nGround Truth (Labels) Summary:")
    print(f"- Labels Shape: {ground_truth.shape} (Width x Height)")
    print(f"- Labels Data Type: {ground_truth.dtype}")
    print(f"- Number of Classes: {len(unique_labels) - 1}")
    print(f"- Class Distribution: {dict(zip(unique_labels[1:], counts[1:]))}\n")


def save_chart(figure, save_path, plot_type, dataset_name, model_name=None):
    """
    Save a plot to the specified directory with an appropriate name.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to save.
        save_path (str): The directory where the plot will be saved.
        plot_type (str): The type of the plot (e.g., 'ConfusionMatrix', 'F1Score').
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.

    Returns:
        None
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if model_name:
        filename = f"{plot_type}_{dataset_name}_{model_name}.png"
    else:
        filename = f"{plot_type}-{dataset_name}.png"
    full_path = os.path.join(save_path, filename)
    figure.savefig(full_path)


def plot_class_distribution_graph(ground_truth, dataset_name, save_path=None):
    """
    Plot the class distribution of the ground truth labels and save the figure if save_path is provided.

    Parameters:
        ground_truth (np.ndarray): Ground truth labels.
        dataset_name (str): Name of the dataset.
        save_path (str, optional): Directory to save the plot.

    Returns:
        None.
    """
    unique_labels, counts = np.unique(ground_truth, return_counts=True)
    defined_labels = unique_labels[1:]
    defined_counts = counts[1:]
    
    plt.figure(figsize=(6, 6))
    plt.bar(defined_labels, defined_counts, color="skyblue")
    plt.title(f"Class Distribution\nDataset: {dataset_name}")
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Pixels")
    plt.xticks(defined_labels)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if save_path:
        save_chart(plt.gcf(), save_path, "CDG", dataset_name)
    plt.show()


def plot_class_distribution_table(ground_truth, label_names, dataset_name, save_path=None):
    """
    Plot a table showing class distribution and save the figure if save_path is provided.

    Parameters:
        ground_truth (np.ndarray): Ground truth labels.
        label_names (list): List of class labels corresponding to the dataset.
        dataset_name (str): Name of the dataset.
        save_path (str, optional): Directory to save the table.

    Returns:
        None.
    """
    unique_labels, counts = np.unique(ground_truth, return_counts=True)
    class_data = [[label, label_names[label], count] for label, count in zip(unique_labels, counts)]
    fig, ax = plt.subplots(figsize=(6, len(class_data) * 0.4))
    ax.set_title(f"Class Distribution Table\nDataset: {dataset_name}")
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=class_data,
        colLabels=["Class Number", "Class Name", "Pixel Count"],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(class_data[0]))))
    if save_path:
        save_chart(fig, save_path, "CDT", dataset_name)
    plt.show()


def visualize_sample_band(data, band, dataset_name, save_path=None):
    """
    Visualizes a single spectral band from the hyperspectral dataset.
    
    Parameters:
        data (np.ndarray): Hyperspectral image data with dimensions (rows, cols, bands).
        band (int): Index of the spectral band to visualize.
        dataset_name (str): Name of the dataset for labeling the output.
        save_path (str, optional): Path to save the visualization. If None, the visualization is not saved.
    
    Returns:
        None
    """
    band = data[:, :, band]
    plt.figure(figsize=(5, 5))
    plt.title(f"Sample Band\nDataset: {dataset_name}")
    plt.imshow(band, cmap='gray')
    plt.axis("off")
    if save_path:
        save_chart(plt.gcf(), save_path, "VSB", dataset_name)
    plt.show()


def visualize_ground_truth(ground_truth, label_names, dataset_name, save_path=None):
    """
    Visualize the ground truth with a colormap, excluding 'Undefined' in the colorbar,
    ensuring readability and distinct class colors.

    Parameters:
        ground_truth (np.ndarray): Ground truth labels.
        label_names (list): List of class labels corresponding to the dataset.
        dataset_name (str): Name of the dataset for visualization.
        save_path (str, optional): Path to save the output figure. Defaults to None.

    Returns:
        None.
    """
    num_classes = len(label_names)
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = ["black"] + [base_colors[i % 20] for i in range(num_classes - 1)]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(ground_truth, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=np.arange(num_classes))
    cbar.ax.set_yticklabels(label_names, fontsize=13)
    cbar.set_label("Classes", rotation=270, labelpad=15, fontsize=15)
    cbar.ax.invert_yaxis()
    ax.set_title(f"Ground Truth Visualization\nDataset: {dataset_name}", fontsize=15)
    ax.axis("off")
    if save_path:
        save_chart(fig, save_path, "GTV", dataset_name)
    plt.show()


# example usage
if __name__ == "__main__":
    from dataset_loader import load_dataset
    dataset_name = "Indian_Pines"
    image, ground_truth = load_dataset(dataset_name)
    label_names = get_dataset_labels(dataset_name)
    print("\nClass Summary:")
    print("- Class Names:", label_names)
    print("- Ignored Class ID:", label_names.index("Undefined"), "\n")
    dataset_summary(image, ground_truth)
    plot_class_distribution_graph(ground_truth)
    plot_class_distribution_table(ground_truth, label_names)
    visualize_ground_truth(ground_truth, label_names)

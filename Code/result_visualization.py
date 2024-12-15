import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dataset_info import save_chart


def plot_confusion_matrix(y_true, y_pred, class_labels, dataset_name, model_name, save_path=None):
    """
    Plot and optionally save the confusion matrix.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_labels (list): List of class label names.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        save_path (str): Directory to save the plot. If None, plot will not be saved.

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix\nDataset: {dataset_name} | Model: {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    if save_path:
        save_chart(plt.gcf(), save_path, "Confusion Matrix", dataset_name, model_name)
    plt.show()


def plot_f1_bar(report, dataset_name, model_name, save_path=None):
    """
    Plot and optionally save F1-score as a bar graph.

    Parameters:
        report (dict): Dictionary containing the metrics for each class.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        save_path (str): Directory to save the plot. If None, plot will not be saved.

    Returns:
        None
    """
    labels = [key for key in report.keys() if key.isdigit()]
    f1_scores = [report[label]['f1-score'] for label in labels]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, f1_scores, color="skyblue", alpha=0.7)
    plt.title(f"F1-Score by Class\nDataset: {dataset_name} | Model: {model_name}", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    if save_path:
        save_chart(plt.gcf(), save_path, "F1Score", dataset_name, model_name)
    plt.show()

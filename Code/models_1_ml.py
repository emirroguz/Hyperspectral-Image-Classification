from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def initialize_model(model_name, **kwargs):
    """
    Initialize a machine learning model based on the model name.

    Parameters:
        model_name (str): Name of the model (e.g., "LogisticRegression", "RandomForest").
        **kwargs: Additional parameters for the model.

    Returns:
        object: An initialized model.
    """
    models = {
        "LogisticRegression": LogisticRegression,
        "LDA": LinearDiscriminantAnalysis,
        "SVC-Linear": lambda **kwargs: SVC(kernel="linear", **kwargs),
        "SVC-Poly": lambda **kwargs: SVC(kernel="poly", **kwargs),
        "SVC-RBF": lambda **kwargs: SVC(kernel="rbf", **kwargs),
        "KNN": KNeighborsClassifier,
        "GNB": GaussianNB,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "AdaBoost": AdaBoostClassifier,
        "CatBoost": CatBoostClassifier,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "MLP": MLPClassifier,
    }
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return models[model_name](**kwargs)


def evaluate_model(y_test, y_pred):
    """
    Evaluate a trained model's predictions against true labels.

    Parameters:
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns:
        tuple:
            - dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
            - dict: Detailed classification report containing metrics for each class.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("- Classification Report:\n\n", classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}, report


# example usage
if __name__ == "__main__":
    print("\n- Loading the dataset...\n")
    from dataset_loader import load_dataset
    dataset_name = "Indian_Pines"
    image, ground_truth = load_dataset(dataset_name)

    print("- Preprocessing the data...\n")
    from preprocess import reshape_features, filter_valid_samples
    X, y = reshape_features(image, ground_truth)
    X, y = filter_valid_samples(X, y)

    print("- Splitting the data into train and test sets...\n")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("- Initializing and training the Logistic Regression model...\n")
    model_name = "LogisticRegression"
    model = initialize_model(model_name, C=100, penalty="l1", solver="saga", max_iter=1000, tol=1e-3, multi_class="multinomial", random_state=42)
    model.fit(X_train, y_train)

    print("- Evaluating the model...\n")
    y_pred = model.predict(X_test)
    results = evaluate_model(y_test, y_pred)

    print("- General Evaluation Results:")
    for metric, value in results.items():
        if value is not None:
            print(f"  - {metric}: {value:.4f}")
    print()

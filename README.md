# HSI Classification: Machine Learning, Deep Learning, and Semi-Supervised Learning

A comprehensive project for **Hyperspectral Image (HSI) Classification** using traditional machine learning, deep learning, and semi-supervised learning techniques. The project utilizes well-known benchmark datasets to evaluate and compare the performance of various methods.

## Features

- **Machine Learning Models:**
   Classical models such as Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forest, and ensemble techniques like XGBoost and LightGBM are implemented for baseline performance.

- **Deep Learning Models:**
   Advanced models including 2D-CNN, 3D-CNN, and Vision Transformer (ViT) are utilized for feature extraction and classification.

- **Semi-Supervised Learning:**
   - **Self-Training:** Combining labeled data with pseudo-labeled data to improve model performance.
   - **Semi-Supervised GAN:** Leveraging labeled and unlabeled data through adversarial learning.

- **Preprocessing Techniques:**
   - Image reshaping and ground truth flattening.
   - Noise reduction using Gaussian filters.
   - Dimensionality reduction with **Principal Component Analysis (PCA)**.

- **Evaluation Metrics:**
   Performance is assessed using standard metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

## Datasets

The project uses three widely adopted benchmark datasets in hyperspectral image classification:

1. **Indian Pines Dataset**
   - Captured using the AVIRIS sensor in Northwest Indiana.
   - Contains 145x145 pixels with 200 spectral bands after water absorption removal.
   - Includes 16 ground-truth classes.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/bd93e4f7-79ee-4517-86a8-91ab892f0e9d" width="400" height="250" alt="Indian Pines">
  </p>

2. **Pavia University Dataset**
   - Collected using the ROSIS sensor over an urban area in Pavia, Italy.
   - Comprises 610x340 pixels with 103 spectral bands.
   - Includes 9 ground-truth classes.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/4d655fe5-b0a8-445d-8811-8c9b133068b4" width="300" height="290" alt="Pavia University">
  </p>

3. **Salinas Dataset**
   - Captured with the AVIRIS sensor over the Salinas Valley, California.
   - Contains 512x217 pixels and 204 spectral bands.
   - Includes 16 classes consisting of vegetables, bare soil, and vineyard fields.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/367b60af-3623-4f57-b2c0-327fbc903bec" width="300" height="290" alt="Salinas">
  </p>

## Workflow

The project workflow is structured as follows:

1. **Data Preprocessing:**
   Raw hyperspectral images are reshaped, noise-filtered, and dimensionality-reduced using PCA for improved efficiency.

2. **Model Training:**
   Models (machine learning, deep learning, and semi-supervised) are trained on the preprocessed datasets.

3. **Model Evaluation:**
   Models are evaluated using Accuracy, Precision, Recall, and F1-Score metrics.

4. **Comparison and Analysis:**
   Results from various models are compared to identify the most effective methods for hyperspectral image classification.

## Results

The project generates the following outputs:
- Performance comparison of machine learning, deep learning, and semi-supervised learning models.
- Graphical results illustrating accuracy and loss curves for each method.
- Model-specific insights for hyperspectral image classification.

## Future Work

Potential future improvements include:
- Expanding the dataset with additional HSI benchmarks.
- Implementing advanced semi-supervised learning techniques.
- Exploring hybrid models combining deep learning with traditional approaches.

---

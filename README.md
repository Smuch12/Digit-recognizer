# Digit recognizer

This project is designed to classify handwritten digits using machine learning techniques. The primary objective is to implement a pipeline for training, testing, and evaluating models to achieve high accuracy in recognizing digits. The code leverages popular libraries such as scikit-learn and tensorflow for machine learning and deep learning tasks.


### Features:

- Data Preparation: Preprocesses data for training and testing, including scaling and dimensionality reduction.
- Model Training: Implements various machine learning algorithms, including:
- Random Forest Classifier
- Support Vector Machines (SVMs)
- Neural networks via TensorFlow
- Evaluation Metrics: Provides performance evaluation using:
- Confusion matrices
- Classification reports
- Accuracy scores
- Cross-Validation: Uses techniques like K-Fold cross-validation for robust model evaluation.
- Dimensionality Reduction: Integrates Principal Component Analysis (PCA) for feature reduction.
- Outlier Detection: Uses One-Class SVM for detecting anomalies in the dataset

### The project requires the following Python libraries:

- scikit-learn for machine learning models and evaluation.
- tensorflow for building and training deep learning models.
- numpy for numerical operations.
- matplotlib for visualizations.
- warnings, math, and time for auxiliary functionalities.

### How to run:

Install packages with:
```
pip install -r requirements.txt
```

and then just run the jupyter notebook.
**ML Model Comparison Script Documentation**

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [File Structure](#file-structure)
4. [Function Reference](#function-reference)

   * `preprocess_data`
   * `train_logistic_regression`
   * `train_random_forest`
   * `train_svm`
   * `train_neural_network`
   * `plot_scatter`
   * `main`
5. [Data Workflow](#data-workflow)
6. [Model Training & Evaluation](#model-training--evaluation)
7. [Visualization](#visualization)
8. [Usage](#usage)
9. [Key Hyperparameters & Settings](#key-hyperparameters--settings)
10. [Extending the Script](#extending-the-script)
11. [Keywords & Glossary](#keywords--glossary)

---

## 1. Overview

This Python script implements and compares four different classification models—Logistic Regression, Random Forest, Support Vector Machine (SVM), and a simple feed‑forward Neural Network—on a tabular dataset of client records. It automates the full machine learning pipeline from data loading and preprocessing through model training, evaluation, and visualization of results.

## 2. Prerequisites

* Python 3.7+
* Libraries:

  * pandas
  * scikit-learn
  * matplotlib
  * tensorflow

Install dependencies via:

```bash
pip install pandas scikit-learn matplotlib tensorflow
```

## 3. File Structure

```
project_root/
├── completedclient.csv    # Input dataset (CSV)
├── ml_compare.py          # This script
└── requirements.txt       # (recommended)
```

## 4. Function Reference

### `preprocess_data(file_path) -> pd.DataFrame`

**Description:**

* Loads a CSV file into a DataFrame.
* Converts any object‑dtype columns to string to avoid mixed‑type warnings.
* Encodes all categorical features (object columns) using `LabelEncoder`.

**Inputs:**

* `file_path` (str): Path to the CSV file.

**Output:**

* `data` (DataFrame): Preprocessed and encoded dataset.

---

### `train_logistic_regression(X_train, y_train, X_test, y_test) -> (model, accuracy)`

**Description:**

* Trains a `LogisticRegression` classifier.
* Evaluates on the test set and returns the accuracy.

**Outputs:**

* `lr_model`: Trained logistic regression model.
* `lr_accuracy` (float): Test set accuracy.

---

### `train_random_forest(X_train, y_train, X_test, y_test) -> (model, accuracy)`

**Description:**

* Trains a `RandomForestClassifier`.
* Returns trained model and test accuracy.

---

### `train_svm(X_train, y_train, X_test, y_test) -> (model, accuracy)`

**Description:**

* Trains an `SVC` (Support Vector Machine).
* Returns trained SVM and accuracy.

---

### `train_neural_network(X_train, y_train, X_test, y_test) -> (model, accuracy)`

**Description:**

* Builds a simple feed‑forward neural network (two hidden layers: 64 and 32 units).
* Uses binary crossentropy loss and the Adam optimizer.
* Trains for 10 epochs and returns the model and its test accuracy.

---

### `plot_scatter(true_labels, predicted_labels, model_name, accuracy)`

**Description:**

* Generates a scatter plot comparing true vs. predicted labels.
* Annotates the plot with model name and accuracy.

---

### `main()`

**Description:**

* Entry point of the script.
* Checks for TensorFlow GPU availability.
* Loads and preprocesses `completedclient.csv`.
* Splits data into training and test sets (80/20 split).
* Calls each training function in turn, prints accuracies.
* Plots scatter charts for each model’s predictions.

---

## 5. Data Workflow

1. **Load**: Read `completedclient.csv`.
2. **Clean**: Force all mixed‑type columns to string.
3. **Encode**: Label‑encode categorical features.
4. **Split**: 80% training, 20% testing.

## 6. Model Training & Evaluation

* Logistic Regression, Random Forest, and SVM use default hyperparameters.
* Neural Network:

  * Architecture: 64→32→1 neurons
  * Activations: ReLU (hidden), Sigmoid (output)
  * Loss: Binary crossentropy
  * Optimizer: Adam
  * Epochs: 10, Batch size: 32, Validation split: 20%

## 7. Visualization

For each model, a scatter plot of true vs. predicted labels is generated:

* **X‑axis**: True labels
* **Y‑axis**: Predicted labels
* Diagonal reference line (ideal prediction)

## 8. Usage

```bash
python ml_compare.py
```

* Ensure `completedclient.csv` resides in the same directory or adjust the path in `main()`.
* View printed accuracies and pop‑up scatter plots.

## 9. Key Hyperparameters & Settings

* Train/Test split: 80/20
* Neural network epochs: 10
* Batch size (NN): 32
* SVM kernel: RBF (default)
* Random Forest: 100 trees (default)

## 10. Extending the Script

* **Hyperparameter Tuning:** Integrate `GridSearchCV` or `RandomizedSearchCV`.
* **Additional Metrics:** Compute precision, recall, F1-score.
* **Save/Load Models:** Use `joblib.dump()` and `joblib.load()` for persistence.
* **Cross‑Validation:** Replace single split with k‑fold cross‑validation.

## 11. Keywords & Glossary

* **Classification**: Predicting discrete labels.
* **Label Encoding**: Converting categorical text to numeric codes.
* **Train/Test Split**: Partitioning data for unbiased evaluation.
* **Logistic Regression**: Linear model for binary classification.
* **Random Forest**: Ensemble of decision trees.
* **SVM (Support Vector Machine)**: Classifier maximizing margin.
* **Neural Network**: Multi‑layer perceptron (MLP) model.
* **Binary Crossentropy**: Loss function for two‑class problems.
* **ReLU**: Activation function (rectified linear unit).
* **Sigmoid**: Activation giving output in \[0,1].
* **Adam**: Adaptive optimizer for gradient descent.
* **Scatter Plot**: Visualization of prediction quality.

---
## Smaron biswas 2024


*End of Documentation*

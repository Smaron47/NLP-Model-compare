import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# Function to preprocess data
def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Handle mixed data types warning
    mixed_cols = data.select_dtypes(include=['object']).columns
    data[mixed_cols] = data[mixed_cols].astype(str)

    # Encode categorical variables
    encoder = LabelEncoder()
    for col in data.select_dtypes(include=['object']):
        data[col] = encoder.fit_transform(data[col])

    return data

# Function to train logistic regression model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    # Train logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    return lr_model, lr_accuracy

# Function to train random forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    # Train random forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    return rf_model, rf_accuracy

# Function to train SVM model
def train_svm(X_train, y_train, X_test, y_test):
    # Train SVM model
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    return svm_model, svm_accuracy

# Function to train neural network model
def train_neural_network(X_train, y_train, X_test, y_test):
    # Define neural network architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate model
    _, accuracy = model.evaluate(X_test, y_test)
    return model, accuracy

# Function to plot scatter plot of predictions
def plot_scatter(true_labels, predicted_labels, model_name, accuracy):
    plt.figure(figsize=(12, 6))
    plt.scatter(true_labels, predicted_labels, alpha=0.5)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title(f'Scatter Plot of Predictions ({model_name} - Accuracy: {accuracy:.2f})')
    plt.grid(True)
    plt.show()

# Main function
def main():
    # Check if TensorFlow is using GPU
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    file_path = "completedclient.csv"
    data = preprocess_data(file_path)

    # Split data into features and target
    X = data.drop(columns=['client_id'])
    y = data['client_id']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    lr_model, lr_accuracy = train_logistic_regression(X_train, y_train, X_test, y_test)

    # Train random forest model
    rf_model, rf_accuracy = train_random_forest(X_train, y_train, X_test, y_test)

    # Train SVM model
    svm_model, svm_accuracy = train_svm(X_train, y_train, X_test, y_test)

    # Train neural network model
    nn_model, nn_accuracy = train_neural_network(X_train, y_train, X_test, y_test)

    # Plot scatter plot of predictions
    plot_scatter(y_test, lr_model.predict(X_test), 'Logistic Regression', lr_accuracy)
    plot_scatter(y_test, rf_model.predict(X_test), 'Random Forest', rf_accuracy)
    plot_scatter(y_test, svm_model.predict(X_test), 'SVM', svm_accuracy)
    plot_scatter(y_test, nn_model.predict(X_test), 'Neural Network', nn_accuracy)

if __name__ == "__main__":
    main()

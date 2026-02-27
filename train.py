import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import mlflow
import mlflow.sklearn
from preprocessing import load_and_preprocess_data

# 1. Set up MLflow Tracking
# This creates a local directory 'mlruns' to store experiments
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("Flipkart_Sentiment_Analysis")

def plot_confusion_matrix(y_true, y_pred, run_name):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {run_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save artifact
    plot_filename = f"conf_matrix_{run_name}.png"
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

def train_model():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('data.csv')
    
    X = df['Cleaned_Review']
    y = df['Sentiment']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define Hyperparameters to iterate over for demonstration
    params_list = [
        {'C': 1.0, 'max_iter': 100, 'solver': 'lbfgs'},
        {'C': 0.1, 'max_iter': 100, 'solver': 'lbfgs'},
        {'C': 10.0, 'max_iter': 200, 'solver': 'liblinear'}
    ]

    vectorizer_params = {'max_features': 1000}

    print("Starting MLflow Runs...")

    for i, params in enumerate(params_list):
        # 2. Customizing MLflow UI with Run Names
        run_name = f"LogReg_Run_{i+1}_C_{params['C']}"

        with mlflow.start_run(run_name=run_name):
            print(f"Training {run_name}...")
            
            # Feature Extraction (TF-IDF)
            tfidf = TfidfVectorizer(max_features=vectorizer_params['max_features'])
            X_train_vec = tfidf.fit_transform(X_train)
            X_test_vec = tfidf.transform(X_test)

            # Model Training
            model = LogisticRegression(**params)
            model.fit(X_train_vec, y_train)

            # Predictions
            y_pred = model.predict(X_test_vec)

            # Metrics
            f1 = f1_score(y_test, y_pred, average='weighted')
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')

            # 3. Log Parameters
            mlflow.log_params(params)
            mlflow.log_param("vectorizer", "TF-IDF")
            mlflow.log_param("max_features", vectorizer_params['max_features'])

            # 4. Log Metrics
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            # 5. Log Artifacts (Confusion Matrix Plot)
            plot_file = plot_confusion_matrix(y_test, y_pred, run_name)
            mlflow.log_artifact(plot_file)
            
            # Clean up plot file after logging
            os.remove(plot_file)

            # 6. Log Model (Manual Method for reliability)
            # Save the model to a temporary file
            model_filename = "model.joblib"
            joblib.dump(model, model_filename)
            
            # Log the file as an artifact
            mlflow.log_artifact(model_filename, artifact_path="model")
            
            # Clean up the temporary file
            os.remove(model_filename)

            print(f"Run {run_name} completed. F1 Score: {f1:.4f}")

    print("Training finished. Check MLflow UI.")

if __name__ == "__main__":
    train_model()
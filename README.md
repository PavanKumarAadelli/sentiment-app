# 🛒 Flipkart Reviews Sentiment Analysis

This project is an end-to-end Machine Learning application that classifies customer reviews from Flipkart as **Positive** or **Negative**. It includes data preprocessing, model training with MLflow tracking, workflow orchestration with Prefect, and deployment using Streamlit.

## 📌 Project Overview

The objective is to analyze customer sentiments to understand product feedback better. The application uses Natural Language Processing (NLP) techniques to process review text and a Logistic Regression model to predict sentiment.

## ✨ Features

- **Data Preprocessing:** Cleaning text, removing stopwords, and lemmatization using NLTK.
- **Feature Extraction:** TF-IDF Vectorization.
- **Model Training:** Logistic Regression classification.
- **Experiment Tracking:** Logged parameters, metrics, and artifacts using **MLflow**.
- **Workflow Orchestration:** Automated pipeline using **Prefect**.
- **Deployment:** Interactive web application built with **Streamlit**.

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Frameworks:** Streamlit, MLflow, Prefect
- **Deployment:** Streamlit Cloud

## 🚀 Getting Started

### Prerequisites
Install the required libraries:
```bash
pip install -r requirements.txt


### Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sentiment-app.git
   cd sentiment-app
   

2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   

## 📂 Project Structure

Sentiment_Project/
│
├── app.py                # Streamlit application code
├── train.py              # Model training script (MLflow integrated)
├── workflow.py           # Prefect workflow orchestration
├── save_final_model.py   # Script to save model/vectorizer as pickle files
├── data.csv              # Dataset
├── model.pkl             # Trained model file
├── vectorizer.pkl        # TF-IDF Vectorizer file
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

## 📊 MLflow Dashboard

To view experiment tracking:
1. Run `mlflow ui` in your terminal.
2. Open `http://localhost:5000` in your browser.

## ⚙️ Prefect Workflow

To visualize the pipeline:
1. Run `prefect server start`.
2. Run `python workflow.py`.
3. View the dashboard at `http://localhost:4200`.

## 🌐 Live Deployment

The application is deployed on Streamlit Cloud. 
(https://flipkart-sentiment-analysis9.streamlit.app/)

Made with ❤️ by [Pavan kumar Aadelli]

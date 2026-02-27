import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    # 1. Find the best run based on F1 Score
    experiment_name = "Flipkart_Sentiment_Analysis"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Search for the best run
    best_run = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )[0]

    best_run_id = best_run.info.run_id
    print(f"Best Run ID: {best_run_id} with F1 Score: {best_run.data.metrics['f1_score']}")

    # 2. Register the Model
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "Flipkart_Sentiment_Best"

    # Register the model in the Model Registry
    model_version = mlflow.register_model(model_uri, model_name)
    print(f"Model registered as {model_name} version {model_version.version}")

    # 3. Tagging the Model Version
    # Adding tags helps in managing models (e.g., marking as 'Staging' or 'Production')
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="stage",
        value="production"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="task",
        value="sentiment-analysis"
    )

    print(f"Tags added to model version {model_version.version}")

if __name__ == "__main__":
    register_best_model()
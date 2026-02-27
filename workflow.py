import prefect
from prefect import flow, task
from train import train_model
from register_model import register_best_model

# Define Tasks
@task
def run_training():
    """Task to run the training process"""
    print("Starting training task...")
    train_model()
    return "Training Completed"

@task
def run_registration():
    """Task to register the best model"""
    print("Starting model registration task...")
    register_best_model()
    return "Registration Completed"

# Define Flow
@flow(name="Sentiment Analysis Pipeline")
def sentiment_pipeline():
    print("Starting Sentiment Analysis Pipeline")
    
    # Run tasks
    # In Prefect 2, tasks run concurrently by default. 
    # We pass the result of the first task to the second to ensure order.
    train_status = run_training()
    reg_status = run_registration(wait_for=[train_status])
    
    print(f"Pipeline finished: {reg_status}")

if __name__ == "__main__":
    # Run the flow locally
    sentiment_pipeline()
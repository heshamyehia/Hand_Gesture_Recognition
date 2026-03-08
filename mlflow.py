import mlflow
import mlflow.sklearn
import json
import tempfile


def log_models(models_dict, metrics_dict):
    """
    Log multiple trained models to MLflow with their parameters, metrics, and artifacts.
    
    Args:
        models_dict: Dict of {model_name: trained_model}
        metrics_dict: Dict of {model_name: metrics_dict}
                     e.g., {"SVM": {"accuracy": 0.93, "f1": 0.92}}
    
    Returns:
        Dict of {model_name: run_id}
    """
    run_ids = {}
    
    for model_name, model in models_dict.items():
        metrics = metrics_dict[model_name]
        
        with mlflow.start_run(run_name=model_name):
            # Log model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            # Create and log artifacts
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                summary = {
                    "model_name": model_name,
                    "model_type": type(model).__name__,
                    "metrics": metrics,
                    "params": params if hasattr(model, 'get_params') else {}
                }
                json.dump(summary, f, indent=2, default=str)
                f.flush()
                mlflow.log_artifact(f.name, "artifacts")
            
            run_ids[model_name] = mlflow.active_run().info.run_id
    
    return run_ids

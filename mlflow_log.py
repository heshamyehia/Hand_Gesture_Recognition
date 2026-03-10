import os
import tempfile
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def log_models(models_dict, metrics_dict, X_test, y_test, experiment_name="Hand_Gesture_Recognition"):

    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    run_ids = {}

    for model_name, model in models_dict.items():
        metrics = metrics_dict[model_name]

        with mlflow.start_run(run_name=model_name):

            # log parameters
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # log metrics (accuracy, f1_score, ...)
            mlflow.log_metrics(metrics)

            # log the model itself
            mlflow.sklearn.log_model(model, artifact_path="model")

            # generate and log confusion matrix as artifact
            y_pred = model.predict(X_test)
            labels = sorted(set(y_test))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
                ax=ax, colorbar=False, xticks_rotation="vertical"
            )
            ax.set_title(f"Confusion Matrix — {model_name}")
            fig.tight_layout()

            with tempfile.TemporaryDirectory() as tmp_dir:
                cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
                fig.savefig(cm_path, dpi=120)
                mlflow.log_artifact(cm_path, artifact_path="plots")

            plt.close(fig)

            # save run id
            run_ids[model_name] = mlflow.active_run().info.run_id
            
    return run_ids
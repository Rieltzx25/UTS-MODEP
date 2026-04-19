"""Train classifier Pipeline, log ke MLflow."""
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from preprocessing import build_preprocessor


def train_classifier(x_train, y_train):
    n_estimators = 200
    max_depth    = None

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Placement Classification")

    with mlflow.start_run(run_name="rf_classifier") as run:
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", str(max_depth))
        mlflow.log_param("class_weight", "balanced")

        pipe.fit(x_train, y_train)

        mlflow.sklearn.log_model(
            pipe, name="model",
            registered_model_name="placement_classifier",
        )
        print(f"Classifier trained. Run ID: {run.info.run_id}")

    return run.info.run_id

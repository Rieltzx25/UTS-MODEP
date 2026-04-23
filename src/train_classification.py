import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocessing import build_preprocessor


def train_classifier(x_train, y_train):
    C = 0.1
    solver = "lbfgs"
    max_iter = 500

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(
            C=C, solver=solver, max_iter=max_iter,
            class_weight="balanced", random_state=42,
        )),
    ])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Placement Classification")

    with mlflow.start_run(run_name="logreg_classifier") as run:
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("class_weight", "balanced")

        pipe.fit(x_train, y_train)

        mlflow.sklearn.log_model(
            pipe, name="model",
            registered_model_name="placement_classifier",
        )
        print(f"Classifier trained. Run ID: {run.info.run_id}")

    return run.info.run_id

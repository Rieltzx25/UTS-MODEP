import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from preprocessing import build_preprocessor


def train_regressor(x_train, y_train):
    alpha = 1.0
    solver = "auto"
    fit_intercept = True

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", Ridge(alpha=alpha, solver=solver, fit_intercept=fit_intercept)),
    ])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Salary Regression")

    with mlflow.start_run(run_name="ridge_regressor") as run:
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("solver", solver)
        mlflow.log_param("fit_intercept", fit_intercept)

        pipe.fit(x_train, y_train)

        mlflow.sklearn.log_model(
            pipe, name="model",
            registered_model_name="salary_regressor",
        )
        print(f"Regressor trained. Run ID: {run.info.run_id}")

    return run.info.run_id

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)


def evaluate_classifier(x_test, y_test, run_id):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(x_test)
    placed_idx = list(model.classes_).index("Placed")
    proba = model.predict_proba(x_test)[:, placed_idx]

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score((y_test == "Placed").astype(int), proba)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("roc_auc",  auc)

    print(f"Classifier | Accuracy={acc:.3f} | F1={f1:.3f} | ROC-AUC={auc:.3f}")
    return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc}


def evaluate_regressor(x_test, y_test, run_id):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(x_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("r2",   r2)

    print(f"Regressor | RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

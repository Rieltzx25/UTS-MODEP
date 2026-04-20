"""
Extract model .pkl dari MLflow registry ke folder models/.
Dijalankan setelah `python src/main.py` selesai.

Alasan: Streamlit Cloud tidak punya akses ke MLflow registry lokal,
jadi kita copy model ke folder biasa yang ikut ter-commit ke GitHub.
"""
import os
import shutil
import mlflow.sklearn


def export():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    os.makedirs("models", exist_ok=True)

    targets = {
        "placement_classifier": "models/placement_classifier.pkl",
        "salary_regressor":     "models/salary_regressor.pkl",
    }

    for name, out in targets.items():
        model_uri = f"models:/{name}/latest"
        local_dir = mlflow.artifacts.download_artifacts(model_uri)
        src_pkl   = os.path.join(local_dir, "model.pkl")
        shutil.copy(src_pkl, out)
        print(f"Exported {name} -> {out}")


if __name__ == "__main__":
    export()

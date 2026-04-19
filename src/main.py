"""
Part 2 - Pipeline Runner (sklearn Pipeline + MLflow)
"""

from sklearn.model_selection import train_test_split

from data_ingestion import load_data
from train_classification import train_classifier
from train_regression import train_regressor
from evaluate import evaluate_classifier, evaluate_regressor


F1_THRESHOLD = 0.60
RMSE_THRESHOLD = 2.0


def run_pipeline():
    print("Step 1: Data Ingestion")
    df = load_data()

    print("\nStep 2: Train/Test Split")
    X = df.drop(columns=["placement_status", "salary_lpa"])
    y = df["placement_status"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    placed = df[df["placement_status"] == "Placed"]
    Xr = placed.drop(columns=["placement_status", "salary_lpa"])
    yr = placed["salary_lpa"]
    xr_train, xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42)

    print("\nStep 3: Classification Training (with sklearn Pipeline)")
    cls_run = train_classifier(x_train, y_train)

    print("\nStep 4: Classification Evaluation")
    cls_metrics = evaluate_classifier(x_test, y_test, cls_run)

    print("\nStep 5: Regression Training (with sklearn Pipeline)")
    reg_run = train_regressor(xr_train, yr_train)

    print("\nStep 6: Regression Evaluation")
    reg_metrics = evaluate_regressor(xr_test, yr_test, reg_run)

    print("\n" + "=" * 50)
    if cls_metrics["f1_macro"] >= F1_THRESHOLD:
        print(f"Classifier APPROVED (f1_macro={cls_metrics['f1_macro']:.3f})")
    else:
        print(f"Classifier REJECTED (f1_macro={cls_metrics['f1_macro']:.3f} < {F1_THRESHOLD})")

    if reg_metrics["rmse"] <= RMSE_THRESHOLD:
        print(f"Regressor  APPROVED (rmse={reg_metrics['rmse']:.3f})")
    else:
        print(f"Regressor  REJECTED (rmse={reg_metrics['rmse']:.3f} > {RMSE_THRESHOLD})")


if __name__ == "__main__":
    run_pipeline()

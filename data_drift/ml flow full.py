import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
from sklearn.model_selection import train_test_split

#  Load datasets
reference_data_path = r"C:\Users\Minfy.DESKTOP-81ME0ME\Downloads\dataset loan.csv"
current_data_path = r"C:\Users\Minfy.DESKTOP-81ME0ME\Downloads\New Customer Bank_Personal_Loan.csv"

reference_full_df = pd.read_csv(reference_data_path)
reference_features = reference_full_df.drop(columns=["Personal Loan"])
reference_labels = reference_full_df["Personal Loan"]

current_df = pd.read_csv(current_data_path)

#  Set base experiment
mlflow.set_experiment("Drift Metrics Analysis")

def log_data_drift_metrics(reference_df, current_df, run_name):
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    drift_report.save_html("drift_report.html")
    report_dict = drift_report.as_dict()

    mlflow.set_experiment("Evidently_Drift_Metrics")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact("drift_report.html")

        for metric in report_dict["metrics"]:
            if metric.get("metric") == "DataDriftTable":
                result = metric["result"]
                n_drifted = result.get("number_of_drifted_columns", 0)
                n_total = result.get("number_of_columns", 1)
                drift_ratio = n_drifted / n_total
                mlflow.log_metric("drifted_column_count", drift_ratio)

                for feature, stats in result["drift_by_columns"].items():
                    score = stats.get("drift_score", 0)
                    mlflow.log_metric(f"drift_{feature}", score)

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(reference_features, reference_labels, test_size=0.2, random_state=42)

#  Run 1: Train vs Test
log_data_drift_metrics(reference_df=X_train, current_df=X_test, run_name="Train_vs_Test_UnderstandingOf_DataDrift")

#  Run 2: Historical vs New Customers
log_data_drift_metrics(reference_df=reference_features, current_df=current_df, run_name="Historical_vs_New_DataDrift_Understanding")

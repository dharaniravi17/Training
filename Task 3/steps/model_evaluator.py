from zenml import step
from evaluation import Evaluator
import mlflow

@step
def evaluate_models(lr_model, rf_model, X_test, y_test):
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    lr_eval = Evaluator(y_test, lr_pred)
    rf_eval = Evaluator(y_test, rf_pred)

    results = {
        "Linear Regression": {
            "MSE": lr_eval.mse(),
            "RMSE": lr_eval.rmse(),
            "R2": lr_eval.r2()
        },
        "Random Forest": {
            "MSE": rf_eval.mse(),
            "RMSE": rf_eval.rmse(),
            "R2": rf_eval.r2()
        }
    }

    for model_name, metrics in results.items():
        print(f"📊 {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            mlflow.log_metric(f"{model_name}_{metric}", value)
    return results

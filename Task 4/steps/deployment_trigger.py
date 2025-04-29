from zenml import step
from sklearn.metrics import r2_score

@step
def deployment_trigger(r2: float) -> bool:
    """Check if R2 score is above 0.7 to trigger model deployment."""
    if r2 >= 0.7:
        print("✅ R2 score is sufficient. Proceeding with deployment.")
        return True
    else:
        print("❌ R2 score is too low. Deployment will not occur.")
        return False

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zenml.client import Client
from pipelines.regression_pipeline import regression_pipeline

if __name__ == "__main__":
    active_stack = Client().active_stack
    if active_stack.experiment_tracker:
        print(f"✅ Active experiment tracker: {active_stack.experiment_tracker.flavor}")
    else:
        print("⚠️ No experiment tracker set.")

    regression_pipeline()

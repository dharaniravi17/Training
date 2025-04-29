from zenml import step
import pandas as pd
import numpy as np

@step
def dynamic_importer() -> np.ndarray:
    test_data = pd.DataFrame({
        "feature1": [5.1],
        "feature2": [3.5],
        "feature3": [1.4],
        "feature4": [0.2]
    })
    return test_data.values

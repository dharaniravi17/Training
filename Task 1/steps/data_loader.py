# steps/data_loader.py
from zenml import step
from sklearn.datasets import load_diabetes
import pandas as pd

@step
def load_data() -> pd.DataFrame:
    data = load_diabetes(as_frame=True)
    df = data.frame
    return df

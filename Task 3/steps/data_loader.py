from zenml import step
import pandas as pd
from sklearn.datasets import load_diabetes

@step
def load_data() -> pd.DataFrame:
    data = load_diabetes(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df

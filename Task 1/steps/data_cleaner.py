# steps/data_cleaner.py
from zenml import step
import pandas as pd

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

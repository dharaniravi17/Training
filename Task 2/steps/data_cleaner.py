from zenml import step
import pandas as pd

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.dropna()
    return cleaned_df

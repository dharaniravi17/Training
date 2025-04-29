import pandas as pd
from sklearn.preprocessing import LabelEncoder
from zenml.steps import step

@step
def custom_data_loader() -> pd.DataFrame:
    # Load your data
    df = pd.read_csv("D:\\Bootcamp\\Phase 1\\level18\\Mall_Customers.csv")

    # Check for categorical columns (e.g., 'Gender')
    if 'Genre' in df.columns:
        le = LabelEncoder()
        df['Genre'] = le.fit_transform(df['Genre'])
    
    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')  # Coerce non-numeric values to NaN
    df = df.dropna()  # Drop rows with NaN values

    return df

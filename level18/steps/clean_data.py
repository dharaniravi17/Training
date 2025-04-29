# steps/clean_data.py

import pandas as pd
from zenml import step

# Step 2: Define a cleaning step
@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns (e.g., 'CustomerID')
    data = data.drop(columns=['CustomerID'], errors='ignore')
    
    # Fill missing values:
    # For numerical columns, use the median for filling null values
    data.fillna(data.median(), inplace=True)
    
    # For categorical columns (e.g., 'Gender', 'SpendingScore'), fill with a constant
    data['Genre'].fillna('Unknown', inplace=True)  # Replace missing 'Gender' with 'Unknown'
    data['Spending Score (1-100)'].fillna('no review', inplace=True)  # Replace missing 'SpendingScore' with 'no review'
    
    # You can also use more sophisticated techniques for filling missing values based on column types or models
    
    return data

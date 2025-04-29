import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Load dataset from a local file
    df = pd.read_csv("D:\\titanic\\archive\\train_and_test2.csv")  # Make sure the file is in the same folder or provide the correct path

    # Select relevant columns
    df = df[['Pclass', 'Age', 'Fare', 'Survived']]

    # Handle missing data (e.g., fill Age with mean)
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # Encode target variable (Survived: 1 for survival, 0 for not-survival)
    X = df[['Pclass', 'Age', 'Fare']].values
    y = df['Survived'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

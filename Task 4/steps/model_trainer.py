from zenml import step
from sklearn.linear_model import LinearRegression

@step
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    model = LinearRegression()
    model.fit(X, y)
    return model

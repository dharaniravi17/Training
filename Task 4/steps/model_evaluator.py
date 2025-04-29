from zenml import step
from sklearn.metrics import r2_score

@step
def evaluate_model(model, df):
    X = df.drop('target', axis=1)
    y = df['target']
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    return r2

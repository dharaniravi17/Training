from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Evaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def rmse(self):
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

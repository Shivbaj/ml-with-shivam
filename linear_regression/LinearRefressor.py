import numpy as np

class LinearRegressor:
    
    def __init__(self, learning_rate):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        
    def fit(self, X, y, n_iterations):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(n_iterations):
            # Forward pass
            y_pred = X @ self.w + self.b
            
            # Compute error
            error = y_pred - y
            
            # Backpropagation (gradient computation)
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            
            # Update weights
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
    def predict(self, X):
        return X @ self.w + self.b
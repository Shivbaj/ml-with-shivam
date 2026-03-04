import numpy as np

class LinearRegressor:
    
    def __init__(self, learning_rate=0.01, reg_type=None, alpha=0.0):
        """
        reg_type: None, 'l1' (Lasso), or 'l2' (Ridge)
        alpha: regularization strength
        """
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.reg_type = reg_type
        self.alpha = alpha
        
    def fit(self, X, y, n_iterations=1000):
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(n_iterations):
            # Forward pass
            y_pred = X @ self.w + self.b
            
            # Compute error
            error = y_pred - y
            
            # Gradient computation
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            
            # Add regularization
            if self.reg_type == 'l2':  # Ridge
                dw += 2 * self.alpha * self.w
            elif self.reg_type == 'l1':  # Lasso
                dw += self.alpha * np.sign(self.w)
            
            # Update weights
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
    def predict(self, X):
        return X @ self.w + self.b
    


if __name__ == "__main__":
    # Example usage
    
    # Sample data
    X = np.array([[1,2],[3,4],[5,6]])
    y = np.array([3, 7, 11])

    # Plain Linear Regression
    model = LinearRegressor(learning_rate=0.01)
    model.fit(X, y, n_iterations=1000)
    print("Weights (Plain):", model.w)

    # Ridge Regression
    ridge = LinearRegressor(learning_rate=0.01, reg_type='l2', alpha=0.1)
    ridge.fit(X, y, n_iterations=1000)
    print("Weights (Ridge):", ridge.w)

    # Lasso Regression
    lasso = LinearRegressor(learning_rate=0.01, reg_type='l1', alpha=0.1)
    lasso.fit(X, y, n_iterations=1000)
    print("Weights (Lasso):", lasso.w)
    
    predictions = model.predict(X)
    print("Predictions:", predictions)
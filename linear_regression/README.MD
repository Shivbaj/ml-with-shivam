# Linear Regression Implementation

This folder contains implementations of linear regression from scratch using NumPy, including variants with regularization.

## 📁 Files

### 1. `LinearRefressor.py`
**Basic Linear Regression with Gradient Descent**

A simple implementation of linear regression using gradient descent optimization. This is the foundational version without any regularization.

**Features:**
- Gradient descent optimization
- Configurable learning rate
- Customizable number of iterations
- Matrix operations using NumPy for efficiency

**Usage:**
```python
from LinearRefressor import LinearRegressor

model = LinearRegressor(learning_rate=0.01)
model.fit(X_train, y_train, n_iterations=1000)
predictions = model.predict(X_test)
```

**When to use:**
- Small datasets with few features
- When features are mostly uncorrelated
- As a baseline model before trying regularization

---

### 2. `LinearRegressor_regularisation.py`
**Linear Regression with Ridge (L2) and Lasso (L1) Regularization**

An enhanced version that supports both Ridge and Lasso regularization to handle overfitting and multicollinearity.

**Features:**
- All features from basic linear regression
- Ridge (L2) regularization support
- Lasso (L1) regularization support
- Configurable regularization strength (alpha)
- Example usage with all three variants

**Usage:**
```python
from LinearRegressor_regularisation import LinearRegressor

# Plain Linear Regression
model = LinearRegressor(learning_rate=0.01)
model.fit(X, y, n_iterations=1000)

# Ridge Regression (L2)
ridge = LinearRegressor(learning_rate=0.01, reg_type='l2', alpha=0.1)
ridge.fit(X, y, n_iterations=1000)

# Lasso Regression (L1)
lasso = LinearRegressor(learning_rate=0.01, reg_type='l1', alpha=0.1)
lasso.fit(X, y, n_iterations=1000)
```

**When to use:**
- Datasets with many features
- When features are correlated (multicollinearity)
- When you suspect overfitting
- When you want automatic feature selection (Lasso)

---

## 🎯 Problem: Why Ridge & Lasso Regularization?

### The Problems They Solve

**1. Overfitting**
- If you have many features, your model may fit noise instead of signal
- Model performs well on training data but poorly on new data

**2. Multicollinearity**
- Highly correlated features → weights explode → unstable predictions
- Small changes in data lead to large changes in predictions

**Solution:** Add a penalty term to discourage large weights, keeping the model stable.

---

## 🛡️ Ridge & Lasso: A Safety Net

Even if you try to pick uncorrelated features, real datasets are messy. Ridge & Lasso act as a safety net.

### Ridge Regression (L2 Regularization)

**What it does:**
- Shrinks all weights toward zero
- Prevents weights from exploding when features are correlated
- Keeps all features (none become exactly zero)

**Intuition:**
- Adds penalty proportional to **square of weights**: `α * Σ(w²)`
- Larger weights get penalized more heavily
- Reduces overfitting while keeping all features

**Best for:**
- When all features are potentially useful
- When you have multicollinearity
- When you want smooth, stable predictions

---

### Lasso Regression (L1 Regularization)

**What it does:**
- Can shrink unimportant or redundant features **exactly to zero**
- Performs automatic feature selection
- Helps when you have many irrelevant features

**Intuition:**
- Adds penalty proportional to **absolute value of weights**: `α * Σ|w|`
- Can drive small weights all the way to zero
- Creates sparse models (many zero weights)

**Best for:**
- When you have many features but only some are important
- When you want feature selection
- When you want an interpretable model with fewer features

---

## 📊 How Lasso Drives Weights to Zero

Here's an example showing how Lasso can eliminate a weak feature:

| Feature | Current w | MSE Gradient | α=0.02 | Update Step                   | New w  |
|---------|-----------|--------------|--------|-------------------------------|--------|
| x1      | 0.5       | 0.1          | 0.02   | w - (0.1+0.02) = 0.5-0.12     | 0.38   |
| x2      | 0.01      | 0.005        | 0.02   | w - (0.005+0.02) = 0.01-0.025 | -0.015 |
| x2 next | -0.015    | 0.005        | 0.02   | w - (-0.015) = 0               | 0 ✅   |

Notice how the small weight for feature x2 gets pushed to exactly zero!

---

## 🚀 Quick Start

1. **Start with basic linear regression** (`LinearRefressor.py`) to establish a baseline
2. **If overfitting occurs**, try Ridge regression with small alpha (e.g., 0.01)
3. **If you have many features**, try Lasso to identify important ones
4. **Tune alpha** using cross-validation to find the optimal regularization strength

---

## 🔧 Parameters Guide

- **learning_rate**: Step size for gradient descent (typical: 0.01 to 0.1)
- **n_iterations**: Number of training iterations (typical: 1000 to 10000)
- **reg_type**: `None`, `'l1'` (Lasso), or `'l2'` (Ridge)
- **alpha**: Regularization strength (typical: 0.001 to 1.0)
  - Smaller alpha = less regularization
  - Larger alpha = more regularization

---

## 📚 Mathematical Formulas

**Plain Linear Regression Loss:**
```
Loss = (1/n) * Σ(y_pred - y_true)²
```

**Ridge (L2) Loss:**
```
Loss = (1/n) * Σ(y_pred - y_true)² + α * Σ(w²)
```

**Lasso (L1) Loss:**
```
Loss = (1/n) * Σ(y_pred - y_true)² + α * Σ|w|
```

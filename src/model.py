import numpy as np
# Switching to Regressors for predicting continuous percentage change
from sklearn.linear_model import LinearRegression # Simple Linear Model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR # Support Vector Regressor
from sklearn.naive_bayes import GaussianNB # Naive Bayes doesn't have a direct regressor counterpart, so we omit for now.

# --- 1. LINEAR BASELINE ---
class IntelliTradeLinReg(LinearRegression):
    """Simple Linear Regression Baseline."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# --- 2. RANDOM FOREST REGRESSOR VARIATIONS ---
class IntelliTradeRF(RandomForestRegressor):
    """Base Random Forest Regressor."""
    def __init__(self, **kwargs):
        super().__init__(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, **kwargs)

class IntelliTradeRFMassive(RandomForestRegressor):
    """RF Regressor: Massive Estimators (Highest Previous Scorer Family)."""
    def __init__(self, **kwargs):
        super().__init__(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1, **kwargs)

class IntelliTradeRFMinLeaf(RandomForestRegressor):
    """RF Regressor: Smoother Trees (Min Samples Leaf=10)."""
    def __init__(self, **kwargs):
        super().__init__(n_estimators=150, max_depth=15, min_samples_leaf=10, random_state=42, n_jobs=-1, **kwargs)


# --- 3. GRADIENT BOOSTING REGRESSOR VARIATIONS ---
class IntelliTradeGBC(GradientBoostingRegressor):
    """Base Gradient Boosting Regressor."""
    def __init__(self, **kwargs):
        super().__init__(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, **kwargs)


# --- 4. OTHER REGRESSOR TYPES ---
class IntelliTradeKNN(KNeighborsRegressor):
    """K-Nearest Neighbors Regressor."""
    def __init__(self, **kwargs):
        super().__init__(n_neighbors=15, **kwargs)

class IntelliTradeMLP(MLPRegressor):
    """Multilayer Perceptron Regressor (Neural Network)."""
    def __init__(self, **kwargs):
        super().__init__(hidden_layer_sizes=(100, 50, 10), max_iter=1000, early_stopping=True, random_state=42, **kwargs)
        
class IntelliTradeSVC(SVR):
    """Support Vector Regressor (The Previous Winner Family)."""
    def __init__(self, **kwargs):
        super().__init__(kernel='rbf', C=1.0, gamma='scale', **kwargs)


# Export all model classes for use in pipeline.py
__all__ = [
    'IntelliTradeLinReg', 
    'IntelliTradeRF', 'IntelliTradeRFMassive', 'IntelliTradeRFMinLeaf', 
    'IntelliTradeGBC',
    'IntelliTradeKNN', 'IntelliTradeMLP', 'IntelliTradeSVC'
]
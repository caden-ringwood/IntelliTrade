import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# --- 1. LINEAR BASELINE (Kept for benchmark) ---
class IntelliTradeLogReg(LogisticRegression):
    def __init__(self, **kwargs):
        # Default parameters with increased max_iter for convergence
        super().__init__(random_state=42, **kwargs)

# --- 2. RANDOM FOREST VARIATIONS ---
class IntelliTradeRF(RandomForestClassifier):
    def __init__(self, **kwargs):
        # Base Model: Balanced number of estimators and depth
        super().__init__(n_estimators=100, max_depth=10, random_state=42, **kwargs)

class IntelliTradeRFSparse(RandomForestClassifier):
    def __init__(self, **kwargs):
        # Sparse/Fast: Fewer estimators, less depth
        super().__init__(n_estimators=50, max_depth=5, random_state=42, **kwargs)

class IntelliTradeRFDeep(RandomForestClassifier):
    def __init__(self, **kwargs):
        # Deep/Complex: More estimators, higher depth (current highest scorer)
        super().__init__(n_estimators=200, max_depth=20, random_state=42, **kwargs)

class IntelliTradeRFMassive(RandomForestClassifier):
    def __init__(self, **kwargs):
        # NEW: Massive Estimators (A lot of trees)
        super().__init__(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1, **kwargs)

class IntelliTradeRFShallow(RandomForestClassifier):
    def __init__(self, **kwargs):
        # NEW: Highly Regularized (Very shallow trees)
        super().__init__(n_estimators=150, max_depth=3, random_state=42, **kwargs)
        
class IntelliTradeRFMinLeaf(RandomForestClassifier):
    def __init__(self, **kwargs):
        # NEW: Min Samples Leaf (Forces larger, smoother leaf nodes)
        super().__init__(n_estimators=150, max_depth=15, min_samples_leaf=10, random_state=42, **kwargs)


# --- 3. GRADIENT BOOSTING VARIATIONS ---
class IntelliTradeGBC(GradientBoostingClassifier):
    def __init__(self, **kwargs):
        # Base Model: Standard settings
        super().__init__(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, **kwargs)

class IntelliTradeGBCFast(GradientBoostingClassifier):
    def __init__(self, **kwargs):
        # Fast Learning: Higher learning rate, fewer trees
        super().__init__(n_estimators=50, max_depth=3, learning_rate=0.2, random_state=42, **kwargs)

class IntelliTradeGBCDeep(GradientBoostingClassifier):
    def __init__(self, **kwargs):
        # Deep/Precise: More trees, higher depth
        super().__init__(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, **kwargs)

# --- 4. OTHER MODEL TYPES ---
class IntelliTradeKNN(KNeighborsClassifier):
    def __init__(self, **kwargs):
        # K-Nearest Neighbors: uses 15 neighbors
        super().__init__(n_neighbors=15, **kwargs)

class IntelliTradeMLP(MLPClassifier):
    def __init__(self, **kwargs):
        # Neural Network: Simple 3-layer net
        super().__init__(hidden_layer_sizes=(100, 50, 10), max_iter=1000, early_stopping=True, random_state=42, **kwargs)
        
class IntelliTradeGNB(GaussianNB):
    def __init__(self, **kwargs):
        # Naive Bayes: Fast probabilistic model
        super().__init__(**kwargs)

# --- 5. SUPPORT VECTOR MACHINE (SVC) VARIATIONS (Now a Secondary Focus) ---
class IntelliTradeSVC(SVC):
    def __init__(self, **kwargs):
        # SVC (Base): Standard C=1.0, gamma='scale' (Tested highest in previous runs)
        super().__init__(kernel='rbf', C=1.0, gamma='scale', random_state=42, **kwargs)

class IntelliTradeSVCSmooth(SVC):
    def __init__(self, **kwargs):
        # SVC (Smooth): Low C (C=0.1) for high regularization (smoother boundary)
        super().__init__(kernel='rbf', C=0.1, gamma='scale', random_state=42, **kwargs)
        
class IntelliTradeSVCSharp(SVC):
    def __init__(self, **kwargs):
        # SVC (Sharp): High C (C=10.0) for low regularization (sharper boundary)
        super().__init__(kernel='rbf', C=10.0, gamma='scale', random_state=42, **kwargs)

class IntelliTradeSVCFast(SVC):
    def __init__(self, **kwargs):
        # SVC (Low Gamma): Large influence, simpler decision boundary (gamma=0.01)
        super().__init__(kernel='rbf', C=1.0, gamma=0.01, random_state=42, **kwargs)

class IntelliTradeSVCComplex(SVC):
    def __init__(self, **kwargs):
        # SVC (High Gamma): Small influence, highly complex/overfit boundary (gamma=1.0)
        super().__init__(kernel='rbf', C=1.0, gamma=1.0, random_state=42, **kwargs)

# Export all model classes for use in pipeline.py
__all__ = [
    'IntelliTradeLogReg', 
    'IntelliTradeRF', 'IntelliTradeRFSparse', 'IntelliTradeRFDeep', 
    'IntelliTradeRFMassive', 'IntelliTradeRFShallow', 'IntelliTradeRFMinLeaf', # NEW RF EXPORTS
    'IntelliTradeGBC', 'IntelliTradeGBCFast', 'IntelliTradeGBCDeep', 
    'IntelliTradeKNN', 'IntelliTradeMLP', 'IntelliTradeGNB', 
    'IntelliTradeSVC', 'IntelliTradeSVCSmooth', 'IntelliTradeSVCSharp', 
    'IntelliTradeSVCFast', 'IntelliTradeSVCComplex' 
]

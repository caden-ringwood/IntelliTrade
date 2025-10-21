import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC # New Import: Support Vector Classifier
from sklearn.naive_bayes import GaussianNB # New Import: Naive Bayes
import warnings

# Suppress ConvergenceWarning from LogReg and other warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- BASE MODELS (Used for initial comparison) ---

class IntelliTradeLogReg(LogisticRegression):
    """Simple Logistic Regression Baseline."""
    def __init__(self, max_iter=5000, random_state=42, **kwargs):
        super().__init__(max_iter=max_iter, random_state=random_state, **kwargs)

class IntelliTradeRF(RandomForestClassifier):
    """General Random Forest (Medium complexity)."""
    def __init__(self, n_estimators=150, max_depth=8, random_state=42, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs)

class IntelliTradeGBC(GradientBoostingClassifier):
    """General Gradient Boosting (Medium complexity)."""
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state, **kwargs)


# --- TUNED VARIATIONS (Optimizing Hyperparameters) ---

class IntelliTradeRFSparse(RandomForestClassifier):
    """Random Forest: Sparse/Fast. Uses few trees and shallow depth."""
    def __init__(self, n_estimators=50, max_depth=5, random_state=42, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs)

class IntelliTradeRFDeep(RandomForestClassifier):
    """Random Forest: Complex/Deep. Uses many trees and greater depth."""
    def __init__(self, n_estimators=300, max_depth=12, random_state=42, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=12, random_state=random_state, **kwargs)

class IntelliTradeGBCFast(GradientBoostingClassifier):
    """Gradient Boosting: Fast Learning. Higher learning rate for quicker convergence."""
    def __init__(self, learning_rate=0.2, n_estimators=100, max_depth=3, random_state=42, **kwargs):
        super().__init__(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs)

class IntelliTradeGBCDeep(GradientBoostingClassifier):
    """Gradient Boosting: Deep/Precise. Deeper trees and more estimators for precision."""
    def __init__(self, learning_rate=0.1, n_estimators=150, max_depth=5, random_state=42, **kwargs):
        super().__init__(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=5, random_state=random_state, **kwargs)
        
        
# --- NEW MODELS: KNN and MLP ---

class IntelliTradeKNN(KNeighborsClassifier):
    """K-Nearest Neighbors: Instance-based classification."""
    def __init__(self, n_neighbors=15, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs) 
        
class IntelliTradeMLP(MLPClassifier):
    """Multilayer Perceptron (Shallow Neural Network)."""
    def __init__(self, hidden_layer_sizes=(10, 5), max_iter=500, random_state=42, **kwargs):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state, **kwargs)

# --- NEW MODELS: SVM and Naive Bayes ---

class IntelliTradeSVC(SVC):
    """Support Vector Classifier: Finds the optimal separating hyperplane."""
    def __init__(self, kernel='rbf', gamma='scale', random_state=42, **kwargs):
        # We use 'rbf' kernel for non-linear separation
        super().__init__(kernel=kernel, gamma=gamma, random_state=random_state, **kwargs)

class IntelliTradeGNB(GaussianNB):
    """Gaussian Naive Bayes: Fast, probabilistic classifier."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

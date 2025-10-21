import pandas as pd
import numpy as np
import yfinance as yf 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.compose import ColumnTransformer # Added ColumnTransformer
from sklearn.metrics import accuracy_score
import joblib
import os
from collections import namedtuple

# Import the model classes we defined in model.py
from model import IntelliTradeLogReg, IntelliTradeRF, IntelliTradeGBC, \
                   IntelliTradeRFSparse, IntelliTradeRFDeep, \
                   IntelliTradeGBCFast, IntelliTradeGBCDeep, \
                   IntelliTradeKNN, IntelliTradeMLP, \
                   IntelliTradeSVC, IntelliTradeGNB 

# --- Configuration ---
# Your revised, verified list of 30 tickers (RR replaced with RTX - Raytheon)
TICKERS = [
    'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'META',
    'JPM', 'GS', 'V', 
    'JNJ', 'PFE', 'UNH', 
    'XOM', 'CVX', 'COP',
    'WMT', 'KO', 'PG', 
    'RGR', 'AAT', 'BKE', 'RTX', # RTX added (Aerospace/Defense)
    'TSLA', 'F', 'GM',
    'AMZN', 'EBAY', 'BABA',
    'SBUX', 'MCD', 'PEP'
] 
START_DATE = '2010-01-01' 
SCALER_OUTPUT_PATH = '../assets/scaler.joblib'
PREPROCESSOR_OUTPUT_PATH = '../assets/preprocessor.joblib' # New path for the full preprocessor

# Define the candidate models to test
ModelCandidate = namedtuple('ModelCandidate', ['name', 'instance'])
MODEL_CANDIDATES = [
    # 1. LINEAR BASELINE
    ModelCandidate("LogReg (Base)", IntelliTradeLogReg(max_iter=5000)),
    
    # 2. RANDOM FOREST VARIATIONS
    ModelCandidate("RF (Base)", IntelliTradeRF()),
    ModelCandidate("RF (Sparse/Fast)", IntelliTradeRFSparse()), 
    ModelCandidate("RF (Deep/Complex)", IntelliTradeRFDeep()),   
    
    # 3. GRADIENT BOOSTING VARIATIONS
    ModelCandidate("GBC (Base)", IntelliTradeGBC()),
    ModelCandidate("GBC (Fast Learn)", IntelliTradeGBCFast()),   
    ModelCandidate("GBC (Deep/Precise)", IntelliTradeGBCDeep()), 
    
    # 4. OTHER MODEL TYPES
    ModelCandidate("KNN (15 Neighbors)", IntelliTradeKNN()),
    ModelCandidate("MLP (Neural Net)", IntelliTradeMLP()),
    ModelCandidate("SVC (RBF Kernel)", IntelliTradeSVC()),
    ModelCandidate("Naive Bayes (GNB)", IntelliTradeGNB()),
]

# --- 1. Data Ingestion & Feature Engineering ---

def load_data(tickers=TICKERS, start=START_DATE):
    """
    Fetches historical stock data for multiple tickers, using the most robust 
    method to handle the MultiIndex structure.
    """
    print(f"-> Downloading data for {len(tickers)} assets from {start}...")
    
    try:
        # Download all data (returns a MultiIndex DataFrame)
        df = yf.download(tickers, start=start, auto_adjust=True) 
        print("✅ Data download successful.")
        
        # Check if the DataFrame columns are fully empty after download
        if df.empty:
            raise ValueError("Downloaded DataFrame is entirely empty.")
        
        # --- ROBUST MULTIINDEX EXTRACTION ---
        if isinstance(df.columns, pd.MultiIndex):
            
            # Select relevant data columns
            valid_metrics = ['Close', 'Volume']
            df_cleaned = []
            
            for metric in valid_metrics:
                if metric not in df.columns.get_level_values(0):
                    print(f"⚠️ Warning: Missing metric '{metric}' in downloaded data.")
                    continue
                
                # Extract the metric columns for all tickers
                metric_cols = [col for col in df.columns if col[0] == metric]
                temp_df = df[metric_cols].copy()
                
                # Rename columns from ('Metric', 'Ticker') to 'Ticker_Metric'
                temp_df.columns = [f"{col[1]}_{metric}" for col in metric_cols]
                df_cleaned.append(temp_df)

            df_combined = pd.concat(df_cleaned, axis=1)
            
            if df_combined.empty:
                raise ValueError("DataFrame is empty after cleanup. Check tickers/dates.")
                
            return df_combined
        
        # Fallback for single ticker (unlikely but safe)
        elif all(col in df.columns for col in ['Close', 'Volume']):
            return df[['Close', 'Volume']]
        else:
            raise ValueError("Downloaded data structure is unusable.")

    except Exception as e:
        print(f"❌ CRITICAL DATA PROCESSING ERROR: {e}")
        return pd.DataFrame()


def feature_engineering(df_raw):
    """
    Creates the target variable and multiple, more predictive feature variables 
    across all tickers, including Volume and Day-of-Week effects.
    """
    print("-> Engineering Features...")
    
    all_X = []
    all_y = []
    
    # Identify unique tickers based on column names (e.g., 'MSFT_Close')
    ticker_cols = [col.split('_')[0] for col in df_raw.columns if col.endswith('_Close')]
    
    # Iterate over the tickers
    for ticker_symbol in set(ticker_cols):
        
        # Create a temporary DataFrame for feature calculation
        # Use .copy() to avoid SettingWithCopyWarning
        ticker_df = pd.DataFrame({
            'Close': df_raw[f'{ticker_symbol}_Close'],
            'Volume': df_raw[f'{ticker_symbol}_Volume']
        }).copy()
        
        # --- 1. TARGET (y): Price direction prediction (1=Up, 0=Down)
        ticker_df['Target'] = (ticker_df['Close'].shift(-1) > ticker_df['Close']).astype(int)
        
        # --- 2. CORE FEATURES (Momentum and Volatility) ---
        
        # Feature 1: Moving Average Delta 
        ticker_df['MA_20'] = ticker_df['Close'].rolling(window=20).mean()
        ticker_df['Feature_MA_Delta'] = ticker_df['MA_20'].pct_change(fill_method=None) * 100
        
        # Feature 2: Relative Strength Index (RSI - 14 day standard)
        delta = ticker_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        with np.errstate(divide='ignore'):
            RS = gain / loss
        ticker_df['Feature_RSI'] = 100 - (100 / (1 + RS))
        
        # Feature 3: Bollinger Bands Percentage (%B)
        std = ticker_df['Close'].rolling(window=20).std()
        upper_band = ticker_df['MA_20'] + (2 * std)
        lower_band = ticker_df['MA_20'] - (2 * std)
        with np.errstate(divide='ignore', invalid='ignore'):
            ticker_df['Feature_BBPct'] = (ticker_df['Close'] - lower_band) / (upper_band - lower_band)
            
        # --- 3. NEW FEATURES (Volume and Time) ---
        
        # Feature 4: Volume Shock (Z-Score of Volume)
        volume_mean = ticker_df['Volume'].rolling(window=20).mean()
        volume_std = ticker_df['Volume'].rolling(window=20).std()
        ticker_df['Feature_Volume_ZScore'] = (ticker_df['Volume'] - volume_mean) / volume_std
        
        # Feature 5: Day of Week (Categorical Feature)
        # 0=Monday, 4=Friday. (The model needs to see if Friday behavior is unique)
        ticker_df['Feature_DayOfWeek'] = ticker_df.index.dayofweek.astype('category')
        
        # --- CLEANUP ---
        
        # Remove any rows with NaN values created by rolling windows/shifts (RSI/BB/MA need 14-20 days)
        ticker_df.dropna(inplace=True) 
        
        # CRITICAL CHECK: Ensure the DataFrame is NOT empty
        if not ticker_df.empty:
            all_X.append(ticker_df)
            all_y.append(ticker_df['Target'])
        else:
            print(f"⚠️ Skipping {ticker_symbol}: Not enough data after feature engineering.")

    # Check if we have any data at all before concatenating
    if not all_X:
        print("❌ ERROR: No usable data remaining across all tickers.")
        return pd.DataFrame(), pd.Series() 

    # Combine all usable tickers 
    X_raw = pd.concat(all_X)
    y = pd.concat(all_y)
    
    # Identify the final features
    numerical_features = ['Feature_MA_Delta', 'Feature_RSI', 'Feature_BBPct', 'Feature_Volume_ZScore']
    categorical_features = ['Feature_DayOfWeek']
    
    X = X_raw[numerical_features + categorical_features]
    
    # Shuffle the combined data to mix the tickers before splitting
    combined = pd.concat([X, y], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_shuffled = combined[numerical_features + categorical_features]
    y_shuffled = combined['Target']
    
    print(f"✅ Total usable samples for training: {len(X_shuffled)}")
    return X_shuffled, y_shuffled


# --- 2. Model Training and Evaluation ---

def train_and_compare(X, y, candidates):
    """
    Trains multiple model candidates, compares their test accuracy, 
    and returns the best-performing model instance.
    """
    if X.empty or y.empty:
        print("\n❌ Comparison aborted: Input data is empty.")
        return None
    
    # Identify feature types for Sklearn preprocessing
    numerical_features = ['Feature_MA_Delta', 'Feature_RSI', 'Feature_BBPct', 'Feature_Volume_ZScore']
    categorical_features = ['Feature_DayOfWeek']

    # Create the preprocessing pipeline
    # Numerical data is Scaled (StandardScaler)
    # Categorical data is One-Hot Encoded (OneHotEncoder)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Split the data once for consistent comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit the preprocessor only on the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the full preprocessor, not just the scaler, as it now handles encoding too!
    save_artifacts(preprocessor, PREPROCESSOR_OUTPUT_PATH, type="Preprocessor (Scaler+Encoder)")
    
    best_accuracy = 0
    best_model = None
    results = {}

    print("\n" + "="*50)
    print(f"STARTING MODEL COMPARISON ({len(candidates)} CANDIDATES) with new features.")
    print("="*50)

    for name, model_instance in candidates:
        print(f"\n--- Training {name} ---")
        
        # Train on the fully processed (scaled and encoded) data
        model_instance.fit(X_train_processed, y_train)
        
        # Evaluate
        y_pred = model_instance.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"Test Accuracy for {name}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (name, model_instance)

    print("\n" + "="*50)
    print("COMPARISON RESULTS:")
    for name, acc in results.items():
        print(f" - {name}: {acc:.4f}")
        
    if best_model:
        best_name, best_instance = best_model
        print(f"\n🏆 BEST MODEL: {best_name} with Accuracy: {best_accuracy:.4f}")
        return best_model
    else:
        print("\nFailed to train any models.")
        return None

# --- 3. Deployment/Serialization ---

def save_artifacts(artifact, path, type="Model"):
    """Saves the trained model or scaler."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    joblib.dump(artifact, path)
    print(f"✅ {type} saved to: {path}")


# --- Main Pipeline Execution ---

if __name__ == '__main__':
    
    # 1. Load Data
    data = load_data()
    
    if data.empty:
        print("Pipeline aborted due to data loading error.")
    else:
        # 2. Engineer Features
        X, y = feature_engineering(data)
        
        # 3. Train and Compare
        if not X.empty:
            best_model = train_and_compare(X, y, MODEL_CANDIDATES)
        else:
            best_model = None
        
        # 4. Save the BEST Model
        if best_model:
            best_name, best_instance = best_model 
            
            final_path = f'../assets/{best_name.lower().replace(" ", "_")}_model.joblib'
            save_artifacts(best_instance, final_path, type=f"Best Model ({best_name})")

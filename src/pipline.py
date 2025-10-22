import pandas as pd
import numpy as np
import yfinance as yf 
# REMOVED: from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import accuracy_score
import joblib
import os
from collections import namedtuple

# Import the model classes we defined in model.py
from model import IntelliTradeLogReg, IntelliTradeRF, IntelliTradeGBC, \
                   IntelliTradeRFSparse, IntelliTradeRFDeep, \
                   IntelliTradeGBCFast, IntelliTradeGBCDeep, \
                   IntelliTradeKNN, IntelliTradeMLP, \
                   IntelliTradeSVC, IntelliTradeSVCSmooth, IntelliTradeSVCSharp, \
                   IntelliTradeSVCFast, IntelliTradeSVCComplex, \
                   IntelliTradeGNB # Keep GNB import for simplicity, even if not used

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
PREPROCESSOR_OUTPUT_PATH = '../assets/preprocessor.joblib' 
TEMPORAL_SPLIT_DATE = '2023-01-01' # NEW: Split data at this date: Train before, Test after.

# Define the candidate models to test
ModelCandidate = namedtuple('ModelCandidate', ['name', 'instance'])
MODEL_CANDIDATES = [
    # 1. TOP PERFORMERS (The New RF Focus)
    ModelCandidate("RF (Deep/Complex)", IntelliTradeRFDeep()), 
    ModelCandidate("RF (Massive Est)", IntelliTradeRFMassive()), 
    ModelCandidate("RF (Shallow Reg)", IntelliTradeRFShallow()), 
    ModelCandidate("RF (Min Leaf 10)", IntelliTradeRFMinLeaf()), 

    # 2. COMPETITIVE BENCHMARKS
    ModelCandidate("SVC (Sharp C=10)", IntelliTradeSVCSharp()), 
    ModelCandidate("KNN (15 Neighbors)", IntelliTradeKNN()),      
    ModelCandidate("GBC (Base)", IntelliTradeGBC()),             

    # 3. SVC TUNING VARIATIONS (To complete the tuning matrix)
    ModelCandidate("SVC (Base C=1)", IntelliTradeSVC()),
    ModelCandidate("SVC (Smooth C=0.1)", IntelliTradeSVCSmooth()),
    ModelCandidate("SVC (Low Gamma 0.01)", IntelliTradeSVCFast()),
    ModelCandidate("SVC (High Gamma 1.0)", IntelliTradeSVCComplex()),
    
    # 4. Neural Network (For comparison)
    ModelCandidate("MLP (Neural Net)", IntelliTradeMLP()),
]

# --- 1. Data Ingestion & Feature Engineering (UNCHANGED) ---

def load_data(tickers=TICKERS, start=START_DATE):
    """
    Fetches historical stock data for multiple tickers, using the most robust 
    method to handle the MultiIndex structure.
    """
    print(f"-> Downloading data for {len(tickers)} assets from {start}...")
    
    try:
        # Download all data (returns a MultiIndex DataFrame)
        df = yf.download(tickers, start=start) 
        print("✅ Data download successful.")
        
        # Check if the DataFrame columns are fully empty after download
        if df.empty:
            raise ValueError("Downloaded DataFrame is entirely empty.")
        
        # --- ROBUST MULTIINDEX EXTRACTION (Extracting Close, High, Low, Volume) ---
        if isinstance(df.columns, pd.MultiIndex):
            
            # Select relevant data columns
            valid_metrics = ['Close', 'High', 'Low', 'Volume'] 
            df_cleaned = []
            
            for metric in valid_metrics:
                # Check for metric existence before processing
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
        elif all(col in df.columns for col in ['Close', 'High', 'Low', 'Volume']):
            return df[['Close', 'High', 'Low', 'Volume']]
        else:
            raise ValueError("Downloaded data structure is unusable.")

    except Exception as e:
        print(f"❌ CRITICAL DATA PROCESSING ERROR: {e}")
        return pd.DataFrame()


def feature_engineering(df_raw):
    """
    Adds Lagged Price, Volume Shock, Day of Week, and Average True Range (ATR).
    Filters tickers to ensure all necessary columns exist before processing.
    """
    print("-> Engineering Features...")
    
    all_X = []
    all_y = []
    
    # 1. Determine which tickers have ALL the required columns
    required_suffixes = ['_Close', '_High', '_Low', '_Volume'] 
    
    # Identify unique tickers that were downloaded successfully
    all_downloaded_tickers = list(set([col.split('_')[0] for col in df_raw.columns if '_' in col]))
    
    # Filter for tickers that have ALL required columns for feature calculation
    valid_ticker_symbols = []
    for ticker in all_downloaded_tickers:
        has_all_data = all(f'{ticker}{suffix}' in df_raw.columns for suffix in required_suffixes)
        if has_all_data:
            valid_ticker_symbols.append(ticker)
        else:
            # This is where EBAY likely failed. We skip it, which is the correct robust behavior.
            print(f"⚠️ Skipping {ticker}: Missing one or more required columns ({[f'{ticker}{suffix}' for suffix in required_suffixes if f'{ticker}{suffix}' not in df_raw.columns]}).")
            
    print(f"Using {len(valid_ticker_symbols)}/{len(TICKERS)} tickers with complete data.")
    
    # 2. Iterate only over the filtered, valid tickers
    for ticker_symbol in valid_ticker_symbols:
        
        # Create a temporary DataFrame for feature calculation
        # Safely extract data columns for the valid ticker
        ticker_df = pd.DataFrame({
            'Close': df_raw[f'{ticker_symbol}_Close'], 
            'High': df_raw[f'{ticker_symbol}_High'],
            'Low': df_raw[f'{ticker_symbol}_Low'],
            'Volume': df_raw[f'{ticker_symbol}_Volume']
        }).copy()
        
        # --- 1. TARGET (y): Price direction prediction (1=Up, 0=Down)
        ticker_df['Target'] = (ticker_df['Close'].shift(-1) > ticker_df['Close']).astype(int)
        
        # --- 2. CORE TECHNICAL FEATURES (Momentum, Volatility, Delta) ---
        
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
            
        # --- 3. NEW FEATURES FOR IMPROVED PREDICTIVE POWER ---
        
        # Feature 4: Volume Shock (Z-Score of Volume)
        volume_mean = ticker_df['Volume'].rolling(window=20).mean()
        volume_std = ticker_df['Volume'].rolling(window=20).std()
        ticker_df['Feature_Volume_ZScore'] = (ticker_df['Volume'] - volume_mean) / volume_std
        
        # Feature 5: Average True Range (ATR) - Volatility
        # Calculate True Range: max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        high_low = ticker_df['High'] - ticker_df['Low']
        # Use Close price for the previous day's close in ATR.
        high_prev_close = np.abs(ticker_df['High'] - ticker_df['Close'].shift(1)) 
        low_prev_close = np.abs(ticker_df['Low'] - ticker_df['Close'].shift(1))
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        # ATR is the 14-day EMA of the True Range
        ticker_df['Feature_ATR'] = true_range.ewm(span=14, adjust=False).mean()

        # Feature 6 & 7: Lagged Price Change (Time Series Dependency)
        ticker_df['Feature_Lag1_Pct'] = ticker_df['Close'].pct_change(periods=1) * 100
        ticker_df['Feature_Lag5_Pct'] = ticker_df['Close'].pct_change(periods=5) * 100
        
        # Feature 8: Day of Week (Categorical Feature)
        ticker_df['Feature_DayOfWeek'] = ticker_df.index.dayofweek.astype('category')
        
        # --- CLEANUP ---
        
        # Identify the final list of features
        features = [
            'Feature_MA_Delta', 'Feature_RSI', 'Feature_BBPct', 
            'Feature_Volume_ZScore', 'Feature_ATR', 'Feature_Lag1_Pct', 
            'Feature_Lag5_Pct', 'Feature_DayOfWeek'
        ]
        
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
    
    # Identify the final list of features again for the final DataFrame
    numerical_features = ['Feature_MA_Delta', 'Feature_RSI', 'Feature_BBPct', 'Feature_Volume_ZScore', 'Feature_ATR', 'Feature_Lag1_Pct', 'Feature_Lag5_Pct']
    categorical_features = ['Feature_DayOfWeek']
    
    X = X_raw[numerical_features + categorical_features]
    
    # We remove the random shuffle here and rely on the index for temporal splitting
    combined = pd.concat([X, y], axis=1).sort_index() 
    
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
    
    # -------------------------------------------------------------------
    # NEW: TEMPORAL SPLIT (No more random shuffle - avoids look-ahead bias)
    # -------------------------------------------------------------------
    split_date = pd.to_datetime(TEMPORAL_SPLIT_DATE)
    
    # Find the index where the date crosses the split threshold
    # Since we sorted by index (date) in feature_engineering, this is safe
    split_idx = X.index.get_loc(split_date, method='nearest')
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n📊 Splitting Data Temporally:")
    print(f"   Training samples (Before {TEMPORAL_SPLIT_DATE}): {len(X_train)}")
    print(f"   Testing samples (After {TEMPORAL_SPLIT_DATE}): {len(X_test)}")
    
    # --- Preprocessing (Scaler and Encoder) ---
    numerical_features = ['Feature_MA_Delta', 'Feature_RSI', 'Feature_BBPct', 'Feature_Volume_ZScore', 'Feature_ATR', 'Feature_Lag1_Pct', 'Feature_Lag5_Pct']
    categorical_features = ['Feature_DayOfWeek']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor only on the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the full preprocessor
    save_artifacts(preprocessor, PREPROCESSOR_OUTPUT_PATH, type="Preprocessor (Scaler+Encoder)")
    
    best_accuracy = 0
    best_model = None
    results = {}

    print("\n" + "="*50)
    print(f"STARTING TEMPORAL MODEL COMPARISON ({len(candidates)} CANDIDATES)")
    print("="*50)

    for name, model_instance in candidates:
        print(f"\n--- Training {name} ---")
        
        # Train on the fully processed (scaled and encoded) data
        model_instance.fit(X_train_processed, y_train)
        
        # Evaluate on unseen future data
        y_pred = model_instance.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"Test Accuracy for {name}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (name, model_instance)

    print("\n" + "="*50)
    print("TEMPORAL COMPARISON RESULTS:")
    for name, acc in results.items():
        print(f" - {name}: {acc:.4f}")
        
    if best_model:
        best_name, best_instance = best_model
        print(f"\n🏆 BEST MODEL: {best_name} with TEMPORAL Accuracy: {best_accuracy:.4f}")
        return best_model
    else:
        print("\nFailed to train any models.")
        return None

# --- 3. Deployment/Serialization (UNCHANGED) ---

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

"""
Comprehensive Target Variable Testing Pipeline
Tests multiple target variable definitions to find optimal approach
Author: Optimization Version
Date: 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os
from itertools import product
warnings.filterwarnings('ignore')

# Technical Analysis Libraries
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# ==================== CONFIGURATION ====================
SYMBOLS = {
    'META': 'META',
    'WTI': 'CL=F',
    'URTH': 'URTH'
}

YEARS_OF_DATA = 5
START_DATE = datetime.now() - timedelta(days=YEARS_OF_DATA*365)
END_DATE = datetime.now()

# ==================== DATA COLLECTION ====================
def fetch_stock_data(symbol_name, symbol_ticker, start, end):
    """Fetch historical stock data from Yahoo Finance"""
    print(f"  Fetching data for {symbol_name} ({symbol_ticker})...")
    
    try:
        stock = yf.Ticker(symbol_ticker)
        data = stock.history(start=start, end=end)
        
        if data.empty:
            print(f"  Warning: No data retrieved for {symbol_ticker}")
            return None
            
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data['Symbol'] = symbol_name
        data['Ticker'] = symbol_ticker
        
        print(f"    Retrieved {len(data)} days of data")
        return data
        
    except Exception as e:
        print(f"  Error fetching data for {symbol_ticker}: {str(e)}")
        return None

# ==================== FEATURE ENGINEERING ====================
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Open_Ratio'] = (df['Close'] - df['Open']) / df['Open']
    
    # Volume features
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Momentum features
    for period in [5, 10, 20, 50]:
        df[f'Momentum_{period}d'] = df['Close'].pct_change(period)
        df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
    
    # Volatility features
    for window in [5, 10, 20, 30]:
        df[f'Volatility_{window}d'] = df['Returns'].rolling(window).std()
        df[f'ATR_{window}'] = calculate_atr(df, window)
    
    # RSI variations
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = RSIIndicator(close=df['Close'], window=period).rsi()
    
    # MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
        df[f'EMA_{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
        # Price position relative to MA
        df[f'Close_to_SMA_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']
    
    # Stochastic
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Width'] = bb.bollinger_wband()
    df['BB_Position'] = bb.bollinger_pband()
    
    # Market regime indicators
    df['Trend_Strength'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    
    # Support/Resistance levels
    df['Resistance'] = df['High'].rolling(20).max()
    df['Support'] = df['Low'].rolling(20).min()
    df['SR_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
    
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

# ==================== TARGET VARIABLE DEFINITIONS ====================

def target_simple_returns(df, buy_threshold=-0.01, sell_threshold=0.01, forward_days=1):
    """Simple return-based targets"""
    df = df.copy()
    df['Forward_Return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    
    conditions = [
        df['Forward_Return'] > sell_threshold,
        df['Forward_Return'] < buy_threshold
    ]
    choices = [1, 2]  # 1: Sell, 2: Buy
    df['Target'] = np.select(conditions, choices, default=0)  # 0: Hold
    
    return df, f"SimpleReturns_b{buy_threshold}_s{sell_threshold}_f{forward_days}d"

def target_sharpe_fixed(df, buy_threshold=-1.0, sell_threshold=1.0, vol_window=20, forward_days=1):
    """Sharpe ratio with fixed thresholds"""
    df = df.copy()
    df['Forward_Return'] = df['Returns'].shift(-forward_days)
    df['Rolling_Vol'] = df['Returns'].rolling(vol_window).std()
    df['Sharpe'] = df['Forward_Return'] / (df['Rolling_Vol'] + 1e-10)
    
    conditions = [
        df['Sharpe'] > sell_threshold,
        df['Sharpe'] < buy_threshold
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    return df, f"SharpeFixed_b{buy_threshold}_s{sell_threshold}_v{vol_window}_f{forward_days}d"

def target_sharpe_percentile(df, buy_pct=33, sell_pct=67, vol_window=20, forward_days=1):
    """Sharpe ratio with percentile thresholds"""
    df = df.copy()
    df['Forward_Return'] = df['Returns'].shift(-forward_days)
    df['Rolling_Vol'] = df['Returns'].rolling(vol_window).std()
    df['Sharpe'] = df['Forward_Return'] / (df['Rolling_Vol'] + 1e-10)
    
    df_clean = df.dropna(subset=['Sharpe'])
    buy_threshold = df_clean['Sharpe'].quantile(buy_pct / 100)
    sell_threshold = df_clean['Sharpe'].quantile(sell_pct / 100)
    
    conditions = [
        df['Sharpe'] > sell_threshold,
        df['Sharpe'] < buy_threshold
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    return df, f"SharpePercentile_b{buy_pct}_s{sell_pct}_v{vol_window}_f{forward_days}d"

def target_sharpe_adaptive(df, std_multiplier=0.5, vol_window=20, forward_days=1):
    """Sharpe ratio with adaptive thresholds based on distribution"""
    df = df.copy()
    df['Forward_Return'] = df['Returns'].shift(-forward_days)
    df['Rolling_Vol'] = df['Returns'].rolling(vol_window).std()
    df['Sharpe'] = df['Forward_Return'] / (df['Rolling_Vol'] + 1e-10)
    
    df_clean = df.dropna(subset=['Sharpe'])
    sharpe_mean = df_clean['Sharpe'].mean()
    sharpe_std = df_clean['Sharpe'].std()
    
    buy_threshold = sharpe_mean - (std_multiplier * sharpe_std)
    sell_threshold = sharpe_mean + (std_multiplier * sharpe_std)
    
    conditions = [
        df['Sharpe'] > sell_threshold,
        df['Sharpe'] < buy_threshold
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    return df, f"SharpeAdaptive_std{std_multiplier}_v{vol_window}_f{forward_days}d"

def target_binary_direction(df, threshold=0, forward_days=1):
    """Binary classification: Up vs Down"""
    df = df.copy()
    df['Forward_Return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    df['Target'] = (df['Forward_Return'] > threshold).astype(int)
    
    return df, f"Binary_t{threshold}_f{forward_days}d"

def target_volatility_adjusted(df, vol_multiplier=1.0, forward_days=1):
    """Volatility-adjusted return targets"""
    df = df.copy()
    df['Forward_Return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    df['Rolling_Vol'] = df['Returns'].rolling(20).std()
    
    # Dynamic thresholds based on current volatility
    df['Buy_Threshold'] = -vol_multiplier * df['Rolling_Vol']
    df['Sell_Threshold'] = vol_multiplier * df['Rolling_Vol']
    
    conditions = [
        df['Forward_Return'] > df['Sell_Threshold'],
        df['Forward_Return'] < df['Buy_Threshold']
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    return df, f"VolAdjusted_m{vol_multiplier}_f{forward_days}d"

def target_trend_following(df, lookback=20, forward_days=1):
    """Trend-based targets using moving average crossovers"""
    df = df.copy()
    df['Forward_Return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    
    # Buy when price crosses above MA, Sell when crosses below
    df['MA'] = df['Close'].rolling(lookback).mean()
    df['Position'] = np.where(df['Close'] > df['MA'], 1, -1)
    df['Position_Change'] = df['Position'].diff()
    
    # Simplified: Buy on uptrend, Sell on downtrend
    df['Target'] = np.where(df['Position_Change'] > 0, 2,  # Buy
                            np.where(df['Position_Change'] < 0, 1,  # Sell
                                    0))  # Hold
    
    return df, f"TrendFollowing_lb{lookback}_f{forward_days}d"

def target_mean_reversion(df, z_score_threshold=1.5, lookback=20, forward_days=1):
    """Mean reversion targets using z-scores"""
    df = df.copy()
    df['Forward_Return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    
    # Calculate z-score
    df['MA'] = df['Close'].rolling(lookback).mean()
    df['Std'] = df['Close'].rolling(lookback).std()
    df['Z_Score'] = (df['Close'] - df['MA']) / df['Std']
    
    # Buy when oversold (low z-score), Sell when overbought (high z-score)
    conditions = [
        df['Z_Score'] > z_score_threshold,   # Overbought -> Sell
        df['Z_Score'] < -z_score_threshold    # Oversold -> Buy
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    return df, f"MeanReversion_z{z_score_threshold}_lb{lookback}_f{forward_days}d"

def target_rsi_based(df, oversold=30, overbought=70, forward_days=1):
    """RSI-based targets"""
    df = df.copy()
    df['Forward_Return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    
    if 'RSI_14' not in df.columns:
        df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    conditions = [
        df['RSI_14'] > overbought,   # Overbought -> Sell
        df['RSI_14'] < oversold       # Oversold -> Buy
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    return df, f"RSI_os{oversold}_ob{overbought}_f{forward_days}d"

def target_combined_signals(df, forward_days=1):
    """Combined multiple indicators for target"""
    df = df.copy()
    df['Forward_Return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    
    # Calculate various signals
    df['MA_Signal'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
    df['RSI_Signal'] = np.where(df['RSI_14'] > 70, -1, np.where(df['RSI_14'] < 30, 1, 0))
    df['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
    
    # Combine signals
    df['Combined_Signal'] = df['MA_Signal'] + df['RSI_Signal'] + df['MACD_Signal']
    
    # Create targets based on combined signal strength
    conditions = [
        df['Combined_Signal'] >= 2,    # Strong Buy
        df['Combined_Signal'] <= -2     # Strong Sell
    ]
    choices = [2, 1]  # Buy, Sell
    df['Target'] = np.select(conditions, choices, default=0)
    
    return df, f"CombinedSignals_f{forward_days}d"

# ==================== PREPROCESSING ====================
def preprocess_data(df, feature_cols):
    """Standard preprocessing pipeline"""
    df = df.copy()
    
    # Remove low variance features
    variances = df[feature_cols].var()
    low_var_features = variances[variances < 0.01].index.tolist()
    feature_cols = [col for col in feature_cols if col not in low_var_features]
    
    # Handle outliers
    for col in feature_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                z_scores = np.abs((df[col] - mean_val) / std_val)
                df.loc[z_scores > 3, col] = df[col].quantile(0.99)
                df.loc[z_scores < -3, col] = df[col].quantile(0.01)
    
    # Impute missing values
    if df[feature_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    return df, feature_cols

# ==================== MODEL TRAINING ====================
def train_models(X_train, X_test, y_train, y_test, use_smote=False):
    """Train multiple models and return results"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance if needed
    if use_smote and len(np.unique(y_train)) > 1:
        try:
            min_class = min(np.bincount(y_train))
            if min_class > 5:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class-1))
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        except:
            pass
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, model in models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'cv_accuracy': cv_scores.mean(),
                'test_accuracy': accuracy,
                'test_f1': f1,
                'test_precision': precision,
                'test_recall': recall,
                'combined_score': (cv_scores.mean() + accuracy + f1) / 3
            }
            
        except Exception as e:
            results[name] = {
                'cv_accuracy': 0,
                'test_accuracy': 0,
                'test_f1': 0,
                'test_precision': 0,
                'test_recall': 0,
                'combined_score': 0,
                'error': str(e)
            }
    
    return results

# ==================== MAIN TESTING PIPELINE ====================
def test_all_target_variations():
    """Test all combinations of target variables and parameters"""
    
    print("="*80)
    print("COMPREHENSIVE TARGET VARIABLE TESTING PIPELINE")
    print("="*80)
    print(f"Testing period: {START_DATE.date()} to {END_DATE.date()}")
    print("="*80)
    
    # Store all results
    all_results = []
    
    # Define parameter grids for each target type
    target_configs = {
        'simple_returns': {
            'function': target_simple_returns,
            'params': {
                'buy_threshold': [-0.02, -0.01, -0.005],
                'sell_threshold': [0.005, 0.01, 0.02],
                'forward_days': [1, 3, 5]
            }
        },
        'sharpe_fixed': {
            'function': target_sharpe_fixed,
            'params': {
                'buy_threshold': [-1.0, -0.5, -0.25],
                'sell_threshold': [0.25, 0.5, 1.0],
                'vol_window': [10, 20, 30],
                'forward_days': [1, 3, 5]
            }
        },
        'sharpe_percentile': {
            'function': target_sharpe_percentile,
            'params': {
                'buy_pct': [25, 33, 40],
                'sell_pct': [60, 67, 75],
                'vol_window': [20],
                'forward_days': [1, 3, 5]
            }
        },
        'sharpe_adaptive': {
            'function': target_sharpe_adaptive,
            'params': {
                'std_multiplier': [0.25, 0.5, 0.75, 1.0],
                'vol_window': [20],
                'forward_days': [1, 3, 5]
            }
        },
        'binary_direction': {
            'function': target_binary_direction,
            'params': {
                'threshold': [-0.001, 0, 0.001],
                'forward_days': [1, 3, 5]
            }
        },
        'volatility_adjusted': {
            'function': target_volatility_adjusted,
            'params': {
                'vol_multiplier': [0.5, 1.0, 1.5],
                'forward_days': [1, 3, 5]
            }
        },
        'trend_following': {
            'function': target_trend_following,
            'params': {
                'lookback': [10, 20, 50],
                'forward_days': [1, 3, 5]
            }
        },
        'mean_reversion': {
            'function': target_mean_reversion,
            'params': {
                'z_score_threshold': [1.0, 1.5, 2.0],
                'lookback': [20, 30],
                'forward_days': [1, 3, 5]
            }
        },
        'rsi_based': {
            'function': target_rsi_based,
            'params': {
                'oversold': [20, 30, 35],
                'overbought': [65, 70, 80],
                'forward_days': [1, 3, 5]
            }
        },
        'combined_signals': {
            'function': target_combined_signals,
            'params': {
                'forward_days': [1, 3, 5]
            }
        }
    }
    
    # Process each asset
    for symbol_name, symbol_ticker in SYMBOLS.items():
        print(f"\n{'='*30} {symbol_name} {'='*30}")
        
        # Fetch and prepare data
        df = fetch_stock_data(symbol_name, symbol_ticker, START_DATE, END_DATE)
        if df is None:
            continue
        
        # Calculate all technical indicators
        df = calculate_technical_indicators(df)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Ticker', 
                        'Target', 'Forward_Return']]
        
        # Test each target configuration
        for target_type, config in target_configs.items():
            print(f"\n  Testing {target_type}...")
            
            # Get all parameter combinations
            param_names = list(config['params'].keys())
            param_values = list(config['params'].values())
            
            for param_combo in product(*param_values):
                # Create parameter dictionary
                params = dict(zip(param_names, param_combo))
                
                try:
                    # Apply target function
                    df_target, target_name = config['function'](df, **params)
                    
                    # Clean data
                    df_clean = df_target.dropna(subset=['Target'] + feature_cols)
                    if len(df_clean) < 100:
                        continue
                    
                    # Check class distribution
                    class_dist = df_clean['Target'].value_counts(normalize=True)
                    n_classes = len(class_dist)
                    
                    # Skip if single class
                    if n_classes < 2:
                        continue
                    
                    # Prepare data
                    X = df_clean[feature_cols]
                    y = df_clean['Target']
                    
                    # Preprocess
                    X, selected_features = preprocess_data(X, feature_cols)
                    
                    # Split data
                    split_idx = int(len(X) * 0.7)
                    X_train = X.iloc[:split_idx]
                    X_test = X.iloc[split_idx:]
                    y_train = y.iloc[:split_idx]
                    y_test = y.iloc[split_idx:]
                    
                    # Train models
                    model_results = train_models(X_train, X_test, y_train, y_test, use_smote=True)
                    
                    # Find best model
                    best_model = max(model_results.items(), key=lambda x: x[1]['combined_score'])
                    
                    # Store results
                    result = {
                        'asset': symbol_name,
                        'target_type': target_type,
                        'target_name': target_name,
                        'parameters': params,
                        'n_classes': n_classes,
                        'class_distribution': class_dist.to_dict(),
                        'best_model': best_model[0],
                        'best_accuracy': best_model[1]['test_accuracy'],
                        'best_f1': best_model[1]['test_f1'],
                        'best_cv_accuracy': best_model[1]['cv_accuracy'],
                        'combined_score': best_model[1]['combined_score'],
                        'all_models': model_results
                    }
                    
                    all_results.append(result)
                    
                    # Print progress
                    if best_model[1]['test_accuracy'] > 0.6:  # Only print good results
                        print(f"    ✓ {target_name}: {best_model[0]} = {best_model[1]['test_accuracy']:.3f}")
                    
                except Exception as e:
                    print(f"    ✗ Error with {params}: {str(e)[:50]}")
                    continue
    
    return all_results

def analyze_results(results):
    """Analyze and summarize all results"""
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("TOP CONFIGURATIONS BY ASSET")
    print("="*80)
    
    # Find best configuration for each asset
    for asset in SYMBOLS.keys():
        asset_results = df_results[df_results['asset'] == asset]
        
        if len(asset_results) == 0:
            continue
        
        # Sort by combined score
        top_results = asset_results.nlargest(10, 'combined_score')
        
        print(f"\n{asset} - Top 10 Configurations:")
        print("-"*60)
        
        for idx, row in top_results.iterrows():
            print(f"\n{idx+1}. {row['target_type']} - {row['best_model']}")
            print(f"   Accuracy: {row['best_accuracy']:.3f}, F1: {row['best_f1']:.3f}")
            print(f"   CV Accuracy: {row['best_cv_accuracy']:.3f}")
            print(f"   Combined Score: {row['combined_score']:.3f}")
            print(f"   Parameters: {row['parameters']}")
            print(f"   Classes: {row['n_classes']}, Distribution: {row['class_distribution']}")
    
    # Overall best configurations
    print("\n" + "="*80)
    print("OVERALL BEST CONFIGURATIONS")
    print("="*80)
    
    top_overall = df_results.nlargest(10, 'combined_score')
    
    for idx, row in top_overall.iterrows():
        print(f"\n{idx+1}. {row['asset']} - {row['target_type']} - {row['best_model']}")
        print(f"   Accuracy: {row['best_accuracy']:.3f}, F1: {row['best_f1']:.3f}")
        print(f"   Combined Score: {row['combined_score']:.3f}")
        print(f"   Parameters: {row['parameters']}")
    
    # Best target type analysis
    print("\n" + "="*80)
    print("BEST TARGET TYPES SUMMARY")
    print("="*80)
    
    target_performance = df_results.groupby('target_type').agg({
        'combined_score': ['mean', 'max', 'std'],
        'best_accuracy': ['mean', 'max'],
        'best_f1': ['mean', 'max']
    }).round(3)
    
    print(target_performance)
    
    # Best model type analysis
    print("\n" + "="*80)
    print("BEST MODEL TYPES SUMMARY")
    print("="*80)
    
    model_performance = df_results.groupby('best_model').agg({
        'combined_score': ['mean', 'count'],
        'best_accuracy': 'mean',
        'best_f1': 'mean'
    }).round(3)
    
    print(model_performance.sort_values(('combined_score', 'mean'), ascending=False))
    
    return df_results

def save_results(df_results):
    """Save results to files"""
    
    # Create output directory
    output_dir = 'target_optimization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    df_results.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    
    # Save best configurations per asset
    best_per_asset = df_results.loc[df_results.groupby('asset')['combined_score'].idxmax()]
    best_per_asset.to_csv(os.path.join(output_dir, 'best_per_asset.csv'), index=False)
    
    # Save summary statistics
    summary = {
        'total_configurations_tested': len(df_results),
        'best_overall_accuracy': df_results['best_accuracy'].max(),
        'best_overall_f1': df_results['best_f1'].max(),
        'best_overall_combined': df_results['combined_score'].max(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}/")
    
    return output_dir

# ==================== MAIN EXECUTION ====================
def main():
    """Run comprehensive target variable testing"""
    
    # Run all tests
    results = test_all_target_variations()
    
    # Analyze results
    df_results = analyze_results(results)
    
    # Save results
    output_dir = save_results(df_results)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    print(f"Tested {len(results)} configurations")
    print(f"Results saved to {output_dir}/")
    
    # Print final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    for asset in SYMBOLS.keys():
        asset_results = df_results[df_results['asset'] == asset]
        if len(asset_results) > 0:
            best = asset_results.iloc[asset_results['combined_score'].argmax()]
            print(f"\n{asset}:")
            print(f"  Use: {best['target_type']}")
            print(f"  Model: {best['best_model']}")
            print(f"  Expected Accuracy: {best['best_accuracy']:.1%}")
            print(f"  Parameters: {best['parameters']}")
    
    return df_results

if __name__ == "__main__":
    results_df = main()
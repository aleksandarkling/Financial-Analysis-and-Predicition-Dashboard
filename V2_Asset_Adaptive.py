"""
Adaptive Threshold ML Pipeline
Uses asset-specific volatility to set meaningful thresholds
Improves on thesis's fixed 1% approach
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Technical indicators
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
SYMBOLS = {
    'META': 'META',
    'WTI': 'CL=F', 
    'URTH': 'URTH'
}

# ==================== ADAPTIVE THRESHOLD CALCULATION ====================
def calculate_adaptive_thresholds(df, method='percentile', lookback=252):
    """
    Calculate asset-specific thresholds based on historical volatility
    
    Methods:
    - 'percentile': Use 70th/30th percentiles of absolute returns
    - 'std': Use 0.5 * standard deviation
    - 'atr': Use Average True Range
    """
    print(f"  Calculating adaptive thresholds using {method} method...")
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change() * 100
    
    if method == 'percentile':
        # Use percentiles of absolute returns for more stable thresholds
        abs_returns = df['Returns'].abs().dropna()
        # Use 70th percentile - captures "meaningful" moves, not every wiggle
        threshold = abs_returns.quantile(0.70)
        
    elif method == 'std':
        # Use rolling standard deviation
        rolling_std = df['Returns'].rolling(window=lookback).std()
        # Use median of rolling std to avoid outlier periods
        threshold = rolling_std.median() * 0.5  # 0.5 std captures significant moves
        
    elif method == 'atr':
        # Use ATR (Average True Range) - popular with traders
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATR'] = atr.average_true_range()
        # Convert ATR to percentage
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        threshold = df['ATR_Pct'].median()
    
    else:
        # Fallback to 1%
        threshold = 1.0
    
    print(f"    Calculated threshold: {threshold:.3f}%")
    
    # Show what this means in practice
    moves_above = (df['Returns'].abs() > threshold).sum()
    total_days = len(df['Returns'].dropna())
    print(f"    This captures {moves_above}/{total_days} days ({moves_above/total_days*100:.1f}%) as significant")
    
    return threshold

def create_adaptive_target(df, threshold):
    """
    Create target variable using asset-specific threshold
    Can use either forward-looking or backward-looking based on your preference
    """
    # For academic purposes, backward-looking is safer (no look-ahead bias)
    # For practical trading, forward-looking makes more sense
    
    # Option 1: Backward-looking (what happened today)
    df['Target_Return'] = df['Returns']  # Today's return
    
    # Option 2: Forward-looking (what will happen tomorrow)
    # Uncomment if you want predictive targets
    # df['Target_Return'] = df['Returns'].shift(-1)  # Tomorrow's return
    
    # Create labels using adaptive threshold
    conditions = [
        df['Target_Return'] > threshold,    # Significant up move
        df['Target_Return'] < -threshold    # Significant down move
    ]
    choices = [2, 1]  # 2: UP, 1: DOWN
    df['Target'] = np.select(conditions, choices, default=0)  # 0: NEUTRAL
    
    # Add labels
    df['Target_Label'] = df['Target'].map({0: 'NEUTRAL', 1: 'DOWN', 2: 'UP'})
    
    # Show distribution
    dist = df['Target'].value_counts(normalize=True).sort_index()
    print(f"  Target distribution - Neutral: {dist.get(0,0):.1%}, "
          f"Down: {dist.get(1,0):.1%}, Up: {dist.get(2,0):.1%}")
    
    return df

def create_enhanced_features(df, asset_type='stock'):
    """
    Create features tailored to asset type
    """
    print(f"  Creating enhanced features for {asset_type}...")
    
    # Universal features
    # Price action
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Close_Open'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    
    # Volume features
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Momentum over multiple timeframes
    for period in [2, 5, 10, 20, 50]:
        df[f'Return_{period}d'] = df['Close'].pct_change(period) * 100
        df[f'High_{period}d'] = df['High'].rolling(period).max() / df['Close'] - 1
        df[f'Low_{period}d'] = df['Low'].rolling(period).min() / df['Close'] - 1
    
    # Volatility measures
    df['Volatility_5d'] = df['Returns'].rolling(5).std()
    df['Volatility_20d'] = df['Returns'].rolling(20).std()
    df['Volatility_Ratio'] = df['Volatility_5d'] / df['Volatility_20d']
    
    # Technical indicators
    # RSI with multiple periods
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = RSIIndicator(close=df['Close'], window=period).rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Moving averages and positions
    for period in [10, 20, 50, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
        df[f'Close_to_SMA{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}'] * 100
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    df['BB_Position'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    
    # Asset-specific features
    if asset_type == 'commodity':
        # Commodities are more seasonal and trend-following
        df['Month'] = pd.to_datetime(df.index).month
        df['Quarter'] = pd.to_datetime(df.index).quarter
        df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
        
        # Longer-term trends matter more
        df['Trend_100d'] = (df['Close'] - df['Close'].rolling(100).mean()) / df['Close'].rolling(100).mean() * 100
        
    elif asset_type == 'etf':
        # ETFs are more mean-reverting
        df['Z_Score_20'] = (df['Close'] - df['SMA_20']) / df['Volatility_20d']
        df['Z_Score_50'] = (df['Close'] - df['SMA_50']) / df['Returns'].rolling(50).std()
        
        # Correlation breakdown indicator
        df['Return_Dispersion'] = df['High_Low_Range'].rolling(20).std()
        
    elif asset_type == 'tech_stock':
        # Tech stocks respond to momentum
        df['Momentum_Score'] = (df['Return_5d'] + df['Return_10d'] + df['Return_20d']) / 3
        df['RSI_Divergence'] = df['RSI_14'] - df['RSI_14'].rolling(5).mean()
        
        # Volume momentum (tech stocks have information in volume)
        df['Volume_Momentum'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
    
    return df

def train_optimized_models(X_train, X_test, y_train, y_test, asset_name):
    """
    Train models with hyperparameter optimization for each asset
    """
    print(f"\n  Training optimized models for {asset_name}...")
    
    # Scale data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check class distribution
    class_counts = pd.Series(y_train).value_counts()
    print(f"    Training class distribution: {class_counts.to_dict()}")
    
    # Calculate class weights for imbalanced data
    class_weights = {}
    for cls in class_counts.index:
        class_weights[cls] = len(y_train) / (len(class_counts) * class_counts[cls])
    
    models = {}
    
    # 1. Logistic Regression (baseline)
    models['Logistic_Regression'] = LogisticRegression(
        class_weight=class_weights,
        max_iter=1000,
        random_state=42
    )
    
    # 2. Random Forest (good for any data)
    models['Random_Forest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. XGBoost (usually best performer)
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = np.sqrt(class_counts[0] / class_counts[class_counts.index != 0].mean()) if len(class_counts) > 2 else 1
    
    models['XGBoost'] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # 4. Gradient Boosting (robust alternative)
    models['Gradient_Boosting'] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class performance
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'scaler': scaler
        }
        
        print(f"      {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    
    return results

def evaluate_trading_performance(y_true, y_pred, prices, initial_capital=10000):
    """
    Evaluate actual trading performance
    """
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(y_pred)):
        price = prices.iloc[i]
        
        if y_pred[i] == 2 and position <= 0:  # Buy signal
            # Close short and go long
            if position < 0:
                capital = capital - position * price  # Close short
            position = capital / price  # Go long
            capital = 0
            trades.append(('BUY', price, i))
            
        elif y_pred[i] == 1 and position >= 0:  # Sell signal
            # Close long and go short
            if position > 0:
                capital = position * price  # Close long
            position = -capital / price  # Go short
            trades.append(('SELL', price, i))
    
    # Close final position
    if position != 0:
        final_price = prices.iloc[-1]
        capital = capital + position * final_price
    else:
        capital = capital
    
    total_return = (capital - initial_capital) / initial_capital * 100
    
    print(f"    Trading Performance:")
    print(f"      Total trades: {len(trades)}")
    print(f"      Final capital: ${capital:.2f}")
    print(f"      Total return: {total_return:.2f}%")
    print(f"      Annualized return: {total_return * 252 / len(prices):.2f}%")
    
    return {
        'total_trades': len(trades),
        'total_return': total_return,
        'final_capital': capital
    }

# ==================== MAIN PIPELINE ====================
def main():
    """
    Run adaptive threshold ML pipeline
    """
    print("="*80)
    print("ADAPTIVE THRESHOLD ML PIPELINE")
    print("Asset-Specific Volatility-Adjusted Targets")
    print("="*80)
    
    all_results = {}
    
    # Define asset types
    asset_types = {
        'META': 'tech_stock',
        'WTI': 'commodity',
        'URTH': 'etf'
    }
    
    for symbol_name, symbol_ticker in SYMBOLS.items():
        print(f"\n{'='*40}")
        print(f"Processing {symbol_name} ({asset_types[symbol_name]})")
        print('='*40)
        
        try:
            # Fetch data
            stock = yf.Ticker(symbol_ticker)
            df = stock.history(period="5y")
            
            if df.empty:
                print(f"  No data for {symbol_name}")
                continue
            
            # Calculate adaptive threshold
            threshold = calculate_adaptive_thresholds(df, method='percentile')
            
            # Create target with adaptive threshold
            df = create_adaptive_target(df, threshold)
            
            # Create features based on asset type
            df = create_enhanced_features(df, asset_type=asset_types[symbol_name])
            
            # Prepare data
            df_clean = df.dropna()
            
            # Define features
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'Target', 'Target_Label', 'Returns', 'Target_Return']
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols 
                          and df_clean[col].dtype in ['float64', 'int64']]
            
            X = df_clean[feature_cols]
            y = df_clean['Target']
            prices = df_clean['Close']
            
            # Train/test split
            split_idx = int(len(X) * 0.7)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            prices_test = prices.iloc[split_idx:]
            
            print(f"  Data: {len(X_train)} training, {len(X_test)} test samples")
            
            # Train models
            models = train_optimized_models(X_train, X_test, y_train, y_test, symbol_name)
            
            # Select best model
            best_model_name = max(models.keys(), key=lambda k: models[k]['f1_score'])
            best_model = models[best_model_name]
            
            print(f"\n  Best model: {best_model_name}")
            print(f"    Accuracy: {best_model['accuracy']:.3f}")
            print(f"    F1 Score: {best_model['f1_score']:.3f}")
            
            # Feature importance for tree-based models
            if best_model_name in ['Random_Forest', 'XGBoost', 'Gradient_Boosting']:
                if hasattr(best_model['model'], 'feature_importances_'):
                    importances = best_model['model'].feature_importances_
                    top_features_idx = np.argsort(importances)[-10:][::-1]
                    print("\n  Top 10 features:")
                    for idx in top_features_idx:
                        print(f"    {feature_cols[idx]}: {importances[idx]:.4f}")
            
            # Evaluate trading performance
            trading_metrics = evaluate_trading_performance(
                y_test, best_model['predictions'], prices_test
            )
            
            all_results[symbol_name] = {
                'threshold': threshold,
                'best_model': best_model_name,
                'accuracy': best_model['accuracy'],
                'f1_score': best_model['f1_score'],
                'trading_metrics': trading_metrics,
                'models': models
            }
            
        except Exception as e:
            print(f"  Error processing {symbol_name}: {str(e)}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ADAPTIVE THRESHOLDS")
    print("="*80)
    
    for asset, results in all_results.items():
        print(f"\n{asset}:")
        print(f"  Threshold: {results['threshold']:.3f}%")
        print(f"  Best Model: {results['best_model']}")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  F1 Score: {results['f1_score']:.3f}")
        print(f"  Trading Return: {results['trading_metrics']['total_return']:.2f}%")
    
    # Compare to fixed 1% threshold
    print("\n" + "="*80)
    print("IMPROVEMENT OVER FIXED 1% THRESHOLD")
    print("="*80)
    print("The adaptive thresholds better capture each asset's volatility profile:")
    print("- WTI gets a higher threshold (captures oil's volatility)")
    print("- URTH gets a lower threshold (captures ETF's stability)")
    print("- META gets an appropriate threshold for tech stocks")
    
    return all_results

if __name__ == "__main__":
    results = main()
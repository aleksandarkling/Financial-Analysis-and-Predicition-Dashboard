"""
Complete Stock Market ML Pipeline with Sharpe Ratio Target (Percentile-Based)
Implements forward-looking Sharpe ratio methodology with balanced classes
Author: Enhanced Version with Percentile Thresholds
Date: 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Technical Analysis Libraries
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, auc)
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Set random seed for reproducibility
np.random.seed(42)

# ==================== CONFIGURATION ====================
# Stock symbols - META (Tech), WTI (Oil), iShares MSCI World ETF
SYMBOLS = {
    'META': 'META',           # Technology sector
    'WTI': 'CL=F',           # WTI Crude Oil Futures
    'URTH': 'URTH'           # iShares MSCI World ETF
}

# Data parameters
YEARS_OF_DATA = 5
START_DATE = datetime.now() - timedelta(days=YEARS_OF_DATA*365)
END_DATE = datetime.now()

# Percentile thresholds for balanced classification
PERCENTILE_BUY = 33   # Bottom 33% = Buy
PERCENTILE_SELL = 67  # Top 33% = Sell
# Middle 34% = Hold

# Rolling window for volatility calculation
VOLATILITY_WINDOW = 20

# ==================== DATA COLLECTION ====================
def fetch_stock_data(symbol_name, symbol_ticker, start, end):
    """
    Fetch historical stock data from Yahoo Finance
    """
    print(f"Fetching data for {symbol_name} ({symbol_ticker})...")
    
    try:
        stock = yf.Ticker(symbol_ticker)
        data = stock.history(start=start, end=end)
        
        if data.empty:
            print(f"Warning: No data retrieved for {symbol_ticker}")
            return None
            
        # Ensure we have all required columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Add symbol identifier
        data['Symbol'] = symbol_name
        data['Ticker'] = symbol_ticker
        
        print(f"  Retrieved {len(data)} days of data for {symbol_name}")
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol_ticker}: {str(e)}")
        return None

# ==================== TECHNICAL INDICATORS ====================
def calculate_technical_indicators(df):
    """
    Calculate all technical indicators
    """
    print("  Calculating technical indicators...")
    
    # 1. RSI (Relative Strength Index)
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # 2. MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df['MACDs_12_26_9'] = macd.macd_signal()
    df['MACDh_12_26_9'] = macd.macd_diff()
    
    # 3. Moving Averages
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    
    # 4. Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['STOCHk_14_3_3'] = stoch.stoch()
    df['STOCHd_14_3_3'] = stoch.stoch_signal()
    
    # 5. Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    df['BBB_20_2.0'] = bb.bollinger_wband()
    df['BBP_20_2.0'] = bb.bollinger_pband()
    
    # 6. Ichimoku Cloud
    ichimoku = IchimokuIndicator(
        high=df['High'],
        low=df['Low'],
        window1=9,
        window2=26,
        window3=52
    )
    df['ITS_9'] = ichimoku.ichimoku_conversion_line()
    df['IKS_26'] = ichimoku.ichimoku_base_line()
    df['ISA_9'] = ichimoku.ichimoku_a()
    df['ISB_26'] = ichimoku.ichimoku_b()
    df['ICS_26'] = df['Close'].shift(-26)
    
    # 7. Standard Deviation
    df['STDEV_30'] = df['Close'].rolling(window=30).std()
    
    # 8. Additional momentum features for better performance
    df['Momentum_5d'] = df['Close'].pct_change(5)
    df['Momentum_10d'] = df['Close'].pct_change(10)
    df['Momentum_20d'] = df['Close'].pct_change(20)
    
    return df

# ==================== SHARPE RATIO TARGET WITH PERCENTILE THRESHOLDS ====================
def create_sharpe_target_variable(df, percentile_buy=PERCENTILE_BUY, percentile_sell=PERCENTILE_SELL):
    """
    Create target variable based on forward-looking Sharpe ratio
    Using PERCENTILE-BASED thresholds for balanced classes
    """
    print("  Creating Sharpe ratio target variable with percentile-based thresholds...")
    
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate forward return (tomorrow's return)
    df['Forward_Return'] = df['Returns'].shift(-1)
    
    # Calculate rolling 20-day volatility (standard deviation of returns)
    df['Rolling_Volatility'] = df['Returns'].rolling(window=VOLATILITY_WINDOW).std()
    
    # Calculate Sharpe ratio (forward-looking)
    epsilon = 1e-10
    df['Sharpe_Ratio'] = df['Forward_Return'] / (df['Rolling_Volatility'] + epsilon)
    
    # Remove initial NaN values for threshold calculation
    df_clean = df.dropna(subset=['Sharpe_Ratio']).copy()
    
    # PERCENTILE-BASED THRESHOLDS - Key improvement!
    buy_threshold = df_clean['Sharpe_Ratio'].quantile(percentile_buy / 100)
    sell_threshold = df_clean['Sharpe_Ratio'].quantile(percentile_sell / 100)
    
    print(f"    Percentile thresholds calculated:")
    print(f"      Buy threshold (P{percentile_buy}):  {buy_threshold:.4f}")
    print(f"      Sell threshold (P{percentile_sell}): {sell_threshold:.4f}")
    
    # Show distribution statistics
    print(f"    Sharpe ratio statistics:")
    print(f"      Mean: {df_clean['Sharpe_Ratio'].mean():.4f}")
    print(f"      Std:  {df_clean['Sharpe_Ratio'].std():.4f}")
    print(f"      Min:  {df_clean['Sharpe_Ratio'].min():.4f}")
    print(f"      Max:  {df_clean['Sharpe_Ratio'].max():.4f}")
    
    # Create target variable based on percentile thresholds
    conditions = [
        df['Sharpe_Ratio'] > sell_threshold,   # Top 33% = Sell (1)
        df['Sharpe_Ratio'] < buy_threshold     # Bottom 33% = Buy (2)
    ]
    choices = [1, 2]  # 1: Sell, 2: Buy
    df['Target'] = np.select(conditions, choices, default=0)  # 0: Hold
    
    # Add readable labels
    df['Target_Label'] = df['Target'].map({0: 'HOLD', 1: 'SELL', 2: 'BUY'})
    
    # Store thresholds as attributes
    df.attrs['buy_threshold'] = buy_threshold
    df.attrs['sell_threshold'] = sell_threshold
    
    # Calculate and verify target distribution
    target_dist = df['Target'].value_counts(normalize=True).sort_index()
    print(f"    ✅ Target distribution - Hold: {target_dist.get(0, 0):.1%}, "
          f"Sell: {target_dist.get(1, 0):.1%}, Buy: {target_dist.get(2, 0):.1%}")
    
    # Remove rows with NaN in Sharpe ratio (first 20 days and last day)
    initial_len = len(df)
    df = df.dropna(subset=['Sharpe_Ratio', 'Target'])
    print(f"    Removed {initial_len - len(df)} rows with insufficient data")
    
    return df

# ==================== DATA PREPROCESSING ====================
def preprocess_data(df, symbol_name):
    """
    Implement preprocessing steps
    """
    print(f"  Preprocessing data for {symbol_name}...")
    
    # Get feature columns (all indicators plus momentum features)
    feature_cols = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                   'SMA_50', 'EMA_9', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
                   'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                   'ITS_9', 'IKS_26', 'ISA_9', 'ISB_26', 'ICS_26', 'STDEV_30',
                   'Momentum_5d', 'Momentum_10d', 'Momentum_20d', 'Rolling_Volatility']
    
    # Remove features that don't exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # 1. Remove features with variance < 0.1
    variances = df[feature_cols].var()
    low_var_features = variances[variances < 0.1].index.tolist()
    if low_var_features:
        print(f"    Removing low variance features: {low_var_features}")
        feature_cols = [col for col in feature_cols if col not in low_var_features]
    
    # 2. Handle outliers using z-score
    for col in feature_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                z_scores = np.abs((df[col] - mean_val) / std_val)
                df.loc[z_scores > 3, col] = df[col].quantile(0.99)
                df.loc[z_scores < -3, col] = df[col].quantile(0.01)
    
    # 3. Impute missing values using KNN
    if df[feature_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        print(f"    Imputed missing values using KNN")
    
    return df, feature_cols

# ==================== ASSET-SPECIFIC MODEL TRAINING ====================
def train_asset_specific_models(X_train, X_test, y_train, y_test, symbol_name):
    """
    Train multiple models for a specific asset and select the best one
    Uses cross-validation for model selection
    """
    print(f"\n  Training models for {symbol_name}...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # No need for SMOTE with balanced classes!
    X_train_balanced, y_train_balanced = X_train_scaled, y_train
    print(f"    Classes are now balanced - no SMOTE needed!")
    
    # Initialize all models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            objective='multi:softprob',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=min(10, len(X_train_balanced)//10)
        ),
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=100,
            random_state=42,
            algorithm='SAMME'
        )
    }
    
    results = {}
    cv_scores = {}
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"    Performing cross-validation for model selection...")
    
    for name, model in models.items():
        try:
            # Cross-validation scores
            cv_accuracy = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                         cv=tscv, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                   cv=tscv, scoring='f1_weighted')
            
            cv_scores[name] = {
                'cv_accuracy_mean': cv_accuracy.mean(),
                'cv_accuracy_std': cv_accuracy.std(),
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std()
            }
            
            # Train on full training set
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predict on test set
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate test metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC AUC for multiclass
            if y_pred_proba is not None:
                y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                if y_test_bin.shape[1] == y_pred_proba.shape[1]:
                    roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
                else:
                    roc_auc = 0.0
            else:
                roc_auc = 0.0
            
            results[name] = {
                'model': model,
                'scaler': scaler,
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1_score': f1,
                'test_roc_auc': roc_auc,
                'cv_accuracy': cv_scores[name]['cv_accuracy_mean'],
                'cv_f1': cv_scores[name]['cv_f1_mean'],
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            
            print(f"      {name}: CV Acc={cv_scores[name]['cv_accuracy_mean']:.3f}, "
                  f"Test Acc={accuracy:.3f}, F1={f1:.3f}")
            
        except Exception as e:
            print(f"      Error training {name}: {str(e)}")
    
    # Select best model based on combined CV accuracy and F1 score
    best_model_name = None
    best_score = 0
    
    for name, result in results.items():
        # Combined score: average of CV accuracy and CV F1
        combined_score = (result['cv_accuracy'] + result['cv_f1']) / 2
        if combined_score > best_score:
            best_score = combined_score
            best_model_name = name
    
    print(f"    ✓ Best model for {symbol_name}: {best_model_name} "
          f"(Combined CV Score: {best_score:.3f})")
    
    return results, best_model_name, scaler

# ==================== BACKTESTING ====================
def backtest_strategy(df, model, scaler, feature_cols, initial_capital=100):
    """
    Perform backtesting with the selected model
    """
    print("  Running backtesting...")
    
    # Prepare features for prediction
    X = df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Get predictions for entire dataset
    predictions = model.predict(X_scaled)
    
    # Initialize portfolio
    portfolio_value = initial_capital
    cash = initial_capital
    holdings = 0
    portfolio_values = []
    daily_returns = []
    actions = []
    
    # Add predictions to dataframe
    df = df.copy()
    df['Predicted_Signal'] = predictions
    df['Predicted_Label'] = df['Predicted_Signal'].map({0: 'HOLD', 1: 'SELL', 2: 'BUY'})
    
    # Simulate trading
    for idx in range(len(df)):
        row = df.iloc[idx]
        current_price = row['Close']
        signal = row['Predicted_Signal']
        
        # Calculate current portfolio value
        portfolio_value = cash + (holdings * current_price)
        portfolio_values.append(portfolio_value)
        
        # Execute trades
        if signal == 2 and cash > 0:  # BUY
            shares_to_buy = cash / current_price
            holdings += shares_to_buy
            cash = 0
            actions.append('BUY')
        elif signal == 1 and holdings > 0:  # SELL
            cash += holdings * current_price
            holdings = 0
            actions.append('SELL')
        else:
            actions.append('HOLD')
        
        # Calculate daily return
        if idx > 0:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    
    if daily_returns:
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
        max_drawdown = calculate_max_drawdown(portfolio_values)
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Add results to dataframe
    df['Portfolio_Value'] = portfolio_values
    df['Trading_Action'] = actions
    df['Daily_Return'] = [0] + daily_returns
    df['Cumulative_Return'] = ((pd.Series(portfolio_values) / initial_capital) - 1) * 100
    
    print(f"    Backtest Results - Return: {total_return*100:.2f}%, Sharpe: {sharpe_ratio:.2f}")
    
    return df, {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_values[-1]
    }

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    peak = portfolio_values[0]
    max_dd = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd

# ==================== DAILY PREDICTIONS ====================
def generate_daily_predictions(df, model, scaler, feature_cols):
    """
    Generate today's predictions for the dashboard
    """
    # Get latest data
    latest_features = df[feature_cols].iloc[-1:].fillna(0)
    latest_scaled = scaler.transform(latest_features)
    
    # Make prediction
    prediction = model.predict(latest_scaled)[0]
    prediction_proba = model.predict_proba(latest_scaled)[0] if hasattr(model, 'predict_proba') else [0, 0, 0]
    
    signal_map = {0: 'HOLD', 1: 'SELL', 2: 'BUY'}
    
    return {
        'date': df.index[-1],
        'signal': signal_map[prediction],
        'signal_code': prediction,
        'confidence': max(prediction_proba) * 100,
        'current_price': df['Close'].iloc[-1],
        'current_sharpe': df['Sharpe_Ratio'].iloc[-1] if 'Sharpe_Ratio' in df.columns else 0,
        'buy_threshold': df.attrs.get('buy_threshold', 'N/A'),
        'sell_threshold': df.attrs.get('sell_threshold', 'N/A')
    }

# ==================== CREATE COMPREHENSIVE OUTPUT ====================
def create_comprehensive_output(all_data, all_models, all_backtest, best_models):
    """
    Create comprehensive outputs for Power BI and analysis
    """
    print("\nCreating comprehensive outputs...")
    
    # 1. Model Selection Results
    model_selection_results = []
    
    for symbol_name in all_data.keys():
        models = all_models[symbol_name]
        best_model_name = best_models[symbol_name]
        
        for model_name, results in models.items():
            # Skip non-model entries
            if model_name == 'feature_cols':
                continue
                
            is_best = (model_name == best_model_name)
            
            model_selection_results.append({
                'Asset': symbol_name,
                'Model': model_name,
                'CV_Accuracy': results['cv_accuracy'],
                'CV_F1_Score': results['cv_f1'],
                'Test_Accuracy': results['test_accuracy'],
                'Test_F1_Score': results['test_f1_score'],
                'Test_Precision': results['test_precision'],
                'Test_Recall': results['test_recall'],
                'Test_ROC_AUC': results['test_roc_auc'],
                'Is_Best_Model': is_best,
                'Selection_Score': (results['cv_accuracy'] + results['cv_f1']) / 2
            })
    
    model_selection_df = pd.DataFrame(model_selection_results)
    
    # 2. Historical Data with Predictions
    powerbi_frames = []
    
    for symbol_name in all_data.keys():
        df = all_data[symbol_name]
        models = all_models[symbol_name]
        best_model_name = best_models[symbol_name]
        backtest_df, backtest_metrics = all_backtest[symbol_name]
        
        export_df = backtest_df.copy()
        
        # Add time-based columns
        export_df['Date'] = export_df.index
        export_df['Year'] = export_df.index.year
        export_df['Month'] = export_df.index.month
        export_df['Quarter'] = export_df.index.quarter
        export_df['Week'] = export_df.index.isocalendar().week
        export_df['DayOfWeek'] = export_df.index.dayofweek
        export_df['DayName'] = export_df.index.day_name()
        
        # Add model information
        export_df['Best_Model'] = best_model_name
        export_df['Best_Model_Accuracy'] = models[best_model_name]['test_accuracy']
        export_df['Best_Model_F1'] = models[best_model_name]['test_f1_score']
        
        # Add backtest metrics
        export_df['Backtest_TotalReturn'] = backtest_metrics['total_return']
        export_df['Backtest_SharpeRatio'] = backtest_metrics['sharpe_ratio']
        export_df['Backtest_MaxDrawdown'] = backtest_metrics['max_drawdown']
        
        # Add threshold information
        export_df['Buy_Threshold'] = df.attrs.get('buy_threshold', 'N/A')
        export_df['Sell_Threshold'] = df.attrs.get('sell_threshold', 'N/A')
        
        powerbi_frames.append(export_df)
    
    historical_df = pd.concat(powerbi_frames, ignore_index=True)
    
    # 3. Daily Predictions
    daily_predictions = []
    
    for symbol_name in all_data.keys():
        df = all_data[symbol_name]
        best_model_name = best_models[symbol_name]
        best_model = all_models[symbol_name][best_model_name]['model']
        best_scaler = all_models[symbol_name][best_model_name]['scaler']
        
        # Get feature_cols
        feature_cols = all_models[symbol_name].get('feature_cols', [])
        
        today_pred = generate_daily_predictions(
            df,
            best_model,
            best_scaler,
            feature_cols
        )
        today_pred['asset'] = symbol_name
        today_pred['model_used'] = best_model_name
        daily_predictions.append(today_pred)
    
    predictions_df = pd.DataFrame(daily_predictions)
    
    return model_selection_df, historical_df, predictions_df

# ==================== SAVE OUTPUTS ====================
def save_all_outputs(model_selection_df, historical_df, predictions_df):
    """Save all outputs for Power BI and analysis"""
    
    # Create output directory
    output_dir = 'powerbi_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save model selection results
    model_file = os.path.join(output_dir, 'model_selection_results.csv')
    model_selection_df.to_csv(model_file, index=False)
    print(f"  ✓ Saved model selection results: {model_file}")
    
    # 2. Save historical data with predictions
    historical_file = os.path.join(output_dir, 'stock_ml_historical_data.csv')
    historical_df.to_csv(historical_file, index=False)
    print(f"  ✓ Saved historical data: {historical_file} ({len(historical_df)} rows)")
    
    # 3. Save daily predictions
    predictions_file = os.path.join(output_dir, 'daily_predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print(f"  ✓ Saved daily predictions: {predictions_file}")
    
    # 4. Save metadata
    metadata = {
        'last_update': datetime.now().isoformat(),
        'assets': list(SYMBOLS.keys()),
        'data_start': START_DATE.isoformat(),
        'data_end': END_DATE.isoformat(),
        'percentile_buy': PERCENTILE_BUY,
        'percentile_sell': PERCENTILE_SELL,
        'volatility_window': VOLATILITY_WINDOW,
        'total_rows': len(historical_df)
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata: {metadata_file}")
    
    return output_dir

# ==================== PRINT SUMMARY TABLE ====================
def print_summary_table(model_selection_df, predictions_df):
    """Print a clear summary table of results"""
    
    print("\n" + "="*80)
    print("MODEL SELECTION SUMMARY")
    print("="*80)
    
    # Group by asset and show best model
    for asset in model_selection_df['Asset'].unique():
        asset_df = model_selection_df[model_selection_df['Asset'] == asset]
        best_row = asset_df[asset_df['Is_Best_Model'] == True].iloc[0]
        
        print(f"\n{asset}:")
        print(f"  Best Model: {best_row['Model']}")
        print(f"  CV Accuracy: {best_row['CV_Accuracy']:.3f}")
        print(f"  CV F1 Score: {best_row['CV_F1_Score']:.3f}")
        print(f"  Test Accuracy: {best_row['Test_Accuracy']:.3f}")
        print(f"  Test F1 Score: {best_row['Test_F1_Score']:.3f}")
        
        # Show all models comparison
        print(f"\n  All Models Comparison:")
        for _, row in asset_df.iterrows():
            mark = "★" if row['Is_Best_Model'] else " "
            print(f"    {mark} {row['Model']:20s} - Acc: {row['Test_Accuracy']:.3f}, "
                  f"F1: {row['Test_F1_Score']:.3f}, CV Score: {row['Selection_Score']:.3f}")
    
    print("\n" + "="*80)
    print("TODAY'S PREDICTIONS")
    print("="*80)
    
    for _, pred in predictions_df.iterrows():
        print(f"\n{pred['asset']}:")
        print(f"  Signal: {pred['signal']}")
        print(f"  Confidence: {pred['confidence']:.1f}%")
        print(f"  Current Price: ${pred['current_price']:.2f}")
        print(f"  Current Sharpe: {pred['current_sharpe']:.3f}")
        print(f"  Model Used: {pred['model_used']}")
        print(f"  Thresholds: Buy < {pred['buy_threshold']:.3f}, Sell > {pred['sell_threshold']:.3f}")

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("="*80)
    print("ENHANCED ML PIPELINE WITH PERCENTILE-BASED SHARPE RATIO")
    print("="*80)
    print(f"Processing {YEARS_OF_DATA} years of data for: {', '.join(SYMBOLS.keys())}")
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Using percentile thresholds: Buy < P{PERCENTILE_BUY}, Sell > P{PERCENTILE_SELL}")
    print("="*80)
    
    all_data = {}
    all_models = {}
    all_backtest = {}
    best_models = {}
    
    # Process each asset separately
    for symbol_name, symbol_ticker in SYMBOLS.items():
        print(f"\n{'='*30} {symbol_name} {'='*30}")
        
        # 1. Fetch data
        df = fetch_stock_data(symbol_name, symbol_ticker, START_DATE, END_DATE)
        if df is None:
            continue
        
        # 2. Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # 3. Create Sharpe ratio target variable with percentile thresholds
        df = create_sharpe_target_variable(df, PERCENTILE_BUY, PERCENTILE_SELL)
        
        # 4. Preprocess data
        df, feature_cols = preprocess_data(df, symbol_name)
        
        # 5. Prepare ML data
        df = df.dropna()
        X = df[feature_cols]
        y = df['Target']
        
        # 6. Temporal split (70/30)
        split_index = int(len(X) * 0.7)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        print(f"  Data split - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Verify class balance
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        print(f"  Train distribution: Hold={train_dist.get(0,0):.1%}, "
              f"Sell={train_dist.get(1,0):.1%}, Buy={train_dist.get(2,0):.1%}")
        print(f"  Test distribution:  Hold={test_dist.get(0,0):.1%}, "
              f"Sell={test_dist.get(1,0):.1%}, Buy={test_dist.get(2,0):.1%}")
        
        # 7. Train asset-specific models and select best
        models, best_model_name, scaler = train_asset_specific_models(
            X_train, X_test, y_train, y_test, symbol_name
        )
        
        # Store feature columns with results
        models['feature_cols'] = feature_cols
        
        # 8. Backtest with best model
        best_model = models[best_model_name]['model']
        backtest_results = backtest_strategy(df, best_model, scaler, feature_cols)
        
        # Store results
        all_data[symbol_name] = df
        all_models[symbol_name] = models
        all_backtest[symbol_name] = backtest_results
        best_models[symbol_name] = best_model_name
    
    # Create comprehensive outputs
    print("\n" + "="*80)
    model_selection_df, historical_df, predictions_df = create_comprehensive_output(
        all_data, all_models, all_backtest, best_models
    )
    
    # Save all outputs
    output_dir = save_all_outputs(model_selection_df, historical_df, predictions_df)
    
    # Print summary table
    print_summary_table(model_selection_df, predictions_df)
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}/")
    print("\nFiles created for Power BI:")
    print("  1. model_selection_results.csv - Model comparison and selection")
    print("  2. stock_ml_historical_data.csv - Historical data with predictions")
    print("  3. daily_predictions.csv - Today's BUY/SELL/HOLD signals")
    print("  4. metadata.json - Pipeline configuration")
    print("\n✅ Percentile-based thresholds ensure balanced classes!")
    print("   This should significantly improve model performance.")
    print("\nThese CSV files can be directly imported into Power BI!")
    print("="*80)
    
    return model_selection_df, historical_df, predictions_df

# Run the pipeline
if __name__ == "__main__":
    model_results, historical_data, today_predictions = main()
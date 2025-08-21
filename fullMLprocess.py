"""
Complete Stock Market ML Pipeline for Power BI Dashboard
Implements the full thesis methodology with daily updates for Power BI visualization
Author: [Your Name]
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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
# Stock symbols as per thesis - META (Tech), WTI (Oil), iShares MSCI World ETF
SYMBOLS = {
    'META': 'META',           # Technology sector
    'WTI': 'CL=F',           # WTI Crude Oil Futures
    'iShares_MSCI': 'URTH'   # iShares MSCI World ETF
}

# Data parameters
YEARS_OF_DATA = 5
START_DATE = datetime.now() - timedelta(days=YEARS_OF_DATA*365)
END_DATE = datetime.now()

# Thresholds as per thesis - using 1% threshold
BUY_THRESHOLD = -1.0   # Price decrease > 1% = Buy signal
SELL_THRESHOLD = 1.0   # Price increase > 1% = Sell signal

# ==================== DATA COLLECTION ====================
def fetch_stock_data(symbol_name, symbol_ticker, start, end):
    """
    Fetch historical stock data from Yahoo Finance
    Matches thesis data structure: Date, Open, High, Low, Close, Volume
    """
    print(f"Fetching data for {symbol_name} ({symbol_ticker})...")
    
    try:
        stock = yf.Ticker(symbol_ticker)
        data = stock.history(start=start, end=end)
        
        if data.empty:
            print(f"Warning: No data retrieved for {symbol_ticker}")
            return None
            
        # Ensure we have all required columns as per thesis
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Add symbol identifier
        data['Symbol'] = symbol_name
        data['Ticker'] = symbol_ticker
        
        print(f"  Retrieved {len(data)} days of data for {symbol_name}")
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol_ticker}: {str(e)}")
        return None

# ==================== TECHNICAL INDICATORS (AS PER THESIS) ====================
def calculate_technical_indicators(df):
    """
    Calculate all 18 technical indicators mentioned in the thesis
    Grouped into 7 main indicator categories
    """
    print("  Calculating technical indicators...")
    
    # 1. RSI (Relative Strength Index) - 14 period as per thesis
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # 2. MACD (Moving Average Convergence Divergence) - 12, 26, 9 as per thesis
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df['MACDs_12_26_9'] = macd.macd_signal()  # Signal line
    df['MACDh_12_26_9'] = macd.macd_diff()    # Histogram
    
    # 3. Moving Averages - SMA 50 and EMA 9 as per thesis
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    
    # 4. Stochastic Oscillator - 14, 3, 3 parameters as per thesis
    stoch = StochasticOscillator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['STOCHk_14_3_3'] = stoch.stoch()
    df['STOCHd_14_3_3'] = stoch.stoch_signal()
    
    # 5. Bollinger Bands - 20 period, 2 std dev as per thesis
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()   # Lower Band
    df['BBM_20_2.0'] = bb.bollinger_mavg()    # Middle Band
    df['BBU_20_2.0'] = bb.bollinger_hband()   # Upper Band
    df['BBB_20_2.0'] = bb.bollinger_wband()   # Bandwidth
    df['BBP_20_2.0'] = bb.bollinger_pband()   # Percent B
    
    # 6. Ichimoku Cloud - 9, 26, 52 periods as per thesis
    ichimoku = IchimokuIndicator(
        high=df['High'],
        low=df['Low'],
        window1=9,
        window2=26,
        window3=52
    )
    df['ITS_9'] = ichimoku.ichimoku_conversion_line()     # Tenkan-sen
    df['IKS_26'] = ichimoku.ichimoku_base_line()          # Kijun-sen
    df['ISA_9'] = ichimoku.ichimoku_a()                   # Senkou Span A
    df['ISB_26'] = ichimoku.ichimoku_b()                  # Senkou Span B
    
    # Chikou Span (26 period lag) - manual calculation
    df['ICS_26'] = df['Close'].shift(-26)
    
    # 7. Standard Deviation - 30 period as per thesis
    df['STDEV_30'] = df['Close'].rolling(window=30).std()
    
    return df

# ==================== TARGET VARIABLE CREATION (AS PER THESIS) ====================
def create_target_variable(df):
    """
    Create target variable based on thesis methodology:
    - Calculate daily change percentage from opening prices
    - Apply 1% threshold for Buy/Sell signals
    """
    # Calculate daily change as per thesis formula
    df['Daily_Change'] = ((df['Open'] - df['Open'].shift(1)) / df['Open'].shift(1)) * 100
    
    # Create target variable with thesis thresholds
    conditions = [
        df['Daily_Change'] > SELL_THRESHOLD,   # > 1% = Sell (1)
        df['Daily_Change'] < BUY_THRESHOLD     # < -1% = Buy (2)
    ]
    choices = [1, 2]  # 1: Sell, 2: Buy
    df['Target'] = np.select(conditions, choices, default=0)  # 0: Hold
    
    # Add readable labels
    df['Target_Label'] = df['Target'].map({0: 'HOLD', 1: 'SELL', 2: 'BUY'})
    
    # Calculate target distribution for monitoring
    target_dist = df['Target'].value_counts(normalize=True)
    print(f"    Target distribution - Hold: {target_dist.get(0, 0):.1%}, "
          f"Sell: {target_dist.get(1, 0):.1%}, Buy: {target_dist.get(2, 0):.1%}")
    
    return df

# ==================== DATA PREPROCESSING (AS PER THESIS) ====================
def preprocess_data(df, symbol_name):
    """
    Implement preprocessing steps from thesis:
    - Remove low variance features
    - Handle outliers with z-score
    - Impute missing values with KNN
    - NO PCA as requested
    """
    print(f"  Preprocessing data for {symbol_name}...")
    
    # Get feature columns (all indicators)
    feature_cols = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                   'SMA_50', 'EMA_9', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
                   'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                   'ITS_9', 'IKS_26', 'ISA_9', 'ISB_26', 'ICS_26', 'STDEV_30']
    
    # 1. Remove features with variance < 0.1 (as per thesis)
    variances = df[feature_cols].var()
    low_var_features = variances[variances < 0.1].index.tolist()
    if low_var_features:
        print(f"    Removing low variance features: {low_var_features}")
        feature_cols = [col for col in feature_cols if col not in low_var_features]
    
    # 2. Handle outliers using z-score (cap and floor method from thesis)
    for col in feature_cols:
        if col in df.columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df.loc[z_scores > 3, col] = df[col].quantile(0.99)  # Cap
            df.loc[z_scores < -3, col] = df[col].quantile(0.01)  # Floor
    
    # 3. Impute missing values using KNN (as per thesis)
    if df[feature_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        print(f"    Imputed missing values using KNN")
    
    return df, feature_cols

# ==================== MACHINE LEARNING MODELS (AS PER THESIS) ====================
def train_models(X_train, X_test, y_train, y_test, symbol_name):
    """
    Train all models mentioned in thesis:
    - Random Forest
    - Decision Tree
    - XGBoost
    - KNN
    - AdaBoost
    - Ensemble Learning
    """
    print(f"  Training ML models for {symbol_name}...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE (only for iShares as per thesis)
    if symbol_name == 'iShares_MSCI':
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(y_train.value_counts())-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            print(f"    Applied SMOTE balancing for {symbol_name}")
        except:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
            print(f"    SMOTE not applied due to insufficient samples")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Initialize models with thesis parameters
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'XGBoost': XGBClassifier(
            objective='multi:softprob',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            random_state=42
        )
    }
    
    results = {}
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    # Train each model
    for name, model in models.items():
        try:
            # Train
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
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
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
                
            print(f"      {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
        except Exception as e:
            print(f"      Error training {name}: {str(e)}")
    
    # Create Ensemble model
    try:
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft'
        )
        ensemble.fit(X_train_balanced, y_train_balanced)
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        
        results['Ensemble'] = {
            'model': ensemble,
            'scaler': scaler,
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble),
            'predictions': y_pred_ensemble
        }
        print(f"      Ensemble: Accuracy={results['Ensemble']['accuracy']:.3f}")
    except Exception as e:
        print(f"      Error creating ensemble: {str(e)}")
    
    print(f"    Best model: {best_model_name} with accuracy {best_accuracy:.3f}")
    return results, best_model, best_model_name, scaler

# ==================== BACKTESTING (AS PER THESIS) ====================
def backtest_strategy(df, model, scaler, feature_cols, initial_capital=100):
    """
    Perform backtesting as described in thesis
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
        current_price = row['Open']  # Use opening price as per thesis
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
        'current_price': df['Close'].iloc[-1]
    }

# ==================== CREATE POWER BI OUTPUT ====================
def create_powerbi_output(all_data, all_models, all_backtest):
    """
    Create comprehensive DataFrame for Power BI with all required columns
    """
    print("\\nCreating Power BI output...")
    
    powerbi_frames = []
    daily_predictions = []
    model_performance = []
    
    for symbol_name in all_data.keys():
        df = all_data[symbol_name]
        models = all_models[symbol_name]
        backtest_df, backtest_metrics = all_backtest[symbol_name]
        
        # Add all columns from backtest_df
        export_df = backtest_df.copy()
        
        # Add time-based columns for Power BI
        export_df['Date'] = export_df.index
        export_df['Year'] = export_df.index.year
        export_df['Month'] = export_df.index.month
        export_df['Quarter'] = export_df.index.quarter
        export_df['Week'] = export_df.index.isocalendar().week
        export_df['DayOfWeek'] = export_df.index.dayofweek
        export_df['DayName'] = export_df.index.day_name()
        
        # Add performance metrics
        export_df['Backtest_TotalReturn'] = backtest_metrics['total_return']
        export_df['Backtest_SharpeRatio'] = backtest_metrics['sharpe_ratio']
        export_df['Backtest_MaxDrawdown'] = backtest_metrics['max_drawdown']
        
        # Add model accuracy metrics
        for model_name, results in models.items():
            export_df[f'{model_name}_Accuracy'] = results['accuracy']
            export_df[f'{model_name}_F1Score'] = results['f1_score']
            
            # Add confusion matrix values
            cm = results['confusion_matrix']
            export_df[f'{model_name}_CM_Hold_Hold'] = cm[0, 0] if cm.shape[0] > 0 else 0
            export_df[f'{model_name}_CM_Hold_Sell'] = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
            export_df[f'{model_name}_CM_Hold_Buy'] = cm[0, 2] if cm.shape[0] > 0 and cm.shape[1] > 2 else 0
            
            if cm.shape[0] > 1:
                export_df[f'{model_name}_CM_Sell_Hold'] = cm[1, 0] if cm.shape[1] > 0 else 0
                export_df[f'{model_name}_CM_Sell_Sell'] = cm[1, 1] if cm.shape[1] > 1 else 0
                export_df[f'{model_name}_CM_Sell_Buy'] = cm[1, 2] if cm.shape[1] > 2 else 0
            
            if cm.shape[0] > 2:
                export_df[f'{model_name}_CM_Buy_Hold'] = cm[2, 0] if cm.shape[1] > 0 else 0
                export_df[f'{model_name}_CM_Buy_Sell'] = cm[2, 1] if cm.shape[1] > 1 else 0
                export_df[f'{model_name}_CM_Buy_Buy'] = cm[2, 2] if cm.shape[1] > 2 else 0
        
        # Calculate prediction accuracy
        export_df['Prediction_Correct'] = (export_df['Target'] == export_df['Predicted_Signal']).astype(int)
        
        # Add rolling accuracy metrics
        export_df['Accuracy_7Day'] = export_df['Prediction_Correct'].rolling(window=7, min_periods=1).mean()
        export_df['Accuracy_30Day'] = export_df['Prediction_Correct'].rolling(window=30, min_periods=1).mean()
        
        powerbi_frames.append(export_df)
        
        # Store today's prediction
        best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
        today_pred = generate_daily_predictions(
            df,
            models[best_model_name]['model'],
            models[best_model_name]['scaler'],
            all_models[symbol_name]['feature_cols']
        )
        today_pred['symbol'] = symbol_name
        daily_predictions.append(today_pred)
        
        # Store model performance summary
        for model_name, results in models.items():
            model_performance.append({
                'Symbol': symbol_name,
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score'],
                'ROC_AUC': results.get('roc_auc', 0),
                'Update_Date': datetime.now()
            })
    
    # Combine all stock data
    final_df = pd.concat(powerbi_frames, ignore_index=True)
    
    # Create daily predictions dataframe
    predictions_df = pd.DataFrame(daily_predictions)
    
    # Create model performance dataframe
    performance_df = pd.DataFrame(model_performance)
    
    return final_df, predictions_df, performance_df

# ==================== SAVE OUTPUTS ====================
def save_outputs(main_df, predictions_df, performance_df):
    """Save all outputs for Power BI"""
    
    # Create output directory
    output_dir = 'powerbi_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main historical data with all calculations
    main_file = os.path.join(output_dir, 'stock_ml_historical_data.csv')
    main_df.to_csv(main_file, index=False)
    print(f"  Saved historical data: {main_file} ({len(main_df)} rows)")
    
    # Save today's predictions
    predictions_file = os.path.join(output_dir, 'daily_predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print(f"  Saved daily predictions: {predictions_file}")
    
    # Save model performance metrics
    performance_file = os.path.join(output_dir, 'model_performance.csv')
    performance_df.to_csv(performance_file, index=False)
    print(f"  Saved model performance: {performance_file}")
    
    # Save metadata for Power BI
    metadata = {
        'last_update': datetime.now().isoformat(),
        'stocks': list(SYMBOLS.keys()),
        'data_start': START_DATE.isoformat(),
        'data_end': END_DATE.isoformat(),
        'total_rows': len(main_df),
        'features_used': main_df.columns.tolist()
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return output_dir

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("="*70)
    print("STOCK MARKET ML PIPELINE FOR POWER BI")
    print("="*70)
    print(f"Processing {YEARS_OF_DATA} years of data for: {', '.join(SYMBOLS.keys())}")
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print("="*70)
    
    all_data = {}
    all_models = {}
    all_backtest = {}
    
    # Process each stock
    for symbol_name, symbol_ticker in SYMBOLS.items():
        print(f"\\n{'='*30} {symbol_name} {'='*30}")
        
        # 1. Fetch data
        df = fetch_stock_data(symbol_name, symbol_ticker, START_DATE, END_DATE)
        if df is None:
            continue
        
        # 2. Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # 3. Create target variable
        df = create_target_variable(df)
        
        # 4. Preprocess data
        df, feature_cols = preprocess_data(df, symbol_name)
        
        # 5. Prepare ML data
        df = df.dropna()
        X = df[feature_cols]
        y = df['Target']
        
        # 6. Temporal split (70/30 as per thesis)
        split_index = int(len(X) * 0.7)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        print(f"  Data split - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # 7. Train models
        models, best_model, best_model_name, scaler = train_models(
            X_train, X_test, y_train, y_test, symbol_name
        )
        
        # Store feature columns with results
        models['feature_cols'] = feature_cols
        
        # 8. Backtest with best model
        backtest_results = backtest_strategy(df, best_model, scaler, feature_cols)
        
        # Store results
        all_data[symbol_name] = df
        all_models[symbol_name] = models
        all_backtest[symbol_name] = backtest_results
    
    # Create Power BI output
    print("\\n" + "="*70)
    main_df, predictions_df, performance_df = create_powerbi_output(
        all_data, all_models, all_backtest
    )
    
    # Save all outputs
    output_dir = save_outputs(main_df, predictions_df, performance_df)
    
    # Print summary
    print("\\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\\nOutput files saved to: {output_dir}/")
    print("\\nFiles created:")
    print("  1. stock_ml_historical_data.csv - Main dataset for Power BI")
    print("  2. daily_predictions.csv - Today's BUY/SELL/HOLD predictions")
    print("  3. model_performance.csv - Model accuracy metrics")
    print("  4. metadata.json - Pipeline metadata")
    print("\\nToday's Predictions:")
    print(predictions_df[['symbol', 'signal', 'confidence', 'current_price']])
    print("\\nYou can now import these CSV files into Power BI!")
    print("="*70)
    
    return main_df, predictions_df, performance_df

# Run the pipeline
if __name__ == "__main__":
    historical_data, today_predictions, model_metrics = main()
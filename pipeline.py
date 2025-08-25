"""
ML Trading Pipeline for Power BI - Fresh Start
Simplified and robust version with proper error handling
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Technical indicators
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# ========== CONFIGURATION ==========
SYMBOLS = {
    'META': 'META',
    'WTI': 'CL=F', 
    'URTH': 'URTH'
}

THRESHOLDS = {
    'META': 1.5,   # 1.5% daily move
    'WTI': 1.8,    # 1.8% daily move
    'URTH': 0.5    # 0.5% daily move
}

# Create directories
OUTPUT_DIR = Path("powerbi_data")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class TradingPipeline:
    def __init__(self):
        self.results = []
        self.today = datetime.now()
        
    def fetch_data(self, symbol, ticker, period="2y"):
        """Fetch stock data with error handling"""
        print(f"\n📊 Fetching data for {symbol}...")
        
        try:
            # Method 1: Using Ticker.history()
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            # If empty, try Method 2: yf.download()
            if df.empty:
                print(f"  Trying alternative method...")
                df = yf.download(ticker, period=period, progress=False)
            
            if df.empty:
                print(f"  ❌ No data available for {symbol}")
                return None
                
            print(f"  ✅ Fetched {len(df)} days of data")
            print(f"  📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
            return df
            
        except Exception as e:
            print(f"  ❌ Error fetching {symbol}: {str(e)}")
            return None
    
    def prepare_features(self, df, symbol):
        """Create technical indicators and features"""
        print(f"  🔧 Creating features for {symbol}...")
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change() * 100
        df['Tomorrow_Return'] = df['Returns'].shift(-1)
        
        # Price features
        df['High_Low_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
        df['Close_Open_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
        
        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
        df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']
        
        # RSI
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period) * 100
        
        # Volatility
        df['Volatility_20d'] = df['Returns'].rolling(20).std()
        
        # Create target variable (Buy/Sell/Hold)
        threshold = THRESHOLDS[symbol]
        conditions = [
            df['Tomorrow_Return'] > threshold,
            df['Tomorrow_Return'] < -threshold
        ]
        choices = ['BUY', 'SELL']
        df['Target'] = np.select(conditions, choices, default='HOLD')
        
        # Remove last row (no tomorrow return)
        df = df[:-1]
        
        print(f"  ✅ Created {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features")
        
        return df
    
    def train_model(self, df, symbol):
        """Train or load ML model"""
        print(f"  🤖 Training model for {symbol}...")
        
        # Define feature columns
        feature_cols = [
            'High_Low_Pct', 'Close_Open_Pct', 'Volume_Ratio',
            'Price_SMA20_Ratio', 'Price_SMA50_Ratio', 'RSI',
            'MACD', 'MACD_Signal', 'MACD_Diff',
            'BB_Width', 'BB_Position',
            'Return_5d', 'Return_10d', 'Return_20d',
            'Volatility_20d'
        ]
        
        # Clean data
        df_model = df.dropna()
        
        # Prepare features and target
        X = df_model[feature_cols]
        y = df_model['Target']
        
        # Check if we have enough data
        if len(X) < 100:
            print(f"  ⚠️ Not enough data for {symbol} ({len(X)} samples)")
            return None, None, None
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Check for model file
        model_path = MODEL_DIR / f"{symbol}_model.pkl"
        scaler_path = MODEL_DIR / f"{symbol}_scaler.pkl"
        
        if model_path.exists() and scaler_path.exists():
            # Check model age
            model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            if model_age.days < 7:
                print(f"  📂 Loading existing model (age: {model_age.days} days)")
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler, (X_test, y_test)
        
        # Train new model
        print(f"  🔄 Training new model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  📈 Model accuracy: {accuracy:.2%}")
        
        # Save model
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        return model, scaler, (X_test, y_test)
    
    def make_predictions(self, df, model, scaler, symbol):
        """Make predictions including today's prediction"""
        print(f"  🎯 Making predictions for {symbol}...")
        
        feature_cols = [
            'High_Low_Pct', 'Close_Open_Pct', 'Volume_Ratio',
            'Price_SMA20_Ratio', 'Price_SMA50_Ratio', 'RSI',
            'MACD', 'MACD_Signal', 'MACD_Diff',
            'BB_Width', 'BB_Position',
            'Return_5d', 'Return_10d', 'Return_20d',
            'Volatility_20d'
        ]
        
        # Get latest data point for today's prediction
        df_clean = df.dropna()
        if len(df_clean) > 0:
            X_today = df_clean[feature_cols].iloc[-1:].values
            X_today_scaled = scaler.transform(X_today)
            
            # Make prediction
            today_pred = model.predict(X_today_scaled)[0]
            today_proba = model.predict_proba(X_today_scaled)[0]
            
            # Get confidence for each class
            classes = model.classes_
            confidence_dict = {cls: prob for cls, prob in zip(classes, today_proba)}
            
            return {
                'prediction': today_pred,
                'confidence': confidence_dict,
                'current_price': df_clean['Close'].iloc[-1],
                'date': df_clean.index[-1]
            }
        
        return None
    
    def process_asset(self, symbol, ticker):
        """Process a single asset"""
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print('='*60)
        
        # Fetch data
        df = self.fetch_data(symbol, ticker)
        if df is None:
            return None
        
        # Prepare features
        df = self.prepare_features(df, symbol)
        
        # Train model
        model, scaler, test_data = self.train_model(df, symbol)
        if model is None:
            return None
        
        # Make predictions
        prediction = self.make_predictions(df, model, scaler, symbol)
        if prediction is None:
            return None
        
        # Calculate model performance
        X_test, y_test = test_data
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=['HOLD', 'SELL', 'BUY'])
        
        # Prepare data for Power BI
        result = {
            'symbol': symbol,
            'ticker': ticker,
            'data': df,
            'model_accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'today_prediction': prediction['prediction'],
            'today_confidence': prediction['confidence'],
            'current_price': prediction['current_price'],
            'threshold': THRESHOLDS[symbol]
        }
        
        print(f"\n  📊 Results for {symbol}:")
        print(f"     Today's Prediction: {prediction['prediction']}")
        print(f"     Current Price: ${prediction['current_price']:.2f}")
        print(f"     Model Accuracy: {accuracy:.2%}")
        
        return result
    
    def save_to_powerbi(self, results):
        """Save data in Power BI format"""
        print("\n" + "="*60)
        print("💾 Saving data for Power BI")
        print("="*60)
        
        # Prepare fact tables
        fact_prices = []
        fact_predictions = []
        fact_indicators = []
        fact_performance = []
        
        for result in results:
            if result is None:
                continue
            
            symbol = result['symbol']
            df = result['data']
            
            # Get last 90 days of data
            df_recent = df.tail(90)
            
            # FACT: Prices
            for date, row in df_recent.iterrows():
                fact_prices.append({
                    'Date': date.date(),
                    'AssetID': symbol,
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Close': row['Close'],
                    'Volume': row['Volume'],
                    'Returns': row.get('Returns', 0)
                })
            
            # FACT: Predictions (today's)
            fact_predictions.append({
                'Date': datetime.now().date(),
                'AssetID': symbol,
                'Prediction': result['today_prediction'],
                'Confidence_BUY': result['today_confidence'].get('BUY', 0),
                'Confidence_SELL': result['today_confidence'].get('SELL', 0),
                'Confidence_HOLD': result['today_confidence'].get('HOLD', 0),
                'CurrentPrice': result['current_price'],
                'ModelAccuracy': result['model_accuracy']
            })
            
            # FACT: Technical Indicators
            for date, row in df_recent.iterrows():
                fact_indicators.append({
                    'Date': date.date(),
                    'AssetID': symbol,
                    'RSI': row.get('RSI', None),
                    'MACD': row.get('MACD', None),
                    'MACD_Signal': row.get('MACD_Signal', None),
                    'SMA_20': row.get('SMA_20', None),
                    'SMA_50': row.get('SMA_50', None),
                    'SMA_200': row.get('SMA_200', None),
                    'BB_Upper': row.get('BB_Upper', None),
                    'BB_Lower': row.get('BB_Lower', None),
                    'BB_Position': row.get('BB_Position', None),
                    'Volatility': row.get('Volatility_20d', None)
                })
            
            # FACT: Model Performance (confusion matrix)
            cm = result['confusion_matrix']
            for i, actual in enumerate(['HOLD', 'SELL', 'BUY']):
                for j, predicted in enumerate(['HOLD', 'SELL', 'BUY']):
                    fact_performance.append({
                        'Date': datetime.now().date(),
                        'AssetID': symbol,
                        'Actual': actual,
                        'Predicted': predicted,
                        'Count': int(cm[i][j]) if i < len(cm) and j < len(cm[i]) else 0
                    })
        
        # Dimension tables
        dim_assets = [
            {'AssetID': 'META', 'AssetName': 'Meta Platforms', 'AssetType': 'Stock', 'Sector': 'Technology'},
            {'AssetID': 'WTI', 'AssetName': 'WTI Crude Oil', 'AssetType': 'Commodity', 'Sector': 'Energy'},
            {'AssetID': 'URTH', 'AssetName': 'iShares MSCI World ETF', 'AssetType': 'ETF', 'Sector': 'Global'}
        ]
        
        dim_signals = [
            {'SignalID': 'BUY', 'Description': 'Buy Signal', 'Color': '#00FF00', 'Action': 1},
            {'SignalID': 'SELL', 'Description': 'Sell Signal', 'Color': '#FF0000', 'Action': -1},
            {'SignalID': 'HOLD', 'Description': 'Hold Position', 'Color': '#808080', 'Action': 0}
        ]
        
        # Save to CSV
        if fact_prices:
            pd.DataFrame(fact_prices).to_csv(OUTPUT_DIR / 'fact_prices.csv', index=False)
            print(f"  ✅ Saved fact_prices.csv ({len(fact_prices)} rows)")
        
        if fact_predictions:
            pd.DataFrame(fact_predictions).to_csv(OUTPUT_DIR / 'fact_predictions.csv', index=False)
            print(f"  ✅ Saved fact_predictions.csv ({len(fact_predictions)} rows)")
        
        if fact_indicators:
            pd.DataFrame(fact_indicators).to_csv(OUTPUT_DIR / 'fact_indicators.csv', index=False)
            print(f"  ✅ Saved fact_indicators.csv ({len(fact_indicators)} rows)")
        
        if fact_performance:
            pd.DataFrame(fact_performance).to_csv(OUTPUT_DIR / 'fact_performance.csv', index=False)
            print(f"  ✅ Saved fact_performance.csv ({len(fact_performance)} rows)")
        
        pd.DataFrame(dim_assets).to_csv(OUTPUT_DIR / 'dim_assets.csv', index=False)
        print(f"  ✅ Saved dim_assets.csv")
        
        pd.DataFrame(dim_signals).to_csv(OUTPUT_DIR / 'dim_signals.csv', index=False)
        print(f"  ✅ Saved dim_signals.csv")
        
        # Save summary
        summary = {
            'last_update': datetime.now().isoformat(),
            'assets_processed': [r['symbol'] for r in results if r is not None],
            'predictions': {
                r['symbol']: {
                    'signal': r['today_prediction'],
                    'price': float(r['current_price']),
                    'accuracy': float(r['model_accuracy'])
                } for r in results if r is not None
            }
        }
        
        with open(OUTPUT_DIR / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✅ Saved summary.json")
        
        return summary
    
    def run(self):
        """Main execution"""
        print("\n" + "="*60)
        print("🚀 ML TRADING PIPELINE FOR POWER BI")
        print(f"📅 Date: {self.today.strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        results = []
        
        # Process each asset
        for symbol, ticker in SYMBOLS.items():
            result = self.process_asset(symbol, ticker)
            results.append(result)
        
        # Save to Power BI format
        summary = self.save_to_powerbi(results)
        
        # Print final summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
        print("="*60)
        
        if summary['assets_processed']:
            print("\n📊 Today's Signals:")
            for asset, data in summary['predictions'].items():
                print(f"  {asset:6} → {data['signal']:6} (${data['price']:.2f}, Accuracy: {data['accuracy']:.1%})")
        else:
            print("  ⚠️ No assets were successfully processed")
        
        print(f"\n📁 Output saved to: {OUTPUT_DIR.absolute()}")
        
        return summary

# ========== MAIN EXECUTION ==========
def main():
    """Entry point"""
    pipeline = TradingPipeline()
    summary = pipeline.run()
    return summary

if __name__ == "__main__":
    main()
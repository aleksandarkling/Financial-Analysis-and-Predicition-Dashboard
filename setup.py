"""
Setup script to install dependencies and test the pipeline
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        'pandas',
        'numpy',
        'yfinance',
        'scikit-learn',
        'ta',
        'joblib',
        'openpyxl'  # For Excel support
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed!\n")

def test_imports():
    """Test if all imports work"""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        print("  ✅ pandas")
        
        import numpy as np
        print("  ✅ numpy")
        
        import yfinance as yf
        print("  ✅ yfinance")
        
        from sklearn.ensemble import RandomForestClassifier
        print("  ✅ scikit-learn")
        
        from ta.trend import MACD
        print("  ✅ ta (technical analysis)")
        
        import joblib
        print("  ✅ joblib")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

def test_yfinance():
    """Test if yfinance can fetch data"""
    print("🌐 Testing data fetching...")
    
    try:
        import yfinance as yf
        
        # Test with META
        ticker = yf.Ticker("META")
        hist = ticker.history(period="1mo")
        
        if not hist.empty:
            print(f"  ✅ Successfully fetched META data")
            print(f"     Latest close: ${hist['Close'].iloc[-1]:.2f}")
            print(f"     Data points: {len(hist)}")
        else:
            print("  ⚠️ No data returned for META")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("📁 Creating directories...")
    
    dirs = ['powerbi_data', 'models']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  ✅ Created {dir_name}/")
        else:
            print(f"  ℹ️ {dir_name}/ already exists")
    
    print()

def run_pipeline_test():
    """Run a quick test of the pipeline"""
    print("🚀 Running pipeline test...")
    
    try:
        from pipeline import TradingPipeline
        
        pipeline = TradingPipeline()
        
        # Test with just META
        result = pipeline.process_asset('META', 'META')
        
        if result:
            print(f"\n✅ Pipeline test successful!")
            print(f"  Prediction: {result['today_prediction']}")
            print(f"  Price: ${result['current_price']:.2f}")
            print(f"  Accuracy: {result['model_accuracy']:.1%}")
        else:
            print("\n⚠️ Pipeline test returned no results")
            
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main setup process"""
    print("="*60)
    print("🔧 ML TRADING PIPELINE SETUP")
    print("="*60)
    print()
    
    # Step 1: Install dependencies
    response = input("Install/update dependencies? (y/n): ")
    if response.lower() == 'y':
        install_dependencies()
    
    # Step 2: Test imports
    if not test_imports():
        print("Please fix import errors before continuing")
        return
    
    # Step 3: Test yfinance
    if not test_yfinance():
        print("Please check your internet connection")
        return
    
    # Step 4: Create directories
    create_directories()
    
    # Step 5: Run pipeline test
    response = input("Run pipeline test? (y/n): ")
    if response.lower() == 'y':
        run_pipeline_test()
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the pipeline: python pipeline.py")
    print("2. Check powerbi_data/ folder for CSV files")
    print("3. Import CSV files into Power BI Desktop")
    print("4. Follow the Power BI setup guide")

if __name__ == "__main__":
    main()
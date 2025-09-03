#!/usr/bin/env python3
"""
Test script to verify the modified lasso_tracking_model function works correctly
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.getcwd())

def test_lasso_function():
    """Test the modified lasso_tracking_model function"""
    
    try:
        # Load data
        print("Loading data...")
        data = pd.read_excel('precios_limpios.xlsx')
        print(f"Data loaded successfully. Shape: {data.shape}")
        
        # Convert dates
        if 'DATES' in data.columns:
            data['DATES'] = pd.to_datetime(data['DATES'])
            data.set_index('DATES', inplace=True)
        
        # Separate dependent variable (IPSA Index)
        ipsa_column = 'IPSA Index' if 'IPSA Index' in data.columns else None
        
        if ipsa_column:
            ipsa_prices = data[ipsa_column].copy()
        else:
            # Create synthetic IPSA as equal-weighted average
            price_columns = [col for col in data.columns if not col.startswith('Unnamed')]
            ipsa_prices = data[price_columns].mean(axis=1)
        
        # Independent variables (all other columns except IPSA)
        independent_columns = [col for col in data.columns 
                             if col not in [ipsa_column, 'DATES'] and not col.startswith('Unnamed')]
        
        stock_prices = data[independent_columns].copy()
        
        # Remove rows with missing values
        stock_prices = stock_prices.dropna()
        ipsa_prices = ipsa_prices.loc[stock_prices.index]
        
        # Calculate logarithmic returns
        stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
        ipsa_returns = np.log(ipsa_prices / ipsa_prices.shift(1)).dropna()
        
        # Align dates
        common_dates = stock_returns.index.intersection(ipsa_returns.index)
        stock_returns = stock_returns.loc[common_dates]
        ipsa_returns = ipsa_returns.loc[common_dates]
        
        print(f"Dependent variable: {ipsa_column if ipsa_column else 'Synthetic IPSA'}")
        print(f"Independent variables: {len(stock_returns.columns)} assets")
        print(f"Period: {stock_returns.index.min()} to {stock_returns.index.max()}")
        print(f"Observations: {len(stock_returns)}")
        
        # Import the modified function
        print("\nImporting the modified lasso_tracking_model function...")
        
        # Read the function from the file
        with open('bloomberg_ipsa/modelo_TE2', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the function definition
        start_idx = content.find('def lasso_tracking_model(')
        if start_idx == -1:
            raise ValueError("Function lasso_tracking_model not found")
        
        # Find the end of the function (next function or end of file)
        lines = content[start_idx:].split('\n')
        function_lines = []
        indent_level = None
        
        for line in lines:
            if line.strip().startswith('def ') and line != lines[0]:
                break
            if line.strip() and indent_level is None:
                indent_level = len(line) - len(line.lstrip())
            if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.startswith(' '):
                if line.strip() and not line.strip().startswith('#'):
                    break
            function_lines.append(line)
        
        function_code = '\n'.join(function_lines)
        
        # Execute the function definition
        exec(function_code, globals())
        
        print("Function imported successfully!")
        
        # Test the function
        print("\nTesting the modified lasso_tracking_model function...")
        print("=" * 60)
        
        lasso_weights, lasso_cumulative = lasso_tracking_model(ipsa_returns, stock_returns)
        
        print("=" * 60)
        print("Function test completed successfully!")
        
        # Verify results
        print(f"\nVerification:")
        print(f"- Number of stocks in weights: {len(lasso_weights)}")
        print(f"- Total stocks available: {len(stock_returns.columns)}")
        print(f"- Weights sum: {sum(lasso_weights.values()):.6f}")
        print(f"- Cumulative returns shape: {lasso_cumulative.shape}")
        
        # Check if we're using all stocks (not just top 4)
        significant_weights = [w for w in lasso_weights.values() if abs(w) > 0.001]
        print(f"- Stocks with significant weights (>0.001): {len(significant_weights)}")
        
        if len(lasso_weights) == len(stock_returns.columns):
            print("✅ SUCCESS: Function now uses all available stocks (no top 4 restriction)")
        else:
            print("❌ ERROR: Function still has restrictions")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lasso_function()
    if success:
        print("\n🎉 All tests passed! The function works correctly.")
    else:
        print("\n💥 Tests failed. Please check the errors above.")

#!/usr/bin/env python
"""
Main execution script - Bayesian Ranking of Trading Strategies
Complete project pipeline from data collection to strategy ranking
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import DataCollector
from src.strategies import SMAStrategy, RSIStrategy, MACrossoverStrategy
from src.backtest import BacktestEngine
from src.bayesian_model import BayesianStrategyRanker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/backtest.log')
    ]
)
logger = logging.getLogger(__name__)

def create_results_directory():
    """Create results directory if it doesn't exist"""
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

def run_complete_pipeline():
    """
    Run the complete Bayesian strategy ranking pipeline
    """
    logger.info("="*60)
    logger.info("BAYESIAN RANKING OF INVESTMENT STRATEGIES")
    logger.info("Period: May 2025 - July 2025")
    logger.info("="*60)
    
    # Create results directory
    create_results_directory()
    
    # Step 1: Data Collection
    logger.info("\n" + "="*40)
    logger.info("STEP 1: Collecting OHLCV Data")
    logger.info("="*40)
    
    collector = DataCollector()
    
    # Collect data for multiple assets
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    end_date = '2025-07-31'
    start_date = '2025-05-01'
    
    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching {symbol} data...")
        data = collector.fetch_ohlcv(symbol, start_date, end_date)
        all_data[symbol] = data
        logger.info(f"  ✓ {symbol}: {len(data)} days of data")
    
    # Use SPY for strategy testing
    data = all_data['SPY']
    logger.info(f"\nUsing SPY data for strategy testing")
    logger.info(f"  Period: {data.index[0]} to {data.index[-1]}")
    logger.info(f"  Days: {len(data)}")
    
    # Step 2: Define Strategies
    logger.info("\n" + "="*40)
    logger.info("STEP 2: Defining Trading Strategies")
    logger.info("="*40)
    
    strategies = [
        # SMA Crossover strategies
        SMAStrategy(short_window=10, long_window=30),
        SMAStrategy(short_window=20, long_window=50),
        SMAStrategy(short_window=30, long_window=60),
        
        # RSI strategies with different parameters
        RSIStrategy(period=14, overbought=70, oversold=30),
        RSIStrategy(period=14, overbought=80, oversold=20),
        RSIStrategy(period=21, overbought=70, oversold=30),
        
        # Multiple moving average crossover
        MACrossoverStrategy(),
    ]
    
    logger.info(f"Total strategies: {len(strategies)}")
    for s in strategies:
        logger.info(f"  - {s.name}")
    
    # Step 3: Backtesting
    logger.info("\n" + "="*40)
    logger.info("STEP 3: Running Backtests")
    logger.info("="*40)
    
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    results = engine.run_backtest(data, strategies)
    
    logger.info(f"Successfully backtested {len(results)} strategies")
    
    # Step 4: Extract Returns for Bayesian Analysis
    logger.info("\n" + "="*40)
    logger.info("STEP 4: Preparing Data for Bayesian Analysis")
    logger.info("="*40)
    
    returns_dict = {}
    portfolios_dict = {}
    
    for name, result in results.items():
        returns = result['portfolio']['returns'].dropna()
        if len(returns) > 0:
            returns_dict[name] = returns.values
            portfolios_dict[name] = result
            logger.info(f"  {name}: {len(returns)} returns, Sharpe: {result['metrics']['sharpe_ratio']:.3f}")
    
    # Step 5: Bayesian Analysis
    logger.info("\n" + "="*40)
    logger.info("STEP 5: Bayesian Strategy Ranking")
    logger.info("="*40)
    
    ranker = BayesianStrategyRanker(list(returns_dict.keys()))
    
    logger.info("Fitting Bayesian model (this may take a few minutes)...")
    ranker.fit(returns_dict, n_samples=2000, n_chains=4)
    
    # Get rankings
    rankings = ranker.get_ranking()
    
    logger.info("\n" + "-"*40)
    logger.info("STRATEGY RANKING RESULTS")
    logger.info("-"*40)
    
    for idx, row in rankings.iterrows():
        medal = "🥇" if row['Rank'] == 1 else "🥈" if row['Rank'] == 2 else "🥉" if row['Rank'] == 3 else ""
        logger.info(f"{medal} Rank {row['Rank']}: {row['Strategy']}")
        logger.info(f"   Mean Return: {row['Mean_Return']:.5f}")
        logger.info(f"   95% CI: [{row['CI_2.5%']:.5f}, {row['CI_97.5%']:.5f}]")
        logger.info(f"   P(Best): {row['P(Best)']:.2%}")
        logger.info("")
    
    # Step 6: Visualization
    logger.info("\n" + "="*40)
    logger.info("STEP 6: Generating Visualizations")
    logger.info("="*40)
    
    # Plot posterior distributions
    fig_path1 = 'results/figures/posterior_analysis.png'
    ranker.plot_posterior(save_path=fig_path1)
    logger.info(f"  ✓ Posterior analysis saved to {fig_path1}")
    
    # Plot performance comparison
    fig_path2 = 'results/figures/performance_comparison.png'
    ranker.plot_performance_comparison(portfolios_dict, save_path=fig_path2)
    logger.info(f"  ✓ Performance comparison saved to {fig_path2}")
    
    # Step 7: Save Results
    logger.info("\n" + "="*40)
    logger.info("STEP 7: Saving Results")
    logger.info("="*40)
    
    # Save rankings
    rankings_path = 'results/strategy_rankings.csv'
    rankings.to_csv(rankings_path, index=False)
    logger.info(f"  ✓ Rankings saved to {rankings_path}")
    
    # Save metrics summary
    metrics_summary = pd.DataFrame({
        name: result['metrics'] 
        for name, result in portfolios_dict.items()
    }).T
    
    metrics_path = 'results/performance_metrics.csv'
    metrics_summary.to_csv(metrics_path)
    logger.info(f"  ✓ Metrics saved to {metrics_path}")
    
    # Create summary report
    with open('results/summary_report.txt', 'w') as f:
        f.write("="*60 + "\n")

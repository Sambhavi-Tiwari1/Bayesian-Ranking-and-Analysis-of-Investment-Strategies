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
import yaml
import argparse
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

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_complete_pipeline(config_path='config.yaml'):
    """
    Run the complete Bayesian strategy ranking pipeline
    """
    # Load configuration
    config = load_config(config_path)
    
    logger.info("="*60)
    logger.info("BAYESIAN RANKING OF INVESTMENT STRATEGIES")
    logger.info(f"Period: {config['data']['start_date']} to {config['data']['end_date']}")
    logger.info("="*60)
    
    # Create results directory
    create_results_directory()
    
    # Step 1: Data Collection
    logger.info("\n" + "="*40)
    logger.info("STEP 1: Collecting OHLCV Data")
    logger.info("="*40)
    
    collector = DataCollector()
    
    # Collect data for multiple assets
    symbols = config['data']['symbols']
    end_date = config['data']['end_date']
    start_date = config['data']['start_date']
    
    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching {symbol} data...")
        data = collector.fetch_ohlcv(symbol, start_date, end_date)
        all_data[symbol] = data
        logger.info(f"  ✓ {symbol}: {len(data)} days of data")
    
    # Use the primary symbol for strategy testing
    primary_symbol = config['data']['primary_symbol']
    data = all_data[primary_symbol]
    logger.info(f"\nUsing {primary_symbol} data for strategy testing")
    logger.info(f"  Period: {data.index[0]} to {data.index[-1]}")
    logger.info(f"  Days: {len(data)}")
    
    # Step 2: Define Strategies
    logger.info("\n" + "="*40)
    logger.info("STEP 2: Defining Trading Strategies")
    logger.info("="*40)
    
    strategies = []
    
    # SMA Crossover strategies from config
    for sma_config in config['strategies']['sma']:
        strategies.append(
            SMAStrategy(
                short_window=sma_config['short_window'],
                long_window=sma_config['long_window']
            )
        )
    
    # RSI strategies from config
    for rsi_config in config['strategies']['rsi']:
        strategies.append(
            RSIStrategy(
                period=rsi_config['period'],
                overbought=rsi_config['overbought'],
                oversold=rsi_config['oversold']
            )
        )
    
    # MACO strategy if enabled
    if config['strategies']['ma_crossover']['enabled']:
        strategies.append(MACrossoverStrategy())
    
    logger.info(f"Total strategies: {len(strategies)}")
    for s in strategies:
        logger.info(f"  - {s.name}")
    
    # Step 3: Backtesting
    logger.info("\n" + "="*40)
    logger.info("STEP 3: Running Backtests")
    logger.info("="*40)
    
    engine = BacktestEngine(
        initial_capital=config['backtest']['initial_capital'],
        commission=config['backtest']['commission']
    )
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
    ranker.fit(
        returns_dict,
        n_samples=config['bayesian']['n_samples'],
        n_chains=config['bayesian']['n_chains']
    )
    
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
        f.write("BAYESIAN STRATEGY RANKING - SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Primary Symbol: {primary_symbol}\n")
        f.write(f"Total Strategies: {len(strategies)}\n\n")
        
        f.write("RANKINGS:\n")
        f.write("-"*40 + "\n")
        for idx, row in rankings.iterrows():
            f.write(f"Rank {row['Rank']}: {row['Strategy']}\n")
            f.write(f"  Mean Return: {row['Mean_Return']:.5f}\n")
            f.write(f"  P(Best): {row['P(Best)']:.2%}\n\n")
        
        f.write("BEST PERFORMING METRICS:\n")
        f.write("-"*40 + "\n")
        best_strategy = rankings.iloc[0]['Strategy']
        best_metrics = portfolios_dict[best_strategy]['metrics']
        f.write(f"Strategy: {best_strategy}\n")
        f.write(f"  Total Return: {best_metrics['total_return']:.2%}\n")
        f.write(f"  Annualized Return: {best_metrics['annualized_return']:.2%}\n")
        f.write(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.3f}\n")
        f.write(f"  Max Drawdown: {best_metrics['max_drawdown']:.2%}\n")
        f.write(f"  Win Rate: {best_metrics['win_rate']:.2%}\n")
    
    logger.info(f"  ✓ Summary report saved to results/summary_report.txt")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info(f"Results saved in 'results/' directory")
    logger.info("="*60)
    
    return rankings, portfolios_dict

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Bayesian Ranking of Investment Strategies'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run with reduced MCMC samples for faster execution'
    )
    
    args = parser.parse_args()
    
    # If quick mode, override config settings
    if args.quick:
        logger.info("Running in quick mode with reduced samples")
        # We'll handle this in the config loading
    
    run_complete_pipeline(args.config)

if __name__ == "__main__":
    main()

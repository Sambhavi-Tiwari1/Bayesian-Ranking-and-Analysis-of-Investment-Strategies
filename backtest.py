
"""
Backtesting engine for strategy evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Backtesting engine with comprehensive metrics"""
    
    def __init__(self, initial_capital: float = 100000, 
                 commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
    
    def run_backtest(self, data: pd.DataFrame, 
                     strategies: List) -> Dict:
        """
        Run backtest for multiple strategies
        
        Returns:
            Dict with portfolio, metrics for each strategy
        """
        results = {}
        
        for strategy in strategies:
            logger.info(f"Backtesting {strategy.name}...")
            
            try:
                # Generate signals
                signals = strategy.generate_signals(data.copy())
                
                # Simulate trading
                portfolio = self._simulate_trading(data, signals, strategy)
                
                # Calculate metrics
                metrics = self._calculate_metrics(portfolio, data)
                strategy.metrics = metrics
                
                results[strategy.name] = {
                    'portfolio': portfolio,
                    'signals': signals,
                    'metrics': metrics
                }
                
            except Exception as e:
                logger.error(f"Error in {strategy.name}: {e}")
                continue
        
        self.results = results
        return results
    
    def _simulate_trading(self, data: pd.DataFrame, 
                         signals: pd.Series, 
                         strategy) -> pd.DataFrame:
        """Simulate trading with positions"""
        
        portfolio = pd.DataFrame(index=data.index)
        portfolio['price'] = data['close']
        portfolio['signal'] = signals.fillna(0)
        portfolio['position'] = 0
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0
        portfolio['total'] = self.initial_capital
        portfolio['returns'] = 0
        
        position = 0
        cash = self.initial_capital
        
        for i in range(len(portfolio)):
            current_price = portfolio.iloc[i]['price']
            signal = portfolio.iloc[i]['signal']
            
            # Execute trades based on signal changes
            if i > 0:
                prev_signal = signals.iloc[i-1] if i > 0 else 0
                
                if signal != prev_signal:
                    if signal == 1:  # Buy
                        # Invest all cash
                        shares = (cash * 0.95) / current_price
                        cost = shares * current_price * (1 + self.commission)
                        cash -= cost
                        position = shares
                    elif signal == -1:  # Sell
                        # Sell all holdings
                        revenue = position * current_price * (1 - self.commission)
                        cash += revenue
                        position = 0
            
            # Update portfolio
            portfolio.iloc[i, portfolio.columns.get_loc('position')] = position
            portfolio.iloc[i, portfolio.columns.get_loc('cash')] = cash
            portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = position * current_price
            portfolio.iloc[i, portfolio.columns.get_loc('total')] = cash + position * current_price
            
            if i > 0:
                portfolio.iloc[i, portfolio.columns.get_loc('returns')] = (
                    portfolio.iloc[i]['total'] / portfolio.iloc[i-1]['total'] - 1
                )
        
        return portfolio
    
    def _calculate_metrics(self, portfolio: pd.DataFrame, 
                          data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        returns = portfolio['returns'].dropna()
        
        if len(returns) == 0:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        # Total return
        total_return = (portfolio['total'].iloc[-1] / self.initial_capital - 1)
        
        # Annualized return (assuming 252 trading days)
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Maximum consecutive wins/losses
        returns_binary = returns.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for val in returns_binary:
            if val == 1:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif val == -1:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'final_capital': portfolio['total'].iloc[-1]
        }

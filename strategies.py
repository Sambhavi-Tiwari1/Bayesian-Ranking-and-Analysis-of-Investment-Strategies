"""
Trading strategies with SMA and RSI indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseStrategy:
    """Base class for all strategies"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.signals = None
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (to be overridden)"""
        raise NotImplementedError
    
    def calculate_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """Calculate position size"""
        return 1.0 if signal == 1 else 0.0 if signal == -1 else 0.5

class SMAStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(
            name=f"SMA_{short_window}_{long_window}",
            params={'short_window': short_window, 'long_window': long_window}
        )
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy/sell signals based on SMA crossover"""
        # Calculate moving averages
        data = data.copy()
        data['sma_short'] = data['close'].rolling(window=self.short_window).mean()
        data['sma_long'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy when short SMA crosses above long SMA
        signals[(data['sma_short'] > data['sma_long']) & 
                (data['sma_short'].shift(1) <= data['sma_long'].shift(1))] = 1
        
        # Sell when short SMA crosses below long SMA
        signals[(data['sma_short'] < data['sma_long']) & 
                (data['sma_short'].shift(1) >= data['sma_long'].shift(1))] = -1
        
        # Hold previous position (optional)
        signals = signals.replace(0, method='ffill').fillna(0)
        
        self.signals = signals
        return signals

class RSIStrategy(BaseStrategy):
    """Relative Strength Index strategy"""
    
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__(
            name=f"RSI_{period}_{overbought}_{oversold}",
            params={'period': period, 'overbought': overbought, 'oversold': oversold}
        )
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy/sell signals based on RSI"""
        data = data.copy()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy when RSI crosses above oversold
        signals[(data['rsi'] > self.oversold) & 
                (data['rsi'].shift(1) <= self.oversold)] = 1
        
        # Sell when RSI crosses below overbought
        signals[(data['rsi'] < self.overbought) & 
                (data['rsi'].shift(1) >= self.overbought)] = -1
        
        # Hold previous position
        signals = signals.replace(0, method='ffill').fillna(0)
        
        self.signals = signals
        return signals

class MACrossoverStrategy(BaseStrategy):
    """Multiple moving average crossover (extra strategy)"""
    
    def __init__(self):
        super().__init__(name="MA_Crossover", params={})
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()
        data['ma10'] = data['close'].rolling(10).mean()
        data['ma30'] = data['close'].rolling(30).mean()
        data['ma60'] = data['close'].rolling(60).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Buy when MA10 > MA30 > MA60 (uptrend)
        signals[(data['ma10'] > data['ma30']) & 
                (data['ma30'] > data['ma60'])] = 1
        
        # Sell when MA10 < MA30 < MA60 (downtrend)
        signals[(data['ma10'] < data['ma30']) & 
                (data['ma30'] < data['ma60'])] = -1
        
        signals = signals.replace(0, method='ffill').fillna(0)
        return signals

# Bayesian Ranking and Analysis of Investment Strategies

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📊 Overview

A Bayesian statistical framework for evaluating and ranking rule-based trading strategies using probabilistic performance comparison. This project implements common technical indicators (SMA, RSI) and uses Bayesian inference to compare strategy performance robustly.

**Project Period**: May 2025 – July 2025

### Key Features
- 🔬 **Bayesian Performance Comparison**: Probabilistic ranking of strategies with uncertainty quantification
- 📈 **Technical Indicators**: Implementation of SMA, RSI, and other common indicators
- 📊 **Real Market Data**: OHLCV data collection and processing pipeline
- ⚡ **Backtesting Engine**: Comprehensive backtesting with risk-adjusted metrics
- 🎯 **Robust Strategy Selection**: Statistical inference combined with trading signal analysis

## 🏗️ Architecture

```mermaid
graph TD
    A[Data Collection] --> B[Data Processing]
    B --> C[Strategy Execution]
    C --> D[Performance Metrics]
    D --> E[Bayesian Analysis]
    E --> F[Strategy Ranking]
    
    G[OHLCV Data] --> A
    H[SMA/RSI Signals] --> C
    I[Priors] --> E

"""
Bayesian statistical framework for strategy ranking
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BayesianStrategyRanker:
    """
    Bayesian model for ranking and comparing trading strategies
    """
    
    def __init__(self, strategies: List[str]):
        self.strategies = strategies
        self.n_strategies = len(strategies)
        self.trace = None
        self.model = None
        self.rankings = None
        
    def fit(self, returns_dict: Dict[str, np.ndarray], 
            n_samples: int = 2000, 
            n_chains: int = 4) -> None:
        """
        Fit Bayesian model to strategy returns
        
        Args:
            returns_dict: {strategy_name: array_of_returns}
            n_samples: MCMC samples per chain
            n_chains: Number of MCMC chains
        """
        # Align lengths
        min_length = min([len(r) for r in returns_dict.values()])
        returns_aligned = {name: r[:min_length] for name, r in returns_dict.items()}
        
        # Prepare data for PyMC
        returns_stack = np.column_stack([returns_aligned[name] for name in self.strategies])
        n_obs = returns_stack.shape[0]
        
        with pm.Model() as model:
            # Hierarchical prior for strategy means
            mu_prior = pm.Normal('mu_prior', mu=0, sigma=0.1)
            sigma_prior = pm.HalfNormal('sigma_prior', sigma=0.1)
            
            # Strategy-specific means
            mu = pm.Normal('mu', 
                          mu=mu_prior, 
                          sigma=sigma_prior, 
                          shape=self.n_strategies)
            
            # Strategy-specific volatility
            sigma = pm.HalfNormal('sigma', 
                                 sigma=0.1, 
                                 shape=self.n_strategies)
            
            # Observed returns
            returns_obs = pm.Normal('returns_obs',
                                   mu=mu,
                                   sigma=sigma,
                                   observed=returns_stack)
            
            # Ranking
            rank = pm.Deterministic('rank', pm.math.argsort(-mu))
            
            # Probability each strategy is best
            prob_best = pm.Deterministic('prob_best', 
                                        pm.math.eq(rank, 0).astype('float').mean(axis=0))
            
            self.model = model
            
            # Sample
            with model:
                self.trace = pm.sample(
                    draws=n_samples,
                    chains=n_chains,
                    tune=1000,
                    cores=min(4, n_chains),
                    random_seed=42,
                    progressbar=True
                )
    
    def get_ranking(self) -> pd.DataFrame:
        """
        Get strategy ranking with posterior probabilities
        
        Returns:
            DataFrame with ranking summary
        """
        if self.trace is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        # Extract posterior
        mu_samples = self.trace.posterior['mu'].values
        mu_samples = mu_samples.reshape(-1, self.n_strategies)
        
        # Calculate statistics
        means = mu_samples.mean(axis=0)
        stds = mu_samples.std(axis=0)
        lower_95 = np.percentile(mu_samples, 2.5, axis=0)
        upper_95 = np.percentile(mu_samples, 97.5, axis=0)
        lower_50 = np.percentile(mu_samples, 25, axis=0)
        upper_50 = np.percentile(mu_samples, 75, axis=0)
        
        # Probability of being best
        rankings = np.argsort(-mu_samples, axis=1)
        prob_best = (rankings == 0).mean(axis=0)
        
        # Probability of being in top 3
        prob_top3 = (rankings < 3).mean(axis=0)
        
        # Create results
        results = pd.DataFrame({
            'Strategy': self.strategies,
            'Mean_Return': means,
            'Std_Return': stds,
            'CI_2.5%': lower_95,
            'CI_97.5%': upper_95,
            'CI_25%': lower_50,
            'CI_75%': upper_50,
            'P(Best)': prob_best,
            'P(Top3)': prob_top3
        })
        
        # Sort by mean return
        results = results.sort_values('Mean_Return', ascending=False).reset_index(drop=True)
        results['Rank'] = results.index + 1
        
        self.rankings = results
        return results
    
    def plot_posterior(self, save_path: str = None) -> None:
        """Plot posterior distributions of strategy returns"""
        if self.trace is None:
            raise ValueError("Model not fitted yet!")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 1. Posterior distributions
        ax = axes[0]
        mu_samples = self.trace.posterior['mu'].values.reshape(-1, self.n_strategies)
        
        for i, name in enumerate(self.strategies):
            sns.kdeplot(mu_samples[:, i], label=name, ax=ax)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Posterior Distributions of Strategy Returns')
        ax.set_xlabel('Expected Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Ranking probabilities
        ax = axes[1]
        rankings = self.get_ranking()
        colors = ['gold' if i == 0 else 'silver' if i == 1 else 'bronze' if i == 2 else 'lightgray' 
                  for i in range(len(rankings))]
        ax.barh(rankings['Strategy'], rankings['P(Best)'], color=colors)
        ax.set_title('Probability of Being Best Strategy')
        ax.set_xlabel('P(Best)')
        ax.grid(True, alpha=0.3)
        
        # 3. Forest plot
        ax = axes[2]
        rankings_sorted = rankings.sort_values('Mean_Return')
        
        y_pos = range(len(rankings_sorted))
        ax.errorbar(rankings_sorted['Mean_Return'], y_pos,
                   xerr=[rankings_sorted['Mean_Return'] - rankings_sorted['CI_2.5%'],
                         rankings_sorted['CI_97.5%'] - rankings_sorted['Mean_Return']],
                   fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=10)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rankings_sorted['Strategy'])
        ax.set_xlabel('Expected Return')
        ax.set_title('Strategy Ranking with 95% Credible Intervals')
        ax.grid(True, alpha=0.3)
        
        # 4. Pairwise comparison
        ax = axes[3]
        # Calculate probability each strategy beats the others
        mu_samples = self.trace.posterior['mu'].values.reshape(-1, self.n_strategies)
        comparison_matrix = np.zeros((self.n_strategies, self.n_strategies))
        
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                if i != j:
                    comparison_matrix[i, j] = (mu_samples[:, i] > mu_samples[:, j]).mean()
        
        sns.heatmap(comparison_matrix, 
                   xticklabels=self.strategies,
                   yticklabels=self.strategies,
                   annot=True, fmt='.2f', cmap='RdYlGn_r',
                   ax=ax)
        ax.set_title('P(Strategy beats competitor)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, portfolios: Dict, 
                                   save_path: str = None) -> None:
        """Plot cumulative performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative returns
        ax = axes[0, 0]
        for name, result in portfolios.items():
            cum_returns = (1 + result['portfolio']['returns']).cumprod()
            ax.plot(cum_returns.index, cum_returns, label=name, linewidth=2)
        
        ax.axhline(1, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Cumulative Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        for name, result in portfolios.items():
            cum_returns = (1 + result['portfolio']['returns']).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max * 100
            ax.fill_between(drawdown.index, 0, drawdown, label=name, alpha=0.5)
        
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Metrics comparison
        ax = axes[1, 0]
        metrics_df = pd.DataFrame({name: result['metrics'] 
                                  for name, result in portfolios.items()}).T
        
        # Normalize metrics for radar chart
        metrics_to_plot = ['sharpe_ratio', 'calmar_ratio', 'win_rate', 'profit_factor']
        metrics_df_norm = metrics_df[metrics_to_plot].copy()
        metrics_df_norm = (metrics_df_norm - metrics_df_norm.min()) / (metrics_df_norm.max() - metrics_df_norm.min())
        
        # Bar chart instead of radar for simplicity
        metrics_df_norm.plot(kind='bar', ax=ax)
        ax.set_title('Normalized Performance Metrics')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Normalized Score')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 4. Strategy ranking summary
        ax = axes[1, 1]
        rankings = self.get_ranking()
        
        # Create summary table
        ax.axis('tight')
        ax.axis('off')
        
        # Format table
        table_data = rankings[['Rank', 'Strategy', 'Mean_Return', 'P(Best)']].round(4)
        table_data['Mean_Return'] = table_data['Mean_Return'] * 100
        table_data['P(Best)'] = table_data['P(Best)'] * 100
        
        table = ax.table(cellText=table_data.values,
                        colLabels=['Rank', 'Strategy', 'Mean Return (%)', 'P(Best) (%)'],
                        cellLoc='center',
                        loc='center',
                        colColours=['#4472C4']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color coding for best strategies
        for i in range(len(table_data)):
            if table_data.iloc[i]['Rank'] == 1:
                table[(i+1, 0)].set_facecolor('#FFD700')  # Gold
            elif table_data.iloc[i]['Rank'] == 2:
                table[(i+1, 0)].set_facecolor('#C0C0C0')  # Silver
            elif table_data.iloc[i]['Rank'] == 3:
                table[(i+1, 0)].set_facecolor('#CD7F32')  # Bronze
        
        ax.set_title('Strategy Ranking Summary', fontsize=12, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

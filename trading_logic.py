import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from typing import List, Dict, Tuple

class RVStrategy:
    def __init__(self, aum = 10e6, max_trades = 10, avg_trades = 5):
        # Portfolio parameters
        self.aum = aum  # Assets Under Management
        self.max_trades = max_trades  # Maximum number of trades
        self.avg_trades = avg_trades  # Average number of trades per day
        self.initial_aum = aum  # Initial AUM for performance calculations

        # Trading parameters
        self.entry_z = 1.5
        self.exit_z = 0.5
        self.tier1_sl_z = 2.5 # Tier 1 stop loss z-score
        self.tier2_sl_z = 3.5 # Tier 2 stop loss z-score
        self.tier2_entry_z = 2.0 # Higher entry z-score for tier 2
        self.max_holding_days = 32 # Given constraint
        self.risk_per_trade = 0.01  # 1% of AUM

        # State tracking variables
        self.open_trades: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.portfolio_nav: List[Dict] = []
        self.daily_positions: List[Dict] = []

        # Rolling window sizes for regression and z-score computation
        self.ROLL_REG = 60 # Days for regression (hedge ratio estimation)
        self.ROLL_Z = 20 # Days for z-score stats (mean and std of spread)

    @property
    def capital_per_trade(self):
        return self.aum / self.avg_trades

    def fit_spread(self, x, y):
        """
        Fits a linear regression x = alpha + beta*y to estimate hedge ratio.
        Calculates spread and rolling z-score.
        """
        X = add_constant(y)
        model = OLS(x, X).fit()
        spread = x - (model.params['const'] + model.params[y.name] * y)

        mean = spread.rolling(self.ROLL_Z).mean()
        std = spread.rolling(self.ROLL_Z).std()
        zscore = (spread - mean) / std

        return spread, zscore, model.params['const'], model.params[y.name] # ...alpha, beta
    
    def update_trades(self, current_date, df_prices):
        """
        Updates all open trades:
        - Recomputes spread and z-score
        - Checks for exit (take-profit, stop-loss, or max holding)
        - Updates NAV
        """
        px = df_prices.loc[current_date]
        new_open_trades = []
        daily_pnl = 0

        for trade in self.open_trades:
            a1, a2 = trade['pair']
            current_spread = px[a1] - (trade['alpha'] + trade['beta'] * px[a2])
            current_z = (current_spread - trade['spread_mean']) / trade['spread_std']

            # PnL calculation
            pnl = (current_spread - trade['entry_spread']) * trade['direction'] * trade['size']
            daily_pnl += pnl

            # Determine if exit conditions are met
            days_held = (current_date - trade['entry_date']).days
            exit_reason = None

            if abs(current_z) < self.exit_z:
                exit_reason = 'take_profit'
            elif abs(current_z) > trade['sl_z']:
                exit_reason = 'stop_loss'
            elif days_held >= self.max_holding_days:
                exit_reason = 'time_exit'

            if exit_reason:
                closed_trade = trade.copy()
                closed_trade.update({
                    'exit_date': current_date,
                    'exit_price1': px[a1],
                    'exit_price2': px[a2],
                    'exit_spread': current_spread,
                    'exit_z': current_z,
                    'exit_pnl': pnl,
                    'exit_reason': exit_reason,
                    'holding_days': days_held
                })
                self.closed_trades.append(closed_trade)
            else:
                trade['current_pnl'] = pnl
                trade['current_z'] = current_z
                new_open_trades.append(trade)

        self.open_trades = new_open_trades
        self.aum += daily_pnl  # Update AUM with daily PnL

        # Record daily portfolio snapshot
        self.portfolio_nav.append({
            'date': current_date,
            'nav': self.aum,
            'daily_pnl': daily_pnl,
            'num_trades': len(self.open_trades)
        })

    def enter_trades(self, current_date, df_prices, tier1, tier2):
        """
        Attempts to open new trades:
        - Applies separate Z-score thresholds for Tier 1 and Tier 2
        - Ensures risk sizing and no duplicate trades
        """

        if len(self.open_trades) >= self.max_trades:
            return False

        px_window = df_prices.loc[:current_date].iloc[-(self.ROLL_REG + self.ROLL_Z):]
        px_today = df_prices.loc[current_date]

        def try_enter(pair, z_entry, sl_z, tier):
            a1, a2 = pair
            if a1 not in px_today or a2 not in px_today:
                return False
            
            if any(t['pair'] == pair for t in self.open_trades):
                return False
            
            x, y = px_window[a1], px_window[a2]
            if x.isna().any() or y.isna().any():
                return False
            
            # Fit the spread and get the latest z-score
            spread, zscore, alpha, beta = self.fit_spread(x, y)
            current_z = zscore.iloc[-1]

            direction = 0
            if current_z > z_entry:
                direction = -1  # Short
            elif current_z < -z_entry:
                direction = 1   # Long
            else:
                return False

            spread_mean = spread.rolling(self.ROLL_Z).mean().iloc[-1]
            spread_std = spread.rolling(self.ROLL_Z).std().iloc[-1]
            spread_at_entry = spread.iloc[-1]
            spread_at_stop = spread_mean + sl_z * spread_std * (-direction)
            distance_to_sl = abs(spread_at_stop - spread_at_entry)

            if distance_to_sl == 0 or np.isnan(distance_to_sl):
                return False
            
            # Calculate position size based on target risk per trade
            size = min(self.capital_per_trade / max(distance_to_sl, 1e-6), self.aum * 0.1)

            # Create trade record
            trade = {
                'entry_date': current_date,
                'pair': pair,
                'alpha': alpha,
                'beta': beta,
                'spread_mean': spread.rolling(self.ROLL_Z).mean().iloc[-1],
                'spread_std': spread_std,
                'entry_spread': spread.iloc[-1],
                'entry_z': current_z,
                'direction': direction,
                'size': size,
                'sl_z': sl_z,
                'tier': tier,
                'entry_price1': px_today[a1],
                'entry_price2': px_today[a2]
            }

            self.open_trades.append(trade)
            return True
        
        slots_available = self.max_trades - len(self.open_trades)
        target_new_trades = min(slots_available, self.avg_trades - len(self.open_trades))

        # Tier 1 entries
        for _, row in tier1.iterrows():
            if target_new_trades <= 0:
                break
            if len(self.open_trades) >= self.max_trades:
                break
            success = try_enter(row['Pair'], self.entry_z, self.tier1_sl_z, tier='Tier1')
            if success:
                target_new_trades -= 1

        # Tier 2 entries
        for _, row in tier2.iterrows():
            if target_new_trades <= 0:
                break
            if len(self.open_trades) >= self.max_trades:
                break
            sucess = try_enter(row['Pair'], self.tier2_entry_z, self.tier2_sl_z, tier='Tier2')
            if success:
                target_new_trades -= 1

    def run_backtest(self, df_prices: pd.DataFrame, tier1: pd.DataFrame, tier2: pd.DataFrame) -> None:
        """
        Runs the complete backtest over the price data.
        
        Parameters:
            df_prices: DataFrame with daily prices for all assets
            tier1: DataFrame of Tier 1 pairs (strong cointegration, fast mean reversion)
            tier2: DataFrame of Tier 2 pairs (very strong cointegration, any mean reversion)
        """
        # Clean data and ensure proper indexing
        df_prices.index = pd.to_datetime(df_prices.index)
        
        # Initialize state
        self.open_trades = []
        self.closed_trades = []
        self.portfolio_nav = []
        self.aum = self.initial_aum
        
        # Main backtest loop
        start_idx = self.ROLL_REG + self.ROLL_Z
        dates = df_prices.index[start_idx:]
        
        for current_date in dates:
            # Update existing trades first
            self.update_trades(current_date, df_prices)
            
            # Then look for new entries
            if len(self.open_trades) < self.max_trades:
                self.enter_trades(current_date, df_prices, tier1, tier2)

    def calculate_performance(self) -> Dict:
        """
        Calculates performance metrics from the backtest results.
        
        Returns:
            Dictionary containing all performance metrics
        """
        if not self.portfolio_nav:
            raise ValueError("No backtest results available. Run backtest first.")
            
        # Create NAV series
        nav_df = pd.DataFrame(self.portfolio_nav)
        nav_df['date'] = pd.to_datetime(nav_df['date'])
        nav_df = nav_df.set_index('date')
        nav_series = nav_df['nav']
        
        # Calculate returns
        returns = nav_series.pct_change().dropna()
        daily_returns = returns
        annualized_return = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (252/len(nav_series)) - 1
        
        # Calculate metrics
        max_drawdown = (nav_series / nav_series.cummax() - 1).min()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Trade analysis
        closed_trades = pd.DataFrame(self.closed_trades)
        win_rate = len(closed_trades[closed_trades['exit_pnl'] > 0]) / len(closed_trades) if len(closed_trades) > 0 else 0
        
        return {
            'Initial_AUM': self.initial_aum,
            'Final_AUM': nav_series.iloc[-1],
            'Cumulative_PnL': nav_series.iloc[-1] - nav_series.iloc[0],
            'Cumulative_Return': (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100,
            'Annualized_Return': annualized_return * 100,
            'Annualized_Volatility': volatility * 100,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_drawdown * 100,
            'Calmar_Ratio': calmar,
            'Total_Trades': len(closed_trades),
            'Win_Rate': win_rate * 100,
            'Avg_Trade_Duration': closed_trades['holding_days'].mean() if len(closed_trades) > 0 else 0,
            'Avg_Concurrent_Trades': nav_df['num_trades'].mean(),
            'Trades_by_Exit_Reason': closed_trades['exit_reason'].value_counts().to_dict(),
            'Trades_by_Tier': closed_trades['tier'].value_counts().to_dict() if len(closed_trades) > 0 else {}
        }

    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Returns backtest results in convenient formats.
        
        Returns:
            nav_df: DataFrame with daily NAV and positions
            trades_df: DataFrame of all closed trades
            metrics: Dictionary of performance metrics
        """
        nav_df = pd.DataFrame(self.portfolio_nav)
        trades_df = pd.DataFrame(self.closed_trades)
        metrics = self.calculate_performance()
        return nav_df, trades_df, metrics

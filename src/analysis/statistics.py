"""
Statistical Analysis Module
Advanced statistical computations for FLNG operations
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class OperationalStatistics:
   """Calculate operational statistics and trends"""
   
   def __init__(self):
       self.scaler = StandardScaler()
   
   def calculate_summary_statistics(self, data: pd.DataFrame) -> Dict:
       """Calculate comprehensive summary statistics"""
       
       if data.empty:
           return {}
       
       numeric_columns = data.select_dtypes(include=[np.number]).columns
       
       summary = {
           'total_records': len(data),
           'time_range': {
               'start': data['timestamp'].min(),
               'end': data['timestamp'].max(),
               'duration_hours': (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600
           }
       }
       
       # Calculate statistics for each numeric column
       for col in numeric_columns:
           if col != 'timestamp':
               summary[col] = {
                   'mean': data[col].mean(),
                   'std': data[col].std(),
                   'min': data[col].min(),
                   'max': data[col].max(),
                   'p25': data[col].quantile(0.25),
                   'median': data[col].median(),
                   'p75': data[col].quantile(0.75),
                   'skewness': data[col].skew(),
                   'kurtosis': data[col].kurtosis()
               }
       
       return summary
   
   def calculate_correlations(self, data: pd.DataFrame, 
                            target_column: str = 'safety_score') -> pd.DataFrame:
       """Calculate correlations with target variable"""
       
       numeric_data = data.select_dtypes(include=[np.number])
       
       if target_column not in numeric_data.columns:
           return pd.DataFrame()
       
       correlations = numeric_data.corr()[target_column].sort_values(ascending=False)
       
       # Calculate p-values for correlations
       p_values = {}
       for col in numeric_data.columns:
           if col != target_column:
               corr, p_val = stats.pearsonr(numeric_data[col].dropna(), 
                                           numeric_data[target_column].dropna())
               p_values[col] = p_val
       
       correlation_df = pd.DataFrame({
           'correlation': correlations,
           'p_value': pd.Series(p_values),
           'significant': pd.Series(p_values) < 0.05
       })
       
       return correlation_df
   
   def perform_trend_analysis(self, data: pd.DataFrame, 
                            variable: str,
                            period: str = 'D') -> Dict:
       """Perform trend analysis on time series data"""
       
       if variable not in data.columns:
           return {}
       
       # Resample data by period
       time_series = data.set_index('timestamp')[variable].resample(period).mean()
       
       # Calculate trend using linear regression
       x = np.arange(len(time_series))
       y = time_series.values
       
       # Remove NaN values
       mask = ~np.isnan(y)
       x_clean = x[mask]
       y_clean = y[mask]
       
       if len(x_clean) < 2:
           return {}
       
       slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
       
       trend_analysis = {
           'slope': slope,
           'intercept': intercept,
           'r_squared': r_value ** 2,
           'p_value': p_value,
           'trend_direction': 'increasing' if slope > 0 else 'decreasing',
           'trend_significant': p_value < 0.05,
           'forecast_next_period': slope * len(time_series) + intercept
       }
       
       return trend_analysis
   
   def calculate_operational_efficiency(self, data: pd.DataFrame) -> Dict:
       """Calculate operational efficiency metrics"""
       
       if 'safety_score' not in data.columns:
           return {}
       
       # Define operational thresholds
       safe_threshold = 80
       operational_threshold = 60
       
       total_hours = len(data)
       safe_hours = len(data[data['safety_score'] >= safe_threshold])
       operational_hours = len(data[data['safety_score'] >= operational_threshold])
       
       efficiency_metrics = {
           'availability_percentage': (operational_hours / total_hours) * 100,
           'safety_percentage': (safe_hours / total_hours) * 100,
           'downtime_percentage': ((total_hours - operational_hours) / total_hours) * 100,
           'average_safety_score': data['safety_score'].mean(),
           'safety_score_stability': data['safety_score'].std(),
           'operational_windows': self._count_operational_windows(data, operational_threshold)
       }
       
       # Calculate efficiency by time of day
       data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
       hourly_efficiency = data.groupby('hour')['safety_score'].apply(
           lambda x: (x >= operational_threshold).sum() / len(x) * 100
       )
       
       efficiency_metrics['hourly_efficiency'] = hourly_efficiency.to_dict()
       efficiency_metrics['best_operational_hours'] = hourly_efficiency.nlargest(3).index.tolist()
       efficiency_metrics['worst_operational_hours'] = hourly_efficiency.nsmallest(3).index.tolist()
       
       return efficiency_metrics
   
   def _count_operational_windows(self, data: pd.DataFrame, threshold: float) -> int:
       """Count number of continuous operational windows"""
       
       is_operational = data['safety_score'] >= threshold
       windows = 0
       in_window = False
       
       for operational in is_operational:
           if operational and not in_window:
               windows += 1
               in_window = True
           elif not operational:
               in_window = False
       
       return windows
   
   def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
       """Calculate risk-related metrics"""
       
       risk_metrics = {}
       
       # Value at Risk (VaR) for safety score
       if 'safety_score' in data.columns:
           risk_metrics['var_95'] = np.percentile(data['safety_score'], 5)
           risk_metrics['var_99'] = np.percentile(data['safety_score'], 1)
           
           # Conditional Value at Risk (CVaR)
           var_95_mask = data['safety_score'] <= risk_metrics['var_95']
           risk_metrics['cvar_95'] = data.loc[var_95_mask, 'safety_score'].mean()
       
       # Risk factor frequency
       if 'limiting_factors' in data.columns:
           all_factors = []
           for factors in data['limiting_factors'].dropna():
               if isinstance(factors, list):
                   all_factors.extend(factors)
           
           if all_factors:
               factor_counts = pd.Series(all_factors).value_counts()
               risk_metrics['top_risk_factors'] = factor_counts.head(5).to_dict()
       
       return risk_metrics
   
   def generate_report(self, data: pd.DataFrame) -> str:
       """Generate statistical report"""
       
       report = []
       report.append("FLNG Operations Statistical Report")
       report.append("=" * 50)
       
       # Summary statistics
       summary = self.calculate_summary_statistics(data)
       report.append(f"\nData Period: {summary['time_range']['start']} to {summary['time_range']['end']}")
       report.append(f"Total Records: {summary['total_records']}")
       report.append(f"Duration: {summary['time_range']['duration_hours']:.1f} hours")
       
       # Efficiency metrics
       efficiency = self.calculate_operational_efficiency(data)
       report.append(f"\nOperational Efficiency:")
       report.append(f"  Availability: {efficiency['availability_percentage']:.1f}%")
       report.append(f"  Safety: {efficiency['safety_percentage']:.1f}%")
       report.append(f"  Downtime: {efficiency['downtime_percentage']:.1f}%")
       report.append(f"  Operational Windows: {efficiency['operational_windows']}")
       
       # Risk metrics
       risk = self.calculate_risk_metrics(data)
       if 'var_95' in risk:
           report.append(f"\nRisk Metrics:")
           report.append(f"  VaR (95%): {risk['var_95']:.1f}")
           report.append(f"  CVaR (95%): {risk['cvar_95']:.1f}")
       
       return '\n'.join(report)

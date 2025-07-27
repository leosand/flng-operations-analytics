"""
Weather Window Analysis Module
Analyzes marine weather data to determine safe operational windows for FLNG operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
import yaml
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OperationalWindow:
   """Represents a safe operational window"""
   start: datetime
   end: datetime
   duration_hours: float
   average_score: float
   min_score: float
   confidence: float
   limiting_factors: List[str] = field(default_factory=list)
   location: str = ""
   
   @property
   def is_viable(self) -> bool:
       """Check if window is viable for operations (min 4 hours)"""
       return self.duration_hours >= 4.0 and self.min_score >= 60.0

@dataclass
class OperationalStatus:
   """Current operational status"""
   timestamp: datetime
   location: str
   is_safe: bool
   score: float
   status: str  # SAFE, CAUTION, RESTRICTED, SUSPENDED
   wind_speed: float
   wave_height: float
   wave_period: float
   visibility: float
   current_speed: float
   limiting_factors: List[str] = field(default_factory=list)
   recommendations: List[str] = field(default_factory=list)

class SafetyAnalyzer:
   """Analyzes safety conditions based on OCIMF guidelines"""
   
   def __init__(self, thresholds_path: str = "config/safety_thresholds.yml"):
       self.thresholds = self._load_thresholds(thresholds_path)
       self.weight_factors = {
           'wind': 0.25,
           'wave': 0.35,
           'visibility': 0.15,
           'current': 0.15,
           'wave_period': 0.10
       }
   
   def _load_thresholds(self, path: str) -> Dict:
       """Load safety thresholds from YAML file"""
       default_thresholds = {
           'wind_speed': {
               'normal': 15.0,
               'caution': 20.0,
               'critical': 25.0
           },
           'wave_height': {
               'normal': 2.0,
               'caution': 3.0,
               'critical': 4.0
           },
           'wave_period': {
               'min': 6.0,
               'max': 15.0
           },
           'visibility': {
               'min': 1000.0
           },
           'current_speed': {
               'max': 2.0
           },
           'combined_sea_state': {
               'max_wind_wave_ratio': 0.7  # Wind waves should not exceed 70% of total
           }
       }
       
       try:
           with open(path, 'r') as f:
               loaded = yaml.safe_load(f)
               # Merge with defaults
               for key in default_thresholds:
                   if key in loaded:
                       default_thresholds[key].update(loaded[key])
               return default_thresholds
       except:
           logger.warning(f"Could not load thresholds from {path}, using defaults")
           return default_thresholds
   
   def calculate_safety_score(self, weather_data: pd.Series) -> Tuple[float, List[str]]:
       """
       Calculate safety score (0-100) based on weather conditions
       Returns: (score, limiting_factors)
       """
       score = 100.0
       limiting_factors = []
       
       # Wind speed analysis
       wind_score = self._analyze_wind(weather_data.get('wind_speed_10m', 0))
       if wind_score < 100:
           limiting_factors.append(f"Wind: {weather_data.get('wind_speed_10m', 0):.1f} m/s")
       score = score * (wind_score / 100) * self.weight_factors['wind'] + \
               score * (1 - self.weight_factors['wind'])
       
       # Wave height analysis
       wave_score = self._analyze_waves(
           weather_data.get('wave_height_significant', 0),
           weather_data.get('wind_wave_height', 0),
           weather_data.get('swell_height_primary', 0)
       )
       if wave_score < 100:
           limiting_factors.append(f"Waves: {weather_data.get('wave_height_significant', 0):.1f} m")
       score = score * (wave_score / 100) * self.weight_factors['wave'] + \
               score * (1 - self.weight_factors['wave'])
       
       # Wave period analysis
       period_score = self._analyze_wave_period(weather_data.get('wave_period_mean', 10))
       if period_score < 100:
        # Suite de la commande PowerShell...
            limiting_factors.append(f"Wave period: {weather_data.get('wave_period_mean', 10):.1f} s")
       score = score * (period_score / 100) * self.weight_factors['wave_period'] + \
               score * (1 - self.weight_factors['wave_period'])
       
       # Visibility analysis
       vis_score = self._analyze_visibility(weather_data.get('visibility', 10000))
       if vis_score < 100:
           limiting_factors.append(f"Visibility: {weather_data.get('visibility', 10000):.0f} m")
       score = score * (vis_score / 100) * self.weight_factors['visibility'] + \
               score * (1 - self.weight_factors['visibility'])
       
       # Current speed analysis
       current_score = self._analyze_current(weather_data.get('current_speed_surface', 0))
       if current_score < 100:
           limiting_factors.append(f"Current: {weather_data.get('current_speed_surface', 0):.1f} m/s")
       score = score * (current_score / 100) * self.weight_factors['current'] + \
               score * (1 - self.weight_factors['current'])
       
       # Additional safety checks
       # Check for cross seas (dangerous when swell and wind waves from different directions)
       if self._check_cross_seas(weather_data):
           score *= 0.8
           limiting_factors.append("Cross seas detected")
       
       # Check for rapid weather deterioration
       if hasattr(weather_data, 'pressure_msl') and weather_data.get('pressure_msl', 1013) < 1000:
           score *= 0.9
           limiting_factors.append("Low pressure system")
       
       return max(0, min(100, score)), limiting_factors
   
   def _analyze_wind(self, wind_speed: float) -> float:
       """Analyze wind conditions"""
       thresholds = self.thresholds['wind_speed']
       
       if wind_speed >= thresholds['critical']:
           return 0
       elif wind_speed >= thresholds['caution']:
           # Linear decrease from 60 to 0
           return 60 * (thresholds['critical'] - wind_speed) / \
                  (thresholds['critical'] - thresholds['caution'])
       elif wind_speed >= thresholds['normal']:
           # Linear decrease from 100 to 60
           return 100 - 40 * (wind_speed - thresholds['normal']) / \
                  (thresholds['caution'] - thresholds['normal'])
       else:
           return 100
   
   def _analyze_waves(self, sig_height: float, wind_wave: float, swell: float) -> float:
       """Analyze wave conditions including sea state composition"""
       thresholds = self.thresholds['wave_height']
       
       # Check significant wave height
       if sig_height >= thresholds['critical']:
           return 0
       elif sig_height >= thresholds['caution']:
           score = 60 * (thresholds['critical'] - sig_height) / \
                  (thresholds['critical'] - thresholds['caution'])
       elif sig_height >= thresholds['normal']:
           score = 100 - 40 * (sig_height - thresholds['normal']) / \
                  (thresholds['caution'] - thresholds['normal'])
       else:
           score = 100
       
       # Check wind wave contribution
       if sig_height > 0:
           wind_wave_ratio = wind_wave / sig_height
           if wind_wave_ratio > self.thresholds['combined_sea_state']['max_wind_wave_ratio']:
               # High wind waves indicate active weather system
               score *= 0.85
       
       return score
   
   def _analyze_wave_period(self, period: float) -> float:
       """Analyze wave period - both too short and too long are problematic"""
       thresholds = self.thresholds['wave_period']
       
       if period < thresholds['min']:
           # Short period waves are dangerous
           return max(0, 100 * (period / thresholds['min']) ** 2)
       elif period > thresholds['max']:
           # Very long period swells can cause resonance
           excess = period - thresholds['max']
           return max(60, 100 - excess * 5)
       else:
           return 100
   
   def _analyze_visibility(self, visibility: float) -> float:
       """Analyze visibility conditions"""
       min_vis = self.thresholds['visibility']['min']
       
       if visibility >= min_vis:
           return 100
       else:
           return max(0, 100 * (visibility / min_vis) ** 0.5)
   
   def _analyze_current(self, current_speed: float) -> float:
       """Analyze current conditions"""
       max_current = self.thresholds['current_speed']['max']
       
       if current_speed <= max_current:
           return 100
       else:
           excess = current_speed - max_current
           return max(0, 100 - excess * 50)
   
   def _check_cross_seas(self, weather_data: pd.Series) -> bool:
       """Check for dangerous cross seas condition"""
       wind_dir = weather_data.get('wind_wave_direction', 0)
       swell_dir = weather_data.get('swell_direction_primary', 0)
       
       # Calculate angle difference
       angle_diff = abs(wind_dir - swell_dir)
       if angle_diff > 180:
           angle_diff = 360 - angle_diff
       
       # Cross seas are dangerous when angle > 45 degrees and both are significant
       return (angle_diff > 45 and 
               weather_data.get('wind_wave_height', 0) > 0.5 and
               weather_data.get('swell_height_primary', 0) > 0.5)
   
   def determine_status(self, score: float) -> str:
       """Determine operational status based on score"""
       if score >= 80:
           return "SAFE"
       elif score >= 60:
           return "CAUTION"
       elif score >= 40:
           return "RESTRICTED"
       else:
           return "SUSPENDED"

class WeatherWindowAnalyzer:
   """Main analyzer for identifying safe operational windows"""
   
   def __init__(self, safety_analyzer: Optional[SafetyAnalyzer] = None):
       self.safety_analyzer = safety_analyzer or SafetyAnalyzer()
       self.min_window_hours = 4  # Minimum viable window duration
       self.forecast_confidence_decay = 0.95  # Confidence decreases with forecast time
   
   def analyze_weather_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
       """Add safety scores and analysis to weather data"""
       if weather_df.empty:
           return weather_df
       
       # Calculate safety scores
       scores = []
       factors = []
       
       for idx, row in weather_df.iterrows():
           score, limiting = self.safety_analyzer.calculate_safety_score(row)
           scores.append(score)
           factors.append(limiting)
       
       weather_df['safety_score'] = scores
       weather_df['limiting_factors'] = factors
       weather_df['operational_status'] = weather_df['safety_score'].apply(
           self.safety_analyzer.determine_status
       )
       
       # Add temporal features
       weather_df['hour'] = pd.to_datetime(weather_df['timestamp']).dt.hour
       weather_df['day_of_week'] = pd.to_datetime(weather_df['timestamp']).dt.dayofweek
       weather_df['month'] = pd.to_datetime(weather_df['timestamp']).dt.month
       
       return weather_df
   
   def find_operational_windows(self, weather_df: pd.DataFrame, 
                              min_score: float = 60.0,
                              location: Optional[str] = None) -> List[OperationalWindow]:
       """Find continuous windows of safe operational conditions"""
       
       if weather_df.empty:
           return []
       
       # Filter by location if specified
       if location:
           weather_df = weather_df[weather_df['location_name'] == location]
       
       # Sort by timestamp
       weather_df = weather_df.sort_values('timestamp')
       
       # Find windows where score >= min_score
       weather_df['is_safe'] = weather_df['safety_score'] >= min_score
       
       windows = []
       current_window_start = None
       window_scores = []
       
       for idx, row in weather_df.iterrows():
           if row['is_safe']:
               if current_window_start is None:
                   current_window_start = row['timestamp']
                   window_scores = [row['safety_score']]
               else:
                   window_scores.append(row['safety_score'])
           else:
               if current_window_start is not None:
                   # End of window
                   window_end = weather_df.loc[idx-1, 'timestamp'] if idx > 0 else row['timestamp']
                   duration = (window_end - current_window_start).total_seconds() / 3600
                   
                   if duration >= self.min_window_hours:
                       # Calculate forecast confidence
                       hours_ahead = (current_window_start - datetime.now()).total_seconds() / 3600
                       confidence = self.forecast_confidence_decay ** max(0, hours_ahead / 24)
                       
                       window = OperationalWindow(
                           start=current_window_start,
                           end=window_end,
                           duration_hours=duration,
                           average_score=np.mean(window_scores),
                           min_score=np.min(window_scores),
                           confidence=confidence,
                           location=location or "All"
                       )
                       windows.append(window)
                   
                   current_window_start = None
                   window_scores = []
       
       # Check if last window extends to end
       if current_window_start is not None:
           window_end = weather_df.iloc[-1]['timestamp']
           duration = (window_end - current_window_start).total_seconds() / 3600
           
           if duration >= self.min_window_hours:
               hours_ahead = (current_window_start - datetime.now()).total_seconds() / 3600
               confidence = self.forecast_confidence_decay ** max(0, hours_ahead / 24)
               
               window = OperationalWindow(
                   start=current_window_start,
                   end=window_end,
                   duration_hours=duration,
                   average_score=np.mean(window_scores),
                   min_score=np.min(window_scores),
                   confidence=confidence,
                   location=location or "All"
               )
               windows.append(window)
       
       return windows
   
   def get_current_status(self, weather_df: pd.DataFrame, 
                         location: str) -> Optional[OperationalStatus]:
       """Get current operational status for a location"""
       
       # Get most recent data for location
       location_data = weather_df[weather_df['location_name'] == location]
       if location_data.empty:
           return None
       
       current = location_data.loc[location_data['timestamp'].idxmax()]
       
       # Calculate current score
       score, limiting_factors = self.safety_analyzer.calculate_safety_score(current)
       status = self.safety_analyzer.determine_status(score)
       
       # Generate recommendations
       recommendations = self._generate_recommendations(current, score, limiting_factors)
       
       return OperationalStatus(
           timestamp=current['timestamp'],
           location=location,
           is_safe=score >= 60,
           score=score,
           status=status,
           wind_speed=current.get('wind_speed_10m', 0),
           wave_height=current.get('wave_height_significant', 0),
           wave_period=current.get('wave_period_mean', 10),
           visibility=current.get('visibility', 10000),
           current_speed=current.get('current_speed_surface', 0),
           limiting_factors=limiting_factors,
           recommendations=recommendations
       )
   
   def _generate_recommendations(self, weather_data: pd.Series, 
                               score: float, 
                               limiting_factors: List[str]) -> List[str]:
       """Generate operational recommendations based on conditions"""
       recommendations = []
       
       if score >= 80:
           recommendations.append(" Conditions favorable for all FLNG operations")
       elif score >= 60:
           recommendations.append(" Enhanced monitoring required")
           recommendations.append("Consider postponing non-critical operations")
       else:
           recommendations.append(" Suspend all transfer operations")
           recommendations.append("Prepare for emergency disconnection if needed")
       
       # Specific recommendations based on limiting factors
       for factor in limiting_factors:
           if "Wind" in factor:
               wind_speed = float(factor.split(":")[1].split()[0])
               if wind_speed > 20:
                   recommendations.append("Monitor mooring line tensions closely")
               if wind_speed > 15:
                   recommendations.append("Reduce loading/unloading rates")
           
           elif "Waves" in factor:
               wave_height = float(factor.split(":")[1].split()[0])
               if wave_height > 3:
                   recommendations.append("Activate dynamic positioning if available")
               if wave_height > 2:
                   recommendations.append("Increase fender monitoring frequency")
           
           elif "Cross seas" in factor:
               recommendations.append(" Cross seas detected - monitor vessel motions")
               recommendations.append("Consider vessel reorientation if possible")
           
           elif "Visibility" in factor:
               recommendations.append("Enhance radar and AIS monitoring")
               recommendations.append("Post additional lookouts")
       
       return recommendations
   
   def generate_statistics(self, weather_df: pd.DataFrame) -> Dict:
       """Generate statistical analysis of operational conditions"""
       
       if weather_df.empty:
           return {}
       
       analyzed_df = self.analyze_weather_data(weather_df)
       
       stats = {
           'total_hours': len(analyzed_df),
           'safe_hours': len(analyzed_df[analyzed_df['safety_score'] >= 80]),
           'caution_hours': len(analyzed_df[
               (analyzed_df['safety_score'] >= 60) & 
               (analyzed_df['safety_score'] < 80)
           ]),
           'restricted_hours': len(analyzed_df[
               (analyzed_df['safety_score'] >= 40) & 
               (analyzed_df['safety_score'] < 60)
           ]),
           'suspended_hours': len(analyzed_df[analyzed_df['safety_score'] < 40]),
           'average_score': analyzed_df['safety_score'].mean(),
           'score_std': analyzed_df['safety_score'].std()
       }
       
       # Calculate percentages
       stats['safe_percentage'] = (stats['safe_hours'] / stats['total_hours']) * 100
       stats['operational_percentage'] = (
           (stats['safe_hours'] + stats['caution_hours']) / stats['total_hours']
       ) * 100
       
       # Hourly statistics
       hourly_stats = analyzed_df.groupby('hour')['safety_score'].agg(['mean', 'std'])
       stats['best_hours'] = hourly_stats.nlargest(3, 'mean').index.tolist()
       stats['worst_hours'] = hourly_stats.nsmallest(3, 'mean').index.tolist()
       
       # Monthly statistics
       if 'month' in analyzed_df.columns:
           monthly_stats = analyzed_df.groupby('month')['safety_score'].agg(['mean', 'std'])
           stats['best_months'] = monthly_stats.nlargest(3, 'mean').index.tolist()
           stats['worst_months'] = monthly_stats.nsmallest(3, 'mean').index.tolist()
       
       # Find longest safe window
       windows = self.find_operational_windows(analyzed_df, min_score=80)
       if windows:
           longest_window = max(windows, key=lambda w: w.duration_hours)
           stats['longest_safe_window_hours'] = longest_window.duration_hours
           stats['average_window_duration'] = np.mean([w.duration_hours for w in windows])
       else:
           stats['longest_safe_window_hours'] = 0
           stats['average_window_duration'] = 0
       
       return stats

class MLWindowPredictor:
   """Machine learning based window prediction (optional advanced feature)"""
   
   def __init__(self):
       self.scaler = MinMaxScaler()
       self.feature_columns = [
           'wind_speed_10m', 'wave_height_significant', 'wave_period_mean',
           'swell_height_primary', 'current_speed_surface', 'pressure_msl',
           'hour', 'month'
       ]
   
   def prepare_features(self, weather_df: pd.DataFrame) -> np.ndarray:
       """Prepare features for ML prediction"""
       # Add temporal features if not present
       if 'hour' not in weather_df.columns:
           weather_df['hour'] = pd.to_datetime(weather_df['timestamp']).dt.hour
       if 'month' not in weather_df.columns:
           weather_df['month'] = pd.to_datetime(weather_df['timestamp']).dt.month
       
       # Select and scale features
       features = weather_df[self.feature_columns].fillna(0)
       return self.scaler.fit_transform(features)
   
   def predict_windows(self, weather_df: pd.DataFrame, 
                      model_path: Optional[str] = None) -> pd.DataFrame:
       """Predict operational windows using ML model"""
       # This is a placeholder for ML integration
       # In production, this would load a trained model
       logger.info("ML prediction not implemented - using rule-based analysis")
       return weather_df

# Example usage
def main():
   """Example usage of weather window analyzer"""
   
   # Create sample data
   dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
   sample_data = []
   
   for date in dates:
       # Simulate diurnal patterns
       hour = date.hour
       hour_factor = np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else 0
       
       sample_data.append({
           'timestamp': date,
           'location_name': 'Tokyo Bay',
           'wind_speed_10m': 10 + hour_factor * 5 + np.random.normal(0, 2),
           'wave_height_significant': 1.5 + hour_factor * 0.5 + np.random.normal(0, 0.2),
           'wave_period_mean': 8 + np.random.normal(0, 1),
           'visibility': 5000 + hour_factor * 2000,
           'current_speed_surface': 0.5 + np.random.normal(0, 0.2),
           'wind_wave_height': 0.8 + hour_factor * 0.3,
           'swell_height_primary': 1.0 + np.random.normal(0, 0.1),
           'wind_wave_direction': np.random.uniform(0, 360),
           'swell_direction_primary': np.random.uniform(0, 360),
           'pressure_msl': 1013 + np.random.normal(0, 5)
       })
   
   weather_df = pd.DataFrame(sample_data)
   
   # Initialize analyzer
   analyzer = WeatherWindowAnalyzer()
   
   # Analyze weather data
   analyzed_df = analyzer.analyze_weather_data(weather_df)
   
   print("Safety Score Statistics:")
   print(f"Average: {analyzed_df['safety_score'].mean():.1f}")
   print(f"Std Dev: {analyzed_df['safety_score'].std():.1f}")
   print(f"Min: {analyzed_df['safety_score'].min():.1f}")
   print(f"Max: {analyzed_df['safety_score'].max():.1f}")
   
   # Find operational windows
   windows = analyzer.find_operational_windows(analyzed_df, min_score=70)
   
   print(f"\nFound {len(windows)} operational windows:")
   for window in windows[:5]:  # Show first 5
       print(f"  {window.start} to {window.end}")
       print(f"  Duration: {window.duration_hours:.1f} hours")
       print(f"  Average score: {window.average_score:.1f}")
       print(f"  Confidence: {window.confidence:.1%}")
       print()
   
   # Get current status
   current_status = analyzer.get_current_status(analyzed_df, 'Tokyo Bay')
   if current_status:
       print(f"\nCurrent Status for Tokyo Bay:")
       print(f"  Status: {current_status.status}")
       print(f"  Score: {current_status.score:.1f}")
       print(f"  Wind: {current_status.wind_speed:.1f} m/s")
       print(f"  Waves: {current_status.wave_height:.1f} m")
       print(f"  Limiting factors: {', '.join(current_status.limiting_factors)}")
       print(f"  Recommendations:")
       for rec in current_status.recommendations:
           print(f"    - {rec}")
   
   # Generate statistics
   stats = analyzer.generate_statistics(analyzed_df)
   print(f"\nOperational Statistics:")
   print(f"  Safe hours: {stats['safe_percentage']:.1f}%")
   print(f"  Operational hours: {stats['operational_percentage']:.1f}%")
   print(f"  Best hours: {stats['best_hours']}")
   print(f"  Longest safe window: {stats['longest_safe_window_hours']:.1f} hours")

if __name__ == "__main__":
   main()

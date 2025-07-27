"""
Unit tests for weather window analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from analysis.weather_window import SafetyAnalyzer, WeatherWindowAnalyzer

class TestSafetyAnalyzer:
   """Test SafetyAnalyzer class"""
   
   def setup_method(self):
       """Setup test data"""
       self.analyzer = SafetyAnalyzer()
       self.sample_data = pd.Series({
           'wind_speed_10m': 12.0,
           'wave_height_significant': 1.8,
           'wave_period_mean': 9.0,
           'visibility': 5000,
           'current_speed_surface': 0.8,
           'wind_wave_height': 1.0,
           'swell_height_primary': 0.8,
           'wind_wave_direction': 225,
           'swell_direction_primary': 180,
           'pressure_msl': 1013
       })
   
   def test_calculate_safety_score_safe_conditions(self):
       """Test safety score calculation for safe conditions"""
       score, factors = self.analyzer.calculate_safety_score(self.sample_data)
       
       assert 70 <= score <= 100
       assert len(factors) == 0 or all('Wind' not in f for f in factors)
   
   def test_calculate_safety_score_dangerous_conditions(self):
       """Test safety score calculation for dangerous conditions"""
       dangerous_data = self.sample_data.copy()
       dangerous_data['wind_speed_10m'] = 26.0
       dangerous_data['wave_height_significant'] = 4.5
       
       score, factors = self.analyzer.calculate_safety_score(dangerous_data)
       
       assert score == 0
       assert len(factors) > 0
   
   def test_determine_status(self):
       """Test status determination"""
       assert self.analyzer.determine_status(85) == "SAFE"
       assert self.analyzer.determine_status(70) == "CAUTION"
       assert self.analyzer.determine_status(50) == "RESTRICTED"
       assert self.analyzer.determine_status(30) == "SUSPENDED"

class TestWeatherWindowAnalyzer:
   """Test WeatherWindowAnalyzer class"""
   
   def setup_method(self):
       """Setup test data"""
       self.analyzer = WeatherWindowAnalyzer()
       
       # Create sample weather data
       dates = pd.date_range(start='2024-01-01', periods=48, freq='H')
       self.sample_df = pd.DataFrame({
           'timestamp': dates,
           'location_name': 'Tokyo Bay',
           'wind_speed_10m': np.random.uniform(5, 15, 48),
           'wave_height_significant': np.random.uniform(0.5, 2.5, 48),
           'wave_period_mean': np.random.uniform(7, 11, 48),
           'visibility': np.random.uniform(3000, 10000, 48),
           'current_speed_surface': np.random.uniform(0.2, 1.5, 48)
       })
   
   def test_analyze_weather_data(self):
       """Test weather data analysis"""
       analyzed = self.analyzer.analyze_weather_data(self.sample_df)
       
       assert 'safety_score' in analyzed.columns
       assert 'operational_status' in analyzed.columns
       assert 'limiting_factors' in analyzed.columns
       assert len(analyzed) == len(self.sample_df)
   
   def test_find_operational_windows(self):
       """Test finding operational windows"""
       analyzed = self.analyzer.analyze_weather_data(self.sample_df)
       windows = self.analyzer.find_operational_windows(analyzed, min_score=60)
       
       assert isinstance(windows, list)
       # Check window properties if any found
       if windows:
           window = windows[0]
           assert hasattr(window, 'start')
           assert hasattr(window, 'end')
           assert hasattr(window, 'duration_hours')
           assert window.duration_hours >= 4  # Minimum window duration

if __name__ == "__main__":
   pytest.main([__file__])

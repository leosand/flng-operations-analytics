"""
Weather and Ocean Data Collection Module
Integrates multiple APIs for comprehensive marine weather data
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
import logging
from functools import lru_cache
import yaml
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarineWeatherData:
   """Comprehensive marine weather data structure"""
   timestamp: datetime
   latitude: float
   longitude: float
   location_name: str
   
   # Wind parameters
   wind_speed_10m: float  # m/s
   wind_direction_10m: float  # degrees
   wind_gust_10m: float  # m/s
   
   # Wave parameters
   wave_height_significant: float  # meters
   wave_height_max: float  # meters
   wave_period_mean: float  # seconds
   wave_period_peak: float  # seconds
   wave_direction_mean: float  # degrees
   
   # Swell parameters
   swell_height_primary: float  # meters
   swell_period_primary: float  # seconds
   swell_direction_primary: float  # degrees
   swell_height_secondary: Optional[float] = None
   swell_period_secondary: Optional[float] = None
   swell_direction_secondary: Optional[float] = None
   
   # Wind wave parameters
   wind_wave_height: float = 0.0  # meters
   wind_wave_period: float = 0.0  # seconds
   wind_wave_direction: float = 0.0  # degrees
   
   # Ocean currents
   current_speed_surface: float = 0.0  # m/s
   current_direction_surface: float = 0.0  # degrees
   
   # Atmospheric parameters
   pressure_msl: float = 1013.0  # hPa
   temperature_air: float = 20.0  # Celsius
   humidity_relative: float = 70.0  # percentage
   visibility: float = 10000.0  # meters
   precipitation_rate: float = 0.0  # mm/hr
   
   # Sea state
   sea_surface_temperature: float = 20.0  # Celsius
   sea_surface_salinity: float = 35.0  # PSU
   
   # Data quality
   data_source: str = ""
   quality_flag: int = 1  # 1=good, 2=questionable, 3=bad
   
   def to_dict(self) -> Dict:
       """Convert to dictionary with ISO timestamp"""
       data = asdict(self)
       data['timestamp'] = self.timestamp.isoformat()
       return data

class MarineWeatherAPI:
   """Base class for marine weather API integration"""
   
   def __init__(self, api_config: Dict):
       self.api_key = api_config.get('api_key', '')
       self.base_url = api_config.get('base_url', '')
       self.rate_limit = api_config.get('rate_limit', 10)  # requests per second
       self.last_request_time = 0
       self.cache_dir = "data/cache"
       os.makedirs(self.cache_dir, exist_ok=True)
   
   async def _rate_limit_wait(self):
       """Implement rate limiting"""
       current_time = time.time()
       time_since_last_request = current_time - self.last_request_time
       min_interval = 1.0 / self.rate_limit
       
       if time_since_last_request < min_interval:
           await asyncio.sleep(min_interval - time_since_last_request)
       
       self.last_request_time = time.time()
   
   @lru_cache(maxsize=128)
   def _get_cache_key(self, lat: float, lon: float, timestamp: datetime) -> str:
       """Generate cache key for location and time"""
       return f"{self.__class__.__name__}_{lat:.2f}_{lon:.2f}_{timestamp.strftime('%Y%m%d_%H')}"
   
   async def fetch_data(self, session: aiohttp.ClientSession, 
                       lat: float, lon: float, 
                       start_time: datetime, end_time: datetime) -> List[MarineWeatherData]:
       """Fetch data from API - to be implemented by subclasses"""
       raise NotImplementedError

class OpenMeteoMarineAPI(MarineWeatherAPI):
   """Open-Meteo Marine Weather API integration"""
   
   def __init__(self):
       super().__init__({
           'base_url': 'https://marine-api.open-meteo.com/v1/marine',
           'rate_limit': 10
       })
   
   async def fetch_data(self, session: aiohttp.ClientSession,
                       lat: float, lon: float,
                       start_time: datetime, end_time: datetime) -> List[MarineWeatherData]:
       """Fetch marine weather data from Open-Meteo"""
       
       await self._rate_limit_wait()
       
       params = {
           'latitude': lat,
           'longitude': lon,
           'hourly': [
               'wave_height', 'wave_direction', 'wave_period',
               'wind_wave_height', 'wind_wave_direction', 'wind_wave_period', 'wind_wave_peak_period',
               'swell_wave_height', 'swell_wave_direction', 'swell_wave_period', 'swell_wave_peak_period'
           ],
           'current_weather': True,
           'timezone': 'Asia/Tokyo',
           'start_date': start_time.strftime('%Y-%m-%d'),
           'end_date': end_time.strftime('%Y-%m-%d')
       }
       
       try:
           async with session.get(self.base_url, params=params) as response:
               if response.status == 200:
                   data = await response.json()
                   return self._parse_response(data, lat, lon)
               else:
                   logger.error(f"Open-Meteo API error: {response.status}")
                   return []
       except Exception as e:
           logger.error(f"Error fetching Open-Meteo data: {e}")
           return []
   
   def _parse_response(self, data: Dict, lat: float, lon: float) -> List[MarineWeatherData]:
       """Parse Open-Meteo response into MarineWeatherData objects"""
       weather_data = []
       
       if 'hourly' not in data:
           return weather_data
       
       hourly = data['hourly']
       times = pd.to_datetime(hourly['time'])
       
       for i in range(len(times)):
           weather = MarineWeatherData(
               timestamp=times[i].to_pydatetime(),
               latitude=lat,
               longitude=lon,
               location_name=f"{lat:.2f}N, {lon:.2f}E",
               
               # Wind (estimated from waves)
               wind_speed_10m=self._estimate_wind_from_waves(
                   hourly.get('wind_wave_height', [0])[i] or 0
               ),
               wind_direction_10m=hourly.get('wind_wave_direction', [0])[i] or 0,
               wind_gust_10m=0,  # Not available
               
               # Waves
               wave_height_significant=hourly.get('wave_height', [0])[i] or 0,
               wave_height_max=hourly.get('wave_height', [0])[i] * 1.8 if hourly.get('wave_height', [0])[i] else 0,
               wave_period_mean=hourly.get('wave_period', [0])[i] or 10,
               wave_period_peak=hourly.get('wave_period', [0])[i] or 10,
               wave_direction_mean=hourly.get('wave_direction', [0])[i] or 0,
               
               # Swell
               swell_height_primary=hourly.get('swell_wave_height', [0])[i] or 0,
               swell_period_primary=hourly.get('swell_wave_period', [0])[i] or 12,
               swell_direction_primary=hourly.get('swell_wave_direction', [0])[i] or 0,
               
               # Wind waves
               wind_wave_height=hourly.get('wind_wave_height', [0])[i] or 0,
               wind_wave_period=hourly.get('wind_wave_period', [0])[i] or 6,
               wind_wave_direction=hourly.get('wind_wave_direction', [0])[i] or 0,
               
               data_source='open_meteo',
               quality_flag=1
           )
           weather_data.append(weather)
       
       return weather_data
   
   def _estimate_wind_from_waves(self, wave_height: float) -> float:
       """Estimate wind speed from wave height using empirical relationship"""
       # Simplified relationship: U10  6 * sqrt(Hs)
       return min(6 * np.sqrt(wave_height), 30.0)

class CopernicusMarineAPI(MarineWeatherAPI):
   """Copernicus Marine Service API integration"""
   
   def __init__(self, api_config: Dict):
       super().__init__(api_config)
       self.dataset_id = 'GLOBAL_ANALYSISFORECAST_WAV_001_027'
       self.variables = [
           'VHM0',  # Significant wave height
           'VMDR',  # Mean wave direction
           'VTM10', # Mean wave period
           'VHM0_SW1',  # Primary swell significant height
           'VMDR_SW1',  # Primary swell direction
           'VTM10_SW1', # Primary swell mean period
           'VHM0_WW',   # Wind wave significant height
           'VMDR_WW',   # Wind wave direction
           'VTM10_WW'   # Wind wave mean period
       ]
   
   async def fetch_data(self, session: aiohttp.ClientSession,
                       lat: float, lon: float,
                       start_time: datetime, end_time: datetime) -> List[MarineWeatherData]:
       """Fetch data from Copernicus Marine Service"""
       
       # Note: This is a simplified example. Actual Copernicus API requires authentication
       # and uses their Python client library (copernicus-marine-client)
       
       logger.info("Copernicus Marine API integration requires copernicus-marine-client")
       # Placeholder for actual implementation
       return []

class JMAOceanAPI(MarineWeatherAPI):
   """Japan Meteorological Agency Ocean Data API"""
   
   def __init__(self):
       super().__init__({
           'base_url': 'https://www.jma.go.jp/bosai/wave/',
           'rate_limit': 5
       })
       self.buoy_stations = {
           'Tokyo Bay': {'id': '21004', 'lat': 35.5, 'lon': 139.8},
           'Osaka Bay': {'id': '21005', 'lat': 34.5, 'lon': 135.3},
           'Nagoya': {'id': '21006', 'lat': 35.0, 'lon': 136.9},
           'Sendai Bay': {'id': '21002', 'lat': 38.0, 'lon': 141.5},
           'Kagoshima Bay': {'id': '21008', 'lat': 31.5, 'lon': 130.5}
       }
   
   async def fetch_buoy_data(self, session: aiohttp.ClientSession,
                            station_id: str) -> Optional[Dict]:
       """Fetch data from JMA ocean buoy"""
       
       await self._rate_limit_wait()
       
       url = f"{self.base_url}data/{station_id}.json"
       
       try:
           async with session.get(url) as response:
               if response.status == 200:
                   return await response.json()
               else:
                   logger.error(f"JMA API error for station {station_id}: {response.status}")
                   return None
       except Exception as e:
           logger.error(f"Error fetching JMA data: {e}")
           return None

class StormGlassAPI(MarineWeatherAPI):
   """StormGlass.io Marine Weather API"""
   
   def __init__(self, api_config: Dict):
       super().__init__(api_config)
       self.base_url = 'https://api.stormglass.io/v2/weather/point'
       self.params = [
           'waveHeight', 'wavePeriod', 'waveDirection',
           'windSpeed', 'windDirection', 'windGust',
           'swellHeight', 'swellPeriod', 'swellDirection',
           'secondarySwellHeight', 'secondarySwellPeriod', 'secondarySwellDirection',
           'waterTemperature', 'currentSpeed', 'currentDirection',
           'airTemperature', 'pressure', 'humidity', 'visibility'
       ]
   
   async def fetch_data(self, session: aiohttp.ClientSession,
                       lat: float, lon: float,
                       start_time: datetime, end_time: datetime) -> List[MarineWeatherData]:
       """Fetch comprehensive marine weather data from StormGlass"""
       
       await self._rate_limit_wait()
       
       headers = {
           'Authorization': self.api_key
       }
       
       params = {
           'lat': lat,
           'lng': lon,
           'params': ','.join(self.params),
           'start': start_time.timestamp(),
           'end': end_time.timestamp(),
           'source': 'noaa,icon,dwd'  # Multiple sources for better accuracy
       }
       
       try:
           async with session.get(self.base_url, headers=headers, params=params) as response:
               if response.status == 200:
                   data = await response.json()
                   return self._parse_stormglass_response(data, lat, lon)
               elif response.status == 402:
                   logger.error("StormGlass API quota exceeded")
                   return []
               else:
                   logger.error(f"StormGlass API error: {response.status}")
                   return []
       except Exception as e:
           logger.error(f"Error fetching StormGlass data: {e}")
           return []
   
   def _parse_stormglass_response(self, data: Dict, lat: float, lon: float) -> List[MarineWeatherData]:
       """Parse StormGlass response into MarineWeatherData objects"""
       weather_data = []
       
       for hour_data in data.get('hours', []):
           # Extract values from the first available source
           def get_value(param: str, default: float = 0.0) -> float:
               if param in hour_data:
                   sources = hour_data[param]
                   if isinstance(sources, dict) and sources:
                       return list(sources.values())[0]
               return default
           
           weather = MarineWeatherData(
               timestamp=datetime.fromisoformat(hour_data['time'].replace('Z', '+00:00')),
               latitude=lat,
               longitude=lon,
               location_name=f"{lat:.2f}N, {lon:.2f}E",
               
               # Wind
               wind_speed_10m=get_value('windSpeed'),
               wind_direction_10m=get_value('windDirection'),
               wind_gust_10m=get_value('windGust'),
               
               # Waves
               wave_height_significant=get_value('waveHeight'),
               wave_height_max=get_value('waveHeight') * 1.8,
               wave_period_mean=get_value('wavePeriod'),
               wave_period_peak=get_value('wavePeriod'),
               wave_direction_mean=get_value('waveDirection'),
               
               # Swell
               swell_height_primary=get_value('swellHeight'),
               swell_period_primary=get_value('swellPeriod'),
               swell_direction_primary=get_value('swellDirection'),
               swell_height_secondary=get_value('secondarySwellHeight'),
               swell_period_secondary=get_value('secondarySwellPeriod'),
               swell_direction_secondary=get_value('secondarySwellDirection'),
               
               # Ocean
               current_speed_surface=get_value('currentSpeed'),
               current_direction_surface=get_value('currentDirection'),
               sea_surface_temperature=get_value('waterTemperature'),
               
               # Atmospheric
               pressure_msl=get_value('pressure'),
               temperature_air=get_value('airTemperature'),
               humidity_relative=get_value('humidity'),
               visibility=get_value('visibility', 10000),
               
               data_source='stormglass',
               quality_flag=1
           )
           weather_data.append(weather)
       
       return weather_data

class MarineWeatherCollector:
   """Main collector that aggregates data from multiple sources"""
   
   def __init__(self, config_path: str = "config/api_keys.yml"):
       self.config = self._load_config(config_path)
       self.apis = self._initialize_apis()
       self.japan_regions = {
           'Tokyo Bay': [(35.3, 139.6), (35.5, 139.8), (35.4, 139.9)],
           'Osaka Bay': [(34.4, 135.2), (34.5, 135.3), (34.6, 135.4)],
           'Nagoya': [(34.9, 136.8), (35.0, 136.9), (35.1, 137.0)],
           'Yokohama': [(35.3, 139.5), (35.4, 139.6), (35.5, 139.7)],
           'Kobe': [(34.6, 135.1), (34.7, 135.2), (34.8, 135.3)],
           'Sendai Bay': [(37.9, 141.4), (38.0, 141.5), (38.1, 141.6)],
           'Kagoshima Bay': [(31.4, 130.4), (31.5, 130.5), (31.6, 130.6)]
       }
   
   def _load_config(self, config_path: str) -> Dict:
       """Load API configuration from YAML file"""
       if os.path.exists(config_path):
           with open(config_path, 'r') as f:
               return yaml.safe_load(f)
       else:
           logger.warning(f"Config file not found: {config_path}")
           return {}
   
   def _initialize_apis(self) -> Dict[str, MarineWeatherAPI]:
       """Initialize available APIs based on configuration"""
       apis = {
           'open_meteo': OpenMeteoMarineAPI()  # Always available (no key required)
       }
       
       if 'stormglass' in self.config and self.config['stormglass'].get('api_key'):
           apis['stormglass'] = StormGlassAPI(self.config['stormglass'])
       
       if 'copernicus' in self.config and self.config['copernicus'].get('username'):
           apis['copernicus'] = CopernicusMarineAPI(self.config['copernicus'])
       
       apis['jma'] = JMAOceanAPI()  # Always available
       
       logger.info(f"Initialized APIs: {list(apis.keys())}")
       return apis
   
   async def collect_region_data(self, region: str, 
                                start_time: datetime,
                                end_time: datetime,
                                sources: List[str] = None) -> pd.DataFrame:
       """Collect weather data for a specific region from multiple sources"""
       
       if region not in self.japan_regions:
           logger.error(f"Unknown region: {region}")
           return pd.DataFrame()
       
       if sources is None:
           sources = list(self.apis.keys())
       
       all_data = []
       coordinates = self.japan_regions[region]
       
       async with aiohttp.ClientSession() as session:
           tasks = []
           
           for source in sources:
               if source in self.apis:
                   api = self.apis[source]
                   for lat, lon in coordinates:
                       task = api.fetch_data(session, lat, lon, start_time, end_time)
                       tasks.append((source, lat, lon, task))
           
           # Execute all tasks concurrently
           results = await asyncio.gather(*[t[3] for t in tasks], return_exceptions=True)
           
           # Process results
           for (source, lat, lon, _), result in zip(tasks, results):
               if isinstance(result, Exception):
                   logger.error(f"Error fetching {source} data for {lat},{lon}: {result}")
               elif result:
                   for data_point in result:
                       data_point.location_name = region
                       all_data.append(data_point)
       
       # Convert to DataFrame
       if all_data:
           df = pd.DataFrame([d.to_dict() for d in all_data])
           df['timestamp'] = pd.to_datetime(df['timestamp'])
           
           # Remove duplicates and aggregate by timestamp and location
           df = df.groupby(['timestamp', 'latitude', 'longitude']).first().reset_index()
           
           logger.info(f"Collected {len(df)} data points for {region}")
           return df
       else:
           logger.warning(f"No data collected for {region}")
           return pd.DataFrame()
   
   async def collect_all_regions(self, hours: int = 168) -> pd.DataFrame:
       """Collect data for all Japanese regions"""
       end_time = datetime.now()
       start_time = end_time - timedelta(hours=hours)
       
       all_data = []
       
       for region in self.japan_regions.keys():
           logger.info(f"Collecting data for {region}")
           region_data = await self.collect_region_data(region, start_time, end_time)
           if not region_data.empty:
               all_data.append(region_data)
       
       if all_data:
           combined_df = pd.concat(all_data, ignore_index=True)
           
           # Save to cache
           cache_file = os.path.join(self.cache_dir, 
                                    f"marine_data_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet")
           combined_df.to_parquet(cache_file)
           logger.info(f"Saved {len(combined_df)} records to {cache_file}")
           
           return combined_df
       else:
           return pd.DataFrame()

# Example usage
async def main():
   """Example usage of the marine weather collector"""
   collector = MarineWeatherCollector()
   
   # Collect data for Tokyo Bay for the next 48 hours
   end_time = datetime.now() + timedelta(hours=48)
   start_time = datetime.now()
   
   data = await collector.collect_region_data(
       'Tokyo Bay', 
       start_time, 
       end_time,
       sources=['open_meteo']  # Use only free API for demo
   )
   
   if not data.empty:
       print(f"Collected {len(data)} weather data points")
       print("\nSample data:")
       print(data[['timestamp', 'wave_height_significant', 'wind_speed_10m', 
                  'wave_period_mean', 'data_source']].head())
       
       # Calculate average conditions
       avg_conditions = data.groupby('data_source').agg({
           'wave_height_significant': 'mean',
           'wind_speed_10m': 'mean',
           'wave_period_mean': 'mean'
       }).round(2)
       
       print("\nAverage conditions by source:")
       print(avg_conditions)
   else:
       print("No data collected")

if __name__ == "__main__":
   asyncio.run(main())

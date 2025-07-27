"""
Data Caching Module
Efficient caching for API responses and processed data
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataCache:
   """Handle data caching for the application"""
   
   def __init__(self, cache_dir: str = "data/cache", 
                default_ttl: int = 3600):
       self.cache_dir = Path(cache_dir)
       self.cache_dir.mkdir(parents=True, exist_ok=True)
       self.default_ttl = default_ttl  # seconds
       self._cache_index = self._load_cache_index()
   
   def _get_cache_key(self, key: str) -> str:
       """Generate cache filename from key"""
       return hashlib.md5(key.encode()).hexdigest()
   
   def _get_cache_path(self, cache_key: str, extension: str = '.pkl') -> Path:
       """Get full cache file path"""
       return self.cache_dir / f"{cache_key}{extension}"
   
   def _load_cache_index(self) -> Dict:
       """Load cache index"""
       index_path = self.cache_dir / "cache_index.json"
       if index_path.exists():
           try:
               with open(index_path, 'r') as f:
                   return json.load(f)
           except:
               return {}
       return {}
   
   def _save_cache_index(self):
       """Save cache index"""
       index_path = self.cache_dir / "cache_index.json"
       with open(index_path, 'w') as f:
           json.dump(self._cache_index, f)
   
   def get(self, key: str) -> Optional[Any]:
       """Get item from cache"""
       cache_key = self._get_cache_key(key)
       
       # Check if exists and not expired
       if cache_key in self._cache_index:
           entry = self._cache_index[cache_key]
           if datetime.now().timestamp() < entry['expires']:
               cache_path = self._get_cache_path(cache_key)
               if cache_path.exists():
                   try:
                       with open(cache_path, 'rb') as f:
                           data = pickle.load(f)
                       logger.debug(f"Cache hit: {key}")
                       return data
                   except Exception as e:
                       logger.error(f"Error loading cache: {e}")
       
       logger.debug(f"Cache miss: {key}")
       return None
   
   def set(self, key: str, value: Any, ttl: Optional[int] = None):
       """Set item in cache"""
       cache_key = self._get_cache_key(key)
       cache_path = self._get_cache_path(cache_key)
       
       # Save data
       try:
           with open(cache_path, 'wb') as f:
               pickle.dump(value, f)
           
           # Update index
           self._cache_index[cache_key] = {
               'key': key,
               'created': datetime.now().timestamp(),
               'expires': datetime.now().timestamp() + (ttl or self.default_ttl),
               'size': cache_path.stat().st_size
           }
           self._save_cache_index()
           
           logger.debug(f"Cached: {key}")
       except Exception as e:
           logger.error(f"Error saving cache: {e}")
   
   def delete(self, key: str):
       """Delete item from cache"""
       cache_key = self._get_cache_key(key)
       cache_path = self._get_cache_path(cache_key)
       
       if cache_path.exists():
           cache_path.unlink()
       
       if cache_key in self._cache_index:
           del self._cache_index[cache_key]
           self._save_cache_index()
   
   def clear_expired(self):
       """Clear expired cache entries"""
       current_time = datetime.now().timestamp()
       expired_keys = []
       
       for cache_key, entry in self._cache_index.items():
           if current_time > entry['expires']:
               expired_keys.append(cache_key)
       
       for cache_key in expired_keys:
           cache_path = self._get_cache_path(cache_key)
           if cache_path.exists():
               cache_path.unlink()
           del self._cache_index[cache_key]
       
       if expired_keys:
           self._save_cache_index()
           logger.info(f"Cleared {len(expired_keys)} expired cache entries")
   
   def get_cache_size(self) -> int:
       """Get total cache size in bytes"""
       total_size = 0
       for cache_file in self.cache_dir.glob("*.pkl"):
           total_size += cache_file.stat().st_size
       return total_size
   
   def get_cache_info(self) -> Dict:
       """Get cache information"""
       return {
           'total_entries': len(self._cache_index),
           'total_size_mb': self.get_cache_size() / (1024 * 1024),
           'cache_dir': str(self.cache_dir),
           'entries': [
               {
                   'key': entry['key'],
                   'created': datetime.fromtimestamp(entry['created']),
                   'expires': datetime.fromtimestamp(entry['expires']),
                   'size_kb': entry['size'] / 1024
               }
               for entry in self._cache_index.values()
           ]
       }

# Specialized cache for dataframes
class DataFrameCache(DataCache):
   """Specialized cache for pandas DataFrames"""
   
   def get(self, key: str) -> Optional[pd.DataFrame]:
       """Get DataFrame from cache"""
       cache_key = self._get_cache_key(key)
       cache_path = self._get_cache_path(cache_key, '.parquet')
       
       if cache_key in self._cache_index:
           entry = self._cache_index[cache_key]
           if datetime.now().timestamp() < entry['expires']:
               if cache_path.exists():
                   try:
                       df = pd.read_parquet(cache_path)
                       logger.debug(f"DataFrame cache hit: {key}")
                       return df
                   except Exception as e:
                       logger.error(f"Error loading DataFrame cache: {e}")
       
       return None
   
   def set(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None):
       """Set DataFrame in cache"""
       cache_key = self._get_cache_key(key)
       cache_path = self._get_cache_path(cache_key, '.parquet')
       
       try:
           df.to_parquet(cache_path, compression='snappy')
           
           # Update index
           self._cache_index[cache_key] = {
               'key': key,
               'created': datetime.now().timestamp(),
               'expires': datetime.now().timestamp() + (ttl or self.default_ttl),
               'size': cache_path.stat().st_size,
               'rows': len(df),
               'columns': list(df.columns)
           }
           self._save_cache_index()
           
           logger.debug(f"Cached DataFrame: {key} ({len(df)} rows)")
       except Exception as e:
           logger.error(f"Error saving DataFrame cache: {e}")

# Global cache instances
_cache = None
_df_cache = None

def get_cache() -> DataCache:
   """Get global cache instance"""
   global _cache
   if _cache is None:
       _cache = DataCache()
   return _cache

def get_df_cache() -> DataFrameCache:
   """Get global DataFrame cache instance"""
   global _df_cache
   if _df_cache is None:
       _df_cache = DataFrameCache()
   return _df_cache

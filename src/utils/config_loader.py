"""
Configuration Management Module
Handles loading and validation of configuration files
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
   """Main configuration container"""
   api_keys: Dict[str, Any]
   safety_thresholds: Dict[str, Any]
   regions: Dict[str, Any]
   cache_settings: Dict[str, Any]
   
   def get_api_key(self, service: str) -> Optional[str]:
       """Get API key for a specific service"""
       return self.api_keys.get(service, {}).get('api_key')
   
   def get_safety_threshold(self, parameter: str, level: str) -> Optional[float]:
       """Get safety threshold for a specific parameter and level"""
       return self.safety_thresholds.get(parameter, {}).get(level)

class ConfigLoader:
   """Load and manage application configuration"""
   
   def __init__(self, config_dir: str = "config"):
       self.config_dir = Path(config_dir)
       self.config_cache = {}
       
   def load_config(self) -> Config:
       """Load all configuration files"""
       api_keys = self._load_api_keys()
       safety_thresholds = self._load_yaml_file("safety_thresholds.yml")
       regions = self._load_yaml_file("regions.yml")
       
       # Default cache settings
       cache_settings = {
           'cache_dir': 'data/cache',
           'cache_duration': 3600,  # 1 hour
           'max_cache_size_mb': 500
       }
       
       return Config(
           api_keys=api_keys,
           safety_thresholds=safety_thresholds,
           regions=regions,
           cache_settings=cache_settings
       )
   
   def _load_api_keys(self) -> Dict[str, Any]:
       """Load API keys from configuration file"""
       api_keys_file = self.config_dir / "api_keys.yml"
       
       if not api_keys_file.exists():
           # Try to copy from example file
           example_file = self.config_dir / "api_keys.example.yml"
           if example_file.exists():
               logger.warning("api_keys.yml not found. Using example file.")
               return self._load_yaml_file("api_keys.example.yml")
           else:
               logger.error("No API keys configuration found!")
               return {}
       
       return self._load_yaml_file("api_keys.yml")
   
   def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
       """Load a YAML configuration file"""
       filepath = self.config_dir / filename
       
       if filepath in self.config_cache:
           return self.config_cache[filepath]
       
       try:
           with open(filepath, 'r') as f:
               data = yaml.safe_load(f) or {}
               self.config_cache[filepath] = data
               return data
       except FileNotFoundError:
           logger.error(f"Configuration file not found: {filepath}")
           return {}
       except yaml.YAMLError as e:
           logger.error(f"Error parsing YAML file {filepath}: {e}")
           return {}
   
   def validate_config(self, config: Config) -> bool:
       """Validate configuration completeness"""
       is_valid = True
       
       # Check safety thresholds
       required_thresholds = [
           ('wind_speed', ['normal', 'caution', 'critical']),
           ('wave_height', ['normal', 'caution', 'critical']),
           ('wave_period', ['min', 'max']),
           ('visibility', ['min']),
           ('current_speed', ['max'])
       ]
       
       for param, levels in required_thresholds:
           if param not in config.safety_thresholds:
               logger.error(f"Missing safety threshold parameter: {param}")
               is_valid = False
           else:
               for level in levels:
                   if level not in config.safety_thresholds[param]:
                       logger.error(f"Missing safety threshold: {param}.{level}")
                       is_valid = False
       
       # Check regions
       if not config.regions:
           logger.error("No regions configured")
           is_valid = False
       
       return is_valid
   
   def save_config(self, config: Dict[str, Any], filename: str):
       """Save configuration to file"""
       filepath = self.config_dir / filename
       
       try:
           with open(filepath, 'w') as f:
               yaml.dump(config, f, default_flow_style=False)
           logger.info(f"Configuration saved to {filepath}")
       except Exception as e:
           logger.error(f"Error saving configuration: {e}")

# Singleton instance
_config_loader = None
_config = None

def get_config_loader() -> ConfigLoader:
   """Get singleton ConfigLoader instance"""
   global _config_loader
   if _config_loader is None:
       _config_loader = ConfigLoader()
   return _config_loader

def get_config() -> Config:
   """Get singleton Config instance"""
   global _config
   if _config is None:
       loader = get_config_loader()
       _config = loader.load_config()
       
       if not loader.validate_config(_config):
           logger.warning("Configuration validation failed - using defaults")
   
   return _config

# Convenience functions
def get_api_key(service: str) -> Optional[str]:
   """Get API key for a service"""
   return get_config().get_api_key(service)

def get_safety_threshold(parameter: str, level: str) -> Optional[float]:
   """Get safety threshold value"""
   return get_config().get_safety_threshold(parameter, level)

def get_regions() -> Dict[str, Any]:
   """Get configured regions"""
   return get_config().regions

# Example usage
if __name__ == "__main__":
   config = get_config()
   print(f"Loaded configuration with {len(config.regions)} regions")
   print(f"Wind speed critical threshold: {get_safety_threshold('wind_speed', 'critical')} m/s")

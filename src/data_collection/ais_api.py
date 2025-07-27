"""
AIS Vessel Tracking Module for FLNG Operations
Integrates with various AIS data sources to monitor vessel traffic
"""

import asyncio
import aiohttp
import websockets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import logging
from geopy.distance import geodesic
import folium
from folium import plugins
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Vessel:
   """Vessel information from AIS data"""
   mmsi: str
   imo: Optional[str] = None
   name: str = "Unknown"
   callsign: Optional[str] = None
   vessel_type: str = "Unknown"
   length: Optional[float] = None
   width: Optional[float] = None
   draught: Optional[float] = None
   destination: Optional[str] = None
   eta: Optional[datetime] = None
   
   # Position data
   latitude: float = 0.0
   longitude: float = 0.0
   speed: float = 0.0  # knots
   course: float = 0.0  # degrees
   heading: float = 0.0  # degrees
   timestamp: datetime = field(default_factory=datetime.now)
   
   # Calculated fields
   distance_to_platform: Optional[float] = None
   time_to_cpa: Optional[float] = None  # Closest Point of Approach
   cpa_distance: Optional[float] = None
   risk_level: str = "LOW"

@dataclass
class FLNGPlatform:
   """FLNG platform location and safety zones"""
   name: str
   latitude: float
   longitude: float
   safety_zone_radius: float = 500  # meters
   caution_zone_radius: float = 2000  # meters
   monitoring_zone_radius: float = 5000  # meters
   
   def get_position(self) -> Tuple[float, float]:
       return (self.latitude, self.longitude)

class VesselTypeClassifier:
   """Classify vessels based on AIS type codes"""
   
   # AIS vessel type mapping
   VESSEL_TYPES = {
       # Tankers (relevant for FLNG)
       80: "Tanker",
       81: "Tanker - Hazardous A",
       82: "Tanker - Hazardous B",
       83: "Tanker - Hazardous C",
       84: "Tanker - Hazardous D",
       85: "Tanker - Reserved",
       86: "Tanker - Reserved",
       87: "Tanker - Reserved",
       88: "Tanker - Reserved",
       89: "Tanker - No Info",
       
       # Cargo vessels
       70: "Cargo",
       71: "Cargo - Hazardous A",
       72: "Cargo - Hazardous B",
       73: "Cargo - Hazardous C",
       74: "Cargo - Hazardous D",
       
       # Support vessels
       31: "Tug",
       32: "Tug",
       52: "Tug",
       53: "Port Tender",
       
       # Other relevant types
       30: "Fishing",
       35: "Military",
       36: "Sailing",
       37: "Pleasure",
       40: "HSC",  # High Speed Craft
       50: "Pilot",
       51: "SAR",  # Search and Rescue
       55: "Law Enforcement",
       58: "Medical",
       60: "Passenger",
       90: "Other"
   }
   
   @classmethod
   def classify(cls, type_code: int) -> str:
       """Classify vessel based on AIS type code"""
       return cls.VESSEL_TYPES.get(type_code, "Unknown")
   
   @classmethod
   def is_lng_carrier(cls, vessel: Vessel) -> bool:
       """Check if vessel is likely an LNG carrier"""
       lng_indicators = ['lng', 'liquefied', 'gas carrier', 'methane']
       
       # Check vessel name
       if vessel.name:
           name_lower = vessel.name.lower()
           if any(indicator in name_lower for indicator in lng_indicators):
               return True
       
       # Check vessel type
       if vessel.vessel_type and 'tanker' in vessel.vessel_type.lower():
           # Check dimensions (typical LNG carrier: 280-345m length)
           if vessel.length and 250 <= vessel.length <= 350:
               return True
       
       return False

class AISDataCollector:
   """Base class for AIS data collection"""
   
   def __init__(self, api_config: Dict):
       self.api_key = api_config.get('api_key', '')
       self.base_url = api_config.get('base_url', '')
       self.platforms = []
   
   def add_platform(self, platform: FLNGPlatform):
       """Add FLNG platform to monitor"""
       self.platforms.append(platform)
   
   def calculate_cpa(self, vessel: Vessel, platform: FLNGPlatform) -> Tuple[float, float]:
       """Calculate Closest Point of Approach (CPA) for vessel to platform"""
       # Simplified CPA calculation
       # In production, use more sophisticated algorithms
       
       vessel_pos = (vessel.latitude, vessel.longitude)
       platform_pos = platform.get_position()
       
       # Current distance
       current_distance = geodesic(vessel_pos, platform_pos).meters
       
       # If vessel is stationary or moving away
       if vessel.speed < 0.5:  # Less than 0.5 knots
           return current_distance, 0
       
       # Project vessel position forward
       # Convert speed from knots to m/s
       speed_ms = vessel.speed * 0.514444
       
       # Simple projection (ignoring earth curvature for short distances)
       lat_change = speed_ms * np.cos(np.radians(vessel.course)) / 111111
       lon_change = speed_ms * np.sin(np.radians(vessel.course)) / (111111 * np.cos(np.radians(vessel.latitude)))
       
       # Check distances at future time points
       min_distance = current_distance
       time_to_cpa = 0
       
       for minutes in range(0, 120, 5):  # Check up to 2 hours
           future_lat = vessel.latitude + (lat_change * minutes * 60)
           future_lon = vessel.longitude + (lon_change * minutes * 60)
           future_pos = (future_lat, future_lon)
           
           distance = geodesic(future_pos, platform_pos).meters
           
           if distance < min_distance:
               min_distance = distance
               time_to_cpa = minutes
           elif distance > min_distance * 1.1:  # Moving away
               break
       
       return min_distance, time_to_cpa
   
   def assess_vessel_risk(self, vessel: Vessel, platform: FLNGPlatform) -> str:
       """Assess risk level of vessel relative to platform"""
       
       # Calculate CPA
       cpa_distance, time_to_cpa = self.calculate_cpa(vessel, platform)
       vessel.cpa_distance = cpa_distance
       vessel.time_to_cpa = time_to_cpa
       
       # Risk assessment based on CPA and vessel type
       if cpa_distance < platform.safety_zone_radius:
           risk = "CRITICAL"
       elif cpa_distance < platform.caution_zone_radius:
           if VesselTypeClassifier.is_lng_carrier(vessel):
               risk = "HIGH"  # LNG carrier approaching
           else:
               risk = "MEDIUM"
       elif cpa_distance < platform.monitoring_zone_radius:
           risk = "LOW"
       else:
           risk = "MINIMAL"
       
       # Adjust for vessel size and type
       if vessel.length and vessel.length > 200:  # Large vessel
           if risk == "LOW":
               risk = "MEDIUM"
           elif risk == "MINIMAL":
               risk = "LOW"
       
       vessel.risk_level = risk
       return risk

class AISStreamClient(AISDataCollector):
   """Real-time AIS data from AISStream.io WebSocket"""
   
   def __init__(self, api_config: Dict):
       super().__init__(api_config)
       self.websocket_url = api_config.get('websocket_url', 'wss://stream.aisstream.io/v0/stream')
       self.vessels = {}
   
   async def connect_and_stream(self, platforms: List[FLNGPlatform], 
                               duration_minutes: int = 60):
       """Connect to AIS stream and collect data"""
       
       # Define bounding boxes around platforms
       bboxes = []
       for platform in platforms:
           # Create ~10nm box around platform
           lat_offset = 10 / 60  # 10 nautical miles in degrees
           lon_offset = 10 / (60 * np.cos(np.radians(platform.latitude)))
           
           bbox = {
               "latMin": platform.latitude - lat_offset,
               "latMax": platform.latitude + lat_offset,
               "lonMin": platform.longitude - lon_offset,
               "lonMax": platform.longitude + lon_offset
           }
           # Suite de la commande PowerShell...
           bboxes.append(bbox)
       
       # Subscription message
       subscribe_message = {
           "APIKey": self.api_key,
           "BoundingBoxes": bboxes,
           "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
       }
       
       try:
           async with websockets.connect(self.websocket_url) as websocket:
               # Subscribe
               await websocket.send(json.dumps(subscribe_message))
               logger.info("Connected to AISStream WebSocket")
               
               # Set timeout
               end_time = datetime.now() + timedelta(minutes=duration_minutes)
               
               while datetime.now() < end_time:
                   try:
                       message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                       await self._process_message(json.loads(message))
                   except asyncio.TimeoutError:
                       logger.debug("No message received in 30 seconds")
                       continue
                   except Exception as e:
                       logger.error(f"Error processing message: {e}")
               
       except Exception as e:
           logger.error(f"WebSocket connection error: {e}")
   
   async def _process_message(self, message: Dict):
       """Process AIS message"""
       
       msg_type = message.get("MessageType")
       
       if msg_type == "PositionReport":
           # Update vessel position
           ais_msg = message.get("Message", {}).get("PositionReport", {})
           
           mmsi = str(ais_msg.get("UserID", ""))
           if not mmsi:
               return
           
           if mmsi not in self.vessels:
               self.vessels[mmsi] = Vessel(mmsi=mmsi)
           
           vessel = self.vessels[mmsi]
           vessel.latitude = ais_msg.get("Latitude", 0)
           vessel.longitude = ais_msg.get("Longitude", 0)
           vessel.speed = ais_msg.get("Sog", 0)  # Speed over ground
           vessel.course = ais_msg.get("Cog", 0)  # Course over ground
           vessel.heading = ais_msg.get("TrueHeading", 0)
           vessel.timestamp = datetime.now()
           
           # Update vessel type if available
           if "Type" in ais_msg:
               vessel.vessel_type = VesselTypeClassifier.classify(ais_msg["Type"])
           
           # Calculate distance to platforms
           for platform in self.platforms:
               vessel_pos = (vessel.latitude, vessel.longitude)
               platform_pos = platform.get_position()
               vessel.distance_to_platform = geodesic(vessel_pos, platform_pos).meters
               
               # Assess risk
               self.assess_vessel_risk(vessel, platform)
       
       elif msg_type == "ShipStaticData":
           # Update vessel static information
           static_msg = message.get("Message", {}).get("ShipStaticData", {})
           
           mmsi = str(static_msg.get("UserID", ""))
           if not mmsi:
               return
           
           if mmsi not in self.vessels:
               self.vessels[mmsi] = Vessel(mmsi=mmsi)
           
           vessel = self.vessels[mmsi]
           vessel.imo = static_msg.get("ImoNumber")
           vessel.name = static_msg.get("Name", "Unknown")
           vessel.callsign = static_msg.get("CallSign")
           vessel.vessel_type = VesselTypeClassifier.classify(
               static_msg.get("Type", 0)
           )
           
           # Dimensions
           dim_a = static_msg.get("DimensionA", 0)
           dim_b = static_msg.get("DimensionB", 0)
           dim_c = static_msg.get("DimensionC", 0)
           dim_d = static_msg.get("DimensionD", 0)
           
           vessel.length = dim_a + dim_b if (dim_a and dim_b) else None
           vessel.width = dim_c + dim_d if (dim_c and dim_d) else None
           vessel.draught = static_msg.get("MaximumDraught", 0) / 10  # Convert to meters
           
           # ETA and destination
           vessel.destination = static_msg.get("Destination")
           eta_data = static_msg.get("Eta")
           if eta_data:
               # Parse ETA (format varies by provider)
               try:
                   vessel.eta = datetime.fromisoformat(eta_data)
               except:
                   vessel.eta = None
   
   def get_vessels_in_zone(self, platform: FLNGPlatform, 
                          zone_radius: float) -> List[Vessel]:
       """Get all vessels within specified radius of platform"""
       vessels_in_zone = []
       
       for vessel in self.vessels.values():
           if vessel.distance_to_platform and vessel.distance_to_platform <= zone_radius:
               vessels_in_zone.append(vessel)
       
       return sorted(vessels_in_zone, key=lambda v: v.distance_to_platform or float('inf'))

class VesselTrafficAnalyzer:
   """Analyze vessel traffic patterns around FLNG platforms"""
   
   def __init__(self):
       self.traffic_data = []
   
   def add_snapshot(self, vessels: List[Vessel], timestamp: datetime):
       """Add traffic snapshot for analysis"""
       snapshot = {
           'timestamp': timestamp,
           'vessel_count': len(vessels),
           'lng_carriers': sum(1 for v in vessels if VesselTypeClassifier.is_lng_carrier(v)),
           'tankers': sum(1 for v in vessels if 'tanker' in v.vessel_type.lower()),
           'support_vessels': sum(1 for v in vessels if v.vessel_type in ['Tug', 'Port Tender']),
           'high_risk': sum(1 for v in vessels if v.risk_level in ['HIGH', 'CRITICAL']),
           'vessels': vessels
       }
       self.traffic_data.append(snapshot)
   
   def get_traffic_statistics(self) -> Dict:
       """Calculate traffic statistics"""
       if not self.traffic_data:
           return {}
       
       df = pd.DataFrame(self.traffic_data)
       
       stats = {
           'average_vessels': df['vessel_count'].mean(),
           'max_vessels': df['vessel_count'].max(),
           'average_lng_carriers': df['lng_carriers'].mean(),
           'average_high_risk': df['high_risk'].mean(),
           'peak_hours': self._identify_peak_hours(df),
           'quiet_hours': self._identify_quiet_hours(df)
       }
       
       return stats
   
   def _identify_peak_hours(self, df: pd.DataFrame) -> List[int]:
       """Identify hours with highest traffic"""
       df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
       hourly_avg = df.groupby('hour')['vessel_count'].mean()
       return hourly_avg.nlargest(3).index.tolist()
   
   def _identify_quiet_hours(self, df: pd.DataFrame) -> List[int]:
       """Identify hours with lowest traffic"""
       df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
       hourly_avg = df.groupby('hour')['vessel_count'].mean()
       return hourly_avg.nsmallest(3).index.tolist()
   
   def create_traffic_heatmap(self) -> go.Figure:
       """Create vessel traffic heatmap"""
       if not self.traffic_data:
           return go.Figure()
       
       df = pd.DataFrame(self.traffic_data)
       df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
       df['day'] = pd.to_datetime(df['timestamp']).dt.day_name()
       
       # Create pivot table
       pivot = df.pivot_table(
           values='vessel_count',
           index='hour',
           columns='day',
           aggfunc='mean'
       )
       
       # Create heatmap
       fig = go.Figure(data=go.Heatmap(
           z=pivot.values,
           x=pivot.columns,
           y=pivot.index,
           colorscale='Blues',
           colorbar=dict(title="Avg<br>Vessels"),
           text=np.round(pivot.values, 1),
           texttemplate='%{text}',
           textfont={"size": 10},
           hovertemplate='%{x}<br>%{y}:00<br>Avg Vessels: %{z:.1f}<extra></extra>'
       ))
       
       fig.update_layout(
           title="Vessel Traffic Density - Hourly Average",
           xaxis_title="Day of Week",
           yaxis_title="Hour of Day",
           height=500,
           template="plotly_dark"
       )
       
       return fig

class VesselMapVisualizer:
   """Create interactive maps of vessel positions"""
   
   def __init__(self, platform: FLNGPlatform):
       self.platform = platform
   
   def create_vessel_map(self, vessels: List[Vessel]) -> folium.Map:
       """Create interactive map with vessels and platform"""
       
       # Create base map centered on platform
       m = folium.Map(
           location=[self.platform.latitude, self.platform.longitude],
           zoom_start=10,
           tiles='OpenStreetMap'
       )
       
       # Add platform marker
       folium.Marker(
           [self.platform.latitude, self.platform.longitude],
           popup=f"FLNG Platform: {self.platform.name}",
           icon=folium.Icon(color='red', icon='anchor'),
       ).add_to(m)
       
       # Add safety zones
       # Safety zone (red)
       folium.Circle(
           location=[self.platform.latitude, self.platform.longitude],
           radius=self.platform.safety_zone_radius,
           color='red',
           fill=True,
           fillColor='red',
           fillOpacity=0.2,
           popup=f"Safety Zone ({self.platform.safety_zone_radius}m)"
       ).add_to(m)
       
       # Caution zone (orange)
       folium.Circle(
           location=[self.platform.latitude, self.platform.longitude],
           radius=self.platform.caution_zone_radius,
           color='orange',
           fill=True,
           fillColor='orange',
           fillOpacity=0.1,
           popup=f"Caution Zone ({self.platform.caution_zone_radius}m)"
       ).add_to(m)
       
       # Monitoring zone (yellow)
       folium.Circle(
           location=[self.platform.latitude, self.platform.longitude],
           radius=self.platform.monitoring_zone_radius,
           color='yellow',
           fill=True,
           fillColor='yellow',
           fillOpacity=0.05,
           popup=f"Monitoring Zone ({self.platform.monitoring_zone_radius}m)"
       ).add_to(m)
       
       # Add vessel markers
       for vessel in vessels:
           # Determine marker color based on risk
           color_map = {
               'CRITICAL': 'red',
               'HIGH': 'orange',
               'MEDIUM': 'yellow',
               'LOW': 'green',
               'MINIMAL': 'blue'
           }
           color = color_map.get(vessel.risk_level, 'gray')
           
           # Determine icon based on vessel type
           if VesselTypeClassifier.is_lng_carrier(vessel):
               icon = 'ship'
           elif 'tanker' in vessel.vessel_type.lower():
               icon = 'tint'
           elif vessel.vessel_type in ['Tug', 'Port Tender']:
               icon = 'life-ring'
           else:
               icon = 'circle'
           
           # Create popup text
           popup_text = f"""
           <b>{vessel.name}</b><br>
           Type: {vessel.vessel_type}<br>
           MMSI: {vessel.mmsi}<br>
           Speed: {vessel.speed:.1f} knots<br>
           Course: {vessel.course:.0f}°<br>
           Distance: {vessel.distance_to_platform:.0f}m<br>
           Risk: {vessel.risk_level}<br>
           CPA: {vessel.cpa_distance:.0f}m in {vessel.time_to_cpa:.0f} min
           """
           
           # Add vessel marker
           folium.Marker(
               [vessel.latitude, vessel.longitude],
               popup=popup_text,
               icon=folium.Icon(color=color, icon=icon),
           ).add_to(m)
           
           # Add vessel track (projected path)
           if vessel.speed > 0.5:  # Only for moving vessels
               # Project path for next hour
               track_points = [[vessel.latitude, vessel.longitude]]
               
               speed_ms = vessel.speed * 0.514444
               for minutes in range(10, 61, 10):
                   lat_change = speed_ms * np.cos(np.radians(vessel.course)) * minutes * 60 / 111111
                   lon_change = speed_ms * np.sin(np.radians(vessel.course)) * minutes * 60 / (111111 * np.cos(np.radians(vessel.latitude)))
                   
                   future_lat = vessel.latitude + lat_change
                   future_lon = vessel.longitude + lon_change
                   track_points.append([future_lat, future_lon])
               
               # Add track line
               folium.PolyLine(
                   track_points,
                   color=color,
                   weight=2,
                   opacity=0.6,
                   dash_array='5, 10'
               ).add_to(m)
       
       # Add vessel density heatmap
       if vessels:
           heat_data = [[v.latitude, v.longitude, 1] for v in vessels]
           plugins.HeatMap(heat_data, radius=20, blur=15).add_to(m)
       
       # Add layer control
       folium.LayerControl().add_to(m)
       
       return m
   
   def create_risk_assessment_plot(self, vessels: List[Vessel]) -> go.Figure:
       """Create risk assessment visualization"""
       
       # Categorize vessels by risk
       risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'MINIMAL': 0}
       
       for vessel in vessels:
           risk_counts[vessel.risk_level] = risk_counts.get(vessel.risk_level, 0) + 1
       
       # Create pie chart
       fig = go.Figure(data=[go.Pie(
           labels=list(risk_counts.keys()),
           values=list(risk_counts.values()),
           hole=0.3,
           marker_colors=['#ff0000', '#ff6600', '#ffaa00', '#88ff00', '#0066ff']
       )])
       
       fig.update_layout(
           title="Vessel Risk Distribution",
           height=400,
           template="plotly_dark"
       )
       
       return fig

# Example usage
async def monitor_vessel_traffic():
   """Example monitoring function"""
   
   # Define FLNG platform
   platform = FLNGPlatform(
       name="Tokyo Bay FLNG",
       latitude=35.5,
       longitude=139.8,
       safety_zone_radius=500,
       caution_zone_radius=2000,
       monitoring_zone_radius=5000
   )
   
   # Initialize AIS client (example with mock data)
   # In production, use real API credentials
   ais_config = {
       'api_key': 'your_api_key_here',
       'websocket_url': 'wss://stream.aisstream.io/v0/stream'
   }
   
   client = AISStreamClient(ais_config)
   client.add_platform(platform)
   
   # Create some mock vessels for demonstration
   mock_vessels = [
       Vessel(
           mmsi="123456789",
           name="LNG Sakura",
           vessel_type="Tanker",
           latitude=35.45,
           longitude=139.75,
           speed=12.5,
           course=45,
           length=289
       ),
       Vessel(
           mmsi="987654321",
           name="Pacific Tug",
           vessel_type="Tug",
           latitude=35.52,
           longitude=139.82,
           speed=8.0,
           course=270,
           length=32
       ),
       Vessel(
           mmsi="456789123",
           name="Cargo Express",
           vessel_type="Cargo",
           latitude=35.48,
           longitude=139.85,
           speed=15.0,
           course=180,
           length=180
       )
   ]
   
   # Calculate distances and risks
   for vessel in mock_vessels:
       vessel.distance_to_platform = geodesic(
           (vessel.latitude, vessel.longitude),
           platform.get_position()
       ).meters
       client.assess_vessel_risk(vessel, platform)
   
   # Create visualizations
   analyzer = VesselTrafficAnalyzer()
   analyzer.add_snapshot(mock_vessels, datetime.now())
   
   visualizer = VesselMapVisualizer(platform)
   
   # Create map
   vessel_map = visualizer.create_vessel_map(mock_vessels)
   vessel_map.save("vessel_traffic_map.html")
   
   # Create risk plot
   risk_plot = visualizer.create_risk_assessment_plot(mock_vessels)
   
   # Print summary
   print(f"\nVessel Traffic Summary for {platform.name}")
   print(f"{'='*50}")
   print(f"Total vessels in monitoring zone: {len(mock_vessels)}")
   
   for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
       count = sum(1 for v in mock_vessels if v.risk_level == risk_level)
       if count > 0:
           print(f"{risk_level} risk vessels: {count}")
   
   print(f"\nClosest vessel: {min(mock_vessels, key=lambda v: v.distance_to_platform).name} "
         f"at {min(v.distance_to_platform for v in mock_vessels):.0f}m")
   
   # LNG carriers
   lng_carriers = [v for v in mock_vessels if VesselTypeClassifier.is_lng_carrier(v)]
   if lng_carriers:
       print(f"\nLNG Carriers detected: {len(lng_carriers)}")
       for carrier in lng_carriers:
           print(f"  - {carrier.name}: {carrier.distance_to_platform:.0f}m away, "
                 f"ETA: {carrier.eta or 'Unknown'}")

if __name__ == "__main__":
   asyncio.run(monitor_vessel_traffic())

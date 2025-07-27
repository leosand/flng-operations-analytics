"""
Risk Assessment Module
Comprehensive risk analysis for FLNG operations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskFactor:
   """Individual risk factor assessment"""
   category: str
   parameter: str
   value: float
   threshold: float
   risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
   impact: float  # 0-1 scale
   
@dataclass
class RiskAssessment:
   """Comprehensive risk assessment result"""
   timestamp: datetime
   location: str
   overall_risk: str
   risk_score: float  # 0-100
   risk_factors: List[RiskFactor]
   recommendations: List[str]
   mitigation_measures: List[str]

class RiskAnalyzer:
   """Analyze operational risks for FLNG operations"""
   
   def __init__(self):
       self.risk_weights = {
           'weather': 0.35,
           'ocean': 0.30,
           'vessel_traffic': 0.20,
           'equipment': 0.10,
           'human': 0.05
       }
       
       self.risk_matrix = {
           'LOW': {'color': 'green', 'score_range': (0, 25)},
           'MEDIUM': {'color': 'yellow', 'score_range': (25, 50)},
           'HIGH': {'color': 'orange', 'score_range': (50, 75)},
           'CRITICAL': {'color': 'red', 'score_range': (75, 100)}
       }
   
   def assess_comprehensive_risk(self, 
                                weather_data: pd.DataFrame,
                                vessel_data: Optional[List] = None,
                                equipment_status: Optional[Dict] = None) -> RiskAssessment:
       """Perform comprehensive risk assessment"""
       
       risk_factors = []
       
       # Weather risk assessment
       weather_risks = self._assess_weather_risks(weather_data)
       risk_factors.extend(weather_risks)
       
       # Ocean condition risks
       ocean_risks = self._assess_ocean_risks(weather_data)
       risk_factors.extend(ocean_risks)
       
       # Vessel traffic risks
       if vessel_data:
           traffic_risks = self._assess_traffic_risks(vessel_data)
           risk_factors.extend(traffic_risks)
       
       # Equipment risks
       if equipment_status:
           equipment_risks = self._assess_equipment_risks(equipment_status)
           risk_factors.extend(equipment_risks)
       
       # Calculate overall risk score
       overall_score = self._calculate_overall_risk_score(risk_factors)
       overall_risk = self._determine_risk_level(overall_score)
       
       # Generate recommendations
       recommendations = self._generate_recommendations(risk_factors)
       mitigation_measures = self._generate_mitigation_measures(risk_factors)
       
       return RiskAssessment(
           timestamp=datetime.now(),
           location=weather_data.iloc[0].get('location_name', 'Unknown'),
           overall_risk=overall_risk,
           risk_score=overall_score,
           risk_factors=risk_factors,
           recommendations=recommendations,
           mitigation_measures=mitigation_measures
       )
   
   def _assess_weather_risks(self, weather_data: pd.DataFrame) -> List[RiskFactor]:
       """Assess weather-related risks"""
       risks = []
       
       if weather_data.empty:
           return risks
       
       current = weather_data.iloc[-1]
       
       # Wind risk
       wind_speed = current.get('wind_speed_10m', 0)
       wind_risk = RiskFactor(
           category='weather',
           parameter='wind_speed',
           value=wind_speed,
           threshold=25.0,
           risk_level=self._calculate_parameter_risk(wind_speed, 15, 20, 25),
           impact=self._calculate_impact(wind_speed, 25)
       )
       risks.append(wind_risk)
       
       # Wave risk
       wave_height = current.get('wave_height_significant', 0)
       wave_risk = RiskFactor(
           category='weather',
           parameter='wave_height',
           value=wave_height,
           threshold=4.0,
           risk_level=self._calculate_parameter_risk(wave_height, 2, 3, 4),
           impact=self._calculate_impact(wave_height, 4)
       )
       risks.append(wave_risk)
       
       # Visibility risk
       visibility = current.get('visibility', 10000)
       vis_risk = RiskFactor(
           category='weather',
           parameter='visibility',
           value=visibility,
           threshold=1000,
           risk_level=self._calculate_visibility_risk(visibility),
           impact=self._calculate_impact(1000 - visibility, 1000) if visibility < 1000 else 0
       )
       risks.append(vis_risk)
       
       return risks
   
   def _assess_ocean_risks(self, weather_data: pd.DataFrame) -> List[RiskFactor]:
       """Assess ocean condition risks"""
       risks = []
       
       if weather_data.empty:
           return risks
       
       current = weather_data.iloc[-1]
       
       # Current speed risk
       current_speed = current.get('current_speed_surface', 0) * 1.94384  # Convert m/s to knots
       current_risk = RiskFactor(
           category='ocean',
           parameter='current_speed',
           value=current_speed,
           threshold=2.0,
           risk_level=self._calculate_parameter_risk(current_speed, 1, 1.5, 2),
           impact=self._calculate_impact(current_speed, 2)
       )
       risks.append(current_risk)
       
       # Swell risk
       swell_height = current.get('swell_height_primary', 0)
       swell_risk = RiskFactor(
           category='ocean',
           parameter='swell_height',
           value=swell_height,
           threshold=3.0,
           risk_level=self._calculate_parameter_risk(swell_height, 1.5, 2.5, 3),
           impact=self._calculate_impact(swell_height, 3)
       )
       risks.append(swell_risk)
       
       return risks
   
   def _assess_traffic_risks(self, vessel_data: List) -> List[RiskFactor]:
       """Assess vessel traffic risks"""
       risks = []
       
       # Count high-risk vessels
       high_risk_count = sum(1 for v in vessel_data if v.risk_level in ['HIGH', 'CRITICAL'])
       
       traffic_risk = RiskFactor(
           category='vessel_traffic',
           parameter='high_risk_vessels',
           value=high_risk_count,
           threshold=2,
           risk_level=self._calculate_parameter_risk(high_risk_count, 1, 2, 3),
           impact=self._calculate_impact(high_risk_count, 3)
       )
       risks.append(traffic_risk)
       
       # Closest vessel distance
       if vessel_data:
           closest_distance = min(v.distance_to_platform for v in vessel_data if v.distance_to_platform)
           proximity_risk = RiskFactor(
               category='vessel_traffic',
               parameter='closest_vessel',
               value=closest_distance,
               threshold=500,
               risk_level=self._calculate_proximity_risk(closest_distance),
               impact=self._calculate_impact(500 - closest_distance, 500) if closest_distance < 500 else 0
           )
           risks.append(proximity_risk)
       
       return risks
   
   def _assess_equipment_risks(self, equipment_status: Dict) -> List[RiskFactor]:
       """Assess equipment-related risks"""
       risks = []
       
       # Mooring system status
       mooring_health = equipment_status.get('mooring_system_health', 100)
       mooring_risk = RiskFactor(
           category='equipment',
           parameter='mooring_system',
           value=mooring_health,
           threshold=80,
           risk_level=self._calculate_equipment_risk(mooring_health),
           impact=self._calculate_impact(100 - mooring_health, 100)
       )
       risks.append(mooring_risk)
       
       return risks
   
   def _calculate_parameter_risk(self, value: float, low: float, medium: float, high: float) -> str:
       """Calculate risk level for a parameter"""
       if value < low:
           return 'LOW'
       elif value < medium:
           return 'MEDIUM'
       elif value < high:
           return 'HIGH'
       else:
           return 'CRITICAL'
   
   def _calculate_visibility_risk(self, visibility: float) -> str:
       """Calculate risk level for visibility"""
       if visibility >= 5000:
           return 'LOW'
       elif visibility >= 2000:
           return 'MEDIUM'
       elif visibility >= 1000:
           return 'HIGH'
       else:
           return 'CRITICAL'
   
   def _calculate_proximity_risk(self, distance: float) -> str:
       """Calculate risk level for vessel proximity"""
       if distance >= 2000:
           return 'LOW'
       elif distance >= 1000:
           return 'MEDIUM'
       elif distance >= 500:
           return 'HIGH'
       else:
           return 'CRITICAL'
   
   def _calculate_equipment_risk(self, health: float) -> str:
       """Calculate risk level for equipment health"""
       if health >= 90:
           return 'LOW'
       elif health >= 80:
           return 'MEDIUM'
       elif health >= 70:
           return 'HIGH'
       else:
           return 'CRITICAL'
   
   def _calculate_impact(self, value: float, max_value: float) -> float:
       """Calculate impact factor (0-1)"""
       return min(1.0, max(0.0, value / max_value))
   
   def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
       """Calculate weighted overall risk score"""
       category_scores = {}
       
       for factor in risk_factors:
           if factor.category not in category_scores:
               category_scores[factor.category] = []
           
           # Convert risk level to score
           risk_scores = {'LOW': 10, 'MEDIUM': 40, 'HIGH': 70, 'CRITICAL': 90}
           score = risk_scores.get(factor.risk_level, 0) * factor.impact
           category_scores[factor.category].append(score)
       
       # Calculate weighted average
       overall_score = 0
       for category, scores in category_scores.items():
           if scores:
               category_avg = np.mean(scores)
               weight = self.risk_weights.get(category, 0.1)
               overall_score += category_avg * weight
       
       return min(100, max(0, overall_score))
   
   def _determine_risk_level(self, score: float) -> str:
       """Determine overall risk level from score"""
       for level, info in self.risk_matrix.items():
           if info['score_range'][0] <= score < info['score_range'][1]:
               return level
       return 'CRITICAL'
   
   def _generate_recommendations(self, risk_factors: List[RiskFactor]) -> List[str]:
       """Generate recommendations based on risk factors"""
       recommendations = []
       
       critical_factors = [f for f in risk_factors if f.risk_level == 'CRITICAL']
       high_factors = [f for f in risk_factors if f.risk_level == 'HIGH']
       
       if critical_factors:
           recommendations.append(" CRITICAL: Immediate action required")
           for factor in critical_factors:
               if factor.parameter == 'wind_speed':
                   recommendations.append("- Suspend all operations due to critical wind conditions")
               elif factor.parameter == 'wave_height':
                   recommendations.append("- Prepare for emergency disconnection")
               elif factor.parameter == 'visibility':
                   recommendations.append("- Halt all vessel movements")
       
       if high_factors:
           recommendations.append(" HIGH RISK: Enhanced monitoring required")
           for factor in high_factors:
               if factor.parameter == 'current_speed':
                   recommendations.append("- Monitor mooring line tensions continuously")
               elif factor.parameter == 'high_risk_vessels':
                   recommendations.append("- Maintain radio contact with nearby vessels")
       
       if not critical_factors and not high_factors:
           recommendations.append(" Operations can proceed with standard precautions")
       
       return recommendations
   
   def _generate_mitigation_measures(self, risk_factors: List[RiskFactor]) -> List[str]:
       """Generate mitigation measures for identified risks"""
       measures = []
       
       for factor in risk_factors:
           if factor.risk_level in ['HIGH', 'CRITICAL']:
               if factor.parameter == 'wind_speed':
                   measures.append(f"Reduce loading rate by {int(factor.impact * 50)}%")
               elif factor.parameter == 'wave_height':
                   measures.append("Deploy additional fenders")
               elif factor.parameter == 'current_speed':
                   measures.append("Adjust mooring configuration")
               elif factor.parameter == 'closest_vessel':
                   measures.append("Contact approaching vessel via VHF")
       
       return measures

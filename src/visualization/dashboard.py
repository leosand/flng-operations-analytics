"""
Dashboard Module
Main dashboard UI components
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class Dashboard:
   """Main dashboard controller"""
   
   def __init__(self):
       self.initialize_session_state()
   
   def initialize_session_state(self):
       """Initialize Streamlit session state"""
       if 'last_update' not in st.session_state:
           st.session_state.last_update = datetime.now()
       if 'selected_region' not in st.session_state:
           st.session_state.selected_region = 'Tokyo Bay'
       if 'data_cache' not in st.session_state:
           st.session_state.data_cache = {}
   
   def render_header(self):
       """Render dashboard header"""
       st.markdown("""
       <div style='text-align: center; padding: 1rem 0;'>
           <h1 style='color: #00ff00;'> FLNG Operations Analytics Platform</h1>
           <p style='color: #888;'>Real-time Weather Window Analysis for LNG/FLNG Operations</p>
       </div>
       """, unsafe_allow_html=True)
   
   def render_current_conditions(self, weather_data: pd.DataFrame, 
                                analysis_data: Dict):
       """Render current conditions section"""
       st.header(" Current Conditions")
       
       if weather_data.empty:
           st.warning("No data available")
           return
       
       current = weather_data.iloc[-1]
       
       col1, col2, col3, col4, col5 = st.columns(5)
       
       with col1:
           status = analysis_data.get('status', 'UNKNOWN')
           color = self._get_status_color(status)
           st.markdown(f"""
           <div style='text-align: center; padding: 1rem;
                       background-color: {color}; border-radius: 10px;'>
               <h3 style='margin: 0;'>{status}</h3>
               <p style='margin: 0.5rem 0 0 0;'>Operational Status</p>
           </div>
           """, unsafe_allow_html=True)
       
       with col2:
           score = analysis_data.get('score', 0)
           st.metric("Safety Score", f"{score:.1f}%", 
                    delta=f"{score - 75:.1f}" if score > 75 else None)
       
       with col3:
           wind = current.get('wind_speed_10m', 0)
           st.metric("Wind Speed", f"{wind:.1f} m/s",
                    delta="Normal" if wind < 15 else "High")
       
       with col4:
           wave = current.get('wave_height_significant', 0)
           st.metric("Wave Height", f"{wave:.1f} m",
                    delta=f"Period: {current.get('wave_period_mean', 10):.1f}s")
       
       with col5:
           # Suite de la commande PowerShell...
           vis = current.get('visibility', 10000)
           st.metric("Visibility", f"{vis:.0f} m",
                    delta="Good" if vis > 2000 else "Poor")
   
   def _get_status_color(self, status: str) -> str:
       """Get color for status"""
       colors = {
           'SAFE': '#00ff00',
           'CAUTION': '#ffaa00',
           'RESTRICTED': '#ff6600',
           'SUSPENDED': '#ff0000',
           'UNKNOWN': '#888888'
       }
       return colors.get(status, '#888888')
   
   def render_alerts(self, analysis_data: Dict):
       """Render alerts section"""
       status = analysis_data.get('status', 'UNKNOWN')
       
       if status == 'SUSPENDED':
           st.error(" **SUSPENDED**: Current conditions exceed safe operational limits!")
       elif status == 'RESTRICTED':
           st.warning(" **RESTRICTED**: Limited operations only. Enhanced monitoring required.")
       elif status == 'CAUTION':
           st.warning(" **CAUTION**: Conditions approaching limits. Monitor closely.")
       else:
           st.success(" **SAFE**: Conditions within operational limits.")
   
   def render_forecast_chart(self, forecast_data: pd.DataFrame):
       """Render forecast chart"""
       if forecast_data.empty:
           st.info("No forecast data available")
           return
       
       fig = go.Figure()
       
       # Add safety score line
       fig.add_trace(go.Scatter(
           x=forecast_data['timestamp'],
           y=forecast_data['safety_score'],
           mode='lines',
           name='Safety Score',
           line=dict(color='#00ff00', width=3)
       ))
       
       # Add threshold lines
       fig.add_hline(y=80, line_dash="dash", line_color="green", 
                    annotation_text="Safe")
       fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                    annotation_text="Caution")
       fig.add_hline(y=40, line_dash="dash", line_color="red", 
                    annotation_text="Restricted")
       
       fig.update_layout(
           title="7-Day Operational Forecast",
           xaxis_title="Date/Time",
           yaxis_title="Safety Score (%)",
           height=400,
           template="plotly_dark",
           yaxis=dict(range=[0, 100])
       )
       
       st.plotly_chart(fig, use_container_width=True)
   
   def render_operational_windows(self, windows: List[Dict]):
       """Render operational windows"""
       st.subheader(" Operational Windows")
       
       if not windows:
           st.info("No operational windows found in forecast period")
           return
       
       for window in windows[:5]:  # Show top 5
           start = window['start']
           duration = window['duration_hours']
           score = window['average_score']
           
           col1, col2, col3 = st.columns([2, 1, 1])
           
           with col1:
               st.write(f"**{start.strftime('%Y-%m-%d %H:%M')}**")
           with col2:
               st.write(f"{duration:.1f} hours")
           with col3:
               st.write(f"Avg: {score:.1f}%")
   
   def render_statistics(self, stats: Dict):
       """Render statistics section"""
       st.subheader(" Operational Statistics")
       
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.metric("Availability", 
                    f"{stats.get('operational_percentage', 0):.1f}%")
       
       with col2:
           st.metric("Safe Hours", 
                    f"{stats.get('safe_percentage', 0):.1f}%")
       
       with col3:
           st.metric("Avg Score", 
                    f"{stats.get('average_score', 0):.1f}")
       
       with col4:
           st.metric("Best Hour", 
                    f"{stats.get('best_hours', [0])[0]}:00")

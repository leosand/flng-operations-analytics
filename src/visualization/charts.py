"""
Charts Module
Additional visualization components
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class ChartGenerator:
   """Generate various charts for FLNG operations"""
   
   def __init__(self):
       self.theme = {
           'background': '#0e1117',
           'paper': '#262730',
           'text': '#ffffff',
           'grid': 'rgba(255, 255, 255, 0.1)',
           'safe': '#00ff00',
           'caution': '#ffaa00',
           'restricted': '#ff6600',
           'suspended': '#ff0000'
       }
   
   def create_wind_rose(self, data: pd.DataFrame) -> go.Figure:
       """Create wind rose diagram"""
       
       if 'wind_direction_10m' not in data.columns or 'wind_speed_10m' not in data.columns:
           return go.Figure()
       
       # Bin wind directions
       dir_bins = np.arange(0, 361, 30)
       dir_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 
                    'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W']
       
       # Bin wind speeds
       speed_bins = [0, 5, 10, 15, 20, 25, 100]
       speed_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '>25']
       
       # Create frequency table
       data['dir_bin'] = pd.cut(data['wind_direction_10m'], bins=dir_bins, labels=dir_labels[:-1])
       data['speed_bin'] = pd.cut(data['wind_speed_10m'], bins=speed_bins, labels=speed_labels)
       
       freq_table = data.groupby(['dir_bin', 'speed_bin']).size().unstack(fill_value=0)
       
       # Create wind rose
       fig = go.Figure()
       
       for i, speed_label in enumerate(speed_labels):
           if speed_label in freq_table.columns:
               r_values = freq_table[speed_label].values
               theta_values = dir_labels[:-1]
               
               fig.add_trace(go.Barpolar(
                   r=r_values,
                   theta=theta_values,
                   name=f'{speed_label} m/s',
                   marker_color=px.colors.sequential.Viridis[i]
               ))
       
       fig.update_layout(
           polar=dict(
               radialaxis=dict(
                   tickfont_size=12,
                   showticklabels=True,
                   ticks='outside'
               ),
               angularaxis=dict(
                   tickfont_size=14,
                   direction='clockwise',
                   rotation=90
               )
           ),
           title="Wind Rose Diagram",
           template="plotly_dark",
           showlegend=True,
           legend=dict(
               title="Wind Speed (m/s)",
               x=1.1,
               y=0.5
           )
       )
       
       return fig
   
   def create_scatter_matrix(self, data: pd.DataFrame, 
                            variables: List[str]) -> go.Figure:
       """Create scatter matrix for multiple variables"""
       
       # Filter available variables
       available_vars = [v for v in variables if v in data.columns]
       
       if len(available_vars) < 2:
           return go.Figure()
       
       # Create scatter matrix
       fig = px.scatter_matrix(
           data,
           dimensions=available_vars,
           color='safety_score' if 'safety_score' in data.columns else None,
           color_continuous_scale='Viridis',
           title="Parameter Relationships",
           height=800
       )
       
       fig.update_traces(diagonal_visible=False)
       fig.update_layout(template="plotly_dark")
       
       return fig
   
   def create_3d_risk_surface(self, data: pd.DataFrame) -> go.Figure:
       """Create 3D risk surface plot"""
       
       required_cols = ['wind_speed_10m', 'wave_height_significant', 'safety_score']
       if not all(col in data.columns for col in required_cols):
           return go.Figure()
       
       # Create grid
       wind_range = np.linspace(data['wind_speed_10m'].min(), 
                               data['wind_speed_10m'].max(), 50)
       wave_range = np.linspace(data['wave_height_significant'].min(), 
                               data['wave_height_significant'].max(), 50)
       
       wind_grid, wave_grid = np.meshgrid(wind_range, wave_range)
       
       # Interpolate safety scores
       from scipy.interpolate import griddata
       
       points = data[['wind_speed_10m', 'wave_height_significant']].values
       values = data['safety_score'].values
       
       safety_grid = griddata(points, values, (wind_grid, wave_grid), method='cubic')
       
       # Create 3D surface
       fig = go.Figure(data=[go.Surface(
           x=wind_range,
           y=wave_range,
           z=safety_grid,
           colorscale='Viridis',
           colorbar=dict(title="Safety Score")
       )])
       
       fig.update_layout(
           title="3D Risk Surface",
           scene=dict(
               xaxis_title="Wind Speed (m/s)",
               yaxis_title="Wave Height (m)",
               zaxis_title="Safety Score",
               camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
           ),
           template="plotly_dark",
           height=700
       )
       
       return fig
   
   def create_timeline_gantt(self, windows: List[Dict]) -> go.Figure:
       """Create Gantt chart of operational windows"""
       
       if not windows:
           return go.Figure()
       
       # Prepare data for Gantt
       gantt_data = []
       for i, window in enumerate(windows):
           gantt_data.append({
               'Task': f'Window {i+1}',
               'Start': window['start'],
               'Finish': window['end'],
               'Score': window['average_score']
           })
       
       df = pd.DataFrame(gantt_data)
       
       # Create Gantt chart
       fig = px.timeline(
           df,
           x_start="Start",
           x_end="Finish",
           y="Task",
           color="Score",
           color_continuous_scale="Viridis",
           title="Operational Windows Timeline"
       )
       
       fig.update_yaxes(autorange="reversed")
       fig.update_layout(
           template="plotly_dark",
           height=400,
           showlegend=False
       )
       
       return fig
   
   def create_parameter_evolution(self, data: pd.DataFrame, 
                                 parameters: List[str]) -> go.Figure:
       """Create multi-parameter evolution chart"""
       
       # Filter available parameters
       available_params = [p for p in parameters if p in data.columns]
       
       if not available_params:
           return go.Figure()
       
       # Create subplots
       fig = make_subplots(
           rows=len(available_params),
           cols=1,
           shared_xaxes=True,
           subplot_titles=available_params,
           vertical_spacing=0.05
       )
       
       # Add traces
       for i, param in enumerate(available_params):
           fig.add_trace(
               go.Scatter(
                   x=data['timestamp'],
                   y=data[param],
                   mode='lines',
                   name=param,
                   line=dict(width=2)
               ),
               row=i+1,
               col=1
           )
           
           # Add threshold lines if applicable
           if param == 'wind_speed_10m':
               fig.add_hline(y=25, line_dash="dash", line_color="red", 
                           row=i+1, col=1)
           elif param == 'wave_height_significant':
               fig.add_hline(y=4, line_dash="dash", line_color="red", 
                           row=i+1, col=1)
       
       fig.update_layout(
           title="Parameter Evolution",
           height=200 * len(available_params),
           template="plotly_dark",
           showlegend=False
       )
       
       fig.update_xaxes(title_text="Time", row=len(available_params), col=1)
       
       return fig

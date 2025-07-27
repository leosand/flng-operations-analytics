"""
Advanced Heatmap Generation Module for FLNG Operations
Creates interactive temporal heatmaps with dynamic granularity control
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import colorsys
from dataclasses import dataclass
import json

@dataclass
class HeatmapConfig:
   """Configuration for heatmap generation"""
   title: str
   color_scale: List[Tuple[float, str]]
   show_annotations: bool = True
   annotation_threshold: float = 50.0  # Only annotate values below this
   height: int = 600
   width: Optional[int] = None
   font_size: int = 12
   
   @property
   def plotly_colorscale(self):
       """Convert to Plotly colorscale format"""
       return self.color_scale

class HeatmapGenerator:
   """Generate various types of operational heatmaps"""
   
   def __init__(self):
       # FLNG operational safety colorscale
       self.safety_colorscale = [
           [0.0, '#1a0000'],    # Dark red - Suspended (0-20)
           [0.2, '#660000'],    # Red - Suspended (20-40)
           [0.4, '#ff0000'],    # Bright red - Restricted (40-50)
           [0.5, '#ff6600'],    # Orange - Restricted (50-60)
           [0.6, '#ffaa00'],    # Yellow-orange - Caution (60-70)
           [0.7, '#ffdd00'],    # Yellow - Caution (70-80)
           [0.8, '#88ff00'],    # Yellow-green - Safe (80-90)
           [1.0, '#00ff00']     # Green - Safe (90-100)
       ]
       
       self.default_config = HeatmapConfig(
           title="FLNG Operational Safety Score",
           color_scale=self.safety_colorscale,
           show_annotations=True,
           height=600
       )
   
   def create_hourly_heatmap(self, data: pd.DataFrame, 
                            config: Optional[HeatmapConfig] = None) -> go.Figure:
       """Create hour of day vs day of week heatmap"""
       
       config = config or self.default_config
       
       # Prepare data
       data = data.copy()
       data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
       data['day_name'] = pd.to_datetime(data['timestamp']).dt.day_name()
       data['day_num'] = pd.to_datetime(data['timestamp']).dt.dayofweek
       
       # Create pivot table
       pivot = data.pivot_table(
           values='safety_score',
           index='hour',
           columns=['day_num', 'day_name'],
           aggfunc='mean'
       )
       
       # Sort columns by day number
       pivot = pivot.reindex(columns=sorted(pivot.columns, key=lambda x: x[0]))
       
       # Extract day names for display
       day_names = [col[1] for col in pivot.columns]
       
       # Create heatmap
       fig = go.Figure(data=go.Heatmap(
           z=pivot.values,
           x=day_names,
           y=pivot.index,
           colorscale=config.plotly_colorscale,
           colorbar=dict(
               title="Safety<br>Score",
               titleside="right",
               tickmode="array",
               tickvals=[0, 20, 40, 60, 80, 100],
               ticktext=["0<br>Suspended", "20", "40<br>Restricted", 
                        "60<br>Caution", "80<br>Safe", "100"],
               len=0.9,
               thickness=20
           ),
           text=np.round(pivot.values, 1),
           texttemplate='%{text}',
           textfont={"size": 10, "color": "white"},
           hoverongaps=False,
           hovertemplate='%{x}<br>%{y}:00<br>Score: %{z:.1f}<extra></extra>',
           zmin=0,
           zmax=100
       ))
       
       # Add grid lines for better readability
       shapes = []
       # Horizontal lines every 3 hours
       for hour in range(0, 24, 3):
           shapes.append({
               'type': 'line',
               'x0': -0.5, 'x1': len(day_names) - 0.5,
               'y0': hour - 0.5, 'y1': hour - 0.5,
               'line': {'color': 'rgba(255,255,255,0.2)', 'width': 1}
           })
       
       # Vertical lines between days
       for day in range(len(day_names)):
           shapes.append({
               'type': 'line',
               'x0': day - 0.5, 'x1': day - 0.5,
               'y0': -0.5, 'y1': 23.5,
               'line': {'color': 'rgba(255,255,255,0.2)', 'width': 1}
           })
       
       # Update layout
       fig.update_layout(
           title={
               'text': f"{config.title} - Hourly Pattern",
               'x': 0.5,
               'xanchor': 'center',
               'font': {'size': 24}
           },
           xaxis_title="Day of Week",
           yaxis_title="Hour of Day",
           xaxis={'side': 'bottom', 'tickangle': 0},
           yaxis={'tickmode': 'linear', 'tick0': 0, 'dtick': 1},
           height=config.height,
           width=config.width,
           template="plotly_dark",
           shapes=shapes,
           font=dict(size=config.font_size)
       )
       
       # Add annotations for critical hours
       if config.show_annotations:
           annotations = []
           for i, hour in enumerate(pivot.index):
               for j, day in enumerate(day_names):
                   value = pivot.iloc[i, j]
                   if value < config.annotation_threshold:
                       annotations.append(dict(
                           x=j, y=i,
                           text="",
                           showarrow=False,
                           font=dict(size=16)
                       ))
           fig.update_layout(annotations=annotations)
       
       return fig
   
   def create_monthly_heatmap(self, data: pd.DataFrame,
                             config: Optional[HeatmapConfig] = None) -> go.Figure:
       """Create day of month vs month heatmap"""
       
       config = config or self.default_config
       
       # Prepare data
       data = data.copy()
       data['day'] = pd.to_datetime(data['timestamp']).dt.day
       data['month'] = pd.to_datetime(data['timestamp']).dt.month
       data['month_name'] = pd.to_datetime(data['timestamp']).dt.month_name()
       
       # Create pivot table
       pivot = data.pivot_table(
           values='safety_score',
           index='day',
           columns=['month', 'month_name'],
           aggfunc='mean'
       )
       
       # Sort columns by month number
       pivot = pivot.reindex(columns=sorted(pivot.columns, key=lambda x: x[0]))
       month_names = [col[1] for col in pivot.columns]
       
       # Ensure we have all days 1-31
       full_index = range(1, 32)
       pivot = pivot.reindex(index=full_index)
       
       # Create heatmap
       fig = go.Figure(data=go.Heatmap(
           z=pivot.values,
           x=month_names,
           y=pivot.index,
           colorscale=config.plotly_colorscale,
           colorbar=dict(
               title="Safety<br>Score",
               titleside="right",
               tickmode="array",
               tickvals=[0, 20, 40, 60, 80, 100],
               ticktext=["0<br>Suspended", "20", "40<br>Restricted", 
                        "60<br>Caution", "80<br>Safe", "100"]
           ),
           text=np.round(pivot.values, 1),
           texttemplate='%{text}',
           textfont={"size": 9},
           hoverongaps=False,
           hovertemplate='%{x}<br>Day %{y}<br>Score: %{z:.1f}<extra></extra>',
           zmin=0,
           zmax=100
       ))
       
       # Add month separators
       shapes = []
       for month in range(len(month_names)):
           shapes.append({
               'type': 'line',
               'x0': month - 0.5, 'x1': month - 0.5,
               'y0': 0.5, 'y1': 31.5,
               'line': {'color': 'rgba(255,255,255,0.3)', 'width': 1}
           })
       
       # Update layout
       fig.update_layout(
           title={
                # Suite de la commande PowerShell...
                'text': f"{config.title} - Monthly Pattern",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
           },
           xaxis_title="Month",
           yaxis_title="Day of Month",
           xaxis={'side': 'bottom'},
           yaxis={'tickmode': 'linear', 'tick0': 1, 'dtick': 1},
           height=config.height,
           width=config.width,
           template="plotly_dark",
           shapes=shapes,
           font=dict(size=config.font_size)
       )
       
       return fig
   
   def create_annual_heatmap(self, data: pd.DataFrame,
                            config: Optional[HeatmapConfig] = None) -> go.Figure:
       """Create week of year vs year heatmap"""
       
       config = config or self.default_config
       
       # Prepare data
       data = data.copy()
       data['week'] = pd.to_datetime(data['timestamp']).dt.isocalendar().week
       data['year'] = pd.to_datetime(data['timestamp']).dt.year
       
       # Create pivot table
       pivot = data.pivot_table(
           values='safety_score',
           index='week',
           columns='year',
           aggfunc='mean'
       )
       
       # Ensure we have all weeks 1-52
       full_index = range(1, 53)
       pivot = pivot.reindex(index=full_index)
       
       # Create heatmap
       fig = go.Figure(data=go.Heatmap(
           z=pivot.values,
           x=pivot.columns,
           y=pivot.index,
           colorscale=config.plotly_colorscale,
           colorbar=dict(
               title="Safety<br>Score",
               titleside="right",
               tickmode="array",
               tickvals=[0, 20, 40, 60, 80, 100],
               ticktext=["0<br>Suspended", "20", "40<br>Restricted", 
                        "60<br>Caution", "80<br>Safe", "100"]
           ),
           text=np.round(pivot.values, 1),
           texttemplate='%{text}',
           textfont={"size": 10},
           hoverongaps=False,
           hovertemplate='Year: %{x}<br>Week: %{y}<br>Score: %{z:.1f}<extra></extra>',
           zmin=0,
           zmax=100
       ))
       
       # Add quarter separators
       shapes = []
       for week in [13, 26, 39]:  # End of Q1, Q2, Q3
           shapes.append({
               'type': 'line',
               'x0': pivot.columns[0] - 0.5, 
               'x1': pivot.columns[-1] + 0.5,
               'y0': week + 0.5, 
               'y1': week + 0.5,
               'line': {'color': 'rgba(255,255,255,0.3)', 'width': 2}
           })
       
       # Update layout
       fig.update_layout(
           title={
               'text': f"{config.title} - Annual Pattern",
               'x': 0.5,
               'xanchor': 'center',
               'font': {'size': 24}
           },
           xaxis_title="Year",
           yaxis_title="Week of Year",
           xaxis={'tickformat': 'd'},
           yaxis={'tickmode': 'linear', 'tick0': 1, 'dtick': 1},
           height=config.height,
           width=config.width,
           template="plotly_dark",
           shapes=shapes,
           font=dict(size=config.font_size)
       )
       
       return fig
   
   def create_dynamic_heatmap(self, data: pd.DataFrame,
                             temporal_type: str = 'hourly',
                             filters: Optional[Dict] = None) -> go.Figure:
       """Create dynamic heatmap with filtering capabilities"""
       
       # Apply filters if provided
       if filters:
           filtered_data = data.copy()
           
           if 'hour_range' in filters:
               hour_mask = (pd.to_datetime(filtered_data['timestamp']).dt.hour >= filters['hour_range'][0]) & \
                          (pd.to_datetime(filtered_data['timestamp']).dt.hour <= filters['hour_range'][1])
               filtered_data = filtered_data[hour_mask]
           
           if 'day_range' in filters:
               day_mask = (pd.to_datetime(filtered_data['timestamp']).dt.day >= filters['day_range'][0]) & \
                         (pd.to_datetime(filtered_data['timestamp']).dt.day <= filters['day_range'][1])
               filtered_data = filtered_data[day_mask]
           
           if 'month_range' in filters:
               months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
               start_month = months.index(filters['month_range'][0]) + 1
               end_month = months.index(filters['month_range'][1]) + 1
               month_mask = (pd.to_datetime(filtered_data['timestamp']).dt.month >= start_month) & \
                          (pd.to_datetime(filtered_data['timestamp']).dt.month <= end_month)
               filtered_data = filtered_data[month_mask]
           
           if 'location' in filters and filters['location'] != 'All':
               filtered_data = filtered_data[filtered_data['location_name'] == filters['location']]
       else:
           filtered_data = data
       
       # Create appropriate heatmap based on temporal type
       if temporal_type == 'hourly':
           return self.create_hourly_heatmap(filtered_data)
       elif temporal_type == 'daily':
           return self.create_monthly_heatmap(filtered_data)
       elif temporal_type == 'weekly':
           return self.create_annual_heatmap(filtered_data)
       else:
           return self.create_monthly_heatmap(filtered_data)
   
   def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
       """Create correlation heatmap between weather parameters and safety score"""
       
       # Select relevant columns
       corr_columns = [
           'wind_speed_10m', 'wave_height_significant', 'wave_period_mean',
           'swell_height_primary', 'current_speed_surface', 'visibility',
           'pressure_msl', 'safety_score'
       ]
       
       # Filter columns that exist
       available_columns = [col for col in corr_columns if col in data.columns]
       
       # Calculate correlation matrix
       corr_matrix = data[available_columns].corr()
       
       # Create labels
       labels = {
           'wind_speed_10m': 'Wind Speed',
           'wave_height_significant': 'Wave Height',
           'wave_period_mean': 'Wave Period',
           'swell_height_primary': 'Swell Height',
           'current_speed_surface': 'Current Speed',
           'visibility': 'Visibility',
           'pressure_msl': 'Pressure',
           'safety_score': 'Safety Score'
       }
       
       display_labels = [labels.get(col, col) for col in corr_matrix.columns]
       
       # Create heatmap
       fig = go.Figure(data=go.Heatmap(
           z=corr_matrix.values,
           x=display_labels,
           y=display_labels,
           colorscale='RdBu',
           zmid=0,
           colorbar=dict(
               title="Correlation",
               titleside="right"
           ),
           text=np.round(corr_matrix.values, 2),
           texttemplate='%{text}',
           textfont={"size": 12},
           hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
       ))
       
       # Update layout
       fig.update_layout(
           title={
               'text': "Weather Parameters Correlation Matrix",
               'x': 0.5,
               'xanchor': 'center',
               'font': {'size': 24}
           },
           xaxis={'side': 'bottom', 'tickangle': -45},
           yaxis={'side': 'left'},
           height=700,
           width=700,
           template="plotly_dark"
       )
       
       return fig
   
   def create_temporal_comparison(self, data: pd.DataFrame, 
                                 locations: List[str]) -> go.Figure:
       """Create temporal comparison between multiple locations"""
       
       # Create subplots
       fig = make_subplots(
           rows=len(locations), 
           cols=1,
           subplot_titles=[f"{loc} - Hourly Safety Score Pattern" for loc in locations],
           vertical_spacing=0.05,
           specs=[[{'type': 'heatmap'}] for _ in locations]
       )
       
       for idx, location in enumerate(locations):
           # Filter data for location
           loc_data = data[data['location_name'] == location].copy()
           
           if loc_data.empty:
               continue
           
           # Prepare pivot
           loc_data['hour'] = pd.to_datetime(loc_data['timestamp']).dt.hour
           loc_data['day_name'] = pd.to_datetime(loc_data['timestamp']).dt.day_name()
           loc_data['day_num'] = pd.to_datetime(loc_data['timestamp']).dt.dayofweek
           
           pivot = loc_data.pivot_table(
               values='safety_score',
               index='hour',
               columns=['day_num', 'day_name'],
               aggfunc='mean'
           )
           
           # Sort columns
           pivot = pivot.reindex(columns=sorted(pivot.columns, key=lambda x: x[0]))
           day_names = [col[1] for col in pivot.columns]
           
           # Add heatmap
           fig.add_trace(
               go.Heatmap(
                   z=pivot.values,
                   x=day_names,
                   y=pivot.index,
                   colorscale=self.safety_colorscale,
                   showscale=(idx == 0),  # Only show scale for first subplot
                   text=np.round(pivot.values, 1),
                   texttemplate='%{text}',
                   textfont={"size": 8},
                   hovertemplate='%{x}<br>%{y}:00<br>Score: %{z:.1f}<extra></extra>',
                   zmin=0,
                   zmax=100
               ),
               row=idx+1, 
               col=1
           )
           
           # Update axes
           fig.update_xaxes(title_text="Day of Week" if idx == len(locations)-1 else "", 
                           row=idx+1, col=1)
           fig.update_yaxes(title_text="Hour", row=idx+1, col=1)
       
       # Update layout
       fig.update_layout(
           title={
               'text': "Location Comparison - Hourly Safety Patterns",
               'x': 0.5,
               'xanchor': 'center',
               'font': {'size': 24}
           },
           height=300 * len(locations),
           template="plotly_dark",
           showlegend=False
       )
       
       return fig
   
   def create_risk_matrix(self, data: pd.DataFrame) -> go.Figure:
       """Create risk assessment matrix"""
       
       # Define risk categories based on wind and wave conditions
       data = data.copy()
       
       # Categorize wind speed
       wind_bins = [0, 10, 15, 20, 25, 100]
       wind_labels = ['Light<br>(0-10)', 'Moderate<br>(10-15)', 
                     'Fresh<br>(15-20)', 'Strong<br>(20-25)', 'Gale<br>(>25)']
       data['wind_category'] = pd.cut(data['wind_speed_10m'], 
                                      bins=wind_bins, 
                                      labels=wind_labels)
       
       # Categorize wave height
       wave_bins = [0, 1, 2, 3, 4, 100]
       wave_labels = ['Calm<br>(0-1)', 'Slight<br>(1-2)', 
                     'Moderate<br>(2-3)', 'Rough<br>(3-4)', 'High<br>(>4)']
       data['wave_category'] = pd.cut(data['wave_height_significant'], 
                                      bins=wave_bins, 
                                      labels=wave_labels)
       
       # Create risk matrix
       risk_matrix = data.pivot_table(
           values='safety_score',
           index='wave_category',
           columns='wind_category',
           aggfunc='mean'
       )
       
       # Count occurrences
       count_matrix = data.pivot_table(
           values='safety_score',
           index='wave_category',
           columns='wind_category',
           aggfunc='count'
       )
       
       # Create custom text showing both score and count
       text_matrix = []
       for i in range(len(risk_matrix.index)):
           row_text = []
           for j in range(len(risk_matrix.columns)):
               score = risk_matrix.iloc[i, j]
               count = count_matrix.iloc[i, j]
               if pd.notna(score) and pd.notna(count):
                   row_text.append(f"{score:.0f}<br>({int(count)} obs)")
               else:
                   row_text.append("")
           text_matrix.append(row_text)
       
       # Create heatmap
       fig = go.Figure(data=go.Heatmap(
           z=risk_matrix.values,
           x=risk_matrix.columns,
           y=risk_matrix.index,
           colorscale=self.safety_colorscale,
           colorbar=dict(
               title="Safety<br>Score",
               titleside="right"
           ),
           text=text_matrix,
           texttemplate='%{text}',
           textfont={"size": 11},
           hovertemplate='Wind: %{x}<br>Waves: %{y}<br>Avg Score: %{z:.1f}<extra></extra>',
           zmin=0,
           zmax=100
       ))
       
       # Add diagonal risk zones
       shapes = [
           # High risk zone (top-right)
           dict(
               type='line',
               x0=2.5, y0=2.5,
               x1=4.5, y1=4.5,
               line=dict(color='red', width=3, dash='dot')
           )
       ]
       
       # Update layout
       fig.update_layout(
           title={
               'text': "Risk Assessment Matrix - Wind vs Wave Conditions",
               'x': 0.5,
               'xanchor': 'center',
               'font': {'size': 24}
           },
           xaxis_title="Wind Speed (m/s)",
           yaxis_title="Wave Height (m)",
           xaxis={'side': 'bottom'},
           yaxis={'side': 'left'},
           height=600,
           width=700,
           template="plotly_dark",
           shapes=shapes
       )
       
       return fig
   
   def create_operations_dashboard(self, data: pd.DataFrame, 
                                  current_location: str = "Tokyo Bay") -> go.Figure:
       """Create comprehensive operations dashboard with multiple visualizations"""
       
       # Create subplots
       fig = make_subplots(
           rows=2, cols=2,
           subplot_titles=('Current Week Operations', 'Risk Matrix',
                          'Hourly Patterns', 'Parameter Trends'),
           specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                  [{'type': 'heatmap'}, {'type': 'scatter'}]],
           vertical_spacing=0.12,
           horizontal_spacing=0.1
       )
       
       # Filter data for location
       loc_data = data[data['location_name'] == current_location].copy()
       
       # 1. Current week heatmap
       current_date = pd.Timestamp.now()
       week_start = current_date - timedelta(days=current_date.weekday())
       week_end = week_start + timedelta(days=7)
       
       week_data = loc_data[
           (pd.to_datetime(loc_data['timestamp']) >= week_start) & 
           (pd.to_datetime(loc_data['timestamp']) <= week_end)
       ].copy()
       
       if not week_data.empty:
           week_data['hour'] = pd.to_datetime(week_data['timestamp']).dt.hour
           week_data['date'] = pd.to_datetime(week_data['timestamp']).dt.date
           
           week_pivot = week_data.pivot_table(
               values='safety_score',
               index='hour',
               columns='date',
               aggfunc='mean'
           )
           
           fig.add_trace(
               go.Heatmap(
                   z=week_pivot.values,
                   x=[str(d) for d in week_pivot.columns],
                   y=week_pivot.index,
                   colorscale=self.safety_colorscale,
                   showscale=False,
                   zmin=0,
                   zmax=100
               ),
               row=1, col=1
           )
       
       # 2. Risk matrix (simplified)
       if not loc_data.empty:
           # Create simple risk categories
           loc_data['wind_risk'] = pd.cut(loc_data['wind_speed_10m'], 
                                         bins=[0, 15, 20, 100], 
                                         labels=['Low', 'Med', 'High'])
           loc_data['wave_risk'] = pd.cut(loc_data['wave_height_significant'], 
                                         bins=[0, 2, 3, 100], 
                                         labels=['Low', 'Med', 'High'])
           
           risk_pivot = loc_data.pivot_table(
               values='safety_score',
               index='wave_risk',
               columns='wind_risk',
               aggfunc='mean'
           )
           
           fig.add_trace(
               go.Heatmap(
                   z=risk_pivot.values,
                   x=risk_pivot.columns,
                   y=risk_pivot.index,
                   colorscale=self.safety_colorscale,
                   showscale=True,
                   colorbar=dict(x=1.02, y=0.75, len=0.45),
                   zmin=0,
                   zmax=100
               ),
               row=1, col=2
           )
       
       # 3. Hourly patterns (last 7 days)
       recent_data = loc_data[
           pd.to_datetime(loc_data['timestamp']) > (current_date - timedelta(days=7))
       ].copy()
       
       if not recent_data.empty:
           recent_data['hour'] = pd.to_datetime(recent_data['timestamp']).dt.hour
           recent_data['day_name'] = pd.to_datetime(recent_data['timestamp']).dt.day_name()
           
           hourly_pivot = recent_data.pivot_table(
               values='safety_score',
               index='hour',
               columns='day_name',
               aggfunc='mean'
           )
           
           fig.add_trace(
               go.Heatmap(
                   z=hourly_pivot.values,
                   x=hourly_pivot.columns,
                   y=hourly_pivot.index,
                   colorscale=self.safety_colorscale,
                   showscale=False,
                   zmin=0,
                   zmax=100
               ),
               row=2, col=1
           )
       
       # 4. Parameter trends
       if not recent_data.empty:
           # Add multiple traces for different parameters
           fig.add_trace(
               go.Scatter(
                   x=recent_data['timestamp'],
                   y=recent_data['safety_score'],
                   name='Safety Score',
                   line=dict(color='green', width=2)
               ),
               row=2, col=2
           )
           
           # Add threshold line
           fig.add_hline(
               y=60, 
               line_dash="dash", 
               line_color="orange",
               row=2, col=2
           )
       
       # Update layout
       fig.update_layout(
           title={
               'text': f"FLNG Operations Dashboard - {current_location}",
               'x': 0.5,
               'xanchor': 'center',
               'font': {'size': 28}
           },
           height=900,
           template="plotly_dark",
           showlegend=False
       )
       
       # Update axes labels
       fig.update_xaxes(title_text="Date", row=1, col=1)
       fig.update_yaxes(title_text="Hour", row=1, col=1)
       fig.update_xaxes(title_text="Wind Risk", row=1, col=2)
       fig.update_yaxes(title_text="Wave Risk", row=1, col=2)
       fig.update_xaxes(title_text="Day", row=2, col=1)
       fig.update_yaxes(title_text="Hour", row=2, col=1)
       fig.update_xaxes(title_text="Time", row=2, col=2)
       fig.update_yaxes(title_text="Score", row=2, col=2)
       
       return fig

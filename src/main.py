"""
FLNG Operations Analytics Platform
Main Application Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yaml
import os
from typing import Dict, List, Tuple
import requests
import json
from dataclasses import dataclass
import asyncio
import aiohttp

# Page configuration
st.set_page_config(
    page_title="FLNG Operations Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        background-color: rgba(30, 30, 30, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .plot-container {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class SafetyThresholds:
    """Safety thresholds for FLNG operations"""
    wind_speed_normal: float = 15.0  # m/s
    wind_speed_caution: float = 20.0
    wind_speed_critical: float = 25.0
    wave_height_normal: float = 2.0  # meters
    wave_height_caution: float = 3.0
    wave_height_critical: float = 4.0
    wave_period_min: float = 6.0  # seconds
    wave_period_max: float = 15.0
    visibility_min: float = 1000  # meters
    current_speed_max: float = 2.0  # knots

@dataclass
class WeatherData:
    """Weather data structure"""
    timestamp: datetime
    wind_speed: float
    wind_direction: float
    wave_height: float
    wave_period: float
    wave_direction: float
    visibility: float
    current_speed: float
    temperature: float
    pressure: float

class WeatherDataCollector:
    """Collect weather data from multiple sources"""
    
    def __init__(self):
        self.base_urls = {
            'open_meteo': 'https://marine-api.open-meteo.com/v1/marine',
            'jma': 'https://www.jma.go.jp/bosai/forecast/data/forecast/',
        }
        self.japan_coordinates = {
            'tokyo_bay': (35.5, 139.8),
            'osaka_bay': (34.5, 135.3),
            'nagoya': (35.0, 136.9),
            'yokohama': (35.4, 139.6),
            'kobe': (34.7, 135.2)
        }
    
    async def fetch_open_meteo_data(self, session: aiohttp.ClientSession, 
                                   lat: float, lon: float) -> Dict:
        """Fetch data from Open-Meteo Marine API"""
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'wave_height,wave_direction,wave_period,wind_wave_height,'
                     'wind_wave_direction,wind_wave_period,swell_wave_height,'
                     'swell_wave_direction,swell_wave_period',
            'daily': 'wave_height_max,wave_period_max,wind_wave_height_max',
            'timezone': 'Asia/Tokyo'
        }
        
        try:
            async with session.get(self.base_urls['open_meteo'], params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            st.error(f"Error fetching Open-Meteo data: {e}")
        return {}
    
    async def collect_all_data(self) -> pd.DataFrame:
        """Collect data from all sources"""
        all_data = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for location, coords in self.japan_coordinates.items():
                task = self.fetch_open_meteo_data(session, coords[0], coords[1])
                tasks.append((location, task))
            
            results = await asyncio.gather(*[t[1] for t in tasks])
            
            for (location, _), result in zip(tasks, results):
                if result and 'hourly' in result:
                    hourly_data = result['hourly']
                    times = pd.to_datetime(hourly_data['time'])
                    
                    for i in range(len(times)):
                        all_data.append({
                            'location': location,
                            'timestamp': times[i],
                            'wave_height': hourly_data['wave_height'][i] or 0,
                            'wave_period': hourly_data['wave_period'][i] or 10,
                            'wave_direction': hourly_data['wave_direction'][i] or 0,
                            'wind_wave_height': hourly_data['wind_wave_height'][i] or 0,
                            'swell_wave_height': hourly_data['swell_wave_height'][i] or 0,
                        })
        
        return pd.DataFrame(all_data)

class OperationalAnalyzer:
    """Analyze operational conditions for FLNG operations"""
    
    def __init__(self, thresholds: SafetyThresholds):
        self.thresholds = thresholds
    
    def calculate_operational_score(self, weather: WeatherData) -> float:
        """Calculate operational safety score (0-100)"""
        score = 100.0
        
        # Wind speed impact
        if weather.wind_speed > self.thresholds.wind_speed_critical:
            return 0.0
        elif weather.wind_speed > self.thresholds.wind_speed_caution:
            score -= (weather.wind_speed - self.thresholds.wind_speed_caution) * 10
        elif weather.wind_speed > self.thresholds.wind_speed_normal:
            score -= (weather.wind_speed - self.thresholds.wind_speed_normal) * 5
        
        # Wave height impact
        if weather.wave_height > self.thresholds.wave_height_critical:
            return 0.0
        elif weather.wave_height > self.thresholds.wave_height_caution:
            score -= (weather.wave_height - self.thresholds.wave_height_caution) * 15
        elif weather.wave_height > self.thresholds.wave_height_normal:
            score -= (weather.wave_height - self.thresholds.wave_height_normal) * 7
        
        # Wave period impact
        if weather.wave_period < self.thresholds.wave_period_min:
            score -= (self.thresholds.wave_period_min - weather.wave_period) * 5
        elif weather.wave_period > self.thresholds.wave_period_max:
            score -= (weather.wave_period - self.thresholds.wave_period_max) * 3
        
        # Current speed impact
        if weather.current_speed > self.thresholds.current_speed_max:
            score -= (weather.current_speed - self.thresholds.current_speed_max) * 20
        
        # Visibility impact
        if weather.visibility < self.thresholds.visibility_min:
            visibility_factor = weather.visibility / self.thresholds.visibility_min
            score *= visibility_factor
        
        return max(0, min(100, score))
    
    def determine_operation_status(self, score: float) -> Tuple[str, str]:
        """Determine operation status based on score"""
        if score >= 80:
            return "SAFE", "#00ff00"
        elif score >= 60:
            return "CAUTION", "#ffaa00"
        elif score >= 40:
            return "RESTRICTED", "#ff6600"
        else:
            return "SUSPENDED", "#ff0000"

def create_operational_heatmap(data: pd.DataFrame, temporal_type: str = 'hourly') -> go.Figure:
    """Create operational heatmap visualization"""
    
    if temporal_type == 'hourly':
        # Hour of day vs day of week
        pivot_data = data.pivot_table(
            values='operational_score',
            index=data['timestamp'].dt.hour,
            columns=data['timestamp'].dt.day_name(),
            aggfunc='mean'
        )
        x_label = "Day of Week"
        y_label = "Hour of Day"
        
    elif temporal_type == 'daily':
        # Day of month vs month
        pivot_data = data.pivot_table(
            values='operational_score',
            index=data['timestamp'].dt.day,
            columns=data['timestamp'].dt.month_name(),
            aggfunc='mean'
        )
        x_label = "Month"
        y_label = "Day of Month"
        
    elif temporal_type == 'weekly':
        # Week of year vs year
        pivot_data = data.pivot_table(
            values='operational_score',
            index=data['timestamp'].dt.isocalendar().week,
            columns=data['timestamp'].dt.year,
            aggfunc='mean'
        )
        x_label = "Year"
        y_label = "Week of Year"
        
    else:  # monthly
        # Month vs year
        pivot_data = data.pivot_table(
            values='operational_score',
            index=data['timestamp'].dt.month_name(),
            columns=data['timestamp'].dt.year,
            aggfunc='mean'
        )
        x_label = "Year"
        y_label = "Month"
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=[
            [0, '#1a1a1a'],      # Very bad conditions
            [0.4, '#ff0000'],    # Bad conditions
            [0.6, '#ff6600'],    # Restricted
            [0.8, '#ffaa00'],    # Caution
            [1.0, '#00ff00']     # Good conditions
        ],
        colorbar=dict(
            title="Operational<br>Score",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=20
        ),
        text=np.round(pivot_data.values, 1),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='%{y}<br>%{x}<br>Score: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"FLNG Operational Safety Score - {temporal_type.capitalize()} View",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        template="plotly_dark",
        font=dict(size=14),
        title_font_size=20,
        xaxis=dict(tickangle=-45)
    )
    
    return fig

def create_wind_wave_scatter(data: pd.DataFrame) -> go.Figure:
    """Create wind-wave scatter diagram"""
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=data['wind_speed'],
        y=data['wave_height'],
        mode='markers',
        marker=dict(
            size=8,
            color=data['operational_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Op. Score")
        ),
        text=[f"Score: {score:.1f}" for score in data['operational_score']],
        hovertemplate='Wind: %{x:.1f} m/s<br>Wave: %{y:.1f} m<br>%{text}<extra></extra>'
    ))
    
    # Add safety threshold lines
    thresholds = SafetyThresholds()
    
    # Wind speed thresholds
    fig.add_vline(x=thresholds.wind_speed_normal, line_dash="dash", 
                  line_color="yellow", annotation_text="Normal Wind Limit")
    fig.add_vline(x=thresholds.wind_speed_caution, line_dash="dash", 
                  line_color="orange", annotation_text="Caution Wind Limit")
    fig.add_vline(x=thresholds.wind_speed_critical, line_dash="dash", 
                  line_color="red", annotation_text="Critical Wind Limit")
    
    # Wave height thresholds
    fig.add_hline(y=thresholds.wave_height_normal, line_dash="dash", 
                  line_color="yellow", annotation_text="Normal Wave Limit")
    fig.add_hline(y=thresholds.wave_height_caution, line_dash="dash", 
                  line_color="orange", annotation_text="Caution Wave Limit")
    fig.add_hline(y=thresholds.wave_height_critical, line_dash="dash", 
                  line_color="red", annotation_text="Critical Wave Limit")
    
    fig.update_layout(
        title="Wind Speed vs Wave Height Scatter Diagram",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Significant Wave Height (m)",
        height=600,
        template="plotly_dark",
        showlegend=False
    )
    
    return fig

def create_temporal_analysis(data: pd.DataFrame) -> go.Figure:
    """Create temporal analysis chart"""
    
    # Group by hour and calculate statistics
    hourly_stats = data.groupby(data['timestamp'].dt.hour).agg({
        'operational_score': ['mean', 'std', 'count']
    }).round(2)
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=hourly_stats.index,
        y=hourly_stats['operational_score']['mean'],
        mode='lines+markers',
        name='Average Score',
        line=dict(color='#00ff00', width=3),
        marker=dict(size=8)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=hourly_stats.index,
        y=hourly_stats['operational_score']['mean'] + hourly_stats['operational_score']['std'],
        fill=None,
        mode='lines',
        line_color='rgba(0,255,0,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_stats.index,
        y=hourly_stats['operational_score']['mean'] - hourly_stats['operational_score']['std'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,255,0,0)',
        name='±1 Std Dev',
        fillcolor='rgba(0,255,0,0.2)'
    ))
    
    fig.update_layout(
        title="Average Operational Score by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Operational Score",
        height=400,
        template="plotly_dark",
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2
        ),
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# Main Streamlit App
def main():
    st.title(" FLNG Operations Analytics Platform")
    st.markdown("### Real-time Weather Window Analysis for LNG/FLNG Operations")
    
    # Sidebar configuration
    with st.sidebar:
        st.header(" Configuration")
        
        # Region selection
        region = st.selectbox(
            "Select Region",
            ["Tokyo Bay", "Osaka Bay", "Nagoya", "Yokohama", "Kobe", "All Japan Waters"]
        )
        
        # Threshold adjustment
        st.subheader(" Safety Thresholds")
        
        wind_normal = st.slider("Wind Speed - Normal (m/s)", 10.0, 20.0, 15.0, 0.5)
        wind_caution = st.slider("Wind Speed - Caution (m/s)", 15.0, 25.0, 20.0, 0.5)
        wind_critical = st.slider("Wind Speed - Critical (m/s)", 20.0, 30.0, 25.0, 0.5)
        
        wave_normal = st.slider("Wave Height - Normal (m)", 1.0, 3.0, 2.0, 0.1)
        wave_caution = st.slider("Wave Height - Caution (m)", 2.0, 4.0, 3.0, 0.1)
        wave_critical = st.slider("Wave Height - Critical (m)", 3.0, 5.0, 4.0, 0.1)
        
        # Update thresholds
        thresholds = SafetyThresholds(
            wind_speed_normal=wind_normal,
            wind_speed_caution=wind_caution,
            wind_speed_critical=wind_critical,
            wave_height_normal=wave_normal,
            wave_height_caution=wave_caution,
            wave_height_critical=wave_critical
        )
        
        st.subheader(" Display Options")
        show_vessel_traffic = st.checkbox("Show Vessel Traffic", value=True)
        show_forecast = st.checkbox("Show 7-Day Forecast", value=True)
        refresh_rate = st.selectbox("Refresh Rate", ["1 min", "5 min", "15 min", "30 min"])
    
    # Generate sample data (replace with real API calls)
    @st.cache_data(ttl=300)
    def load_sample_data():
        # Generate sample data for demonstration
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
        locations = ['Tokyo Bay', 'Osaka Bay', 'Nagoya', 'Yokohama', 'Kobe']
        
        data_list = []
        np.random.seed(42)
        
        for date in dates:
            for location in locations:
                # Simulate realistic patterns
                hour_factor = np.sin(date.hour * np.pi / 12) * 0.3
                day_factor = np.sin(date.dayofyear * 2 * np.pi / 365) * 0.2
                
                wind_base = 10 + hour_factor * 5 + day_factor * 3
                wave_base = 1.5 + hour_factor * 0.8 + day_factor * 0.5
                
                weather = WeatherData(
                    timestamp=date,
                    wind_speed=max(0, wind_base + np.random.normal(0, 2)),
                    wind_direction=np.random.uniform(0, 360),
                    wave_height=max(0, wave_base + np.random.normal(0, 0.3)),
                    wave_period=np.random.uniform(6, 12),
                    wave_direction=np.random.uniform(0, 360),
                    visibility=np.random.uniform(500, 5000),
                    current_speed=np.random.uniform(0, 3),
                    temperature=20 + day_factor * 10 + np.random.normal(0, 2),
                    pressure=1013 + np.random.normal(0, 10)
                )
                
                analyzer = OperationalAnalyzer(thresholds)
                score = analyzer.calculate_operational_score(weather)
                
                data_list.append({
                    'timestamp': date,
                    'location': location,
                    'wind_speed': weather.wind_speed,
                    'wind_direction': weather.wind_direction,
                    'wave_height': weather.wave_height,
                    'wave_period': weather.wave_period,
                    'wave_direction': weather.wave_direction,
                    'visibility': weather.visibility,
                    'current_speed': weather.current_speed,
                    'temperature': weather.temperature,
                    'pressure': weather.pressure,
                    'operational_score': score
                })
        
        return pd.DataFrame(data_list)
    
    # Load data
    with st.spinner("Loading weather and ocean data..."):
        data = load_sample_data()
        
        # Filter by region if needed
        if region != "All Japan Waters":
            data = data[data['location'] == region]
    
    # Current conditions overview
    st.header(" Current Conditions")
    
    current_data = data[data['timestamp'] == data['timestamp'].max()].iloc[0]
    analyzer = OperationalAnalyzer(thresholds)
    current_score = current_data['operational_score']
    status, color = analyzer.determine_operation_status(current_score)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Operational Status",
            status,
            delta=f"{current_score:.1f}%",
            delta_color="normal" if current_score >= 60 else "inverse"
        )
    
    with col2:
        st.metric(
            "Wind Speed",
            f"{current_data['wind_speed']:.1f} m/s",
            delta=f"{current_data['wind_direction']:.0f}°"
        )
    
    with col3:
        st.metric(
            "Wave Height",
            f"{current_data['wave_height']:.1f} m",
            delta=f"Period: {current_data['wave_period']:.1f}s"
        )
    
    with col4:
        st.metric(
            "Visibility",
            f"{current_data['visibility']:.0f} m",
            delta="Good" if current_data['visibility'] > 1000 else "Poor"
        )
    
    with col5:
        st.metric(
            "Current Speed",
            f"{current_data['current_speed']:.1f} kts",
            delta=f"{current_data['wave_direction']:.0f}°"
        )
    
    # Alert box for critical conditions
    if status in ["SUSPENDED", "RESTRICTED"]:
        st.error(f" **{status}**: Current conditions exceed safe operational limits!")
    elif status == "CAUTION":
        st.warning(f" **{status}**: Enhanced monitoring required. Conditions approaching limits.")
    else:
        st.success(f" **{status}**: Conditions within safe operational limits.")
    
    # Main visualizations
    st.header(" Operational Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Heatmaps", "Scatter Analysis", "Temporal Trends", "Forecast"])
    
    with tab1:
        st.subheader("Weather Window Heatmaps")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            heatmap_type = st.radio(
                "Select View",
                ["Hourly", "Daily", "Weekly", "Monthly"],
                key="heatmap_radio"
            )
            
            st.markdown("### Granularity Controls")
            
            if heatmap_type == "Hourly":
                hour_range = st.slider(
                    "Hour Range",
                    0, 23, (6, 18),
                    format="%d:00"
                )
                st.info(f"Showing hours {hour_range[0]}:00 to {hour_range[1]}:00")
                
            elif heatmap_type == "Daily":
                day_range = st.slider(
                    "Day of Month",
                    1, 31, (1, 31)
                )
                st.info(f"Showing days {day_range[0]} to {day_range[1]}")
                
            elif heatmap_type == "Weekly":
                week_range = st.slider(
                    "Week of Year",
                    1, 52, (1, 52)
                )
                st.info(f"Showing weeks {week_range[0]} to {week_range[1]}")
                
            else:  # Monthly
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                month_range = st.select_slider(
                    "Month Range",
                    options=months,
                    value=(months[0], months[-1])
                )
        
        with col2:
            # Apply filters based on granularity controls
            filtered_data = data.copy()
            
            if heatmap_type == "Hourly" and 'hour_range' in locals():
                filtered_data = filtered_data[
                    (filtered_data['timestamp'].dt.hour >= hour_range[0]) &
                    (filtered_data['timestamp'].dt.hour <= hour_range[1])
                ]
            
            heatmap_fig = create_operational_heatmap(filtered_data, heatmap_type.lower())
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Wind-Wave Correlation Analysis")
        
        scatter_fig = create_wind_wave_scatter(data)
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        safe_operations = len(data[data['operational_score'] >= 80])
        total_operations = len(data)
        safe_percentage = (safe_operations / total_operations) * 100
        
        with col1:
            st.metric("Safe Operation Windows", f"{safe_percentage:.1f}%")
        with col2:
            st.metric("Average Wind Speed", f"{data['wind_speed'].mean():.1f} m/s")
        with col3:
            st.metric("Average Wave Height", f"{data['wave_height'].mean():.1f} m")
    
    with tab3:
        st.subheader("Temporal Analysis")
        
        temporal_fig = create_temporal_analysis(data)
        st.plotly_chart(temporal_fig, use_container_width=True)
        
        # Best operational windows
        st.subheader(" Optimal Operation Windows")
        
        hourly_avg = data.groupby(data['timestamp'].dt.hour)['operational_score'].mean()
        best_hours = hourly_avg.nlargest(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Hours for Operations:**")
            for hour, score in best_hours.items():
                st.write(f" {hour:02d}:00 - Average Score: {score:.1f}%")
        
        with col2:
            monthly_avg = data.groupby(data['timestamp'].dt.month_name())['operational_score'].mean()
            best_months = monthly_avg.nlargest(3)
            
            st.markdown("**Best Months for Operations:**")
            for month, score in best_months.items():
                st.write(f" {month} - Average Score: {score:.1f}%")
    
    with tab4:
        st.subheader("7-Day Weather Window Forecast")
        
        if show_forecast:
            # Generate forecast data
            forecast_dates = pd.date_range(
                start=datetime.now(),
                end=datetime.now() + timedelta(days=7),
                freq='H'
            )
            
            forecast_data = []
            for date in forecast_dates:
                # Simplified forecast generation
                score = np.random.normal(75, 15)
                score = max(0, min(100, score))
                forecast_data.append({
                    'timestamp': date,
                    'operational_score': score,
                    'confidence': max(50, 100 - (date - datetime.now()).days * 10)
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Create forecast chart
            fig = go.Figure()
            
            # Add operational score line
            fig.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['operational_score'],
                mode='lines',
                name='Forecast Score',
                line=dict(color='#00ff00', width=2)
            ))
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['operational_score'] + (100 - forecast_df['confidence']) / 2,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['operational_score'] - (100 - forecast_df['confidence']) / 2,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Band',
                fillcolor='rgba(0,255,0,0.2)'
            ))
            
            # Add threshold lines
            fig.add_hline(y=80, line_dash="dash", line_color="green", 
                         annotation_text="Safe Operations")
            fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                         annotation_text="Caution")
            fig.add_hline(y=40, line_dash="dash", line_color="red", 
                         annotation_text="Restricted")
            
            fig.update_layout(
                title="7-Day Operational Forecast",
                xaxis_title="Date/Time",
                # Suite de la commande PowerShell...

                yaxis_title="Operational Score (%)",
                height=500,
                template="plotly_dark",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            safe_windows = forecast_df[forecast_df['operational_score'] >= 80]
            total_safe_hours = len(safe_windows)
            
            st.info(f" **Forecast Summary**: {total_safe_hours} hours of safe operational windows in the next 7 days ({total_safe_hours/168*100:.1f}%)")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>FLNG Operations Analytics Platform v1.0 | Data sources: JMA, Copernicus Marine, Open-Meteo</p>
            <p> For operational planning only. Always consult qualified personnel for actual operations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
   main()

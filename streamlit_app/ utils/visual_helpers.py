
"""
Visual Helpers for Streamlit Precision Farming Dashboard

Utility functions for creating interactive visualizations, charts, and UI components
used across the farming dashboard pages.

Author: Precision Farming Team
Date: 2024
License: MIT
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import colorsys


def get_color_palette(name: str = "green") -> Dict[str, str]:
    """Get predefined color palettes for consistent theming"""
    palettes = {
        "green": {
            "primary": "#4CAF50",
            "secondary": "#2E7D32", 
            "light": "#E8F5E8",
            "accent": "#66BB6A",
            "warning": "#FF9800",
            "error": "#F44336"
        },
        "blue": {
            "primary": "#2196F3",
            "secondary": "#1976D2",
            "light": "#E3F2FD", 
            "accent": "#42A5F5",
            "warning": "#FF9800",
            "error": "#F44336"
        },
        "earth": {
            "primary": "#8D6E63",
            "secondary": "#5D4037",
            "light": "#EFEBE9",
            "accent": "#A1887F", 
            "warning": "#FF9800",
            "error": "#F44336"
        }
    }
    return palettes.get(name, palettes["green"])

def create_soil_radar_chart(nutrient_data: Dict[str, float], 
                          title: str = "Soil Nutrient Profile") -> go.Figure:
    """
    Create radar chart for soil nutrient analysis
    
    Args:
        nutrient_data: Dictionary with nutrient names as keys and values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    colors = get_color_palette("green")
    
    # Prepare data for radar chart
    categories = list(nutrient_data.keys())
    values = list(nutrient_data.values())
    
    # Close the radar chart by adding the first value at the end
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=colors["light"],
        line=dict(color=colors["primary"], width=2),
        marker=dict(color=colors["secondary"], size=8),
        name="Nutrient Levels"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20,
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color=colors["secondary"])
        ),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_nutrient_gauge(value: float, 
                         title: str,
                         min_val: float = 0,
                         max_val: float = 100,
                         thresholds: Dict[str, float] = None) -> go.Figure:
    """
    Create gauge chart for individual nutrient levels
    
    Args:
        value: Current nutrient value
        title: Gauge title
        min_val: Minimum value for gauge
        max_val: Maximum value for gauge
        thresholds: Dictionary with threshold levels
        
    Returns:
        Plotly figure object
    """
    if thresholds is None:
        thresholds = {"low": 30, "medium": 60, "high": 90}
    
    colors = get_color_palette("green")
    
    # Determine color based on value
    if value >= thresholds["high"]:
        gauge_color = colors["primary"]
        status = "High"
    elif value >= thresholds["medium"]:
        gauge_color = colors["accent"]
        status = "Medium"
    elif value >= thresholds["low"]:
        gauge_color = colors["warning"]
        status = "Low"
    else:
        gauge_color = colors["error"]
        status = "Critical"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br><span style='font-size:14px'>{status}</span>"},
        delta={'reference': thresholds["medium"]},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, thresholds["low"]], 'color': "lightgray"},
                {'range': [thresholds["low"], thresholds["medium"]], 'color': "gray"},
                {'range': [thresholds["medium"], thresholds["high"]], 'color': "lightgreen"},
                {'range': [thresholds["high"], max_val], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds["high"]
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': colors["secondary"]}
    )
    
    return fig

def create_crop_suitability_chart(crop_data: List[Dict],
                                 title: str = "Crop Suitability Scores") -> go.Figure:
    """
    Create horizontal bar chart for crop suitability scores
    
    Args:
        crop_data: List of dictionaries with crop names and scores
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    colors = get_color_palette("green")
    
    # Sort by score
    sorted_data = sorted(crop_data, key=lambda x: x['score'], reverse=True)
    
    crops = [item['name'] for item in sorted_data]
    scores = [item['score'] for item in sorted_data]
    
    # Color code based on scores
    bar_colors = []
    for score in scores:
        if score >= 80:
            bar_colors.append(colors["primary"])
        elif score >= 60:
            bar_colors.append(colors["accent"])
        elif score >= 40:
            bar_colors.append(colors["warning"])
        else:
            bar_colors.append(colors["error"])
    
    fig = go.Figure(data=[
        go.Bar(
            y=crops,
            x=scores,
            orientation='h',
            marker_color=bar_colors,
            text=[f"{score:.1f}" for score in scores],
            textposition='inside',
            textfont=dict(color='white', size=12)
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Suitability Score",
        yaxis_title="Crops",
        height=max(300, len(crops) * 40),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def create_economic_comparison_chart(economic_data: List[Dict],
                                   metric: str = "profit") -> go.Figure:
    """
    Create comparison chart for economic metrics
    
    Args:
        economic_data: List of dictionaries with crop economic data
        metric: Which metric to display ('profit', 'revenue', 'cost', 'roi')
        
    Returns:
        Plotly figure object
    """
    colors = get_color_palette("green")
    
    crops = [item['crop'] for item in economic_data]
    values = [item[metric] for item in economic_data]
    
    # Create bar chart
    fig = px.bar(
        x=crops,
        y=values,
        title=f"Economic Comparison - {metric.title()}",
        labels={'x': 'Crops', 'y': metric.title()},
        color=values,
        color_continuous_scale=['red', 'yellow', 'green']
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )
    
    return fig

def create_yield_trend_chart(historical_data: pd.DataFrame,
                           predicted_data: pd.DataFrame = None) -> go.Figure:
    """
    Create yield trend chart with historical and predicted data
    
    Args:
        historical_data: DataFrame with year and yield columns
        predicted_data: Optional DataFrame with predicted yield data
        
    Returns:
        Plotly figure object
    """
    colors = get_color_palette("green")
    
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data['year'],
        y=historical_data['yield'],
        mode='lines+markers',
        name='Historical Yield',
        line=dict(color=colors["primary"], width=3),
        marker=dict(size=8, color=colors["secondary"])
    ))
    
    # Add predicted data if provided
    if predicted_data is not None:
        fig.add_trace(go.Scatter(
            x=predicted_data['year'],
            y=predicted_data['yield'],
            mode='lines+markers',
            name='Predicted Yield',
            line=dict(color=colors["accent"], width=3, dash='dash'),
            marker=dict(size=8, color=colors["accent"])
        ))
    
    fig.update_layout(
        title="Yield Trends Over Time",
        xaxis_title="Year",
        yaxis_title="Yield (tonnes/ha)",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0, y=1)
    )
    
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_weather_chart(weather_data: pd.DataFrame) -> go.Figure:
    """
    Create weather data visualization with multiple metrics
    
    Args:
        weather_data: DataFrame with weather information
        
    Returns:
        Plotly figure object
    """
    colors = get_color_palette("blue")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Rainfall', 'Humidity', 'Wind Speed'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=weather_data['date'], y=weather_data['temperature'],
                  name='Temperature', line=dict(color=colors["error"])),
        row=1, col=1
    )
    
    # Rainfall
    fig.add_trace(
        go.Bar(x=weather_data['date'], y=weather_data['rainfall'],
               name='Rainfall', marker_color=colors["primary"]),
        row=1, col=2
    )
    
    # Humidity
    fig.add_trace(
        go.Scatter(x=weather_data['date'], y=weather_data['humidity'],
                  name='Humidity', line=dict(color=colors["accent"])),
        row=2, col=1
    )
    
    # Wind Speed
    fig.add_trace(
        go.Scatter(x=weather_data['date'], y=weather_data['wind_speed'],
                  name='Wind Speed', line=dict(color=colors["secondary"])),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Weather Data Overview",
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def format_recommendation_card(recommendation: str, 
                             priority: str = "medium",
                             icon: str = "💡") -> str:
    """
    Format recommendation as HTML card
    
    Args:
        recommendation: Recommendation text
        priority: Priority level ('high', 'medium', 'low')
        icon: Icon to display
        
    Returns:
        HTML string for the card
    """
    colors = get_color_palette("green")
    
    priority_colors = {
        "high": colors["error"],
        "medium": colors["warning"], 
        "low": colors["primary"]
    }
    
    border_color = priority_colors.get(priority, colors["primary"])
    
    return f"""
    <div style="
        background: white;
        padding: 1rem;
        border-left: 4px solid {border_color};
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: flex-start;">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
            <div>
                <p style="margin: 0; color: #333; line-height: 1.4;">
                    {recommendation}
                </p>
                <span style="
                    font-size: 0.8rem; 
                    color: {border_color}; 
                    font-weight: bold;
                    text-transform: uppercase;
                ">
                    {priority} priority
                </span>
            </div>
        </div>
    </div>
    """

def create_progress_indicator(current: int, total: int, 
                            label: str = "Progress") -> str:
    """
    Create HTML progress indicator
    
    Args:
        current: Current progress value
        total: Total/target value
        label: Progress label
        
    Returns:
        HTML string for progress bar
    """
    colors = get_color_palette("green")
    percentage = min(100, (current / total) * 100) if total > 0 else 0
    
    return f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: bold; color: {colors['secondary']};">{label}</span>
            <span style="color: {colors['primary']};">{current}/{total} ({percentage:.1f}%)</span>
        </div>
        <div style="
            background: #E0E0E0; 
            border-radius: 10px; 
            height: 8px; 
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(90deg, {colors['accent']}, {colors['primary']});
                height: 100%;
                width: {percentage}%;
                border-radius: 10px;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """

def create_metric_card(title: str, value: Union[str, float], 
                      delta: Optional[Union[str, float]] = None,
                      delta_color: str = "green",
                      icon: str = "📊") -> str:
    """
    Create metric display card
    
    Args:
        title: Metric title
        value: Main metric value
        delta: Change/delta value
        delta_color: Color for delta (green/red/gray)
        icon: Icon to display
        
    Returns:
        HTML string for metric card
    """
    colors = get_color_palette("green")
    
    delta_colors = {
        "green": colors["primary"],
        "red": colors["error"],
        "gray": "#666"
    }
    
    delta_html = ""
    if delta is not None:
        delta_html = f"""
        <p style="
            margin: 0.5rem 0 0 0; 
            color: {delta_colors.get(delta_color, '#666')};
            font-size: 0.9rem;
        ">
            {delta}
        </p>
        """
    
    return f"""
    <div style="
        background: linear-gradient(135deg, {colors['light']}, #F8F9FA);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid {colors['accent']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <h3 style="
            margin: 0; 
            color: {colors['secondary']}; 
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        ">{title}</h3>
        <p style="
            font-size: 2rem; 
            font-weight: bold; 
            margin: 0.5rem 0; 
            color: {colors['primary']};
        ">{value}</p>
        {delta_html}
    </div>
    """

def create_alert_box(message: str, alert_type: str = "info", 
                    dismissible: bool = False) -> str:
    """
    Create styled alert box
    
    Args:
        message: Alert message
        alert_type: Type of alert ('success', 'warning', 'error', 'info')
        dismissible: Whether alert can be dismissed
        
    Returns:
        HTML string for alert box
    """
    colors = get_color_palette("green")
    
    alert_config = {
        "success": {
            "bg": colors["light"],
            "border": colors["primary"],
            "icon": "✅",
            "color": colors["secondary"]
        },
        "warning": {
            "bg": "#FFF3E0",
            "border": colors["warning"],
            "icon": "⚠️",
            "color": "#E65100"
        },
        "error": {
            "bg": "#FFEBEE",
            "border": colors["error"],
            "icon": "❌",
            "color": "#B71C1C"
        },
        "info": {
            "bg": "#E3F2FD",
            "border": "#2196F3",
            "icon": "ℹ️",
            "color": "#0D47A1"
        }
    }
    
    config = alert_config.get(alert_type, alert_config["info"])
    
    dismiss_html = ""
    if dismissible:
        dismiss_html = """
        <button onclick="this.parentElement.style.display='none'" style="
            background: none;
            border: none;
            font-size: 1.2rem;
            cursor: pointer;
            float: right;
            color: inherit;
        ">×</button>
        """
    
    return f"""
    <div style="
        background: {config['bg']};
        border: 1px solid {config['border']};
        border-left: 4px solid {config['border']};
        color: {config['color']};
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        position: relative;
    ">
        {dismiss_html}
        <div style="display: flex; align-items: flex-start;">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{config['icon']}</span>
            <div>{message}</div>
        </div>
    </div>
    """

def create_data_table(data: pd.DataFrame, 
                     title: str = "Data Table",
                     max_rows: int = 10) -> str:
    """
    Create styled data table from DataFrame
    
    Args:
        data: DataFrame to display
        title: Table title
        max_rows: Maximum rows to display
        
    Returns:
        HTML string for table
    """
    colors = get_color_palette("green")
    
    # Limit rows if necessary
    if len(data) > max_rows:
        display_data = data.head(max_rows)
        show_more = True
    else:
        display_data = data
        show_more = False
    
    # Generate table HTML
    table_html = f"""
    <div style="margin: 1rem 0;">
        <h3 style="color: {colors['secondary']; margin-bottom: 1rem;">{title}</h3>
        <div style="overflow-x: auto;">
            <table style="
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                <thead>
                    <tr style="background: {colors['primary']}; color: white;">
    """
    
    # Add headers
    for col in display_data.columns:
        table_html += f"""
                        <th style="
                            padding: 1rem;
                            text-align: left;
                            font-weight: bold;
                        ">{col.replace('_', ' ').title()}</th>
        """
    
    table_html += "</tr></thead><tbody>"
    
    # Add rows
    for i, row in display_data.iterrows():
        bg_color = "#F8F9FA" if i % 2 == 0 else "white"
        table_html += f'<tr style="background: {bg_color};">'
        
        for col in display_data.columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.2f}"
            table_html += f"""
                            <td style="
                                padding: 0.75rem 1rem;
                                border-bottom: 1px solid #E0E0E0;
                            ">{value}</td>
            """
        table_html += "</tr>"
    
    table_html += "</tbody></table>"
    
    if show_more:
        table_html += f"""
        <p style="
            text-align: center; 
            color: {colors['secondary']}; 
            font-style: italic;
            margin-top: 1rem;
        ">
            Showing {max_rows} of {len(data)} rows
        </p>
        """
    
    table_html += "</div></div>"
    
    return table_html

def create_timeline_chart(events: List[Dict], 
                         title: str = "Timeline") -> go.Figure:
    """
    Create timeline chart for farm activities or historical events
    
    Args:
        events: List of dictionaries with date, event, and category
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    colors = get_color_palette("green")
    
    # Convert to DataFrame
    df = pd.DataFrame(events)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create timeline
    fig = px.timeline(
        df, 
        x_start="date", 
        x_end="date",
        y="event",
        color="category",
        title=title
    )
    
    fig.update_layout(
        height=max(300, len(events) * 40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="Activities"
    )
    
    return fig

def create_heatmap(data: pd.DataFrame, 
                  x_col: str, y_col: str, value_col: str,
                  title: str = "Heatmap") -> go.Figure:
    """
    Create heatmap visualization
    
    Args:
        data: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis  
        value_col: Column for values/colors
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    colors = get_color_palette("green")
    
    # Pivot data for heatmap
    heatmap_data = data.pivot(index=y_col, columns=x_col, values=value_col)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Greens',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency with proper symbols and formatting"""
    symbols = {
        "USD": "$",
        "EUR": "€", 
        "GBP": "£",
        "CAD": "C$",
        "AUD": "A$"
    }
    
    symbol = symbols.get(currency, "$")
    
    if abs(amount) >= 1_000_000:
        return f"{symbol}{amount/1_000_000:.1f}M"
    elif abs(amount) >= 1_000:
        return f"{symbol}{amount/1_000:.1f}K"
    else:
        return f"{symbol}{amount:.0f}"

def generate_color_gradient(start_color: str, end_color: str, steps: int) -> List[str]:
    """Generate color gradient between two colors"""
    # Convert hex to RGB
    start_rgb = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
    
    gradient = []
    for i in range(steps):
        ratio = i / (steps - 1) if steps > 1 else 0
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
        gradient.append(f"#{r:02x}{g:02x}{b:02x}")
    
    return gradient

def create_comparison_table(data: List[Dict], 
                          highlight_best: bool = True) -> str:
    """
    Create comparison table with highlighting
    
    Args:
        data: List of dictionaries with comparison data
        highlight_best: Whether to highlight best values
        
    Returns:
        HTML string for comparison table
    """
    colors = get_color_palette("green")
    
    if not data:
        return "<p>No data available for comparison.</p>"
    
    # Get all keys for columns
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    columns = sorted(list(all_keys))
    
    # Find best values for highlighting
    best_values = {}
    if highlight_best:
        for col in columns:
            values = [item.get(col, 0) for item in data if isinstance(item.get(col, 0), (int, float))]
            if values:
                best_values[col] = max(values)
    
    table_html = f"""
    <div style="overflow-x: auto; margin: 1rem 0;">
        <table style="
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        ">
            <thead>
                <tr style="background: {colors['primary']}; color: white;">
    """
    
    for col in columns:
        table_html += f"""
                    <th style="padding: 1rem; text-align: left; font-weight: bold;">
                        {col.replace('_', ' ').title()}
                    </th>
        """
    
    table_html += "</tr></thead><tbody>"
    
    for i, row in enumerate(data):
        bg_color = "#F8F9FA" if i % 2 == 0 else "white"
        table_html += f'<tr style="background: {bg_color};">'
        
        for col in columns:
            value = row.get(col, "")
            cell_style = "padding: 0.75rem 1rem; border-bottom: 1px solid #E0E0E0;"
            
            # Highlight best values
            if highlight_best and col in best_values and isinstance(value, (int, float)):
                if value == best_values[col]:
                    cell_style += f" background: {colors['light']}; font-weight: bold;"
            
            # Format value
            if isinstance(value, float):
                if col in ['roi', 'profit_margin', 'percentage']:
                    display_value = f"{value:.1f}%"
                elif col in ['cost', 'revenue', 'profit', 'price']:
                    display_value = format_currency(value)
                else:
                    display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            table_html += f'<td style="{cell_style}">{display_value}</td>'
        
        table_html += "</tr>"
    
    table_html += "</tbody></table></div>"
    
    return table_html


# Utility functions for common Streamlit patterns
def display_metric_row(metrics: List[Dict], columns: int = 4):
    """Display a row of metrics using Streamlit columns"""
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        col_idx = i % columns
        with cols[col_idx]:
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta'),
                delta_color=metric.get('delta_color', 'normal')
            )

def create_expandable_section(title: str, content: str, 
                            expanded: bool = False):
    """Create expandable content section"""
    with st.expander(title, expanded=expanded):
        st.markdown(content, unsafe_allow_html=True)

def show_loading_spinner(text: str = "Loading..."):
    """Show loading spinner with custom text"""
    return st.spinner(text)

def create_sidebar_filters(filter_options: Dict) -> Dict:
    """Create sidebar filters and return selected values"""
    st.sidebar.markdown("### 🔧 Filters")
    
    selected_filters = {}
    
    for filter_name, options in filter_options.items():
        if options['type'] == 'selectbox':
            selected_filters[filter_name] = st.sidebar.selectbox(
                options['label'],
                options['choices'],
                index=options.get('default_index', 0)
            )
        
        elif options['type'] == 'multiselect':
            selected_filters[filter_name] = st.sidebar.multiselect(
                options['label'],
                options['choices'],
                default=options.get('default', [])
            )
        
        elif options['type'] == 'slider':
            selected_filters[filter_name] = st.sidebar.slider(
                options['label'],
                min_value=options['min'],
                max_value=options['max'],
                value=options.get('default', options['min']),
                step=options.get('step', 1)
            )
        
        elif options['type'] == 'date_range':
            selected_filters[filter_name] = st.sidebar.date_input(
                options['label'],
                value=options.get('default', []),
                min_value=options.get('min_date'),
                max_value=options.get('max_date')
            )
    
    return selected_filters

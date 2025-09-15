
"""
Statistics Visualizer for Precision Farming

Comprehensive data visualization module for creating interactive charts,
graphs, and dashboards for agricultural data analysis.

Features:
- Soil health visualizations
- Crop performance charts
- Economic analysis graphs
- Yield prediction plots
- Risk assessment diagrams
- Interactive dashboards

Author: Precision Farming Team
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import colorsys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorPalette:
    """Color palette management for consistent theming"""
    
    PRECISION_FARMING = {
        'primary': '#4CAF50',
        'secondary': '#2E7D32',
        'accent': '#66BB6A',
        'light': '#E8F5E8',
        'warning': '#FF9800',
        'error': '#F44336',
        'info': '#2196F3',
        'success': '#4CAF50'
    }
    
    SOIL_HEALTH = {
        'excellent': '#2E7D32',
        'good': '#4CAF50',
        'fair': '#FF9800',
        'poor': '#F44336',
        'critical': '#B71C1C'
    }
    
    CROP_TYPES = {
        'wheat': '#8D6E63',
        'corn': '#FFC107',
        'rice': '#4CAF50',
        'soybean': '#795548',
        'cotton': '#E0E0E0'
    }
    
    ECONOMICS = {
        'profit': '#4CAF50',
        'cost': '#F44336',
        'revenue': '#2196F3',
        'break_even': '#FF9800'
    }


class StatsVisualizer:
    """
    Main class for creating agricultural data visualizations
    """
    
    def __init__(self, style: str = 'modern', color_palette: str = 'precision_farming'):
        """
        Initialize the visualizer
        
        Args:
            style: Visualization style ('modern', 'classic', 'minimal')
            color_palette: Color palette to use
        """
        self.style = style
        self.color_palette = color_palette
        self.colors = getattr(ColorPalette, color_palette.upper(), ColorPalette.PRECISION_FARMING)
        
        # Set matplotlib style
        if style == 'modern':
            plt.style.use('seaborn-v0_8-whitegrid')
        elif style == 'minimal':
            plt.style.use('seaborn-v0_8-white')
        else:
            plt.style.use('default')
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        logger.info(f"StatsVisualizer initialized with {style} style")
    
    def create_soil_health_chart(self, soil_data: Dict, chart_type: str = 'radar') -> go.Figure:
        """
        Create soil health visualization
        
        Args:
            soil_data: Dictionary with soil parameters
            chart_type: Type of chart ('radar', 'bar', 'gauge')
            
        Returns:
            Plotly figure object
        """
        if chart_type == 'radar':
            return self._create_soil_radar_chart(soil_data)
        elif chart_type == 'bar':
            return self._create_soil_bar_chart(soil_data)
        elif chart_type == 'gauge':
            return self._create_soil_gauge_chart(soil_data)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _create_soil_radar_chart(self, soil_data: Dict) -> go.Figure:
        """Create radar chart for soil health parameters"""
        
        # Prepare data
        categories = []
        values = []
        
        # Normalize values to 0-100 scale
        normalizers = {
            'ph_level': lambda x: min(100, max(0, 100 - abs(x - 6.5) * 20)),
            'nitrogen_ppm': lambda x: min(100, x * 1.5),
            'phosphorus_ppm': lambda x: min(100, x * 2),
            'potassium_ppm': lambda x: min(100, x * 0.3),
            'organic_matter_percent': lambda x: min(100, x * 25)
        }
        
        for param, value in soil_data.items():
            if param in normalizers:
                normalized_value = normalizers[param](value)
                categories.append(param.replace('_', ' ').title())
                values.append(normalized_value)
        
        # Close the radar chart
        categories += [categories[0]]
        values += [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f'{self.colors["primary"]}40',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(color=self.colors['secondary'], size=8),
            name='Soil Health'
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
                text="Soil Health Profile",
                x=0.5,
                font=dict(size=16, color=self.colors['secondary'])
            ),
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_soil_bar_chart(self, soil_data: Dict) -> go.Figure:
        """Create bar chart for soil parameters"""
        
        parameters = []
        values = []
        colors = []
        
        # Define optimal ranges and colors
        optimal_ranges = {
            'ph_level': (6.0, 7.5, 'pH Level'),
            'nitrogen_ppm': (40, 80, 'Nitrogen (ppm)'),
            'phosphorus_ppm': (25, 50, 'Phosphorus (ppm)'),
            'potassium_ppm': (150, 300, 'Potassium (ppm)'),
            'organic_matter_percent': (3, 6, 'Organic Matter (%)')
        }
        
        for param, value in soil_data.items():
            if param in optimal_ranges:
                min_opt, max_opt, label = optimal_ranges[param]
                parameters.append(label)
                values.append(value)
                
                # Color based on optimal range
                if min_opt <= value <= max_opt:
                    colors.append(self.colors['success'])
                elif value < min_opt * 0.8 or value > max_opt * 1.2:
                    colors.append(self.colors['error'])
                else:
                    colors.append(self.colors['warning'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=parameters,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}" for v in values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Soil Parameter Analysis",
            xaxis_title="Parameters",
            yaxis_title="Values",
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_soil_gauge_chart(self, soil_data: Dict) -> go.Figure:
        """Create gauge chart for overall soil health"""
        
        # Calculate overall health score
        health_score = self._calculate_soil_health_score(soil_data)
        
        # Determine color based on score
        if health_score >= 80:
            gauge_color = self.colors['success']
            status = "Excellent"
        elif health_score >= 65:
            gauge_color = self.colors['primary']
            status = "Good"
        elif health_score >= 50:
            gauge_color = self.colors['warning']
            status = "Fair"
        else:
            gauge_color = self.colors['error']
            status = "Poor"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Soil Health Score<br><span style='font-size:14px'>{status}</span>"},
            delta={'reference': 70, 'position': "top"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"},
                    {'range': [70, 85], 'color': "lightgreen"},
                    {'range': [85, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            paper_bgcolor='white',
            font={'color': self.colors['secondary']}
        )
        
        return fig
    
    def create_crop_comparison_chart(self, crop_data: List[Dict], 
                                   metric: str = 'suitability_score') -> go.Figure:
        """
        Create crop comparison visualization
        
        Args:
            crop_data: List of dictionaries with crop data
            metric: Metric to compare ('suitability_score', 'roi', 'yield', etc.)
            
        Returns:
            Plotly figure object
        """
        if not crop_data:
            raise ValueError("No crop data provided")
        
        # Extract data
        crops = [item.get('name', item.get('crop', 'Unknown')) for item in crop_data]
        values = [item.get(metric, 0) for item in crop_data]
        
        # Sort by value
        sorted_data = sorted(zip(crops, values), key=lambda x: x[1], reverse=True)
        crops, values = zip(*sorted_data)
        
        # Color based on values
        colors = []
        for value in values:
            if metric in ['suitability_score', 'roi']:
                if value >= 80:
                    colors.append(self.colors['success'])
                elif value >= 60:
                    colors.append(self.colors['primary'])
                elif value >= 40:
                    colors.append(self.colors['warning'])
                else:
                    colors.append(self.colors['error'])
            else:
                colors.append(self.colors['primary'])
        
        fig = go.Figure(data=[
            go.Bar(
                y=crops,
                x=values,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}" for v in values],
                textposition='inside',
                textfont=dict(color='white', size=12)
            )
        ])
        
        fig.update_layout(
            title=f"Crop Comparison - {metric.replace('_', ' ').title()}",
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title="Crops",
            height=max(400, len(crops) * 40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_economic_dashboard(self, economic_data: Dict) -> go.Figure:
        """
        Create economic analysis dashboard
        
        Args:
            economic_data: Dictionary with economic metrics
            
        Returns:
            Plotly subplot figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cost Breakdown', 'Revenue vs Costs', 
                'Profitability Analysis', 'Break-even Analysis'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Cost breakdown pie chart
        if 'cost_breakdown' in economic_data:
            costs = economic_data['cost_breakdown']
            fig.add_trace(
                go.Pie(
                    labels=list(costs.keys()),
                    values=list(costs.values()),
                    name="Costs"
                ),
                row=1, col=1
            )
        
        # Revenue vs costs bar chart
        if all(key in economic_data for key in ['total_revenue', 'total_costs']):
            fig.add_trace(
                go.Bar(
                    x=['Revenue', 'Costs'],
                    y=[economic_data['total_revenue'], economic_data['total_costs']],
                    marker_color=[self.colors['success'], self.colors['error']],
                    name="Revenue vs Costs"
                ),
                row=1, col=2
            )
        
        # Profitability over time
        if 'profitability_trend' in economic_data:
            trend_data = economic_data['profitability_trend']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trend_data))),
                    y=trend_data,
                    mode='lines+markers',
                    name="Profitability Trend",
                    line=dict(color=self.colors['primary'])
                ),
                row=2, col=1
            )
        
        # Break-even analysis
        if 'break_even_data' in economic_data:
            be_data = economic_data['break_even_data']
            fig.add_trace(
                go.Scatter(
                    x=be_data.get('quantities', []),
                    y=be_data.get('profits', []),
                    mode='lines',
                    name="Break-even Analysis",
                    line=dict(color=self.colors['info'])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Economic Analysis Dashboard",
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_yield_trend_chart(self, yield_data: pd.DataFrame, 
                               prediction_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create yield trend visualization with predictions
        
        Args:
            yield_data: DataFrame with historical yield data
            prediction_data: Optional DataFrame with predicted yields
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Historical yield data
        if 'year' in yield_data.columns and 'yield' in yield_data.columns:
            fig.add_trace(go.Scatter(
                x=yield_data['year'],
                y=yield_data['yield'],
                mode='lines+markers',
                name='Historical Yield',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=8, color=self.colors['secondary'])
            ))
        
        # Prediction data
        if prediction_data is not None and 'year' in prediction_data.columns:
            fig.add_trace(go.Scatter(
                x=prediction_data['year'],
                y=prediction_data['yield'],
                mode='lines+markers',
                name='Predicted Yield',
                line=dict(color=self.colors['info'], width=3, dash='dash'),
                marker=dict(size=8, color=self.colors['info'])
            ))
            
            # Add confidence interval if available
            if 'upper_bound' in prediction_data.columns and 'lower_bound' in prediction_data.columns:
                fig.add_trace(go.Scatter(
                    x=prediction_data['year'],
                    y=prediction_data['upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=prediction_data['year'],
                    y=prediction_data['lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'{self.colors["info"]}30',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title="Yield Trends Over Time",
            xaxis_title="Year",
            yaxis_title="Yield (tonnes/ha)",
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(x=0, y=1)
        )
        
        return fig
    
    def create_risk_assessment_chart(self, risk_data: Dict) -> go.Figure:
        """
        Create risk assessment visualization
        
        Args:
            risk_data: Dictionary with risk factors and scores
            
        Returns:
            Plotly figure object
        """
        # Create risk matrix
        risk_factors = list(risk_data.keys())
        risk_scores = list(risk_data.values())
        
        # Color based on risk level
        colors = []
        for score in risk_scores:
            if score < 20:
                colors.append(self.colors['success'])
            elif score < 40:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['error'])
        
        fig = go.Figure(data=[
            go.Bar(
                y=risk_factors,
                x=risk_scores,
                orientation='h',
                marker_color=colors,
                text=[f"{score:.1f}%" for score in risk_scores],
                textposition='inside'
            )
        ])
        
        fig.update_layout(
            title="Risk Assessment Profile",
            xaxis_title="Risk Score (%)",
            yaxis_title="Risk Factors",
            height=max(400, len(risk_factors) * 40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_weather_impact_chart(self, weather_data: pd.DataFrame, 
                                  yield_data: pd.DataFrame) -> go.Figure:
        """
        Create weather impact visualization
        
        Args:
            weather_data: DataFrame with weather parameters
            yield_data: DataFrame with corresponding yield data
            
        Returns:
            Plotly figure object
        """
        # Create subplots for multiple weather factors
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Temperature vs Yield', 'Rainfall vs Yield',
                'Weather Trends', 'Yield Variability'
            )
        )
        
        # Temperature vs Yield scatter
        if 'temperature' in weather_data.columns and 'yield' in yield_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=weather_data['temperature'],
                    y=yield_data['yield'],
                    mode='markers',
                    name='Temp vs Yield',
                    marker=dict(color=self.colors['error'], size=8)
                ),
                row=1, col=1
            )
        
        # Rainfall vs Yield scatter
        if 'rainfall' in weather_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=weather_data['rainfall'],
                    y=yield_data['yield'],
                    mode='markers',
                    name='Rain vs Yield',
                    marker=dict(color=self.colors['info'], size=8)
                ),
                row=1, col=2
            )
        
        # Weather trends over time
        if 'year' in weather_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=weather_data['year'],
                    y=weather_data.get('temperature', []),
                    name='Temperature',
                    line=dict(color=self.colors['error'])
                ),
                row=2, col=1
            )
            
            if 'rainfall' in weather_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=weather_data['year'],
                        y=weather_data['rainfall'],
                        name='Rainfall',
                        yaxis='y2',
                        line=dict(color=self.colors['info'])
                    ),
                    row=2, col=1
                )
        
        # Yield variability
        if 'year' in yield_data.columns and 'yield' in yield_data.columns:
            rolling_std = yield_data['yield'].rolling(window=3).std()
            fig.add_trace(
                go.Scatter(
                    x=yield_data['year'],
                    y=rolling_std,
                    name='Yield Variability',
                    line=dict(color=self.colors['warning'])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Weather Impact Analysis",
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_nutrient_heatmap(self, field_data: pd.DataFrame) -> go.Figure:
        """
        Create nutrient distribution heatmap
        
        Args:
            field_data: DataFrame with spatial nutrient data
            
        Returns:
            Plotly figure object
        """
        if not all(col in field_data.columns for col in ['x', 'y', 'nutrient_level']):
            raise ValueError("Field data must contain 'x', 'y', and 'nutrient_level' columns")
        
        # Create pivot table for heatmap
        pivot_data = field_data.pivot(index='y', columns='x', values='nutrient_level')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            colorbar=dict(title="Nutrient Level")
        ))
        
        fig.update_layout(
            title="Field Nutrient Distribution",
            xaxis_title="Field X Coordinate",
            yaxis_title="Field Y Coordinate",
            height=500,
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_crop_rotation_timeline(self, rotation_data: List[Dict]) -> go.Figure:
        """
        Create crop rotation timeline
        
        Args:
            rotation_data: List of dictionaries with rotation information
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for i, rotation in enumerate(rotation_data):
            start_date = rotation.get('start_date')
            end_date = rotation.get('end_date')
            crop_name = rotation.get('crop', 'Unknown')
            
            fig.add_trace(go.Scatter(
                x=[start_date, end_date],
                y=[i, i],
                mode='lines+markers',
                name=crop_name,
                line=dict(width=10),
                marker=dict(size=8)
            ))
            
            # Add text annotation
            mid_date = pd.to_datetime(start_date) + (pd.to_datetime(end_date) - pd.to_datetime(start_date)) / 2
            fig.add_annotation(
                x=mid_date,
                y=i,
                text=crop_name,
                showarrow=False,
                font=dict(color='white', size=12)
            )
        
        fig.update_layout(
            title="Crop Rotation Timeline",
            xaxis_title="Date",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(rotation_data))),
                ticktext=[f"Season {i+1}" for i in range(len(rotation_data))]
            ),
            height=400,
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_profitability_matrix(self, crops: List[str], 
                                  yield_scenarios: List[float],
                                  price_scenarios: List[float],
                                  cost_per_ha: float) -> go.Figure:
        """
        Create profitability scenario matrix
        
        Args:
            crops: List of crop names
            yield_scenarios: List of yield scenarios
            price_scenarios: List of price scenarios
            cost_per_ha: Cost per hectare
            
        Returns:
            Plotly figure object
        """
        # Calculate profitability matrix
        profit_matrix = []
        for yield_val in yield_scenarios:
            row = []
            for price in price_scenarios:
                profit = yield_val * price - cost_per_ha
                row.append(profit)
            profit_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=profit_matrix,
            x=[f"${p:.0f}/t" for p in price_scenarios],
            y=[f"{y:.1f} t/ha" for y in yield_scenarios],
            colorscale='RdYlGn',
            colorbar=dict(title="Profit ($/ha)")
        ))
        
        # Add text annotations
        for i, yield_val in enumerate(yield_scenarios):
            for j, price in enumerate(price_scenarios):
                profit = profit_matrix[i][j]
                fig.add_annotation(
                    x=j, y=i,
                    text=f"${profit:.0f}",
                    showarrow=False,
                    font=dict(color='white' if abs(profit) > max(abs(min(min(profit_matrix))), abs(max(max(profit_matrix)))) * 0.5 else 'black')
                )
        
        fig.update_layout(
            title="Profitability Scenario Matrix",
            xaxis_title="Price Scenarios",
            yaxis_title="Yield Scenarios",
            height=500,
            paper_bgcolor='white'
        )
        
        return fig
    
    def _calculate_soil_health_score(self, soil_data: Dict) -> float:
        """Calculate overall soil health score"""
        scores = []
        
        # pH score
        ph = soil_data.get('ph_level', 7.0)
        if 6.0 <= ph <= 7.5:
            ph_score = 100
        elif 5.5 <= ph <= 8.0:
            ph_score = 80 - abs(ph - 6.75) * 10
        else:
            ph_score = 40
        scores.append(ph_score)
        
        # Nutrient scores
        nitrogen = soil_data.get('nitrogen_ppm', 0)
        nitrogen_score = min(100, nitrogen * 2)
        scores.append(nitrogen_score)
        
        phosphorus = soil_data.get('phosphorus_ppm', 0)
        phosphorus_score = min(100, phosphorus * 2.5)
        scores.append(phosphorus_score)
        
        potassium = soil_data.get('potassium_ppm', 0)
        potassium_score = min(100, potassium * 0.4)
        scores.append(potassium_score)
        
        # Organic matter score
        om = soil_data.get('organic_matter_percent', 0)
        om_score = min(100, om * 25)
        scores.append(om_score)
        
        return np.mean(scores)
    
    def export_chart(self, fig: go.Figure, filename: str, 
                    format: str = 'html', width: int = 1200, height: int = 800):
        """
        Export chart to file
        
        Args:
            fig: Plotly figure to export
            filename: Output filename
            format: Export format ('html', 'png', 'pdf', 'svg')
            width: Image width for static exports
            height: Image height for static exports
        """
        try:
            if format.lower() == 'html':
                fig.write_html(filename)
            elif format.lower() == 'png':
                fig.write_image(filename, format='png', width=width, height=height)
            elif format.lower() == 'pdf':
                fig.write_image(filename, format='pdf', width=width, height=height)
            elif format.lower() == 'svg':
                fig.write_image(filename, format='svg', width=width, height=height)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Chart exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export chart: {e}")
            raise
    
    def create_dashboard_layout(self, figures: List[go.Figure], 
                              layout: str = 'grid') -> go.Figure:
        """
        Combine multiple figures into a dashboard layout
        
        Args:
            figures: List of Plotly figures
            layout: Layout style ('grid', 'vertical', 'horizontal')
            
        Returns:
            Combined dashboard figure
        """
        if layout == 'grid':
            rows = int(np.ceil(np.sqrt(len(figures))))
            cols = int(np.ceil(len(figures) / rows))
        elif layout == 'vertical':
            rows = len(figures)
            cols = 1
        elif layout == 'horizontal':
            rows = 1
            cols = len(figures)
        else:
            raise ValueError(f"Unsupported layout: {layout}")
        
        # Create subplot structure
        subplot_titles = [fig.layout.title.text if fig.layout.title else f"Chart {i+1}" 
                         for i, fig in enumerate(figures)]
        
        dashboard = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles
        )
        
        # Add traces from each figure
        for idx, fig in enumerate(figures):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            for trace in fig.data:
                dashboard.add_trace(trace, row=row, col=col)
        
        dashboard.update_layout(
            height=300 * rows,
            title_text="Precision Farming Dashboard",
            paper_bgcolor='white'
        )
        
        return dashboard


# Utility functions for creating specific chart types
def create_soil_health_chart(soil_data: Dict, chart_type: str = 'radar') -> go.Figure:
    """Convenience function for creating soil health charts"""
    visualizer = StatsVisualizer()
    return visualizer.create_soil_health_chart(soil_data, chart_type)


def create_crop_comparison_chart(crop_data: List[Dict], 
                               metric: str = 'suitability_score') -> go.Figure:
    """Convenience function for creating crop comparison charts"""
    visualizer = StatsVisualizer()
    return visualizer.create_crop_comparison_chart(crop_data, metric)


def create_economic_dashboard(economic_data: Dict) -> go.Figure:
    """Convenience function for creating economic dashboards"""
    visualizer = StatsVisualizer()
    return visualizer.create_economic_dashboard(economic_data)


def create_yield_trend_chart(yield_data: pd.DataFrame, 
                           prediction_data: Optional[pd.DataFrame] = None) -> go.Figure:
    """Convenience function for creating yield trend charts"""
    visualizer = StatsVisualizer()
    return visualizer.create_yield_trend_chart(yield_data, prediction_data)


if __name__ == "__main__":
    # Example usage
    visualizer = StatsVisualizer(style='modern', color_palette='precision_farming')
    
    # Sample soil data
    soil_data = {
        'ph_level': 6.5,
        'nitrogen_ppm': 45,
        'phosphorus_ppm': 28,
        'potassium_ppm': 185,
        'organic_matter_percent': 3.2
    }
    
    # Create soil health chart
    fig = visualizer.create_soil_health_chart(soil_data, chart_type='radar')
    fig.show()
    
    # Sample crop comparison data
    crop_data = [
        {'name': 'Wheat', 'suitability_score': 85, 'roi': 15.2},
        {'name': 'Corn', 'suitability_score': 72, 'roi': 18.5},
        {'name': 'Rice', 'suitability_score': 68, 'roi': 12.8},
        {'name': 'Soybean', 'suitability_score': 79, 'roi': 22.1}
    ]
    
    # Create crop comparison chart
    comparison_fig = visualizer.create_crop_comparison_chart(crop_data, 'suitability_score')
    comparison_fig.show()
    
    logger.info("Example visualizations created successfully")

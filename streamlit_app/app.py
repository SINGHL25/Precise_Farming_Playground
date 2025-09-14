
"""
Precision Farming Dashboard - Main Application

A comprehensive web-based platform for precision farming analysis,
crop recommendations, economic modeling, and agricultural insights.

Author: Precision Farming Team
Date: 2024
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure Streamlit page
st.set_page_config(
    page_title="Precision Farming Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/precision-farming/docs',
        'Report a bug': 'https://github.com/precision-farming/issues',
        'About': """
        # Precision Farming Dashboard
        
        Empowering farmers with data-driven insights for sustainable agriculture.
        
        **Features:**
        - Soil analysis and health assessment
        - AI-powered crop recommendations
        - Economic modeling and ROI analysis
        - Yield prediction and planning
        - Farmer feedback and community insights
        
        Built with ❤️ for the global farming community.
        """
    }
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #2E7D32, #4CAF50, #66BB6A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #E8F5E8, #F1F8E9);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #E0E0E0;
    }
    
    .success-message {
        background: linear-gradient(90deg, #C8E6C9, #E8F5E8);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(90deg, #FFF3E0, #FFF8E1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #F5F5F5;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_demo_data():
    """Load sample data for demonstration"""
    # Sample farm data
    farm_data = {
        'total_area': 125.5,
        'fields': 8,
        'soil_samples': 15,
        'avg_health_score': 78.2,
        'last_analysis': '2024-03-15'
    }
    
    # Sample recent activities
    activities = [
        {'date': '2024-03-15', 'action': 'Soil Analysis', 'field': 'North Field', 'status': 'Completed'},
        {'date': '2024-03-12', 'action': 'Crop Recommendation', 'field': 'South Field', 'status': 'Completed'},
        {'date': '2024-03-10', 'action': 'Economic Analysis', 'field': 'East Field', 'status': 'In Progress'},
        {'date': '2024-03-08', 'action': 'Yield Prediction', 'field': 'West Field', 'status': 'Scheduled'}
    ]
    
    # Sample health trends
    health_trend = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=12, freq='W'),
        'health_score': [72, 74, 76, 75, 78, 80, 79, 81, 78, 82, 81, 78]
    })
    
    return farm_data, activities, health_trend

def create_quick_stats_cards(farm_data):
    """Create quick statistics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #2E7D32;">🏡 Farm Area</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #1B5E20;">
                {:.1f} ha
            </p>
            <p style="margin: 0; color: #4CAF50; font-size: 0.9rem;">
                Across {} fields
            </p>
        </div>
        """.format(farm_data['total_area'], farm_data['fields']), 
        unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #2E7D32;">📊 Soil Health</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #1B5E20;">
                {:.1f}/100
            </p>
            <p style="margin: 0; color: #4CAF50; font-size: 0.9rem;">
                Above average
            </p>
        </div>
        """.format(farm_data['avg_health_score']), 
        unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #2E7D32;">🧪 Samples</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #1B5E20;">
                {}
            </p>
            <p style="margin: 0; color: #4CAF50; font-size: 0.9rem;">
                Analyzed this season
            </p>
        </div>
        """.format(farm_data['soil_samples']), 
        unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #2E7D32;">📅 Last Update</h3>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #1B5E20;">
                {}
            </p>
            <p style="margin: 0; color: #4CAF50; font-size: 0.9rem;">
                Recent analysis
            </p>
        </div>
        """.format(farm_data['last_analysis']), 
        unsafe_allow_html=True)

def create_health_trend_chart(health_data):
    """Create soil health trend chart"""
    fig = px.line(
        health_data, 
        x='date', 
        y='health_score',
        title='Soil Health Trend Over Time',
        line_shape='spline'
    )
    
    fig.update_traces(
        line=dict(color='#4CAF50', width=3),
        mode='lines+markers',
        marker=dict(size=8, color='#2E7D32')
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Date",
        yaxis_title="Health Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_feature_overview():
    """Create feature overview section"""
    st.markdown("## 🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #2E7D32; margin-top: 0;">🧪 Soil Analysis</h3>
            <ul>
                <li>Comprehensive soil health assessment</li>
                <li>Nutrient analysis (N-P-K + micronutrients)</li>
                <li>pH and texture classification</li>
                <li>Organic matter evaluation</li>
                <li>Actionable recommendations</li>
            </ul>
            <a href="/1_Soil_Analyzer" target="_self" style="
                background: #4CAF50; 
                color: white; 
                padding: 0.5rem 1rem; 
                border-radius: 5px; 
                text-decoration: none;
                display: inline-block;
                margin-top: 1rem;
            ">Start Analysis →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #2E7D32; margin-top: 0;">🌱 Crop Recommendations</h3>
            <ul>
                <li>AI-powered crop matching</li>
                <li>Climate data integration</li>
                <li>Economic viability scoring</li>
                <li>Risk assessment analysis</li>
                <li>Seasonal planning guidance</li>
            </ul>
            <a href="/2_Crop_Recommender" target="_self" style="
                background: #4CAF50; 
                color: white; 
                padding: 0.5rem 1rem; 
                border-radius: 5px; 
                text-decoration: none;
                display: inline-block;
                margin-top: 1rem;
            ">Get Recommendations →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #2E7D32; margin-top: 0;">💰 Economic Analysis</h3>
            <ul>
                <li>Cost-benefit calculations</li>
                <li>ROI projections</li>
                <li>Market price integration</li>
                <li>Break-even analysis</li>
                <li>Profit optimization</li>
            </ul>
            <a href="/3_Economics_Dashboard" target="_self" style="
                background: #4CAF50; 
                color: white; 
                padding: 0.5rem 1rem; 
                border-radius: 5px; 
                text-decoration: none;
                display: inline-block;
                margin-top: 1rem;
            ">View Economics →</a>
        </div>
        """, unsafe_allow_html=True)

def create_recent_activities(activities):
    """Create recent activities section"""
    st.markdown("## 📋 Recent Activities")
    
    for activity in activities:
        status_color = {
            'Completed': '#4CAF50',
            'In Progress': '#FF9800', 
            'Scheduled': '#2196F3'
        }
        
        status_icon = {
            'Completed': '✅',
            'In Progress': '⏳',
            'Scheduled': '📅'
        }
        
        st.markdown(f"""
        <div style="
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid {status_color.get(activity['status'], '#757575')};
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #1B5E20;">{activity['action']}</strong> - {activity['field']}
                    <br>
                    <small style="color: #666;">{activity['date']}</small>
                </div>
                <div style="color: {status_color.get(activity['status'], '#757575')};">
                    {status_icon.get(activity['status'], '○')} {activity['status']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🌾 Precision Farming Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <p style="font-size: 1.2rem; margin: 0;">
            Empowering farmers with data-driven insights for sustainable agriculture
        </p>
        <p style="margin: 0.5rem 0;">
            Welcome back! Here's your farm overview and latest insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load demo data
    farm_data, activities, health_trend = load_demo_data()
    
    # Quick stats
    create_quick_stats_cards(farm_data)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Health trend chart
        st.plotly_chart(
            create_health_trend_chart(health_trend), 
            use_container_width=True
        )
        
        # Feature overview
        create_feature_overview()
    
    with col2:
        # Recent activities
        create_recent_activities(activities)
        
        # Quick tips
        st.markdown("## 💡 Quick Tips")
        st.markdown("""
        <div class="success-message">
            <strong>🌱 Spring Planning</strong><br>
            Start soil testing 6-8 weeks before planting for optimal crop selection.
        </div>
        
        <div class="warning-message">
            <strong>⚠️ Weather Alert</strong><br>
            Check 7-day forecast before fertilizer application to avoid nutrient loss.
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation help
        st.markdown("## 🧭 Quick Navigation")
        st.markdown("""
        **Getting Started:**
        1. 📤 Upload your soil data
        2. 🌾 Get crop recommendations  
        3. 💰 Analyze economics
        4. 📈 Predict yields
        5. 📝 Share feedback
        
        **Need Help?** Check our documentation or contact support.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>
            Built with ❤️ for sustainable agriculture | 
            <a href="https://github.com/precision-farming" style="color: #4CAF50;">Open Source</a> | 
            <a href="/docs" style="color: #4CAF50;">Documentation</a> | 
            <a href="/support" style="color: #4CAF50;">Support</a>
        </p>
        <p style="font-size: 0.9rem;">
            © 2024 Precision Farming Team - Transforming Agriculture Through Technology
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

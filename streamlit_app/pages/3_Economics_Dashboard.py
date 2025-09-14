
"""
Economics Dashboard Page

Comprehensive economic analysis tool for farm profitability assessment,
cost-benefit analysis, ROI calculations, and financial planning.

Features:
- Detailed cost breakdowns
- Revenue projections
- Profitability analysis
- Scenario modeling
- Market price integration
- Risk assessment

Author: Precision Farming Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.visual_helpers import (
    create_metric_card, create_comparison_table, format_currency,
    get_color_palette, create_alert_box
)

# Page configuration
st.set_page_config(
    page_title="Economics Dashboard - Precision Farming",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .economics-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    .profit-positive { color: #2E7D32; font-weight: bold; }
    .profit-negative { color: #D32F2F; font-weight: bold; }
    .break-even { color: #FF9800; font-weight: bold; }
    
    .cost-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        border-bottom: 1px solid #E0E0E0;
    }
    
    .cost-item:last-child {
        border-bottom: none;
        font-weight: bold;
        background: #F5F5F5;
    }
    
    .scenario-section {
        background: linear-gradient(135deg, #F8F9FA, #E8F5E8);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #E0E0E0;
    }
    
    .roi-excellent { background: #E8F5E8; color: #2E7D32; }
    .roi-good { background: #F1F8E9; color: #388E3C; }
    .roi-fair { background: #FFF8E1; color: #F57C00; }
    .roi-poor { background: #FFEBEE; color: #D32F2F; }
</style>
""", unsafe_allow_html=True)

# Default crop economic data
DEFAULT_CROP_DATA = {
    'wheat': {
        'name': 'Wheat',
        'expected_yield': 3.5,
        'market_price': 250,
        'seed_cost': 150,
        'fertilizer_cost': 200,
        'labor_cost': 300,
        'machinery_cost': 250,
        'other_costs': 100,
        'growing_days': 120,
        'season': 'Winter'
    },
    'corn': {
        'name': 'Corn',
        'expected_yield': 8.5,
        'market_price': 200,
        'seed_cost': 200,
        'fertilizer_cost': 350,
        'labor_cost': 400,
        'machinery_cost': 300,
        'other_costs': 150,
        'growing_days': 100,
        'season': 'Summer'
    },
    'rice': {
        'name': 'Rice',
        'expected_yield': 4.5,
        'market_price': 300,
        'seed_cost': 180,
        'fertilizer_cost': 280,
        'labor_cost': 450,
        'machinery_cost': 200,
        'other_costs': 120,
        'growing_days': 140,
        'season': 'Monsoon'
    },
    'soybean': {
        'name': 'Soybean',
        'expected_yield': 2.8,
        'market_price': 400,
        'seed_cost': 220,
        'fertilizer_cost': 150,
        'labor_cost': 350,
        'machinery_cost': 280,
        'other_costs': 100,
        'growing_days': 110,
        'season': 'Summer'
    },
    'cotton': {
        'name': 'Cotton',
        'expected_yield': 1.2,
        'market_price': 1500,
        'seed_cost': 300,
        'fertilizer_cost': 400,
        'labor_cost': 600,
        'machinery_cost': 350,
        'other_costs': 200,
        'growing_days': 180,
        'season': 'Summer'
    }
}

def load_crop_data_from_recommendations():
    """Load crop data from previous recommendations if available"""
    if 'selected_crop_for_economics' in st.session_state:
        rec = st.session_state.selected_crop_for_economics
        crop_profile = rec.crop_profile
        
        return {
            'name': crop_profile.name,
            'expected_yield': crop_profile.expected_yield_per_ha,
            'market_price': crop_profile.market_price_per_tonne,
            'seed_cost': crop_profile.seed_cost_per_ha,
            'fertilizer_cost': crop_profile.fertilizer_cost_per_ha,
            'labor_cost': crop_profile.labor_cost_per_ha,
            'machinery_cost': crop_profile.machinery_cost_per_ha,
            'other_costs': 100,  # Default
            'growing_days': crop_profile.days_to_maturity,
            'season': crop_profile.growing_season.value
        }
    return None

def create_economic_input_form():
    """Create form for economic parameters input"""
    st.markdown("## 🔧 Economic Parameters")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📊 From Crop Recommendations", "✍️ Manual Input", "📋 Select Crop Type"],
        horizontal=True
    )
    
    crop_data = None
    
    if input_method == "📊 From Crop Recommendations":
        crop_data = load_crop_data_from_recommendations()
        if crop_data:
            st.success(f"✅ Loaded data for {crop_data['name']} from previous recommendations")
        else:
            st.info("No crop recommendation data found. Please generate recommendations first or use manual input.")
    
    elif input_method == "📋 Select Crop Type":
        selected_crop = st.selectbox(
            "Select crop type:",
            list(DEFAULT_CROP_DATA.keys()),
            format_func=lambda x: DEFAULT_CROP_DATA[x]['name']
        )
        crop_data = DEFAULT_CROP_DATA[selected_crop].copy()
        st.info(f"Using default parameters for {crop_data['name']}")
    
    elif input_method == "✍️ Manual Input":
        st.markdown("### Enter Custom Economic Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Crop Information**")
            crop_name = st.text_input("Crop Name", value="Custom Crop")
            expected_yield = st.number_input("Expected Yield (tonnes/ha)", 0.1, 20.0, 3.5, 0.1)
            market_price = st.number_input("Market Price ($/tonne)", 50.0, 2000.0, 300.0, 10.0)
        
        with col2:
            st.markdown("**Production Costs ($/ha)**")
            seed_cost = st.number_input("Seed Cost", 0.0, 1000.0, 150.0, 10.0)
            fertilizer_cost = st.number_input("Fertilizer Cost", 0.0, 1000.0, 200.0, 10.0)
            labor_cost = st.number_input("Labor Cost", 0.0, 1000.0, 300.0, 10.0)
        
        with col3:
            st.markdown("**Additional Costs ($/ha)**")
            machinery_cost = st.number_input("Machinery Cost", 0.0, 1000.0, 250.0, 10.0)
            other_costs = st.number_input("Other Costs", 0.0, 500.0, 100.0, 10.0)
            growing_days = st.number_input("Growing Period (days)", 60, 300, 120, 1)
        
        crop_data = {
            'name': crop_name,
            'expected_yield': expected_yield,
            'market_price': market_price,
            'seed_cost': seed_cost,
            'fertilizer_cost': fertilizer_cost,
            'labor_cost': labor_cost,
            'machinery_cost': machinery_cost,
            'other_costs': other_costs,
            'growing_days': growing_days,
            'season': 'Custom'
        }
    
    # Farm parameters
    if crop_data:
        st.markdown("### 🏡 Farm Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            farm_size

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
            farm_size = st.number_input("Farm Size (hectares)", 1.0, 1000.0, 25.0, 0.5)
        
        with col2:
            location = st.text_input("Location", "Iowa, USA")
        
        with col3:
            planning_years = st.number_input("Planning Period (years)", 1, 10, 5, 1)
        
        return crop_data, farm_size, location, planning_years
    
    return None, None, None, None

def calculate_economics(crop_data, farm_size):
    """Calculate comprehensive economic metrics"""
    if not crop_data:
        return None
    
    # Per hectare calculations
    total_cost_per_ha = (
        crop_data['seed_cost'] + 
        crop_data['fertilizer_cost'] + 
        crop_data['labor_cost'] + 
        crop_data['machinery_cost'] + 
        crop_data['other_costs']
    )
    
    revenue_per_ha = crop_data['expected_yield'] * crop_data['market_price']
    profit_per_ha = revenue_per_ha - total_cost_per_ha
    
    # Total farm calculations
    total_cost = total_cost_per_ha * farm_size
    total_revenue = revenue_per_ha * farm_size
    total_profit = profit_per_ha * farm_size
    
    # Financial ratios
    roi_percent = (profit_per_ha / total_cost_per_ha * 100) if total_cost_per_ha > 0 else 0
    profit_margin = (profit_per_ha / revenue_per_ha * 100) if revenue_per_ha > 0 else 0
    break_even_price = total_cost_per_ha / crop_data['expected_yield'] if crop_data['expected_yield'] > 0 else 0
    break_even_yield = total_cost_per_ha / crop_data['market_price'] if crop_data['market_price'] > 0 else 0
    
    return {
        'per_hectare': {
            'total_cost': total_cost_per_ha,
            'revenue': revenue_per_ha,
            'profit': profit_per_ha,
            'roi_percent': roi_percent,
            'profit_margin': profit_margin
        },
        'total_farm': {
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'farm_size': farm_size
        },
        'breakeven': {
            'price': break_even_price,
            'yield': break_even_yield
        },
        'cost_breakdown': {
            'seed': crop_data['seed_cost'],
            'fertilizer': crop_data['fertilizer_cost'],
            'labor': crop_data['labor_cost'],
            'machinery': crop_data['machinery_cost'],
            'other': crop_data['other_costs']
        }
    }

def display_key_metrics(economics):
    """Display key economic metrics"""
    if not economics:
        return
    
    st.markdown("### 📊 Key Economic Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit = economics['per_hectare']['profit']
        delta_text = "Profitable" if profit > 0 else "Loss" if profit < 0 else "Break-even"
        delta_color = "normal" if profit > 0 else "inverse" if profit < 0 else "off"
        
        st.metric(
            "Profit per Hectare",
            format_currency(profit),
            delta_text,
            delta_color=delta_color
        )
    
    with col2:
        roi = economics['per_hectare']['roi_percent']
        roi_class = "excellent" if roi >= 25 else "good" if roi >= 15 else "fair" if roi >= 5 else "poor"
        
        st.metric(
            "Return on Investment",
            f"{roi:.1f}%",
            "Excellent" if roi >= 25 else "Good" if roi >= 15 else "Fair" if roi >= 5 else "Poor"
        )
    
    with col3:
        st.metric(
            "Total Farm Profit",
            format_currency(economics['total_farm']['total_profit']),
            f"For {economics['total_farm']['farm_size']:.1f} ha"
        )
    
    with col4:
        margin = economics['per_hectare']['profit_margin']
        st.metric(
            "Profit Margin",
            f"{margin:.1f}%",
            "Healthy" if margin >= 20 else "Moderate" if margin >= 10 else "Low"
        )

def create_cost_breakdown_chart(economics):
    """Create cost breakdown visualization"""
    if not economics:
        return None
    
    costs = economics['cost_breakdown']
    labels = list(costs.keys())
    values = list(costs.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=[label.title() for label in labels],
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Cost Breakdown per Hectare",
        height=400,
        showlegend=True,
        annotations=[
            dict(
                text=f"Total<br>{format_currency(sum(values))}",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )
        ]
    )
    
    return fig

def create_profitability_analysis(economics, crop_data):
    """Create profitability analysis charts"""
    if not economics:
        return None, None
    
    # Price sensitivity analysis
    price_range = np.linspace(
        crop_data['market_price'] * 0.7,
        crop_data['market_price'] * 1.3,
        20
    )
    
    profits_price = []
    for price in price_range:
        revenue = crop_data['expected_yield'] * price
        profit = revenue - economics['per_hectare']['total_cost']
        profits_price.append(profit)
    
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=price_range,
        y=profits_price,
        mode='lines+markers',
        name='Profit',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=6)
    ))
    
    # Add break-even line
    price_fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Break-even"
    )
    
    # Add current price marker
    current_profit = crop_data['expected_yield'] * crop_data['market_price'] - economics['per_hectare']['total_cost']
    price_fig.add_trace(go.Scatter(
        x=[crop_data['market_price']],
        y=[current_profit],
        mode='markers',
        name='Current Price',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    price_fig.update_layout(
        title="Price Sensitivity Analysis",
        xaxis_title="Market Price ($/tonne)",
        yaxis_title="Profit per Hectare ($)",
        height=400,
        showlegend=True
    )
    
    # Yield sensitivity analysis
    yield_range = np.linspace(
        crop_data['expected_yield'] * 0.6,
        crop_data['expected_yield'] * 1.4,
        20
    )
    
    profits_yield = []
    for yield_val in yield_range:
        revenue = yield_val * crop_data['market_price']
        profit = revenue - economics['per_hectare']['total_cost']
        profits_yield.append(profit)
    
    yield_fig = go.Figure()
    yield_fig.add_trace(go.Scatter(
        x=yield_range,
        y=profits_yield,
        mode='lines+markers',
        name='Profit',
        line=dict(color='#2196F3', width=3),
        marker=dict(size=6)
    ))
    
    # Add break-even line
    yield_fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Break-even"
    )
    
    # Add current yield marker
    yield_fig.add_trace(go.Scatter(
        x=[crop_data['expected_yield']],
        y=[current_profit],
        mode='markers',
        name='Expected Yield',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    yield_fig.update_layout(
        title="Yield Sensitivity Analysis",
        xaxis_title="Yield (tonnes/ha)",
        yaxis_title="Profit per Hectare ($)",
        height=400,
        showlegend=True
    )
    
    return price_fig, yield_fig

def create_scenario_analysis(crop_data, economics):
    """Create scenario analysis table"""
    if not economics or not crop_data:
        return None
    
    scenarios = [
        {
            'name': 'Pessimistic',
            'yield_factor': 0.8,
            'price_factor': 0.9,
            'cost_factor': 1.1,
            'description': 'Lower yield, lower prices, higher costs'
        },
        {
            'name': 'Expected',
            'yield_factor': 1.0,
            'price_factor': 1.0,
            'cost_factor': 1.0,
            'description': 'Base case scenario'
        },
        {
            'name': 'Optimistic',
            'yield_factor': 1.2,
            'price_factor': 1.1,
            'cost_factor': 0.95,
            'description': 'Higher yield, better prices, lower costs'
        }
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        adjusted_yield = crop_data['expected_yield'] * scenario['yield_factor']
        adjusted_price = crop_data['market_price'] * scenario['price_factor']
        adjusted_cost = economics['per_hectare']['total_cost'] * scenario['cost_factor']
        
        revenue = adjusted_yield * adjusted_price
        profit = revenue - adjusted_cost
        roi = (profit / adjusted_cost * 100) if adjusted_cost > 0 else 0
        
        scenario_results.append({
            'scenario': scenario['name'],
            'yield_tonnes_ha': adjusted_yield,
            'price_per_tonne': adjusted_price,
            'cost_per_ha': adjusted_cost,
            'revenue_per_ha': revenue,
            'profit_per_ha': profit,
            'roi_percent': roi,
            'description': scenario['description']
        })
    
    return scenario_results

def display_scenario_analysis(scenario_results, farm_size):
    """Display scenario analysis results"""
    if not scenario_results:
        return
    
    st.markdown("### 📈 Scenario Analysis")
    
    # Create comparison chart
    scenarios = [s['scenario'] for s in scenario_results]
    profits = [s['profit_per_ha'] for s in scenario_results]
    rois = [s['roi_percent'] for s in scenario_results]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Profit per Hectare', 'Return on Investment'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#f44336', '#ff9800', '#4caf50']
    
    fig.add_trace(
        go.Bar(x=scenarios, y=profits, name='Profit', marker_color=colors),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=scenarios, y=rois, name='ROI', marker_color=colors),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(title_text="Profit ($)", row=1, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed scenario table
    st.markdown("#### Detailed Scenario Comparison")
    
    # Convert to display format
    display_data = []
    for result in scenario_results:
        display_data.append({
            'Scenario': result['scenario'],
            'Yield (t/ha)': f"{result['yield_tonnes_ha']:.1f}",
            'Price ($/t)': f"${result['price_per_tonne']:.0f}",
            'Cost ($/ha)': format_currency(result['cost_per_ha']),
            'Revenue ($/ha)': format_currency(result['revenue_per_ha']),
            'Profit ($/ha)': format_currency(result['profit_per_ha']),
            'Total Profit': format_currency(result['profit_per_ha'] * farm_size),
            'ROI (%)': f"{result['roi_percent']:.1f}%"
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True)

def create_break_even_analysis(economics, crop_data):
    """Create break-even analysis"""
    if not economics:
        return
    
    st.markdown("### ⚖️ Break-Even Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Break-Even Price:**")
        be_price = economics['breakeven']['price']
        current_price = crop_data['market_price']
        price_margin = ((current_price - be_price) / be_price * 100) if be_price > 0 else 0
        
        st.metric(
            "Minimum Price Required",
            format_currency(be_price) + "/tonne",
            f"{price_margin:+.1f}% margin" if price_margin != 0 else "At break-even"
        )
        
        if current_price > be_price:
            st.success(f"✅ Current price (${current_price:.0f}) is ${current_price - be_price:.0f} above break-even")
        else:
            st.warning(f"⚠️ Current price (${current_price:.0f}) is ${be_price - current_price:.0f} below break-even")
    
    with col2:
        st.markdown("**Break-Even Yield:**")
        be_yield = economics['breakeven']['yield']
        current_yield = crop_data['expected_yield']
        yield_margin = ((current_yield - be_yield) / be_yield * 100) if be_yield > 0 else 0
        
        st.metric(
            "Minimum Yield Required",
            f"{be_yield:.1f} tonnes/ha",
            f"{yield_margin:+.1f}% margin" if yield_margin != 0 else "At break-even"
        )
        
        if current_yield > be_yield:
            st.success(f"✅ Expected yield ({current_yield:.1f}t) is {current_yield - be_yield:.1f}t above break-even")
        else:
            st.warning(f"⚠️ Expected yield ({current_yield:.1f}t) is {be_yield - current_yield:.1f}t below break-even")

def create_multi_year_projection(economics, crop_data, years):
    """Create multi-year financial projection"""
    if not economics:
        return
    
    st.markdown(f"### 📅 {years}-Year Financial Projection")
    
    # Assumptions
    with st.expander("📋 Projection Assumptions"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inflation_rate = st.slider("Annual Inflation Rate (%)", 0.0, 10.0, 2.5, 0.1)
            yield_improvement = st.slider("Annual Yield Improvement (%)", -2.0, 5.0, 1.0, 0.1)
        
        with col2:
            price_volatility = st.slider("Price Volatility (%)", 0.0, 20.0, 5.0, 0.5)
            cost_increase = st.slider("Annual Cost Increase (%)", 0.0, 10.0, 3.0, 0.1)
        
        with col3:
            discount_rate = st.slider("Discount Rate (%)", 2.0, 15.0, 8.0, 0.1)
    
    # Calculate projections
    projections = []
    base_profit = economics['per_hectare']['profit']
    base_cost = economics['per_hectare']['total_cost']
    base_yield = crop_data['expected_yield']
    base_price = crop_data['market_price']
    
    for year in range(1, years + 1):
        # Apply yearly adjustments
        adjusted_yield = base_yield * ((1 + yield_improvement/100) ** year)
        adjusted_price = base_price * ((1 + inflation_rate/100) ** year)
        adjusted_cost = base_cost * ((1 + cost_increase/100) ** year)
        
        # Add price volatility (simplified)
        price_factor = 1 + (np.random.random() - 0.5) * (price_volatility/100)
        adjusted_price *= price_factor
        
        revenue = adjusted_yield * adjusted_price
        profit = revenue - adjusted_cost
        
        # Present value calculation
        pv_profit = profit / ((1 + discount_rate/100) ** year)
        
        projections.append({
            'year': year,
            'yield': adjusted_yield,
            'price': adjusted_price,
            'cost': adjusted_cost,
            'revenue': revenue,
            'profit': profit,
            'pv_profit': pv_profit,
            'cumulative_profit': sum(p['profit'] for p in projections) + profit,
            'cumulative_pv': sum(p['pv_profit'] for p in projections) + pv_profit
        })
    
    # Create visualization
    df = pd.DataFrame(projections)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Annual Profit', 'Cumulative Profit', 'Revenue vs Costs', 'Yield Trend'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Annual profit
    fig.add_trace(
        go.Scatter(x=df['year'], y=df['profit'], name='Annual Profit', 
                  line=dict(color='#4CAF50', width=3)),
        row=1, col=1
    )
    
    # Cumulative profit
    fig.add_trace(
        go.Scatter(x=df['year'], y=df['cumulative_profit'], name='Cumulative Profit',
                  fill='tonexty', line=dict(color='#2196F3')),
        row=1, col=2
    )
    
    # Revenue vs Costs
    fig.add_trace(
        go.Scatter(x=df['year'], y=df['revenue'], name='Revenue',
                  line=dict(color='#4CAF50')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['year'], y=df['cost'], name='Costs',
                  line=dict(color='#f44336')),
        row=2, col=1
    )
    
    # Yield trend
    fig.add_trace(
        go.Scatter(x=df['year'], y=df['yield'], name='Yield',
                  line=dict(color='#FF9800')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    total_profit = sum(p['profit'] for p in projections)
    total_pv = sum(p['pv_profit'] for p in projections)
    avg_annual_profit = total_profit / years
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Profit", format_currency(total_profit), f"{years} years")
    with col2:
        st.metric("Net Present Value", format_currency(total_pv), f"@ {discount_rate}% discount")
    with col3:
        st.metric("Avg Annual Profit", format_currency(avg_annual_profit), "Per hectare")

def main():
    """Main economics dashboard function"""
    st.title("💰 Farm Economics Dashboard")
    st.markdown("""
    Comprehensive economic analysis tool for farm profitability assessment, cost-benefit analysis,
    and financial planning. Analyze costs, revenues, ROI, and run scenario modeling.
    """)
    
    # Input parameters
    crop_data, farm_size, location, planning_years = create_economic_input_form()
    
    if crop_data and farm_size:
        # Calculate economics
        economics = calculate_economics(crop_data, farm_size)
        
        if economics:
            # Display key metrics
            display_key_metrics(economics)
            
            # Main analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Cost Analysis", "📈 Profitability", "⚖️ Break-Even", 
                "🎯 Scenarios", "📅 Multi-Year"
            ])
            
            with tab1:
                st.markdown("## 💸 Cost Analysis")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Cost breakdown chart
                    cost_fig = create_cost_breakdown_chart(economics)
                    if cost_fig:
                        st.plotly_chart(cost_fig, use_container_width=True)
                
                with col2:
                    # Detailed cost breakdown
                    st.markdown("### Cost Details per Hectare")
                    costs = economics['cost_breakdown']
                    
                    for cost_type, amount in costs.items():
                        percentage = (amount / sum(costs.values())) * 100
                        st.markdown(f"""
                        <div class="cost-item">
                            <span>{cost_type.title()}</span>
                            <span><strong>{format_currency(amount)}</strong> ({percentage:.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="cost-item">
                        <span><strong>Total Cost</strong></span>
                        <span><strong>{format_currency(sum(costs.values()))}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("## 📈 Profitability Analysis")
                
                # Sensitivity analysis
                price_fig, yield_fig = create_profitability_analysis(economics, crop_data)
                
                if price_fig and yield_fig:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(price_fig, use_container_width=True)
                    with col2:
                        st.plotly_chart(yield_fig, use_container_width=True)
                
                # Profitability summary
                st.markdown("### 💡 Profitability Insights")
                
                profit = economics['per_hectare']['profit']
                roi = economics['per_hectare']['roi_percent']
                
                if profit > 0:
                    if roi >= 25:
                        st.success("🌟 **Excellent profitability!** This crop shows outstanding economic potential.")
                    elif roi >= 15:
                        st.success("✅ **Good profitability.** This crop is a solid economic choice.")
                    elif roi >= 5:
                        st.info("📊 **Moderate profitability.** Consider optimizing costs or yields.")
                    else:
                        st.warning("⚠️ **Low profitability.** Economic viability is marginal.")
                else:
                    st.error("❌ **Unprofitable.** Current projections show losses.")
            
            with tab3:
                create_break_even_analysis(economics, crop_data)
            
            with tab4:
                scenario_results = create_scenario_analysis(crop_data, economics)
                if scenario_results:
                    display_scenario_analysis(scenario_results, farm_size)
            
            with tab5:
                create_multi_year_projection(economics, crop_data, planning_years)
            
            # Export options
            st.markdown("---")
            st.markdown("### 📥 Export Economic Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export summary as JSON
                export_data = {
                    'crop_info': crop_data,
                    'farm_size': farm_size,
                    'economics': economics,
                    'analysis_date': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📄 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"economic_analysis_{crop_data['name']}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Export as CSV
                summary_data = {
                    'Metric': [
                        'Crop', 'Farm Size (ha)', 'Cost per Ha', 'Revenue per Ha',
                        'Profit per Ha', 'Total Profit', 'ROI (%)', 'Profit Margin (%)',
                        'Break-even Price', 'Break-even Yield'
                    ],
                    'Value': [
                        crop_data['name'], farm_size, 
                        format_currency(economics['per_hectare']['total_cost']),
                        format_currency(economics['per_hectare']['revenue']),
                        format_currency(economics['per_hectare']['profit']),
                        format_currency(economics['total_farm']['total_profit']),
                        f"{economics['per_hectare']['roi_percent']:.1f}",
                        f"{economics['per_hectare']['profit_margin']:.1f}",
                        format_currency(economics['breakeven']['price']),
                        f"{economics['breakeven']['yield']:.1f} t/ha"
                    ]
                }
                
                df = pd.DataFrame(summary_data)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"economic_summary_{crop_data['name']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                st.button(
                    "📋 Generate Report",
                    help="PDF report generation coming soon!",
                    disabled=True
                )
            
            # Next steps
            st.markdown("### 🚀 Next Steps")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Yield Prediction", type="secondary", use_container_width=True):
                    st.switch_page("pages/4_Yield_Predictor.py")
            
            with col2:
                if st.button("🌱 Crop Recommender", type="secondary", use_container_width=True):
                    st.switch_page("pages/2_Crop_Recommender.py")
            
            with col3:
                if st.button("📝 Share Feedback", type="secondary", use_container_width=True):
                    st.switch_page("pages/5_Feedback_Form.py")
    
    # Help section
    with st.expander("❓ How to Use the Economics Dashboard"):
        st.markdown("""
        ### Understanding Economic Analysis
        
        **1. Input Methods:**
        - **From Recommendations:** Import data from crop recommendations
        - **Select Crop Type:** Use predefined crop parameters
        - **Manual Input:** Enter custom economic parameters
        
        **2. Key Metrics Explained:**
        - **ROI (Return on Investment):** Profit as percentage of investment
        - **Profit Margin:** Profit as percentage of revenue
        - **Break-even Price:** Minimum price needed to cover costs
        - **Break-even Yield:** Minimum yield needed to cover costs
        
        **3. Analysis Tabs:**
        - **Cost Analysis:** Breakdown of production costs
        - **Profitability:** Sensitivity analysis and projections
        - **Break-Even:** Risk assessment and safety margins
        - **Scenarios:** Compare optimistic/pessimistic outcomes
        - **Multi-Year:** Long-term financial projections
        
        **4. Best Practices:**
        - Consider multiple scenarios for robust planning
        - Monitor break-even points for risk management
        - Use sensitivity analysis to identify key drivers
        - Plan for cost inflation and market volatility
        
        **5. Integration:**
        - Results inform crop selection decisions
        - Use with yield predictions for accuracy
        - Consider soil and climate factors
        """)

if __name__ == "__main__":
    main()

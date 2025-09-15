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
            price_adjustment = st.slider("Market Price Adjustment (%)", -50.0, 50.0, 0.0, 5.0,
                                       help="Adjust market price based on local conditions")
        
        with col3:
            yield_adjustment = st.slider("Yield Adjustment (%)", -30.0, 30.0, 0.0, 5.0,
                                       help="Adjust expected yield based on local conditions")
        
        # Apply adjustments
        adjusted_price = crop_data['market_price'] * (1 + price_adjustment/100)
        adjusted_yield = crop_data['expected_yield'] * (1 + yield_adjustment/100)
        
        return crop_data, farm_size, adjusted_price, adjusted_yield
    
    return None, None, None, None

def calculate_economics(crop_data, farm_size, market_price, expected_yield):
    """Calculate comprehensive economic metrics"""
    
    # Costs per hectare
    cost_per_ha = {
        'seeds': crop_data['seed_cost'],
        'fertilizer': crop_data['fertilizer_cost'],
        'labor': crop_data['labor_cost'],
        'machinery': crop_data['machinery_cost'],
        'other': crop_data['other_costs']
    }
    
    total_cost_per_ha = sum(cost_per_ha.values())
    
    # Revenue calculations
    revenue_per_ha = expected_yield * market_price
    profit_per_ha = revenue_per_ha - total_cost_per_ha
    
    # Farm totals
    total_costs = total_cost_per_ha * farm_size
    total_revenue = revenue_per_ha * farm_size
    total_profit = profit_per_ha * farm_size
    
    # Financial ratios
    roi_percentage = (profit_per_ha / total_cost_per_ha) * 100 if total_cost_per_ha > 0 else 0
    profit_margin = (profit_per_ha / revenue_per_ha) * 100 if revenue_per_ha > 0 else 0
    break_even_price = total_cost_per_ha / expected_yield if expected_yield > 0 else 0
    break_even_yield = total_cost_per_ha / market_price if market_price > 0 else 0
    
    return {
        'cost_breakdown': cost_per_ha,
        'total_cost_per_ha': total_cost_per_ha,
        'revenue_per_ha': revenue_per_ha,
        'profit_per_ha': profit_per_ha,
        'total_costs': total_costs,
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'roi_percentage': roi_percentage,
        'profit_margin': profit_margin,
        'break_even_price': break_even_price,
        'break_even_yield': break_even_yield
    }

def display_economic_summary(economics, crop_name):
    """Display economic summary with key metrics"""
    st.markdown(f"## 📊 Economic Analysis - {crop_name}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit_color = "green" if economics['profit_per_ha'] > 0 else "red"
        st.markdown(create_metric_card(
            "Profit per Hectare",
            format_currency(economics['profit_per_ha']),
            f"Margin: {economics['profit_margin']:.1f}%",
            profit_color,
            "💰"
        ), unsafe_allow_html=True)
    
    with col2:
        roi_color = "green" if economics['roi_percentage'] > 15 else "red" if economics['roi_percentage'] < 0 else "gray"
        st.markdown(create_metric_card(
            "Return on Investment",
            f"{economics['roi_percentage']:.1f}%",
            "Expected ROI",
            roi_color,
            "📈"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Total Revenue",
            format_currency(economics['total_revenue']),
            "Gross income",
            "green",
            "💵"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            "Break-Even Price",
            format_currency(economics['break_even_price']) + "/tonne",
            f"Break-even yield: {economics['break_even_yield']:.1f} t/ha",
            "gray",
            "⚖️"
        ), unsafe_allow_html=True)

def create_cost_breakdown_chart(cost_breakdown):
    """Create cost breakdown pie chart"""
    costs = list(cost_breakdown.values())
    labels = [label.title() for label in cost_breakdown.keys()]
    
    fig = px.pie(
        values=costs,
        names=labels,
        title="Cost Breakdown per Hectare",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Cost: $%{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_profitability_waterfall(economics):
    """Create waterfall chart showing profitability breakdown"""
    
    categories = ['Revenue', 'Seeds', 'Fertilizer', 'Labor', 'Machinery', 'Other', 'Net Profit']
    values = [
        economics['revenue_per_ha'],
        -economics['cost_breakdown']['seeds'],
        -economics['cost_breakdown']['fertilizer'],
        -economics['cost_breakdown']['labor'],
        -economics['cost_breakdown']['machinery'],
        -economics['cost_breakdown']['other'],
        economics['profit_per_ha']
    ]
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Profitability",
        orientation="v",
        measure=["absolute"] + ["relative"] * 5 + ["total"],
        x=categories,
        y=values,
        textposition="outside",
        text=[f"${v:,.0f}" for v in values],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#FF6B6B"}},
        increasing={"marker": {"color": "#4ECDC4"}},
        totals={"marker": {"color": "#45B7D1"}}
    ))
    
    fig.update_layout(
        title="Profitability Waterfall Analysis (per hectare)",
        xaxis_title="Cost Components",
        yaxis_title="Amount ($)",
        height=500,
        showlegend=False
    )
    
    return fig

def create_scenario_analysis(crop_data, farm_size, base_price, base_yield):
    """Create scenario analysis with different price and yield combinations"""
    
    st.markdown("## 🎯 Scenario Analysis")
    st.markdown("Analyze profitability under different market and yield conditions:")
    
    # Scenario parameters
    col1, col2 = st.columns(2)
    
    with col1:
        price_scenarios = st.multiselect(
            "Price Scenarios (% change)",
            [-30, -20, -10, 0, 10, 20, 30],
            default=[-20, 0, 20],
            help="Select price change scenarios to analyze"
        )
    
    with col2:
        yield_scenarios = st.multiselect(
            "Yield Scenarios (% change)",
            [-30, -20, -10, 0, 10, 20, 30],
            default=[-20, 0, 20],
            help="Select yield change scenarios to analyze"
        )
    
    if price_scenarios and yield_scenarios:
        # Create scenario matrix
        scenario_data = []
        
        for price_change in price_scenarios:
            for yield_change in yield_scenarios:
                scenario_price = base_price * (1 + price_change/100)
                scenario_yield = base_yield * (1 + yield_change/100)
                
                economics = calculate_economics(crop_data, farm_size, scenario_price, scenario_yield)
                
                scenario_data.append({
                    'Price Change': f"{price_change:+d}%",
                    'Yield Change': f"{yield_change:+d}%",
                    'Price ($/tonne)': scenario_price,
                    'Yield (t/ha)': scenario_yield,
                    'Profit per Ha': economics['profit_per_ha'],
                    'ROI (%)': economics['roi_percentage'],
                    'Total Profit': economics['total_profit']
                })
        
        df_scenarios = pd.DataFrame(scenario_data)
        
        # Create heatmap for profit per hectare
        pivot_profit = df_scenarios.pivot(
            index='Yield Change',
            columns='Price Change',
            values='Profit per Ha'
        )
        
        fig_heatmap = px.imshow(
            pivot_profit,
            title="Profit per Hectare - Scenario Analysis",
            labels=dict(x="Price Change", y="Yield Change", color="Profit ($/ha)"),
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Scenario table
        st.markdown("### 📋 Detailed Scenario Results")
        
        # Color-code profit values
        def color_profit(val):
            if val > 500:
                return 'background-color: #d4edda'  # Green
            elif val > 0:
                return 'background-color: #fff3cd'  # Yellow
            else:
                return 'background-color: #f8d7da'  # Red
        
        styled_df = df_scenarios.style.applymap(color_profit, subset=['Profit per Ha', 'Total Profit'])
        st.dataframe(styled_df, hide_index=True)

def create_risk_assessment(economics, crop_data):
    """Create risk assessment section"""
    st.markdown("## ⚠️ Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Financial Risk Indicators")
        
        # Risk level based on ROI
        roi = economics['roi_percentage']
        if roi >= 25:
            risk_level = "Low Risk"
            risk_color = "#4CAF50"
        elif roi >= 10:
            risk_level = "Medium Risk"
            risk_color = "#FF9800"
        elif roi >= 0:
            risk_level = "High Risk"
            risk_color = "#F44336"
        else:
            risk_level = "Very High Risk"
            risk_color = "#B71C1C"
        
        st.markdown(f"""
        <div style="
            background: {risk_color}20;
            border-left: 4px solid {risk_color};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        ">
            <h4 style="color: {risk_color}; margin: 0;">Risk Level: {risk_level}</h4>
            <p style="margin: 0.5rem 0 0 0;">Based on expected ROI of {roi:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Break-even analysis
        st.markdown("**Break-Even Analysis:**")
        st.write(f"• Break-even price: {format_currency(economics['break_even_price'])}/tonne")
        st.write(f"• Break-even yield: {economics['break_even_yield']:.2f} tonnes/ha")
        st.write(f"• Safety margin: {((economics['revenue_per_ha'] - economics['total_cost_per_ha'])/economics['revenue_per_ha']*100):.1f}%")
    
    with col2:
        st.markdown("### 🎯 Risk Mitigation Strategies")
        
        recommendations = []
        
        if economics['roi_percentage'] < 10:
            recommendations.extend([
                "Consider crop insurance to protect against yield losses",
                "Diversify with multiple crops to spread risk",
                "Negotiate forward contracts for price stability"
            ])
        
        if economics['profit_margin'] < 20:
            recommendations.extend([
                "Focus on cost reduction strategies",
                "Explore premium market opportunities",
                "Consider precision farming to improve efficiency"
            ])
        
        if crop_data['growing_days'] > 150:
            recommendations.append("Long growing season increases weather risk")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Monitor market prices regularly",
                "Maintain good record keeping",
                "Consider crop rotation benefits",
                "Stay updated on best practices"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

def create_market_comparison():
    """Create market comparison with other crops"""
    st.markdown("## 📊 Market Comparison")
    
    # Get comparison data for all crops
    comparison_crops = ['wheat', 'corn', 'rice', 'soybean', 'cotton']
    comparison_data = []
    
    for crop_key in comparison_crops:
        crop = DEFAULT_CROP_DATA[crop_key]
        economics = calculate_economics(crop, 25.0, crop['market_price'], crop['expected_yield'])
        
        comparison_data.append({
            'Crop': crop['name'],
            'Expected Yield (t/ha)': crop['expected_yield'],
            'Market Price ($/t)': crop['market_price'],
            'Production Cost ($/ha)': economics['total_cost_per_ha'],
            'Profit per Ha ($)': economics['profit_per_ha'],
            'ROI (%)': economics['roi_percentage'],
            'Growing Days': crop['growing_days'],
            'Season': crop['season']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create comparison charts
    tab1, tab2, tab3 = st.tabs(["💰 Profitability", "📈 ROI Comparison", "⏱️ Timeline"])
    
    with tab1:
        profit_fig = px.bar(
            df_comparison,
            x='Crop',
            y='Profit per Ha ($)',
            title="Profit per Hectare Comparison",
            color='Profit per Ha ($)',
            color_continuous_scale='RdYlGn'
        )
        profit_fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(profit_fig, use_container_width=True)
    
    with tab2:
        roi_fig = px.bar(
            df_comparison,
            x='Crop',
            y='ROI (%)',
            title="Return on Investment Comparison",
            color='ROI (%)',
            color_continuous_scale='Viridis'
        )
        roi_fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(roi_fig, use_container_width=True)
    
    with tab3:
        timeline_fig = px.scatter(
            df_comparison,
            x='Growing Days',
            y='Profit per Ha ($)',
            size='Expected Yield (t/ha)',
            color='Season',
            hover_name='Crop',
            title="Profit vs Growing Period",
            labels={'Growing Days': 'Days to Maturity'}
        )
        timeline_fig.update_layout(height=400)
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Comparison table
    st.markdown("### 📋 Detailed Market Comparison")
    comparison_table_html = create_comparison_table(
        df_comparison.to_dict('records'), 
        highlight_best=True
    )
    st.markdown(comparison_table_html, unsafe_allow_html=True)

def export_economic_analysis(economics, crop_data, farm_size):
    """Create export functionality for economic analysis"""
    st.markdown("### 📥 Export Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        export_data = {
            'crop_info': {
                'name': crop_data['name'],
                'expected_yield': crop_data['expected_yield'],
                'market_price': crop_data['market_price'],
                'growing_days': crop_data['growing_days']
            },
            'farm_parameters': {
                'size_hectares': farm_size
            },
            'economic_analysis': {
                'profit_per_hectare': economics['profit_per_ha'],
                'total_profit': economics['total_profit'],
                'roi_percentage': economics['roi_percentage'],
                'profit_margin': economics['profit_margin'],
                'break_even_price': economics['break_even_price'],
                'break_even_yield': economics['break_even_yield']
            },
            'cost_breakdown': economics['cost_breakdown'],
            'analysis_date': datetime.now().isoformat()
        }
        
        st.download_button(
            label="📄 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"economic_analysis_{crop_data['name'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as CSV
        summary_data = {
            'Metric': [
                'Crop Name',
                'Farm Size (ha)',
                'Expected Yield (t/ha)',
                'Market Price ($/t)',
                'Total Cost per Ha ($)',
                'Revenue per Ha ($)',
                'Profit per Ha ($)',
                'Total Profit ($)',
                'ROI (%)',
                'Profit Margin (%)',
                'Break-even Price ($/t)',
                'Break-even Yield (t/ha)'
            ],
            'Value': [
                crop_data['name'],
                farm_size,
                crop_data['expected_yield'],
                crop_data['market_price'],
                economics['total_cost_per_ha'],
                economics['revenue_per_ha'],
                economics['profit_per_ha'],
                economics['total_profit'],
                economics['roi_percentage'],
                economics['profit_margin'],
                economics['break_even_price'],
                economics['break_even_yield']
            ]
        }
        
        df_export = pd.DataFrame(summary_data)
        csv_data = df_export.to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"economic_summary_{crop_data['name'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Generate report (placeholder)
        st.button(
            "📋 Generate PDF Report",
            help="PDF report generation coming soon!",
            disabled=True
        )

def main():
    """Main economics dashboard page"""
    st.title("💰 Farm Economics Dashboard")
    st.markdown("""
    Comprehensive economic analysis tool for farm profitability assessment. Analyze costs, 
    revenues, ROI, and explore different scenarios to make informed financial decisions.
    """)
    
    # Input form
    crop_data, farm_size, adjusted_price, adjusted_yield = create_economic_input_form()
    
    if crop_data and farm_size:
        # Calculate economics
        economics = calculate_economics(crop_data, farm_size, adjusted_price, adjusted_yield)
        
        # Display summary
        display_economic_summary(economics, crop_data['name'])
        
        # Main analysis sections
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Cost breakdown
            st.markdown("### 🥧 Cost Breakdown")
            cost_fig = create_cost_breakdown_chart(economics['cost_breakdown'])
            st.plotly_chart(cost_fig, use_container_width=True)
        
        with col2:
            # Profitability waterfall
            st.markdown("### 📊 Profitability Analysis")
            waterfall_fig = create_profitability_waterfall(economics)
            st.plotly_chart(waterfall_fig, use_container_width=True)
        
        # Scenario analysis
        create_scenario_analysis(crop_data, farm_size, adjusted_price, adjusted_yield)
        
        # Risk assessment
        create_risk_assessment(economics, crop_data)
        
        # Market comparison
        create_market_comparison()
        
        # Export options
        export_economic_analysis(economics, crop_data, farm_size)
        
        # Next steps
        st.markdown("### 🚀 Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Yield Prediction", type="secondary", use_container_width=True):
                st.switch_page("pages/4_Yield_Predictor.py")
        
        with col2:
            if st.button("🧪 Soil Analysis", type="secondary", use_container_width=True):
                st.switch_page("pages/1_Soil_Analyzer.py")
        
        with col3:
            if st.button("📝 Share Feedback", type="secondary", use_container_width=True):
                st.switch_page("pages/5_Feedback_Form.py")
    
    # Help section
    with st.expander("❓ How to Use the Economics Dashboard"):
        st.markdown("""
        ### Getting Started
        
        **1. Input Methods:**
        - **From Crop Recommendations:** Import data from crop recommender results
        - **Select Crop Type:** Choose from predefined crop templates
        - **Manual Input:** Enter custom parameters for any crop
        
        **2. Key Metrics Explained:**
        - **Profit per Hectare:** Net income after all costs
        - **ROI (Return on Investment):** Percentage return on invested capital
        - **Break-even Price:** Minimum selling price to cover costs
        - **Break-even Yield:** Minimum yield needed to cover costs
        
        **3. Analysis Features:**
        - **Cost Breakdown:** Visual breakdown of all production costs
        - **Scenario Analysis:** Test different price and yield scenarios
        - **Risk Assessment:** Evaluate financial risks and mitigation strategies
        - **Market Comparison:** Compare with other crops
        
        **4. Interpreting Results:**
        - **ROI > 25%:** Excellent investment opportunity
        - **ROI 10-25%:** Good profitability
        - **ROI 0-10%:** Marginal profitability
        - **ROI < 0%:** Loss-making scenario
        
        **5. Best Practices:**
        - Use realistic local market prices
        - Factor in your specific growing conditions
        - Consider multiple scenarios for risk management
        - Keep records for future comparisons
        """)

if __name__ == "__main__":
    main()

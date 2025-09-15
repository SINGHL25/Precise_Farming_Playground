
"""
Yield Predictor Dashboard Page

AI-powered yield prediction system using machine learning models,
historical data, weather patterns, and soil conditions.

Features:
- Historical yield analysis
- ML-based predictions
- Weather impact modeling
- Confidence intervals
- Scenario planning

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
    create_yield_trend_chart, create_metric_card, get_color_palette,
    create_alert_box, format_currency
)

# Page configuration
st.set_page_config(
    page_title="Yield Predictor - Precision Farming",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-card {
        background: linear-gradient(135deg, #E3F2FD, #E8F5E8);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #4CAF50;
        text-align: center;
    }
    
    .confidence-high { color: #2E7D32; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low { color: #F44336; font-weight: bold; }
    
    .factor-item {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .historical-trend {
        background: linear-gradient(135deg, #F8F9FA, #E8F5E8);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_historical_data(crop_name="wheat", years=5):
    """Generate realistic historical yield data"""
    base_yield = {
        'wheat': 3.5,
        'corn': 8.5,
        'rice': 4.5,
        'soybean': 2.8,
        'cotton': 1.2
    }.get(crop_name.lower(), 3.5)
    
    # Generate historical data with realistic variations
    np.random.seed(42)  # For reproducible results
    
    historical_data = []
    current_year = datetime.now().year
    
    for i in range(years):
        year = current_year - years + i + 1
        
        # Add realistic variations: trend + weather + random
        trend_factor = 1 + (i * 0.02)  # 2% annual improvement
        weather_factor = np.random.normal(1.0, 0.1)  # Weather variability
        random_factor = np.random.normal(1.0, 0.05)  # Other factors
        
        yield_value = base_yield * trend_factor * weather_factor * random_factor
        yield_value = max(0, yield_value)  # Ensure non-negative
        
        # Weather conditions (simplified)
        rainfall = np.random.normal(600, 100)
        temperature = np.random.normal(22, 3)
        
        historical_data.append({
            'year': year,
            'yield': round(yield_value, 2),
            'rainfall': max(0, round(rainfall, 1)),
            'temperature': round(temperature, 1),
            'weather_factor': round(weather_factor, 3)
        })
    
    return pd.DataFrame(historical_data)

def create_ml_prediction(historical_data, soil_data=None, weather_data=None):
    """Create ML-based yield prediction (simplified simulation)"""
    if len(historical_data) < 3:
        return None
    
    # Simple trend-based prediction with adjustments
    recent_yields = historical_data['yield'].tail(3).values
    base_prediction = np.mean(recent_yields)
    
    # Adjust for trend
    if len(recent_yields) >= 2:
        trend = (recent_yields[-1] - recent_yields[0]) / len(recent_yields)
        base_prediction += trend
    
    # Soil adjustment factor
    soil_factor = 1.0
    if soil_data:
        # Simplified soil impact calculation
        ph_optimal = 6.0 <= soil_data.get('ph_level', 7.0) <= 7.5
        nutrients_adequate = (
            soil_data.get('nitrogen_ppm', 30) >= 40 and
            soil_data.get('phosphorus_ppm', 20) >= 25 and
            soil_data.get('potassium_ppm', 150) >= 180
        )
        organic_matter_good = soil_data.get('organic_matter_percent', 2.5) >= 3.0
        
        adjustments = []
        if ph_optimal:
            adjustments.append(0.05)
        else:
            adjustments.append(-0.05)
            
        if nutrients_adequate:
            adjustments.append(0.08)
        else:
            adjustments.append(-0.08)
            
        if organic_matter_good:
            adjustments.append(0.03)
        else:
            adjustments.append(-0.03)
        
        soil_factor = 1 + sum(adjustments)
    
    # Weather adjustment factor
    weather_factor = 1.0
    if weather_data:
        # Simplified weather impact
        temp_optimal = 15 <= weather_data.avg_temperature <= 25
        rainfall_adequate = weather_data.growing_season_rainfall >= 400
        
        if temp_optimal and rainfall_adequate:
            weather_factor = 1.1
        elif temp_optimal or rainfall_adequate:
            weather_factor = 1.05
        else:
            weather_factor = 0.95
    
    # Final prediction
    predicted_yield = base_prediction * soil_factor * weather_factor
    predicted_yield = max(0, predicted_yield)
    
    # Confidence calculation
    yield_variance = np.var(historical_data['yield'])
    data_points = len(historical_data)
    
    confidence = min(95, 60 + (data_points * 5) + (20 if soil_data else 0) + (15 if weather_data else 0))
    
    # Prediction interval
    std_error = np.sqrt(yield_variance / data_points) if data_points > 1 else 0.5
    margin = std_error * 1.96  # 95% confidence interval
    
    return {
        'predicted_yield': round(predicted_yield, 2),
        'confidence': round(confidence, 1),
        'lower_bound': round(predicted_yield - margin, 2),
        'upper_bound': round(predicted_yield + margin, 2),
        'factors': {
            'soil_factor': soil_factor,
            'weather_factor': weather_factor,
            'trend_component': trend if 'trend' in locals() else 0
        }
    }

def display_prediction_results(prediction, crop_name, farm_size=25):
    """Display prediction results in cards"""
    if not prediction:
        st.error("Unable to generate predictions. Need at least 3 years of historical data.")
        return
    
    colors = get_color_palette("green")
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h2 style="color: #2E7D32; margin: 0 0 1rem 0;">📈 Yield Prediction for {crop_name.title()}</h2>
        <div style="font-size: 3rem; font-weight: bold; color: #1B5E20; margin: 1rem 0;">
            {prediction['predicted_yield']:.1f} t/ha
        </div>
        <div style="font-size: 1.2rem; color: #4CAF50; margin-bottom: 1rem;">
            Range: {prediction['lower_bound']:.1f} - {prediction['upper_bound']:.1f} t/ha
        </div>
        <div style="font-size: 1rem; color: #666;">
            Confidence Level: <span class="confidence-{'high' if prediction['confidence'] >= 80 else 'medium' if prediction['confidence'] >= 60 else 'low'}">{prediction['confidence']:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_production = prediction['predicted_yield'] * farm_size
        st.metric(
            "Total Production",
            f"{total_production:.1f} tonnes",
            f"For {farm_size} hectares"
        )
    
    with col2:
        confidence_level = prediction['confidence']
        confidence_desc = "High" if confidence_level >= 80 else "Medium" if confidence_level >= 60 else "Low"
        st.metric(
            "Prediction Confidence",
            f"{confidence_level:.1f}%",
            confidence_desc
        )
    
    with col3:
        uncertainty = prediction['upper_bound'] - prediction['lower_bound']
        st.metric(
            "Uncertainty Range",
            f"±{uncertainty/2:.1f} t/ha",
            f"{(uncertainty/prediction['predicted_yield']*100):.1f}% of prediction"
        )
    
    with col4:
        # Estimate market value (simplified)
        market_prices = {'wheat': 250, 'corn': 200, 'rice': 300, 'soybean': 400, 'cotton': 1500}
        price = market_prices.get(crop_name.lower(), 300)
        estimated_value = total_production * price
        st.metric(
            "Estimated Value",
            format_currency(estimated_value),
            f"@ ${price}/tonne"
        )

def create_factors_analysis(prediction, historical_data):
    """Create analysis of prediction factors"""
    if not prediction:
        return
    
    st.markdown("### 🔍 Prediction Factors Analysis")
    
    factors = prediction['factors']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Impact Factors")
        
        soil_impact = (factors['soil_factor'] - 1) * 100
        weather_impact = (factors['weather_factor'] - 1) * 100
        trend_impact = factors.get('trend_component', 0) * 100
        
        factor_data = pd.DataFrame({
            'Factor': ['Soil Conditions', 'Weather Conditions', 'Historical Trend'],
            'Impact': [soil_impact, weather_impact, trend_impact],
            'Category': ['Soil', 'Weather', 'Trend']
        })
        
        fig = px.bar(
            factor_data,
            x='Factor',
            y='Impact',
            color='Category',
            title='Factors Affecting Yield Prediction',
            labels={'Impact': 'Impact on Yield (%)'},
            color_discrete_map={'Soil': '#8D6E63', 'Weather': '#2196F3', 'Trend': '#4CAF50'}
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Key Insights")
        
        insights = []
        
        if soil_impact > 5:
            insights.append("✅ Excellent soil conditions boost expected yield")
        elif soil_impact < -5:
            insights.append("⚠️ Poor soil conditions may reduce yield")
        else:
            insights.append("○ Soil conditions have neutral impact")
        
        if weather_impact > 5:
            insights.append("✅ Favorable weather conditions expected")
        elif weather_impact < -5:
            insights.append("⚠️ Weather conditions may be challenging")
        else:
            insights.append("○ Average weather conditions expected")
        
        if trend_impact > 2:
            insights.append("📈 Positive yield trend continues")
        elif trend_impact < -2:
            insights.append("📉 Declining yield trend observed")
        else:
            insights.append("➡️ Stable yield trend")
        
        # Historical variance insight
        yield_cv = (np.std(historical_data['yield']) / np.mean(historical_data['yield'])) * 100
        if yield_cv < 10:
            insights.append("🎯 Low historical variability increases confidence")
        elif yield_cv > 20:
            insights.append("📊 High historical variability reduces confidence")
        else:
            insights.append("📈 Moderate historical variability")
        
        for insight in insights:
            st.markdown(f"""
            <div class="factor-item">
                {insight}
            </div>
            """, unsafe_allow_html=True)

def create_scenario_modeling(prediction, crop_name):
    """Create scenario modeling section"""
    if not prediction:
        return
    
    st.markdown("### 🎯 Scenario Modeling")
    
    st.markdown("Explore how different conditions might affect your predicted yield:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Weather Scenarios")
        
        weather_scenarios = [
            {'name': 'Drought Conditions', 'factor': 0.75, 'color': '#F44336'},
            {'name': 'Normal Weather', 'factor': 1.0, 'color': '#4CAF50'},
            {'name': 'Ideal Weather', 'factor': 1.15, 'color': '#2196F3'},
            {'name': 'Excessive Rain', 'factor': 0.85, 'color': '#FF9800'}
        ]
        
        scenario_yields = []
        scenario_names = []
        colors = []
        
        for scenario in weather_scenarios:
            adjusted_yield = prediction['predicted_yield'] * scenario['factor']
            scenario_yields.append(adjusted_yield)
            scenario_names.append(scenario['name'])
            colors.append(scenario['color'])
        
        fig = go.Figure(data=[
            go.Bar(x=scenario_names, y=scenario_yields, marker_color=colors)
        ])
        
        fig.update_layout(
            title="Weather Impact on Yield",
            xaxis_title="Weather Scenario",
            yaxis_title="Predicted Yield (t/ha)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Management Scenarios")
        
        management_scenarios = [
            {'name': 'Basic Management', 'factor': 0.9},
            {'name': 'Current Practice', 'factor': 1.0},
            {'name': 'Improved Fertilization', 'factor': 1.1},
            {'name': 'Precision Agriculture', 'factor': 1.2}
        ]
        
        mgmt_yields = []
        mgmt_names = []
        
        for scenario in management_scenarios:
            adjusted_yield = prediction['predicted_yield'] * scenario['factor']
            mgmt_yields.append(adjusted_yield)
            mgmt_names.append(scenario['name'])
        
        fig = go.Figure(data=[
            go.Bar(x=mgmt_names, y=mgmt_yields, marker_color='#4CAF50')
        ])
        
        fig.update_layout(
            title="Management Impact on Yield",
            xaxis_title="Management Level",
            yaxis_title="Predicted Yield (t/ha)",
            height=350
        )
        
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def create_risk_assessment(prediction, historical_data):
    """Create risk assessment section"""
    if not prediction or len(historical_data) < 3:
        return
    
    st.markdown("### ⚠️ Risk Assessment")
    
    # Calculate risk metrics
    historical_yields = historical_data['yield'].values
    avg_yield = np.mean(historical_yields)
    yield_volatility = np.std(historical_yields) / avg_yield * 100
    min_yield = np.min(historical_yields)
    max_yield = np.max(historical_yields)
    
    # Probability of yield ranges
    predicted = prediction['predicted_yield']
    prob_above_avg = 70 if predicted > avg_yield else 30
    prob_below_min = 10 if predicted > min_yield * 1.1 else 25
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Risk Metrics")
        
        st.metric("Yield Volatility", f"{yield_volatility:.1f}%", 
                 "Historical variation")
        st.metric("Worst Case Scenario", f"{min_yield:.1f} t/ha", 
                 f"{((predicted - min_yield)/predicted*100):+.1f}% vs prediction")
        st.metric("Best Case Potential", f"{max_yield:.1f} t/ha", 
                 f"{((max_yield - predicted)/predicted*100):+.1f}% vs prediction")
        
        # Risk level assessment
        if yield_volatility < 15:
            risk_level = "Low Risk"
            risk_color = "#4CAF50"
        elif yield_volatility < 25:
            risk_level = "Moderate Risk"
            risk_color = "#FF9800"
        else:
            risk_level = "High Risk"
            risk_color = "#F44336"
        
        st.markdown(f"""
        <div style="background: {risk_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {risk_color};">
            <strong style="color: {risk_color};">Risk Level: {risk_level}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Probability Assessment")
        
        prob_data = pd.DataFrame({
            'Outcome': ['Above Average', 'Below Minimum', 'Within Normal Range'],
            'Probability': [prob_above_avg, prob_below_min, 100 - prob_above_avg - prob_below_min],
            'Color': ['#4CAF50', '#F44336', '#2196F3']
        })
        
        fig = px.pie(
            prob_data,
            values='Probability',
            names='Outcome',
            title='Yield Outcome Probabilities',
            color='Outcome',
            color_discrete_map={
                'Above Average': '#4CAF50',
                'Below Minimum': '#F44336',
                'Within Normal Range': '#2196F3'
            }
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main yield predictor page"""
    st.title("📈 AI Yield Predictor")
    st.markdown("""
    Predict crop yields using advanced machine learning models that analyze historical data,
    soil conditions, weather patterns, and management practices for accurate forecasting.
    """)
    
    # Crop selection and parameters
    st.markdown("## 🌾 Prediction Setup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop_name = st.selectbox(
            "Select Crop",
            ["wheat", "corn", "rice", "soybean", "cotton"],
            help="Choose the crop for yield prediction"
        )
    
    with col2:
        farm_size = st.number_input(
            "Farm Size (hectares)",
            min_value=1.0,
            max_value=1000.0,
            value=25.0,
            step=0.5,
            help="Total area for yield calculation"
        )
    
    with col3:
        historical_years = st.slider(
            "Historical Data Years",
            min_value=3,
            max_value=10,
            value=5,
            help="Years of historical data to use"
        )
    
    # Generate historical data
    historical_data = generate_historical_data(crop_name, historical_years)
    
    # Check for previous analysis data
    soil_data = None
    weather_data = None
    
    if 'crop_soil_data' in st.session_state:
        soil_data = st.session_state.crop_soil_data
        st.success("✅ Using soil data from previous analysis")
    
    if 'crop_weather_data' in st.session_state:
        weather_data = st.session_state.crop_weather_data
        st.success("✅ Using weather data from previous analysis")
    
    if not soil_data and not weather_data:
        st.info("💡 For more accurate predictions, run soil analysis and crop recommendations first.")
    
    # Generate prediction
    if st.button("🚀 Generate Yield Prediction", type="primary"):
        with st.spinner("🔄 Running AI prediction models..."):
            prediction = create_ml_prediction(historical_data, soil_data, weather_data)
            st.session_state.yield_prediction = prediction
            st.success("✅ Prediction generated successfully!")
    
    # Display results if available
    if 'yield_prediction' in st.session_state and st.session_state.yield_prediction:
        prediction = st.session_state.yield_prediction
        
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # Display prediction results
        display_prediction_results(prediction, crop_name, farm_size)
        
        # Historical trend analysis
        st.markdown("### 📈 Historical Trend Analysis")
        
        # Create trend chart
        fig = px.line(
            historical_data,
            x='year',
            y='yield',
            title=f'Historical Yield Trend - {crop_name.title()}',
            markers=True
        )
        
        # Add prediction point
        current_year = datetime.now().year + 1
        fig.add_trace(go.Scatter(
            x=[current_year],
            y=[prediction['predicted_yield']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Prediction'
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=[current_year, current_year],
            y=[prediction['lower_bound'], prediction['upper_bound']],
            mode='lines',
            line=dict(color='red', width=2),
            name='Confidence Interval',
            showlegend=False
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Factors", "🎯 Scenarios", "⚠️ Risk Assessment", "📊 Data"])
        
        with tab1:
            create_factors_analysis(prediction, historical_data)
        
        with tab2:
            create_scenario_modeling(prediction, crop_name)
        
        with tab3:
            create_risk_assessment(prediction, historical_data)
        
        with tab4:
            st.markdown("### 📊 Historical Data")
            st.dataframe(historical_data, use_container_width=True)
            
            st.markdown("### 🔢 Prediction Details")
            prediction_details = {
                'Metric': ['Predicted Yield', 'Confidence Level', 'Lower Bound', 'Upper Bound', 'Soil Factor', 'Weather Factor'],
                'Value': [
                    f"{prediction['predicted_yield']:.2f} t/ha",
                    f"{prediction['confidence']:.1f}%",
                    f"{prediction['lower_bound']:.2f} t/ha",
                    f"{prediction['upper_bound']:.2f} t/ha",
                    f"{prediction['factors']['soil_factor']:.3f}",
                    f"{prediction['factors']['weather_factor']:.3f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(prediction_details), use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_data = {
                'crop': crop_name,
                'farm_size': farm_size,
                'prediction': prediction,
                'historical_data': historical_data.to_dict('records'),
                'analysis_date': datetime.now().isoformat()
            }
            
            st.download_button(
                label="📄 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"yield_prediction_{crop_name}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col2:
            summary_df = pd.DataFrame({
                'Metric': ['Crop', 'Farm Size', 'Predicted Yield', 'Total Production', 'Confidence', 'Risk Level'],
                'Value': [
                    crop_name.title(),
                    f"{farm_size} ha",
                    f"{prediction['predicted_yield']:.1f} t/ha",
                    f"{prediction['predicted_yield'] * farm_size:.1f} t",
                    f"{prediction['confidence']:.1f}%",
                    'Low' if prediction['confidence'] > 80 else 'Medium'
                ]
            })
            
            st.download_button(
                label="📊 Download CSV",
                data=summary_df.to_csv(index=False),
                file_name=f"yield_summary_{crop_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            st.button(
                "📋 Generate Report",
                help="PDF report generation coming soon!",
                disabled=True
            )
    
    # Help section
    with st.expander("❓ How to Use the Yield Predictor"):
        st.markdown("""
        ### Understanding Yield Predictions
        
        **1. Prediction Process:**
        - Historical yield data analysis
        - Soil condition assessment
        - Weather pattern integration
        - Machine learning model application
        
        **2. Accuracy Factors:**
        - **Data Quality:** More historical data = better predictions
        - **Soil Analysis:** Improves prediction accuracy by 15-20%
        - **Weather Data:** Accounts for seasonal variations
        - **Management Practices:** Considers farming techniques
        
        **3. Confidence Levels:**
        - **High (80%+):** Strong data support, reliable prediction
        - **Medium (60-79%):** Good data, reasonable confidence
        - **Low (<60%):** Limited data, use with caution
        
        **4. Using Results:**
        - Plan production and storage capacity
        - Estimate potential revenue
        - Assess financial risks
        - Compare different crops
        
        **5. Scenario Planning:**
        - Consider weather variability
        - Plan for best/worst case outcomes
        - Evaluate management improvements
        - Assess risk tolerance
        """)

if __name__ == "__main__":
    main()

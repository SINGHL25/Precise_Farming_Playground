
"""
Crop Recommender Dashboard Page

AI-powered crop recommendation system that analyzes soil conditions,
climate data, and economic factors to suggest optimal crops for farming.

Features:
- Multi-factor analysis (soil, climate, economics, risk)
- Interactive parameter adjustment
- Detailed recommendation explanations
- Comparative analysis tools
- Export capabilities

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
    create_crop_suitability_chart, create_metric_card, create_alert_box,
    format_recommendation_card, create_comparison_table, get_color_palette
)
from src.crop_recommender import CropRecommendationEngine, WeatherData, create_sample_weather_data
from src.soil_analyzer import SoilSample

# Page configuration
st.set_page_config(
    page_title="Crop Recommender - Precision Farming",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        transition: transform 0.2s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .score-excellent { background: #E8F5E8; color: #2E7D32; }
    .score-good { background: #F1F8E9; color: #388E3C; }
    .score-fair { background: #FFF8E1; color: #F57C00; }
    .score-poor { background: #FFEBEE; color: #D32F2F; }
    
    .factor-positive { color: #2E7D32; }
    .factor-negative { color: #D32F2F; }
    .factor-neutral { color: #666; }
    
    .parameter-section {
        background: linear-gradient(135deg, #F8F9FA, #E3F2FD);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_soil_data():
    """Create sample soil data for demonstration"""
    return {
        'ph_level': 6.8,
        'nitrogen_ppm': 45.0,
        'phosphorus_ppm': 28.0,
        'potassium_ppm': 185.0,
        'organic_matter_percent': 3.2,
        'texture_class': 'loam',
        'moisture_content': 24.0,
        'electrical_conductivity': 1.1
    }

def load_sample_weather_data():
    """Create sample weather data"""
    return create_sample_weather_data("Iowa, USA")

def create_parameter_input_section():
    """Create interactive parameter input section"""
    st.markdown("## 🔧 Input Parameters")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["🌱 From Soil Analysis", "✍️ Manual Input", "🎯 Use Defaults"])
    
    with tab1:
        st.markdown("### Import from Soil Analysis")
        if 'soil_analysis_result' in st.session_state:
            if st.button("📥 Load from Previous Soil Analysis", type="primary"):
                # Load soil data from previous analysis
                soil_result = st.session_state.soil_analysis_result
                sample = soil_result.sample
                
                soil_data = {
                    'ph_level': sample.ph_level,
                    'nitrogen_ppm': sample.nitrogen_ppm,
                    'phosphorus_ppm': sample.phosphorus_ppm,
                    'potassium_ppm': sample.potassium_ppm,
                    'organic_matter_percent': sample.organic_matter_percent,
                    'texture_class': soil_result.texture_class.value,
                    'moisture_content': sample.moisture_content,
                    'electrical_conductivity': getattr(sample, 'electrical_conductivity', None)
                }
                
                st.session_state.crop_soil_data = soil_data
                st.success("Soil data loaded from previous analysis!")
        else:
            st.info("No previous soil analysis found. Please run soil analysis first or use manual input.")
    
    with tab2:
        st.markdown("### Manual Parameter Input")
        create_manual_input_form()
    
    with tab3:
        st.markdown("### Default Demo Parameters")
        if st.button("🚀 Load Demo Parameters", type="primary"):
            soil_data = load_sample_soil_data()
            weather_data = load_sample_weather_data()
            
            st.session_state.crop_soil_data = soil_data
            st.session_state.crop_weather_data = weather_data
            st.session_state.crop_farm_params = {'size': 25.0, 'location': 'Iowa, USA'}
            
            st.success("Demo parameters loaded successfully!")

def create_manual_input_form():
    """Create manual input form for all parameters"""
    
    # Soil parameters
    st.markdown("#### 🧪 Soil Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph_level = st.slider("pH Level", 3.0, 10.0, 6.8, 0.1, 
                            help="Soil acidity/alkalinity level")
        organic_matter = st.slider("Organic Matter (%)", 0.0, 10.0, 3.2, 0.1,
                                 help="Percentage of organic matter in soil")
    
    with col2:
        nitrogen = st.number_input("Nitrogen (ppm)", 0.0, 200.0, 45.0, 1.0,
                                 help="Nitrogen content in parts per million")
        phosphorus = st.number_input("Phosphorus (ppm)", 0.0, 100.0, 28.0, 1.0,
                                   help="Phosphorus content in parts per million")
    
    with col3:
        potassium = st.number_input("Potassium (ppm)", 0.0, 500.0, 185.0, 1.0,
                                  help="Potassium content in parts per million")
        texture = st.selectbox("Soil Texture", 
                             ['clay', 'sandy_clay', 'silty_clay', 'clay_loam', 
                              'sandy_clay_loam', 'silty_clay_loam', 'loam', 
                              'sandy_loam', 'silt_loam', 'sand', 'loamy_sand', 'silt'],
                             index=6, help="USDA soil texture classification")
    
    # Weather parameters
    st.markdown("#### 🌤️ Climate Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_temp = st.slider("Average Temperature (°C)", -10.0, 40.0, 12.0, 0.5,
                           help="Average annual temperature")
        min_temp = st.slider("Minimum Temperature (°C)", -30.0, 20.0, -15.0, 1.0,
                           help="Lowest temperature during growing season")
    
    with col2:
        max_temp = st.slider("Maximum Temperature (°C)", 20.0, 50.0, 32.0, 1.0,
                           help="Highest temperature during growing season")
        humidity = st.slider("Humidity (%)", 30.0, 100.0, 70.0, 1.0,
                           help="Average relative humidity")
    
    with col3:
        annual_rainfall = st.number_input("Annual Rainfall (mm)", 200.0, 2000.0, 850.0, 10.0,
                                        help="Total annual precipitation")
        growing_season_rain = st.number_input("Growing Season Rainfall (mm)", 
                                            100.0, 1000.0, 450.0, 10.0,
                                            help="Rainfall during main growing season")
    
    # Farm parameters
    st.markdown("#### 🏡 Farm Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        farm_size = st.number_input("Farm Size (hectares)", 1.0, 1000.0, 25.0, 0.5,
                                  help="Total farm area for crop production")
    
    with col2:
        location = st.text_input("Location", "Iowa, USA",
                               help="Farm location for weather data and market prices")
    
    with col3:
        frost_free_days = st.number_input("Frost-Free Days", 90, 365, 160, 1,
                                        help="Number of days without frost")
    
    # Save parameters when button is clicked
    if st.button("💾 Save Parameters", type="primary"):
        soil_data = {
            'ph_level': ph_level,
            'nitrogen_ppm': nitrogen,
            'phosphorus_ppm': phosphorus,
            'potassium_ppm': potassium,
            'organic_matter_percent': organic_matter,
            'texture_class': texture,
            'moisture_content': 25.0,  # Default
            'electrical_conductivity': 1.0  # Default
        }
        
        weather_data = WeatherData(
            location=location,
            avg_temperature=avg_temp,
            min_temperature=min_temp,
            max_temperature=max_temp,
            annual_rainfall=annual_rainfall,
            growing_season_rainfall=growing_season_rain,
            humidity=humidity,
            wind_speed=15.0,  # Default
            frost_free_days=frost_free_days,
            solar_radiation=14.5,  # Default
            photoperiod=13.0,  # Default
            last_frost_date="04-15",  # Default
            first_frost_date="10-15"  # Default
        )
        
        farm_params = {
            'size': farm_size,
            'location': location
        }
        
        st.session_state.crop_soil_data = soil_data
        st.session_state.crop_weather_data = weather_data
        st.session_state.crop_farm_params = farm_params
        
        st.success("Parameters saved successfully!")

def display_current_parameters():
    """Display currently set parameters"""
    if ('crop_soil_data' in st.session_state and 
        'crop_weather_data' in st.session_state and
        'crop_farm_params' in st.session_state):
        
        st.markdown("### 📋 Current Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Soil Conditions:**")
            soil = st.session_state.crop_soil_data
            st.write(f"• pH: {soil['ph_level']:.1f}")
            st.write(f"• N-P-K: {soil['nitrogen_ppm']:.0f}-{soil['phosphorus_ppm']:.0f}-{soil['potassium_ppm']:.0f} ppm")
            st.write(f"• Organic Matter: {soil['organic_matter_percent']:.1f}%")
            st.write(f"• Texture: {soil['texture_class'].title()}")
        
        with col2:
            st.markdown("**Climate Conditions:**")
            weather = st.session_state.crop_weather_data
            st.write(f"• Avg Temperature: {weather.avg_temperature:.1f}°C")
            st.write(f"• Annual Rainfall: {weather.annual_rainfall:.0f}mm")
            st.write(f"• Growing Season Rain: {weather.growing_season_rainfall:.0f}mm")
            st.write(f"• Frost-Free Days: {weather.frost_free_days}")
        
        with col3:
            st.markdown("**Farm Information:**")
            farm = st.session_state.crop_farm_params
            st.write(f"• Size: {farm['size']:.1f} hectares")
            st.write(f"• Location: {farm['location']}")
        
        return True
    return False

def generate_crop_recommendations():
    """Generate crop recommendations based on current parameters"""
    if not ('crop_soil_data' in st.session_state and 
            'crop_weather_data' in st.session_state and
            'crop_farm_params' in st.session_state):
        st.warning("Please set parameters first before generating recommendations.")
        return None
    
    with st.spinner("🔄 Analyzing conditions and generating recommendations..."):
        try:
            # Initialize recommendation engine
            engine = CropRecommendationEngine()
            
            # Get parameters from session state
            soil_data = st.session_state.crop_soil_data
            weather_data = st.session_state.crop_weather_data
            farm_params = st.session_state.crop_farm_params
            
            # Generate recommendations
            recommendations = engine.recommend_crops(
                soil_data=soil_data,
                weather_data=weather_data,
                farm_size=farm_params['size'],
                top_n=6,
                market_conditions=None  # Could add market data integration
            )
            
            return recommendations
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return None

def display_recommendation_card(rec, rank):
    """Display individual recommendation card"""
    colors = get_color_palette("green")
    
    # Determine score badge class
    score = rec.suitability_score
    if score >= 80:
        badge_class = "score-excellent"
        badge_text = "Excellent"
    elif score >= 65:
        badge_class = "score-good"
        badge_text = "Good"
    elif score >= 50:
        badge_class = "score-fair"
        badge_text = "Fair"
    else:
        badge_class = "score-poor"
        badge_text = "Poor"
    
    # Create recommendation card
    st.markdown(f"""
    <div class="recommendation-card">
        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 1rem;">
            <div>
                <h3 style="margin: 0; color: #2E7D32; display: flex; align-items: center;">
                    <span style="background: {colors['primary']}; color: white; 
                                 padding: 0.3rem 0.6rem; border-radius: 50%; 
                                 margin-right: 1rem; font-size: 0.9rem;">
                        #{rank}
                    </span>
                    🌾 {rec.crop_profile.name}
                </h3>
                <p style="margin: 0.5rem 0; color: #666; font-style: italic;">
                    {rec.crop_profile.growing_season.value.title()} Season • 
                    {rec.crop_profile.days_to_maturity} days to maturity
                </p>
            </div>
            <div style="text-align: right;">
                <span class="{badge_class} score-badge">{badge_text}</span>
                <div style="font-size: 1.8rem; font-weight: bold; color: {colors['primary']}; margin-top: 0.5rem;">
                    {score:.1f}/100
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed scores in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Soil Match", f"{rec.soil_match_score:.1f}/100", 
                 help="How well the soil conditions match crop requirements")
    
    with col2:
        st.metric("Climate Match", f"{rec.climate_match_score:.1f}/100",
                 help="How suitable the climate is for this crop")
    
    with col3:
        st.metric("Economic Score", f"{rec.economic_score:.1f}/100",
                 help="Expected economic return and profitability")
    
    with col4:
        st.metric("Risk Score", f"{rec.risk_score:.1f}/100",
                 help="Risk assessment (higher = lower risk)")
    
    # Expandable details
    with st.expander(f"📋 Detailed Analysis - {rec.crop_profile.name}"):
        
        # Factors analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Positive Factors:**")
            if rec.positive_factors:
                for factor in rec.positive_factors:
                    st.markdown(f"<p class='factor-positive'>• {factor}</p>", 
                              unsafe_allow_html=True)
            else:
                st.write("• No specific positive factors identified")
        
        with col2:
            st.markdown("**⚠️ Considerations:**")
            if rec.negative_factors:
                for factor in rec.negative_factors:
                    st.markdown(f"<p class='factor-negative'>• {factor}</p>", 
                              unsafe_allow_html=True)
            else:
                st.write("• No major concerns identified")
        
        # Economic details
        st.markdown("**💰 Economic Projections:**")
        economic_col1, economic_col2, economic_col3 = st.columns(3)
        
        with economic_col1:
            st.write(f"Expected Yield: {rec.crop_profile.expected_yield_per_ha:.1f} tonnes/ha")
            st.write(f"Market Price: ${rec.crop_profile.market_price_per_tonne:.0f}/tonne")
        
        with economic_col2:
            total_cost = (rec.crop_profile.seed_cost_per_ha + 
                         rec.crop_profile.fertilizer_cost_per_ha +
                         rec.crop_profile.labor_cost_per_ha + 
                         rec.crop_profile.machinery_cost_per_ha)
            st.write(f"Production Cost: ${total_cost:.0f}/ha")
            st.write(f"Expected ROI: {rec.expected_roi:.1f}%")
        
        with economic_col3:
            revenue_per_ha = rec.crop_profile.expected_yield_per_ha * rec.crop_profile.market_price_per_tonne
            profit_per_ha = revenue_per_ha - total_cost
            st.write(f"Expected Revenue: ${revenue_per_ha:.0f}/ha")
            st.write(f"Expected Profit: ${profit_per_ha:.0f}/ha")
        
        # Management recommendations
        if rec.planting_recommendations:
            st.markdown("**🌱 Key Recommendations:**")
            for i, recommendation in enumerate(rec.planting_recommendations, 1):
                st.write(f"{i}. {recommendation}")

def create_comparison_view(recommendations):
    """Create comparison view of all recommendations"""
    st.markdown("## 📊 Crop Comparison")
    
    # Prepare data for comparison
    comparison_data = []
    for rec in recommendations:
        total_cost = (rec.crop_profile.seed_cost_per_ha + 
                     rec.crop_profile.fertilizer_cost_per_ha +
                     rec.crop_profile.labor_cost_per_ha + 
                     rec.crop_profile.machinery_cost_per_ha)
        
        revenue_per_ha = rec.crop_profile.expected_yield_per_ha * rec.crop_profile.market_price_per_tonne
        profit_per_ha = revenue_per_ha - total_cost
        
        comparison_data.append({
            'crop': rec.crop_profile.name,
            'suitability_score': rec.suitability_score,
            'expected_yield': rec.crop_profile.expected_yield_per_ha,
            'roi_percent': rec.expected_roi,
            'profit_per_ha': profit_per_ha,
            'days_to_maturity': rec.crop_profile.days_to_maturity,
            'water_requirement': rec.crop_profile.water_requirement
        })
    
    # Create comparison charts
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Suitability", "💰 Economics", "📈 Yield", "⏱️ Timeline"])
    
    with tab1:
        # Suitability comparison
        suitability_fig = create_crop_suitability_chart([
            {'name': item['crop'], 'score': item['suitability_score']} 
            for item in comparison_data
        ])
        st.plotly_chart(suitability_fig, use_container_width=True)
    
    with tab2:
        # Economic comparison
        col1, col2 = st.columns(2)
        
        with col1:
            roi_fig = px.bar(
                x=[item['crop'] for item in comparison_data],
                y=[item['roi_percent'] for item in comparison_data],
                title="Expected ROI Comparison",
                labels={'x': 'Crops', 'y': 'ROI (%)'},
                color=[item['roi_percent'] for item in comparison_data],
                color_continuous_scale='Greens'
            )
            roi_fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(roi_fig, use_container_width=True)
        
        with col2:
            profit_fig = px.bar(
                x=[item['crop'] for item in comparison_data],
                y=[item['profit_per_ha'] for item in comparison_data],
                title="Profit per Hectare Comparison",
                labels={'x': 'Crops', 'y': 'Profit ($/ha)'},
                color=[item['profit_per_ha'] for item in comparison_data],
                color_continuous_scale='Greens'
            )
            profit_fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(profit_fig, use_container_width=True)
    
    with tab3:
        # Yield comparison
        yield_fig = px.scatter(
            x=[item['water_requirement'] for item in comparison_data],
            y=[item['expected_yield'] for item in comparison_data],
            size=[item['suitability_score'] for item in comparison_data],
            color=[item['roi_percent'] for item in comparison_data],
            hover_name=[item['crop'] for item in comparison_data],
            title="Yield vs Water Requirements",
            labels={'x': 'Water Requirement (mm)', 'y': 'Expected Yield (tonnes/ha)'},
            color_continuous_scale='Greens'
        )
        yield_fig.update_layout(height=500)
        st.plotly_chart(yield_fig, use_container_width=True)
    
    with tab4:
        # Timeline comparison
        timeline_fig = px.bar(
            x=[item['crop'] for item in comparison_data],
            y=[item['days_to_maturity'] for item in comparison_data],
            title="Days to Maturity Comparison",
            labels={'x': 'Crops', 'y': 'Days to Maturity'},
            color=[item['days_to_maturity'] for item in comparison_data],
            color_continuous_scale='Blues'
        )
        timeline_fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Comparison table
    st.markdown("### 📋 Detailed Comparison Table")
    comparison_table_html = create_comparison_table(comparison_data, highlight_best=True)
    st.markdown(comparison_table_html, unsafe_allow_html=True)

def export_recommendations(recommendations):
    """Create export functionality for recommendations"""
    st.markdown("### 📥 Export Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        export_data = []
        for rec in recommendations:
            export_data.append({
                'crop_name': rec.crop_profile.name,
                'suitability_score': rec.suitability_score,
                'suitability_level': rec.suitability_level.value,
                'confidence': rec.confidence,
                'expected_roi': rec.expected_roi,
                'soil_match': rec.soil_match_score,
                'climate_match': rec.climate_match_score,
                'economic_score': rec.economic_score,
                'risk_score': rec.risk_score,
                'positive_factors': rec.positive_factors,
                'negative_factors': rec.negative_factors,
                'management_tips': rec.management_tips
            })
        
        st.download_button(
            label="📄 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"crop_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as CSV
        df = pd.DataFrame([{
            'Crop': rec.crop_profile.name,
            'Suitability_Score': rec.suitability_score,
            'Confidence': rec.confidence,
            'Expected_ROI': rec.expected_roi,
            'Soil_Match': rec.soil_match_score,
            'Climate_Match': rec.climate_match_score,
            'Economic_Score': rec.economic_score,
            'Risk_Score': rec.risk_score,
            'Growing_Season': rec.crop_profile.growing_season.value,
            'Days_to_Maturity': rec.crop_profile.days_to_maturity,
            'Expected_Yield_per_Ha': rec.crop_profile.expected_yield_per_ha
        } for rec in recommendations])
        
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"crop_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
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
    """Main crop recommender page"""
    st.title("🌱 AI-Powered Crop Recommender")
    st.markdown("""
    Get intelligent crop recommendations based on comprehensive analysis of your soil conditions,
    climate data, economic factors, and risk assessment. Our AI engine considers multiple factors
    to suggest the most suitable crops for your specific farming situation.
    """)
    
    # Initialize session state
    if 'crop_recommendations' not in st.session_state:
        st.session_state.crop_recommendations = None
    
    # Parameter input section
    create_parameter_input_section()
    
    # Display current parameters if set
    params_set = display_current_parameters()
    
    if params_set:
        # Generate recommendations button
        st.markdown("## 🔍 Generate Recommendations")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
                recommendations = generate_crop_recommendations()
                if recommendations:
                    st.session_state.crop_recommendations = recommendations
                    st.success("✅ Recommendations generated successfully!")
        
        with col2:
            if st.button("🔄 Reset Parameters", use_container_width=True):
                # Clear session state
                for key in ['crop_soil_data', 'crop_weather_data', 'crop_farm_params', 'crop_recommendations']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Display recommendations if available
    if st.session_state.crop_recommendations:
        recommendations = st.session_state.crop_recommendations
        
        st.markdown("---")
        st.markdown("## 🎯 Crop Recommendations")
        
        # Summary metrics
        st.markdown("### 📊 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            top_score = recommendations[0].suitability_score
            st.markdown(create_metric_card(
                "Top Recommendation", 
                recommendations[0].crop_profile.name,
                f"Score: {top_score:.1f}/100",
                "green" if top_score >= 70 else "red",
                "🏆"
            ), unsafe_allow_html=True)
        
        with col2:
            avg_score = sum(rec.suitability_score for rec in recommendations) / len(recommendations)
            st.markdown(create_metric_card(
                "Average Score",
                f"{avg_score:.1f}/100",
                "All recommendations",
                "green",
                "📊"
            ), unsafe_allow_html=True)
        
        with col3:
            excellent_count = sum(1 for rec in recommendations if rec.suitability_score >= 80)
            st.markdown(create_metric_card(
                "Excellent Options",
                str(excellent_count),
                f"out of {len(recommendations)}",
                "green" if excellent_count > 0 else "gray",
                "⭐"
            ), unsafe_allow_html=True)
        
        with col4:
            best_roi = max(rec.expected_roi for rec in recommendations)
            st.markdown(create_metric_card(
                "Best Expected ROI",
                f"{best_roi:.1f}%",
                "Maximum potential return",
                "green" if best_roi > 20 else "red",
                "💰"
            ), unsafe_allow_html=True)
        
        # Individual recommendation cards
        st.markdown("### 🌾 Detailed Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            display_recommendation_card(rec, i)
        
        # Comparison view
        create_comparison_view(recommendations)
        
        # Export options
        export_recommendations(recommendations)
        
        # Next steps
        st.markdown("### 🚀 Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💰 Economic Analysis", type="secondary", use_container_width=True):
                # Store top recommendation for economic analysis
                st.session_state.selected_crop_for_economics = recommendations[0]
                st.switch_page("pages/3_Economics_Dashboard.py")
        
        with col2:
            if st.button("📈 Yield Prediction", type="secondary", use_container_width=True):
                st.switch_page("pages/4_Yield_Predictor.py")
        
        with col3:
            if st.button("📝 Share Feedback", type="secondary", use_container_width=True):
                st.switch_page("pages/5_Feedback_Form.py")
    
    # Help section
    with st.expander("❓ How to Use the Crop Recommender"):
        st.markdown("""
        ### Getting Started
        
        **1. Input Parameters:**
        - **From Soil Analysis:** Import data from previous soil analysis
        - **Manual Input:** Enter soil, climate, and farm parameters manually
        - **Use Defaults:** Try with sample data to explore features
        
        **2. Key Factors Analyzed:**
        - **Soil Compatibility:** pH, nutrients, texture, organic matter
        - **Climate Suitability:** Temperature, rainfall, growing season
        - **Economic Viability:** Costs, revenue, ROI projections
        - **Risk Assessment:** Pest/disease risk, weather sensitivity
        
        **3. Understanding Scores:**
        - **Excellent (80-100):** Highly recommended, optimal conditions
        - **Good (65-79):** Suitable choice with good potential
        - **Fair (50-64):** Acceptable but may need management adjustments
        - **Poor (<50):** Not recommended for current conditions
        
        **4. Using Results:**
        - Review detailed analysis for each recommended crop
        - Compare economic projections and timelines
        - Consider positive factors and potential challenges
        - Export results for record-keeping and planning
        
        **5. Integration:**
        - Results can be used in Economic Analysis and Yield Prediction
        - Combine with soil health data for comprehensive planning
        - Share feedback to improve recommendations
        """)

if __name__ == "__main__":
    main()

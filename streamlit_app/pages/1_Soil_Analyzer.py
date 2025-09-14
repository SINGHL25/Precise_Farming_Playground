
"""
Soil Analyzer Dashboard Page

Interactive interface for soil data upload, analysis, and visualization.
Provides comprehensive soil health assessment with actionable recommendations.

Features:
- File upload (CSV, JSON, Excel)
- Manual data entry
- Interactive visualizations
- Detailed analysis reports
- Export capabilities

Author: Precision Farming Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.visual_helpers import create_soil_radar_chart, create_nutrient_gauge, format_recommendation_card
from src.soil_analyzer import SoilAnalyzer, SoilSample

# Page configuration
st.set_page_config(
    page_title="Soil Analyzer - Precision Farming",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .upload-section {
        background: linear-gradient(135deg, #E3F2FD, #F3E5F5);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #4CAF50;
        text-align: center;
        margin: 1rem 0;
    }
    
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #E8F5E8, #F1F8E9);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .recommendation-item {
        background: #F8F9FA;
        padding: 1rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .warning-item {
        background: #FFF3E0;
        padding: 1rem;
        border-left: 4px solid #FF9800;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .health-score-excellent { color: #2E7D32; }
    .health-score-good { color: #4CAF50; }
    .health-score-fair { color: #FF9800; }
    .health-score-poor { color: #F44336; }
    .health-score-critical { color: #B71C1C; }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample soil data for demonstration"""
    return {
        'sample_id': 'DEMO_FIELD_001',
        'location': 'Demo Farm, Sample County',
        'date_collected': datetime.now().strftime('%Y-%m-%d'),
        'sand_percent': 35.0,
        'silt_percent': 40.0,
        'clay_percent': 25.0,
        'ph_level': 6.8,
        'nitrogen_ppm': 45.0,
        'phosphorus_ppm': 28.0,
        'potassium_ppm': 185.0,
        'organic_matter_percent': 3.2,
        'bulk_density': 1.3,
        'moisture_content': 24.0,
        'cation_exchange_capacity': 16.5,
        'electrical_conductivity': 1.1,
        'temperature_celsius': 18.5
    }

def parse_uploaded_file(uploaded_file):
    """Parse uploaded file and extract soil data"""
    try:
        if uploaded_file.type == "application/json":
            # JSON file
            data = json.load(uploaded_file)
            return [data] if isinstance(data, dict) else data
        
        elif uploaded_file.type == "text/csv":
            # CSV file
            df = pd.read_csv(uploaded_file)
            return df.to_dict('records')
        
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            # Excel file
            df = pd.read_excel(uploaded_file)
            return df.to_dict('records')
        
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return None
            
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None

def create_soil_sample_from_dict(data):
    """Create SoilSample object from dictionary"""
    try:
        return SoilSample(
            sample_id=str(data.get('sample_id', 'UNKNOWN')),
            location=str(data.get('location', 'Unknown Location')),
            date_collected=str(data.get('date_collected', datetime.now().date())),
            sand_percent=float(data.get('sand_percent', 33.3)),
            silt_percent=float(data.get('silt_percent', 33.3)),
            clay_percent=float(data.get('clay_percent', 33.4)),
            bulk_density=float(data.get('bulk_density', 1.3)),
            moisture_content=float(data.get('moisture_content', 25.0)),
            ph_level=float(data.get('ph_level', 7.0)),
            nitrogen_ppm=float(data.get('nitrogen_ppm', 30.0)),
            phosphorus_ppm=float(data.get('phosphorus_ppm', 20.0)),
            potassium_ppm=float(data.get('potassium_ppm', 150.0)),
            organic_matter_percent=float(data.get('organic_matter_percent', 2.5)),
            cation_exchange_capacity=float(data.get('cation_exchange_capacity', 15.0)),
            electrical_conductivity=data.get('electrical_conductivity'),
            temperature_celsius=data.get('temperature_celsius')
        )
    except (ValueError, TypeError) as e:
        st.error(f"Error creating soil sample: {str(e)}")
        return None

def display_soil_health_score(score, category):
    """Display soil health score with color coding"""
    color_class = f"health-score-{category.lower()}"
    
    st.markdown(f"""
    <div class="metric-highlight">
        <h2 class="{color_class}" style="margin: 0; font-size: 3rem;">{score:.1f}/100</h2>
        <h3 class="{color_class}" style="margin: 0.5rem 0;">
            {category.replace('_', ' ').title()} Health
        </h3>
        <p style="margin: 0; color: #666;">Overall Soil Health Score</p>
    </div>
    """, unsafe_allow_html=True)

def create_texture_triangle_plot(sand, silt, clay):
    """Create soil texture triangle visualization"""
    # Create ternary plot for soil texture
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': [clay],
        'b': [silt], 
        'c': [sand],
        'marker': {
            'size': 15,
            'color': '#4CAF50',
            'symbol': 'circle'
        },
        'name': 'Your Soil'
    }))
    
    fig.update_layout(
        title="Soil Texture Classification",
        ternary=dict(
            sum=100,
            aaxis=dict(title='Clay %', min=0, linewidth=2, ticks="outside"),
            baxis=dict(title='Silt %', min=0, linewidth=2, ticks="outside"),
            caxis=dict(title='Sand %', min=0, linewidth=2, ticks="outside")
        ),
        showlegend=True,
        height=500
    )
    
    return fig

def create_nutrient_comparison_chart(analysis_result):
    """Create nutrient level comparison chart"""
    nutrients = []
    values = []
    statuses = []
    
    for nutrient, data in analysis_result.nutrient_analysis.items():
        if isinstance(data, dict) and 'value' in data:
            nutrients.append(nutrient.title())
            values.append(data['value'])
            status = data.get('status', 'Unknown')
            statuses.append(status)
    
    colors = []
    for status in statuses:
        if 'high' in status.lower() or 'excellent' in status.lower():
            colors.append('#2E7D32')
        elif 'good' in status.lower() or 'optimal' in status.lower():
            colors.append('#4CAF50')  
        elif 'fair' in status.lower() or 'medium' in status.lower():
            colors.append('#FF9800')
        else:
            colors.append('#F44336')
    
    fig = px.bar(
        x=nutrients,
        y=values,
        title="Nutrient Levels Analysis",
        labels={'x': 'Nutrients', 'y': 'Concentration'},
        color=colors,
        color_discrete_map='identity'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def display_recommendations(recommendations, title="Recommendations"):
    """Display recommendations in formatted cards"""
    st.markdown(f"### 📋 {title}")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-item">
            <strong>{i}.</strong> {rec}
        </div>
        """, unsafe_allow_html=True)

def display_limitations(limitations):
    """Display limitations and warnings"""
    if limitations:
        st.markdown("### ⚠️ Limitations & Considerations")
        
        for i, limitation in enumerate(limitations, 1):
            st.markdown(f"""
            <div class="warning-item">
                <strong>{i}.</strong> {limitation}
            </div>
            """, unsafe_allow_html=True)

def manual_data_entry():
    """Create manual data entry form"""
    st.markdown("### ✍️ Manual Data Entry")
    st.markdown("Enter your soil test results manually:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Sample Information**")
        sample_id = st.text_input("Sample ID", value="FIELD_001")
        location = st.text_input("Location", value="My Farm")
        date_collected = st.date_input("Collection Date", value=datetime.now())
    
    with col2:
        st.markdown("**Soil Texture**")
        sand_percent = st.slider("Sand %", 0.0, 100.0, 35.0, 0.1)
        silt_percent = st.slider("Silt %", 0.0, 100.0, 40.0, 0.1)
        clay_percent = st.slider("Clay %", 0.0, 100.0, 25.0, 0.1)
        
        # Validate texture percentages
        total_texture = sand_percent + silt_percent + clay_percent
        if abs(total_texture - 100.0) > 1.0:
            st.warning(f"Texture percentages sum to {total_texture:.1f}%. Should equal 100%")
    
    with col3:
        st.markdown("**Chemical Properties**")
        ph_level = st.slider("pH Level", 3.0, 10.0, 6.8, 0.1)
        nitrogen_ppm = st.number_input("Nitrogen (ppm)", 0.0, 200.0, 45.0, 1.0)
        phosphorus_ppm = st.number_input("Phosphorus (ppm)", 0.0, 100.0, 28.0, 1.0)
        potassium_ppm = st.number_input("Potassium (ppm)", 0.0, 500.0, 185.0, 1.0)
        organic_matter = st.slider("Organic Matter %", 0.0, 10.0, 3.2, 0.1)
    
    # Additional properties in expander
    with st.expander("Additional Properties (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            bulk_density = st.number_input("Bulk Density (g/cm³)", 0.8, 2.0, 1.3, 0.1)
            moisture_content = st.number_input("Moisture Content %", 0.0, 50.0, 24.0, 1.0)
            cec = st.number_input("Cation Exchange Capacity", 5.0, 40.0, 16.5, 0.1)
        
        with col2:
            electrical_conductivity = st.number_input("Electrical Conductivity (dS/m)", 0.0, 10.0, 1.1, 0.1)
            temperature = st.number_input("Soil Temperature (°C)", 0.0, 40.0, 18.5, 0.1)
    
    # Create soil sample from manual input
    if st.button("🔍 Analyze Soil", type="primary", use_container_width=True):
        soil_data = {
            'sample_id': sample_id,
            'location': location,
            'date_collected': date_collected.strftime('%Y-%m-%d'),
            'sand_percent': sand_percent,
            'silt_percent': silt_percent,
            'clay_percent': clay_percent,
            'ph_level': ph_level,
            'nitrogen_ppm': nitrogen_ppm,
            'phosphorus_ppm': phosphorus_ppm,
            'potassium_ppm': potassium_ppm,
            'organic_matter_percent': organic_matter,
            'bulk_density': bulk_density,
            'moisture_content': moisture_content,
            'cation_exchange_capacity': cec,
            'electrical_conductivity': electrical_conductivity if electrical_conductivity > 0 else None,
            'temperature_celsius': temperature if temperature > 0 else None
        }
        
        return soil_data
    
    return None

def main():
    """Main soil analyzer page"""
    st.title("🧪 Soil Analyzer")
    st.markdown("""
    Upload your soil test results or enter data manually to get comprehensive soil health analysis,
    nutrient recommendations, and crop suitability insights.
    """)
    
    # Initialize session state
    if 'soil_analysis_result' not in st.session_state:
        st.session_state.soil_analysis_result = None
    
    # Data input methods
    st.markdown("## 📤 Data Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload File", "✍️ Manual Entry", "🎯 Use Demo Data"],
        horizontal=True
    )
    
    soil_data = None
    
    if input_method == "📁 Upload File":
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: #2E7D32; margin-top: 0;">📁 Upload Soil Data</h3>
            <p>Supported formats: CSV, JSON, Excel</p>
            <p style="font-size: 0.9rem; color: #666;">
                Your data should include: pH, N-P-K levels, organic matter, soil texture
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'json', 'xlsx', 'xls'],
            help="Upload soil test results in CSV, JSON, or Excel format"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Parse file
            parsed_data = parse_uploaded_file(uploaded_file)
            
            if parsed_data:
                if len(parsed_data) == 1:
                    soil_data = parsed_data[0]
                    st.info("Single soil sample detected")
                else:
                    st.info(f"Multiple samples detected ({len(parsed_data)}). Using first sample for analysis.")
                    soil_data = parsed_data[0]
                    
                    # Show sample selection
                    sample_options = [f"{i+1}. {data.get('sample_id', f'Sample {i+1}')} - {data.get('location', 'Unknown')}" 
                                    for i, data in enumerate(parsed_data)]
                    
                    selected_idx = st.selectbox("Select sample to analyze:", 
                                              range(len(sample_options)),
                                              format_func=lambda x: sample_options[x])
                    
                    soil_data = parsed_data[selected_idx]
    
    elif input_method == "✍️ Manual Entry":
        soil_data = manual_data_entry()
    
    elif input_method == "🎯 Use Demo Data":
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: #2E7D32; margin-top: 0;">🎯 Demo Data</h3>
            <p>Use sample soil data to explore the analysis features</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Load Demo Data", type="primary"):
            soil_data = create_sample_data()
            st.success("Demo data loaded successfully!")
    
    # Perform analysis if data is available
    if soil_data:
        try:
            # Create soil sample object
            soil_sample = create_soil_sample_from_dict(soil_data)
            
            if soil_sample:
                # Initialize analyzer
                with st.spinner("Analyzing soil sample..."):
                    analyzer = SoilAnalyzer()
                    analysis_result = analyzer.analyze_sample(soil_sample)
                    st.session_state.soil_analysis_result = analysis_result
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    
    # Display results if analysis is available
    if st.session_state.soil_analysis_result:
        analysis_result = st.session_state.soil_analysis_result
        sample = analysis_result.sample
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Sample information header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"""
            **Sample:** {sample.sample_id}  
            **Location:** {sample.location}  
            **Date:** {sample.date_collected}
            """)
        
        with col2:
            st.markdown(f"""
            **Texture:** {analysis_result.texture_class.value.title()}  
            **pH Level:** {sample.ph_level}
            """)
        
        with col3:
            st.markdown(f"""
            **Organic Matter:** {sample.organic_matter_percent}%  
            **Moisture:** {sample.moisture_content}%
            """)
        
        # Health score display
        st.markdown("### 🏆 Soil Health Assessment")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            display_soil_health_score(
                analysis_result.health_score, 
                analysis_result.health_category.value
            )
        
        with col2:
            # Create and display texture triangle
            texture_fig = create_texture_triangle_plot(
                sample.sand_percent, 
                sample.silt_percent, 
                sample.clay_percent
            )
            st.plotly_chart(texture_fig, use_container_width=True)
        
        with col3:
            # Key metrics
            st.markdown("**Key Metrics:**")
            st.metric("pH Level", f"{sample.ph_level:.1f}", 
                     "Optimal" if 6.0 <= sample.ph_level <= 7.5 else "Needs attention")
            st.metric("Nitrogen", f"{sample.nitrogen_ppm:.0f} ppm",
                     "Good" if sample.nitrogen_ppm >= 40 else "Low")
            st.metric("Organic Matter", f"{sample.organic_matter_percent:.1f}%",
                     "Excellent" if sample.organic_matter_percent >= 3.0 else "Fair")
        
        # Detailed analysis
        st.markdown("### 🔬 Detailed Analysis")
        
        # Nutrient analysis chart
        nutrient_fig = create_nutrient_comparison_chart(analysis_result)
        st.plotly_chart(nutrient_fig, use_container_width=True)
        
        # Recommendations and limitations
        col1, col2 = st.columns(2)
        
        with col1:
            display_recommendations(analysis_result.recommendations)
        
        with col2:
            display_limitations(analysis_result.limitations)
        
        # Suitable crops
        if analysis_result.suitability_crops:
            st.markdown("### 🌱 Suitable Crops")
            
            crops_text = ", ".join(analysis_result.suitability_crops[:8])  # Show first 8 crops
            if len(analysis_result.suitability_crops) > 8:
                crops_text += f" and {len(analysis_result.suitability_crops) - 8} more..."
            
            st.markdown(f"""
            <div class="analysis-card">
                <p style="font-size: 1.1rem; margin: 0;">
                    <strong>Recommended crops for your soil conditions:</strong><br>
                    {crops_text}
                </p>
                <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
                    💡 For detailed crop recommendations with economic analysis, 
                    visit the <a href="/2_Crop_Recommender" style="color: #4CAF50;">Crop Recommender</a> page.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("### 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as JSON
            export_data = {
                'sample_info': {
                    'sample_id': sample.sample_id,
                    'location': sample.location,
                    'date_collected': sample.date_collected
                },
                'analysis_results': {
                    'health_score': analysis_result.health_score,
                    'health_category': analysis_result.health_category.value,
                    'texture_class': analysis_result.texture_class.value,
                    'nutrient_analysis': analysis_result.nutrient_analysis
                },
                'recommendations': analysis_result.recommendations,
                'suitable_crops': analysis_result.suitability_crops
            }
            
            st.download_button(
                label="📄 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"soil_analysis_{sample.sample_id}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export as CSV
            csv_data = {
                'Sample_ID': [sample.sample_id],
                'Location': [sample.location],
                'Date': [sample.date_collected],
                'Health_Score': [analysis_result.health_score],
                'Health_Category': [analysis_result.health_category.value],
                'Texture_Class': [analysis_result.texture_class.value],
                'pH': [sample.ph_level],
                'Nitrogen_ppm': [sample.nitrogen_ppm],
                'Phosphorus_ppm': [sample.phosphorus_ppm],
                'Potassium_ppm': [sample.potassium_ppm],
                'Organic_Matter_percent': [sample.organic_matter_percent]
            }
            
            df = pd.DataFrame(csv_data)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📊 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"soil_analysis_{sample.sample_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Generate PDF report (placeholder)
            st.button(
                "📋 Generate PDF Report",
                help="PDF report generation coming soon!",
                disabled=True
            )
        
        # Action buttons
        st.markdown("### 🚀 Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌾 Get Crop Recommendations", type="secondary", use_container_width=True):
                st.switch_page("pages/2_Crop_Recommender.py")
        
        with col2:
            if st.button("💰 Economic Analysis", type="secondary", use_container_width=True):
                st.switch_page("pages/3_Economics_Dashboard.py")
        
        with col3:
            if st.button("📈 Yield Prediction", type="secondary", use_container_width=True):
                st.switch_page("pages/4_Yield_Predictor.py")
    
    # Help section
    with st.expander("❓ Need Help?"):
        st.markdown("""
        ### How to Use the Soil Analyzer
        
        **1. Data Input Options:**
        - **Upload File:** Use lab results in CSV, JSON, or Excel format
        - **Manual Entry:** Enter soil test results using the interactive form  
        - **Demo Data:** Try with sample data to explore features
        
        **2. Required Data:**
        - pH level (0-14 scale)
        - Nitrogen, Phosphorus, Potassium levels (ppm)
        - Organic matter percentage
        - Soil texture (sand, silt, clay percentages)
        
        **3. Understanding Results:**
        - **Health Score:** 0-100 scale indicating overall soil condition
        - **Texture Class:** USDA soil classification (loam, clay, sand, etc.)
        - **Nutrient Analysis:** Individual nutrient status and recommendations
        - **Suitable Crops:** Crops well-matched to your soil conditions
        
        **4. File Format Examples:**
        - CSV: Headers should match parameter names (ph_level, nitrogen_ppm, etc.)
        - JSON: Nested structure with soil properties as key-value pairs
        - Excel: First row as headers, data in subsequent rows
        
        **Need more help?** Contact support or check our documentation.
        """)

if __name__ == "__main__":
    main()

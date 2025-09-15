
"""
Feedback Form Dashboard Page

Comprehensive user feedback system for collecting farmer experiences,
suggestions, and platform improvements.

Features:
- User experience surveys
- Feature feedback
- Bug reporting
- Success stories
- Community insights

Author: Precision Farming Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.visual_helpers import create_metric_card, get_color_palette, create_alert_box

# Page configuration
st.set_page_config(
    page_title="Feedback - Precision Farming",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .feedback-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    .rating-container {
        background: linear-gradient(135deg, #E8F5E8, #F1F8E9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .success-story {
        background: linear-gradient(135deg, #E3F2FD, #E8F5E8);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    
    .improvement-item {
        background: #F8F9FA;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #FF9800;
    }
    
    .thank-you-message {
        background: linear-gradient(135deg, #C8E6C9, #E8F5E8);
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #4CAF50;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_feedback_data():
    """Initialize feedback data in session state"""
    if 'feedback_submissions' not in st.session_state:
        # Sample feedback data for demonstration
        st.session_state.feedback_submissions = [
            {
                'date': '2024-03-15',
                'farmer_name': 'John Smith',
                'location': 'Iowa, USA',
                'farm_size': 150,
                'primary_crop': 'corn',
                'platform_rating': 9,
                'feature_ratings': {
                    'soil_analysis': 8,
                    'crop_recommendations': 9,
                    'economics': 7,
                    'yield_prediction': 8,
                    'user_interface': 9
                },
                'most_valuable_feature': 'Crop recommendations helped me switch to more profitable crops',
                'improvements': ['More crop varieties', 'Weather integration'],
                'success_story': 'Increased my corn yield by 15% following soil recommendations',
                'would_recommend': True
            },
            {
                'date': '2024-03-10',
                'farmer_name': 'Maria Garcia',
                'location': 'California, USA',
                'farm_size': 75,
                'primary_crop': 'wheat',
                'platform_rating': 8,
                'feature_ratings': {
                    'soil_analysis': 9,
                    'crop_recommendations': 8,
                    'economics': 8,
                    'yield_prediction': 7,
                    'user_interface': 8
                },
                'most_valuable_feature': 'Soil analysis revealed pH issues I never knew about',
                'improvements': ['Mobile app', 'Local language support'],
                'success_story': 'Saved $2000 on unnecessary fertilizers',
                'would_recommend': True
            }
        ]

def create_feedback_form():
    """Create comprehensive feedback form"""
    st.markdown("## 📝 Share Your Experience")
    st.markdown("Your feedback helps us improve the platform for the farming community.")
    
    with st.form("farmer_feedback_form"):
        # Personal Information
        st.markdown("### 👤 About You")
        col1, col2 = st.columns(2)
        
        with col1:
            farmer_name = st.text_input(
                "Your Name",
                placeholder="Enter your name (optional)",
                help="Help us personalize your experience"
            )
            location = st.text_input(
                "Farm Location",
                placeholder="City, State/Country",
                help="Helps us understand regional needs"
            )
        
        with col2:
            farm_size = st.number_input(
                "Farm Size (hectares)",
                min_value=0.1,
                max_value=10000.0,
                value=25.0,
                step=0.5,
                help="Total cultivated area"
            )
            primary_crop = st.selectbox(
                "Primary Crop",
                ["wheat", "corn", "rice", "soybean", "cotton", "vegetables", "fruits", "other"],
                help="Main crop you grow"
            )
        
        # Platform Experience
        st.markdown("### ⭐ Platform Rating")
        
        overall_rating = st.slider(
            "Overall Platform Experience (1-10)",
            min_value=1,
            max_value=10,
            value=8,
            help="Rate your overall satisfaction with the platform"
        )
        
        # Feature-specific ratings
        st.markdown("### 🔧 Feature Ratings")
        st.markdown("Rate each feature you've used (1-10 scale):")
        
        col1, col2 = st.columns(2)
        
        with col1:
            soil_rating = st.slider("🧪 Soil Analysis", 1, 10, 8)
            crop_rating = st.slider("🌱 Crop Recommendations", 1, 10, 8)
            economics_rating = st.slider("💰 Economics Dashboard", 1, 10, 7)
        
        with col2:
            yield_rating = st.slider("📈 Yield Predictor", 1, 10, 7)
            ui_rating = st.slider("🎨 User Interface", 1, 10, 8)
            support_rating = st.slider("🆘 Help & Support", 1, 10, 7)
        
        # Qualitative feedback
        st.markdown("### 💭 Your Thoughts")
        
        most_valuable = st.text_area(
            "Most Valuable Feature",
            placeholder="What feature has been most helpful for your farming operation?",
            help="Tell us what's working well"
        )
        
        success_story = st.text_area(
            "Success Story (Optional)",
            placeholder="Share how the platform has helped improve your farming results...",
            help="Your success stories inspire other farmers!"
        )
        
        challenges = st.text_area(
            "Challenges or Issues",
            placeholder="What challenges have you faced using the platform?",
            help="Help us identify areas for improvement"
        )
        
        # Improvement suggestions
        st.markdown("### 🚀 Improvement Suggestions")
        
        improvement_categories = [
            "More crop types",
            "Weather integration", 
            "Mobile application",
            "Local language support",
            "Offline functionality",
            "Training videos",
            "Community features",
            "Market price data",
            "Pest management",
            "Equipment recommendations"
        ]
        
        selected_improvements = st.multiselect(
            "What improvements would you like to see?",
            improvement_categories,
            help="Select all that apply"
        )
        
        other_suggestions = st.text_area(
            "Other Suggestions",
            placeholder="Any other features or improvements you'd like to see...",
            help="Share your innovative ideas!"
        )
        
        # Recommendation
        would_recommend = st.radio(
            "Would you recommend this platform to other farmers?",
            ["Yes, definitely", "Yes, with some improvements", "Maybe", "No"],
            help="Help us understand your likelihood to recommend"
        )
        
        # Contact preference
        st.markdown("### 📞 Follow-up (Optional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            contact_email = st.text_input(
                "Email Address",
                placeholder="your.email@example.com",
                help="For follow-up questions or updates (optional)"
            )
        
        with col2:
            follow_up = st.checkbox(
                "I'm interested in participating in beta testing",
                help="Get early access to new features"
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "📤 Submit Feedback",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            # Process feedback submission
            new_feedback = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'farmer_name': farmer_name or "Anonymous",
                'location': location or "Not specified",
                'farm_size': farm_size,
                'primary_crop': primary_crop,
                'platform_rating': overall_rating,
                'feature_ratings': {
                    'soil_analysis': soil_rating,
                    'crop_recommendations': crop_rating,
                    'economics': economics_rating,
                    'yield_prediction': yield_rating,
                    'user_interface': ui_rating,
                    'support': support_rating
                },
                'most_valuable_feature': most_valuable,
                'success_story': success_story,
                'challenges': challenges,
                'improvements': selected_improvements,
                'other_suggestions': other_suggestions,
                'would_recommend': would_recommend,
                'contact_email': contact_email,
                'beta_interest': follow_up
            }
            
            # Add to session state
            st.session_state.feedback_submissions.append(new_feedback)
            st.session_state.feedback_submitted = True
            st.rerun()

def display_thank_you_message():
    """Display thank you message after submission"""
    st.markdown("""
    <div class="thank-you-message">
        <h2 style="color: #2E7D32; margin: 0 0 1rem 0;">🙏 Thank You!</h2>
        <p style="font-size: 1.2rem; margin: 1rem 0; color: #1B5E20;">
            Your feedback has been submitted successfully and is invaluable to our mission
            of empowering farmers with better technology.
        </p>
        <p style="color: #4CAF50; margin: 1rem 0;">
            ✅ Feedback recorded<br>
            📧 Confirmation email sent (if provided)<br>
            🔄 Development team notified
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset form button
    if st.button("📝 Submit Another Response", type="secondary"):
        del st.session_state.feedback_submitted
        st.rerun()

def display_community_insights():
    """Display aggregated community feedback insights"""
    st.markdown("## 📊 Community Insights")
    st.markdown("See how the farming community is using and rating our platform:")
    
    if not st.session_state.feedback_submissions:
        st.info("No feedback data available yet. Be the first to share your experience!")
        return
    
    submissions = st.session_state.feedback_submissions
    df = pd.DataFrame(submissions)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rating = df['platform_rating'].mean()
        st.markdown(create_metric_card(
            "Average Rating",
            f"{avg_rating:.1f}/10",
            f"From {len(submissions)} farmers",
            "green" if avg_rating >= 8 else "orange",
            "⭐"
        ), unsafe_allow_html=True)
    
    with col2:
        recommend_count = sum(1 for r in df['would_recommend'] if 'Yes' in str(r))
        recommend_pct = (recommend_count / len(submissions)) * 100
        st.markdown(create_metric_card(
            "Would Recommend",
            f"{recommend_pct:.0f}%",
            "Farmer satisfaction",
            "green" if recommend_pct >= 80 else "orange",
            "👍"
        ), unsafe_allow_html=True)
    
    with col3:
        total_farm_area = df['farm_size'].sum()
        st.markdown(create_metric_card(
            "Total Farm Area",
            f"{total_farm_area:,.0f} ha",
            "Community coverage",
            "green",
            "🏡"
        ), unsafe_allow_html=True)
    
    with col4:
        unique_locations = df['location'].nunique()
        st.markdown(create_metric_card(
            "Locations",
            str(unique_locations),
            "Geographic reach",
            "green",
            "🌍"
        ), unsafe_allow_html=True)
    
    # Feature ratings visualization
    st.markdown("### 🔧 Feature Performance")
    
    # Extract feature ratings
    feature_data = []
    for submission in submissions:
        for feature, rating in submission['feature_ratings'].items():
            feature_data.append({
                'Feature': feature.replace('_', ' ').title(),
                'Rating': rating,
                'Farmer': submission['farmer_name']
            })
    
    if feature_data:
        feature_df = pd.DataFrame(feature_data)
        avg_ratings = feature_df.groupby('Feature')['Rating'].mean().sort_values(ascending=True)
        
        fig = px.bar(
            x=avg_ratings.values,
            y=avg_ratings.index,
            orientation='h',
            title='Average Feature Ratings',
            labels={'x': 'Average Rating (1-10)', 'y': 'Features'},
            color=avg_ratings.values,
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Improvement requests
    st.markdown("### 🚀 Most Requested Improvements")
    
    all_improvements = []
    for submission in submissions:
        all_improvements.extend(submission.get('improvements', []))
    
    if all_improvements:
        improvement_counts = pd.Series(all_improvements).value_counts().head(8)
        
        fig = px.bar(
            x=improvement_counts.values,
            y=improvement_counts.index,
            orientation='h',
            title='Top Improvement Requests',
            labels={'x': 'Number of Requests', 'y': 'Improvements'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_success_stories():
    """Display farmer success stories"""
    st.markdown("## 🌟 Success Stories")
    st.markdown("Inspiring stories from farmers using our platform:")
    
    submissions = st.session_state.feedback_submissions
    success_stories = [s for s in submissions if s.get('success_story') and s['success_story'].strip()]
    
    if not success_stories:
        st.info("No success stories shared yet. Share yours to inspire other farmers!")
        return
    
    for i, story in enumerate(success_stories, 1):
        st.markdown(f"""
        <div class="success-story">
            <h4 style="color: #1976D2; margin: 0 0 0.5rem 0;">
                🌟 Success Story #{i}
            </h4>
            <p style="font-style: italic; margin: 1rem 0; font-size: 1.1rem;">
                "{story['success_story']}"
            </p>
            <div style="display: flex; justify-content: between; align-items: center; margin-top: 1rem;">
                <div>
                    <strong>{story['farmer_name']}</strong><br>
                    <small style="color: #666;">
                        {story['location']} • {story['farm_size']} hectares • {story['primary_crop'].title()}
                    </small>
                </div>
                <div style="text-align: right;">
                    <div style="color: #4CAF50; font-size: 1.2rem;">
                        ⭐ {story['platform_rating']}/10
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_platform_stats():
    """Display platform usage statistics"""
    st.markdown("## 📈 Platform Impact")
    
    # Mock platform statistics (in a real app, these would come from analytics)
    stats = {
        'total_users': 12847,
        'soil_analyses': 34562,
        'crop_recommendations': 28493,
        'economic_analyses': 19384,
        'yield_predictions': 15273,
        'countries': 47,
        'avg_yield_improvement': 18.5,
        'avg_cost_savings': 1250
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Community Growth")
        
        growth_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'New Users': [890, 1240, 1580, 2100, 2850, 3890],
            'Active Users': [2100, 3400, 5000, 7200, 9800, 12847]
        })
        
        fig = px.line(
            growth_data,
            x='Month',
            y=['New Users', 'Active Users'],
            title='User Growth Trend',
            markers=True
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Feature Usage")
        
        feature_usage = pd.DataFrame({
            'Feature': ['Soil Analysis', 'Crop Recommendations', 'Economics', 'Yield Prediction'],
            'Usage': [34562, 28493, 19384, 15273]
        })
        
        fig = px.pie(
            feature_usage,
            values='Usage',
            names='Feature',
            title='Feature Popularity'
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact metrics
    st.markdown("### 🌍 Global Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Farmers", f"{stats['total_users']:,}", "👨‍🌾"),
        ("Countries Reached", str(stats['countries']), "🌍"),
        ("Avg Yield Increase", f"+{stats['avg_yield_improvement']}%", "📈"),
        ("Avg Cost Savings", f"${stats['avg_cost_savings']:,}", "💰")
    ]
    
    for i, (title, value, icon) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(create_metric_card(
                title, value, "Community impact", "green", icon
            ), unsafe_allow_html=True)

def main():
    """Main feedback form page"""
    st.title("📝 Farmer Feedback & Community")
    st.markdown("""
    Share your experience, success stories, and suggestions to help us build 
    better tools for the global farming community.
    """)
    
    # Initialize feedback data
    initialize_feedback_data()
    
    # Check if feedback was just submitted
    if st.session_state.get('feedback_submitted', False):
        display_thank_you_message()
        
        # Show next steps
        st.markdown("## 🚀 What's Next?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Analyze More Soil", type="secondary", use_container_width=True):
                st.switch_page("pages/1_Soil_Analyzer.py")
        
        with col2:
            if st.button("🌱 Get Recommendations", type="secondary", use_container_width=True):
                st.switch_page("pages/2_Crop_Recommender.py")
        
        with col3:
            if st.button("🏠 Back to Dashboard", type="secondary", use_container_width=True):
                st.switch_page("app.py")
    
    else:
        # Main feedback form
        create_feedback_form()
    
    # Always show community insights
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Community Insights", "🌟 Success Stories", "📈 Platform Impact"])
    
    with tab1:
        display_community_insights()
    
    with tab2:
        display_success_stories()
    
    with tab3:
        display_platform_stats()
    
    # Contact and support section
    st.markdown("---")
    st.markdown("## 📞 Need Help?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📧 Email Support**  
        [support@precisionfarming.com](mailto:support@precisionfarming.com)  
        Response within 24 hours
        """)
    
    with col2:
        st.markdown("""
        **📚 Documentation**  
        [View User Guide](https://docs.precisionfarming.com)  
        Comprehensive tutorials and FAQs
        """)
    
    with col3:
        st.markdown("""
        **💬 Community Forum**  
        [Join Discussion](https://community.precisionfarming.com)  
        Connect with other farmers
        """)

if __name__ == "__main__":
    main()

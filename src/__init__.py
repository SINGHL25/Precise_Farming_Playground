
"""
Precision Farming Core Library

A comprehensive Python library for precision farming analysis, providing
tools for soil analysis, crop recommendations, economic modeling, and
agricultural insights.

Author: Precision Farming Team
Date: 2024
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Precision Farming Team"
__license__ = "MIT"
__email__ = "support@precisionfarming.com"

# Core module imports
from .soil_analyzer import (
    SoilAnalyzer,
    SoilSample,
    SoilTexture,
    SoilHealth,
    SoilAnalysisResult,
    load_soil_data_from_csv
)

from .crop_recommender import (
    CropRecommendationEngine,
    CropProfile,
    CropType,
    Season,
    SuitabilityLevel,
    CropRecommendation,
    WeatherData,
    create_sample_weather_data
)

from .economics_calculator import (
    EconomicsCalculator,
    EconomicAnalysis,
    CostBreakdown,
    RevenueProjection,
    RiskAssessment,
    ScenarioAnalysis
)

from .stats_visualizer import (
    StatsVisualizer,
    create_soil_health_chart,
    create_crop_comparison_chart,
    create_economic_dashboard,
    create_yield_trend_chart
)

from .feedback_manager import (
    FeedbackManager,
    FeedbackEntry,
    FeedbackAnalytics,
    UserExperience
)

from .utils import (
    load_config,
    save_config,
    validate_soil_data,
    calculate_growing_degree_days,
    convert_units,
    format_results,
    export_data,
    import_data
)

# Package metadata
__all__ = [
    # Soil Analysis
    'SoilAnalyzer',
    'SoilSample', 
    'SoilTexture',
    'SoilHealth',
    'SoilAnalysisResult',
    'load_soil_data_from_csv',
    
    # Crop Recommendations
    'CropRecommendationEngine',
    'CropProfile',
    'CropType',
    'Season',
    'SuitabilityLevel', 
    'CropRecommendation',
    'WeatherData',
    'create_sample_weather_data',
    
    # Economics
    'EconomicsCalculator',
    'EconomicAnalysis',
    'CostBreakdown',
    'RevenueProjection',
    'RiskAssessment',
    'ScenarioAnalysis',
    
    # Visualization
    'StatsVisualizer',
    'create_soil_health_chart',
    'create_crop_comparison_chart',
    'create_economic_dashboard',
    'create_yield_trend_chart',
    
    # Feedback
    'FeedbackManager',
    'FeedbackEntry',
    'FeedbackAnalytics',
    'UserExperience',
    
    # Utilities
    'load_config',
    'save_config',
    'validate_soil_data',
    'calculate_growing_degree_days',
    'convert_units',
    'format_results',
    'export_data',
    'import_data'
]

# Configuration defaults
DEFAULT_CONFIG = {
    'soil_analysis': {
        'ph_optimal_range': [6.0, 7.0],
        'organic_matter_minimum': 2.0,
        'nutrient_thresholds': {
            'nitrogen': {'low': 20, 'medium': 40, 'high': 60},
            'phosphorus': {'low': 15, 'medium': 30, 'high': 50},
            'potassium': {'low': 100, 'medium': 200, 'high': 350}
        }
    },
    'crop_recommendation': {
        'confidence_threshold': 70.0,
        'top_recommendations': 5,
        'weather_weight': 0.3,
        'soil_weight': 0.4,
        'economic_weight': 0.2,
        'risk_weight': 0.1
    },
    'economics': {
        'default_discount_rate': 8.0,
        'inflation_rate': 2.5,
        'risk_premium': 3.0,
        'currency': 'USD'
    },
    'visualization': {
        'color_scheme': 'green',
        'figure_size': (12, 8),
        'dpi': 300,
        'style': 'modern'
    }
}

# Logging configuration
import logging

def setup_logging(level=logging.INFO, filename=None):
    """
    Setup logging configuration for the precision farming library
    
    Args:
        level: Logging level (default: INFO)
        filename: Optional log file path
    """
    logging_config = {
        'level': level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if filename:
        logging_config['filename'] = filename
    
    logging.basicConfig(**logging_config)

# Version checking utilities
def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'plotly', 
        'matplotlib', 'seaborn', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}. "
            f"Please install with: pip install {' '.join(missing_packages)}"
        )
    
    return True

def get_system_info():
    """Get system and library information"""
    import sys
    import platform
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        
        info = {
            'precision_farming_version': __version__,
            'python_version': sys.version,
            'platform': platform.platform(),
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'sklearn_version': sklearn.__version__
        }
        
        return info
    except ImportError as e:
        return {'error': f"Missing dependencies: {e}"}

# Initialize logging by default
setup_logging()

# Welcome message
def welcome():
    """Display welcome message and system information"""
    print(f"""
    🌾 Precision Farming Library v{__version__}
    
    Welcome to the comprehensive precision farming analysis toolkit!
    
    Key Features:
    • 🧪 Advanced soil analysis and health assessment
    • 🌱 AI-powered crop recommendations  
    • 💰 Economic modeling and profitability analysis
    • 📊 Data visualization and reporting
    • 📝 Feedback management and analytics
    
    Documentation: https://docs.precisionfarming.com
    Support: {__email__}
    
    Ready to revolutionize agriculture with data-driven insights!
    """)

# Auto-check dependencies on import
try:
    check_dependencies()
except ImportError as e:
    import warnings
    warnings.warn(f"Dependency check failed: {e}", ImportWarning)

# Optional welcome message (can be disabled)
import os
if os.getenv('PRECISION_FARMING_SHOW_WELCOME', 'true').lower() == 'true':
    welcome()

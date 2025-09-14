"""
Crop Recommendation Engine for Precision Farming

This module provides intelligent crop recommendations based on:
- Soil characteristics and health
- Weather and climate data
- Economic factors and market conditions
- Historical yield data
- Machine learning predictions

Author: Precision Farming Team
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CropType(Enum):
    """Enumeration of supported crop types"""
    WHEAT = "wheat"
    CORN = "corn"
    RICE = "rice"
    SOYBEAN = "soybean"
    COTTON = "cotton"
    BARLEY = "barley"
    OATS = "oats"
    POTATO = "potato"
    TOMATO = "tomato"
    BEANS = "beans"
    PEAS = "peas"
    SUNFLOWER = "sunflower"
    CANOLA = "canola"
    SUGAR_BEET = "sugar_beet"
    ALFALFA = "alfalfa"


class Season(Enum):
    """Growing seasons"""
    SPRING = "spring"
    SUMMER = "summer" 
    FALL = "fall"
    WINTER = "winter"
    YEAR_ROUND = "year_round"


class SuitabilityLevel(Enum):
    """Crop suitability levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    NOT_SUITABLE = "not_suitable"


@dataclass
class CropProfile:
    """
    Comprehensive crop profile with growing requirements
    """
    name: str
    crop_type: CropType
    
    # Environmental requirements
    min_temperature: float  # Celsius
    max_temperature: float  # Celsius
    optimal_temp_range: Tuple[float, float]
    
    # Soil requirements
    ph_range: Tuple[float, float]
    nitrogen_requirement: float  # kg/ha
    phosphorus_requirement: float  # kg/ha
    potassium_requirement: float  # kg/ha
    
    # Water requirements
    water_requirement: float  # mm per season
    drought_tolerance: str  # "low", "medium", "high"
    
    # Growing characteristics
    growing_season: Season
    days_to_maturity: int
    planting_window: Tuple[int, int]  # Julian days
    
    # Economic factors
    seed_cost_per_ha: float  # USD
    fertilizer_cost_per_ha: float  # USD
    labor_cost_per_ha: float  # USD
    machinery_cost_per_ha: float  # USD
    expected_yield_per_ha: float  # tonnes
    market_price_per_tonne: float  # USD
    
    # Risk factors
    pest_risk: str  # "low", "medium", "high"
    disease_risk: str  # "low", "medium", "high"
    weather_risk: str  # "low", "medium", "high"
    
    # Additional characteristics
    soil_texture_preference: List[str] = field(default_factory=list)
    companion_crops: List[str] = field(default_factory=list)
    rotation_benefits: List[str] = field(default_factory=list)


@dataclass 
class WeatherData:
    """Weather and climate information"""
    location: str
    
    # Temperature data
    avg_temperature: float
    min_temperature: float  
    max_temperature: float
    
    # Precipitation
    annual_rainfall: float  # mm
    growing_season_rainfall: float  # mm
    
    # Other factors
    humidity: float  # percentage
    wind_speed: float  # km/h
    frost_free_days: int
    
    # Solar radiation
    solar_radiation: float  # MJ/m2/day
    photoperiod: float  # hours
    
    # Historical data
    last_frost_date: str  # MM-DD format
    first_frost_date: str  # MM-DD format


@dataclass
class CropRecommendation:
    """Individual crop recommendation with details"""
    crop_profile: CropProfile
    suitability_score: float  # 0-100
    suitability_level: SuitabilityLevel
    confidence: float  # 0-100
    
    # Detailed analysis
    soil_match_score: float
    climate_match_score: float
    economic_score: float
    risk_score: float
    
    # Specific recommendations
    planting_recommendations: List[str]
    management_tips: List[str]
    risk_mitigation: List[str]
    expected_roi: float
    
    # Reasoning
    positive_factors: List[str]
    negative_factors: List[str]
    limiting_factors: List[str]


class CropDatabase:
    """Database of crop profiles and characteristics"""
    
    def __init__(self):
        self.crops = {}
        self._initialize_crop_profiles()
    
    def _initialize_crop_profiles(self):
        """Initialize with standard crop profiles"""
        
        # Wheat profile
        wheat = CropProfile(
            name="Wheat",
            crop_type=CropType.WHEAT,
            min_temperature=3, max_temperature=32,
            optimal_temp_range=(15, 25),
            ph_range=(6.0, 7.5),
            nitrogen_requirement=120,
            phosphorus_requirement=60,
            potassium_requirement=40,
            water_requirement=450,
            drought_tolerance="medium",
            growing_season=Season.WINTER,
            days_to_maturity=120,
            planting_window=(260, 320),  # Sept-Nov
            seed_cost_per_ha=150,
            fertilizer_cost_per_ha=200,
            labor_cost_per_ha=300,
            machinery_cost_per_ha=250,
            expected_yield_per_ha=3.5,
            market_price_per_tonne=250,
            pest_risk="medium",
            disease_risk="medium",
            weather_risk="medium",
            soil_texture_preference=["loam", "clay_loam", "silt_loam"],
            companion_crops=["legumes"],
            rotation_benefits=["nitrogen_fixation", "pest_break"]
        )
        self.crops[CropType.WHEAT] = wheat
        
        # Corn profile  
        corn = CropProfile(
            name="Corn",
            crop_type=CropType.CORN,
            min_temperature=10, max_temperature=35,
            optimal_temp_range=(20, 30),
            ph_range=(6.0, 6.8),
            nitrogen_requirement=180,
            phosphorus_requirement=80,
            potassium_requirement=60,
            water_requirement=600,
            drought_tolerance="low",
            growing_season=Season.SUMMER,
            days_to_maturity=100,
            planting_window=(110, 150),  # Apr-May
            seed_cost_per_ha=200,
            fertilizer_cost_per_ha=350,
            labor_cost_per_ha=400,
            machinery_cost_per_ha=300,
            expected_yield_per_ha=8.5,
            market_price_per_tonne=200,
            pest_risk="high",
            disease_risk="medium",
            weather_risk="medium",
            soil_texture_preference=["loam", "silt_loam"],
            companion_crops=["beans", "squash"],
            rotation_benefits=["soil_structure"]
        )
        self.crops[CropType.CORN] = corn
        
        # Rice profile
        rice = CropProfile(
            name="Rice",
            crop_type=CropType.RICE,
            min_temperature=20, max_temperature=37,
            optimal_temp_range=(25, 32),
            ph_range=(5.5, 6.5),
            nitrogen_requirement=150,
            phosphorus_requirement=75,
            potassium_requirement=50,
            water_requirement=1200,
            drought_tolerance="low",
            growing_season=Season.SUMMER,
            days_to_maturity=140,
            planting_window=(120, 180),  # May-Jun
            seed_cost_per_ha=180,
            fertilizer_cost_per_ha=280,
            labor_cost_per_ha=450,
            machinery_cost_per_ha=200,
            expected_yield_per_ha=4.5,
            market_price_per_tonne=300,
            pest_risk="high",
            disease_risk="high",
            weather_risk="high",
            soil_texture_preference=["clay", "clay_loam"],
            companion_crops=["fish", "ducks"],
            rotation_benefits=["water_management"]
        )
        self.crops[CropType.RICE] = rice
        
        # Soybean profile
        soybean = CropProfile(
            name="Soybean",
            crop_type=CropType.SOYBEAN,
            min_temperature=15, max_temperature=35,
            optimal_temp_range=(20, 30),
            ph_range=(6.0, 7.0),
            nitrogen_requirement=0,  # Nitrogen fixing
            phosphorus_requirement=90,
            potassium_requirement=70,
            water_requirement=500,
            drought_tolerance="medium",
            growing_season=Season.SUMMER,
            days_to_maturity=110,
            planting_window=(120, 160),  # May-Jun
            seed_cost_per_ha=220,
            fertilizer_cost_per_ha=150,
            labor_cost_per_ha=350,
            machinery_cost_per_ha=280,
            expected_yield_per_ha=2.8,
            market_price_per_tonne=400,
            pest_risk="medium",
            disease_risk="medium", 
            weather_risk="low",
            soil_texture_preference=["loam", "sandy_loam"],
            companion_crops=["corn"],
            rotation_benefits=["nitrogen_fixation", "soil_health"]
        )
        self.crops[CropType.SOYBEAN] = soybean
        
        # Add more crops as needed...
        
    def get_crop(self, crop_type: CropType) -> CropProfile:
        """Get crop profile by type"""
        return self.crops.get(crop_type)
    
    def get_all_crops(self) -> List[CropProfile]:
        """Get all available crop profiles"""
        return list(self.crops.values())


class CropRecommendationEngine:
    """
    Main engine for crop recommendations using rule-based and ML approaches
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize recommendation engine
        
        Args:
            model_path: Path to pre-trained ML model
        """
        self.crop_db = CropDatabase()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.ml_model = None
        
        if model_path:
            self.load_model(model_path)
        
        logger.info("Crop recommendation engine initialized")
    
    def calculate_soil_compatibility(self, soil_data: Dict, 
                                   crop_profile: CropProfile) -> Tuple[float, List[str]]:
        """
        Calculate soil compatibility score for a crop
        
        Args:
            soil_data: Dictionary with soil characteristics
            crop_profile: Crop requirements profile
            
        Returns:
            Tuple of (compatibility_score, reasoning_factors)
        """
        score = 0.0
        factors = []
        max_score = 100.0
        
        # pH compatibility (30 points)
        soil_ph = soil_data.get('ph_level', 7.0)
        ph_min, ph_max = crop_profile.ph_range
        
        if ph_min <= soil_ph <= ph_max:
            ph_score = 30.0
            factors.append(f"✓ pH ({soil_ph:.1f}) is optimal for {crop_profile.name}")
        else:
            ph_distance = min(abs(soil_ph - ph_min), abs(soil_ph - ph_max))
            ph_score = max(0, 30.0 - (ph_distance * 10))
            if ph_score < 15:
                factors.append(f"⚠ pH ({soil_ph:.1f}) is not optimal (needs {ph_min}-{ph_max})")
            else:
                nutrient_score = 5.0
                factors.append(f"⚠ Low {nutrient} levels - supplementation needed")
            
            score += nutrient_score
        
        # Soil texture compatibility (20 points)
        soil_texture = soil_data.get('texture_class', 'loam')
        if soil_texture in crop_profile.soil_texture_preference:
            texture_score = 20.0
            factors.append(f"✓ {soil_texture.title()} soil is preferred for {crop_profile.name}")
        elif len(crop_profile.soil_texture_preference) == 0:  # No preference
            texture_score = 15.0
            factors.append("○ Soil texture is acceptable")
        else:
            texture_score = 10.0
            factors.append(f"⚠ {soil_texture.title()} soil is not optimal")
        
        score += texture_score
        
        # Organic matter bonus (10 points)
        organic_matter = soil_data.get('organic_matter_percent', 2.0)
        if organic_matter >= 3.0:
            om_score = 10.0
            factors.append(f"✓ Good organic matter content ({organic_matter:.1f}%)")
        elif organic_matter >= 2.0:
            om_score = 7.0
            factors.append(f"○ Adequate organic matter ({organic_matter:.1f}%)")
        else:
            om_score = 3.0
            factors.append(f"⚠ Low organic matter ({organic_matter:.1f}%)")
        
        score += om_score
        
        return min(100.0, score), factors
    
    def calculate_climate_compatibility(self, weather_data: WeatherData,
                                      crop_profile: CropProfile) -> Tuple[float, List[str]]:
        """
        Calculate climate compatibility score for a crop
        
        Args:
            weather_data: Weather and climate information
            crop_profile: Crop requirements profile
            
        Returns:
            Tuple of (compatibility_score, reasoning_factors)
        """
        score = 0.0
        factors = []
        
        # Temperature compatibility (40 points)
        avg_temp = weather_data.avg_temperature
        min_temp, max_temp = crop_profile.optimal_temp_range
        
        if min_temp <= avg_temp <= max_temp:
            temp_score = 40.0
            factors.append(f"✓ Temperature ({avg_temp:.1f}°C) is optimal")
        elif crop_profile.min_temperature <= avg_temp <= crop_profile.max_temperature:
            temp_score = 25.0
            factors.append(f"○ Temperature ({avg_temp:.1f}°C) is acceptable")
        else:
            temp_score = 10.0
            factors.append(f"⚠ Temperature ({avg_temp:.1f}°C) may be challenging")
        
        score += temp_score
        
        # Water availability (35 points)
        available_water = weather_data.growing_season_rainfall
        required_water = crop_profile.water_requirement
        
        water_ratio = available_water / required_water
        if water_ratio >= 1.0:
            water_score = 35.0
            factors.append("✓ Sufficient rainfall for crop needs")
        elif water_ratio >= 0.8:
            water_score = 28.0
            factors.append("○ Adequate rainfall, minimal irrigation needed")
        elif water_ratio >= 0.6:
            water_score = 20.0
            factors.append("⚠ Moderate irrigation will be required")
        else:
            water_score = 10.0
            factors.append("⚠ Significant irrigation will be required")
        
        # Adjust for drought tolerance
        if crop_profile.drought_tolerance == "high" and water_ratio < 0.8:
            water_score += 10.0
            factors.append("✓ Crop has good drought tolerance")
        
        score += min(35.0, water_score)
        
        # Growing season compatibility (15 points)
        frost_free = weather_data.frost_free_days
        maturity_days = crop_profile.days_to_maturity
        
        if frost_free >= maturity_days + 30:  # 30 day buffer
            season_score = 15.0
            factors.append("✓ Sufficient growing season length")
        elif frost_free >= maturity_days:
            season_score = 12.0
            factors.append("○ Adequate growing season length")
        else:
            season_score = 5.0
            factors.append("⚠ Short growing season may be limiting")
        
        score += season_score
        
        # Solar radiation compatibility (10 points)
        if weather_data.solar_radiation >= 15:  # MJ/m2/day
            solar_score = 10.0
            factors.append("✓ Good solar radiation levels")
        elif weather_data.solar_radiation >= 12:
            solar_score = 8.0
            factors.append("○ Adequate solar radiation")
        else:
            solar_score = 5.0
            factors.append("⚠ Limited solar radiation")
        
        score += solar_score
        
        return min(100.0, score), factors
    
    def calculate_economic_viability(self, crop_profile: CropProfile,
                                   farm_size: float,
                                   market_conditions: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate economic viability score and metrics
        
        Args:
            crop_profile: Crop profile with cost data
            farm_size: Farm size in hectares
            market_conditions: Optional market price adjustments
            
        Returns:
            Tuple of (economic_score, financial_metrics)
        """
        # Calculate costs per hectare
        total_cost_per_ha = (
            crop_profile.seed_cost_per_ha +
            crop_profile.fertilizer_cost_per_ha +
            crop_profile.labor_cost_per_ha +
            crop_profile.machinery_cost_per_ha
        )
        
        # Calculate revenue per hectare
        market_price = crop_profile.market_price_per_tonne
        if market_conditions and crop_profile.crop_type.value in market_conditions:
            market_price *= market_conditions[crop_profile.crop_type.value]
        
        revenue_per_ha = crop_profile.expected_yield_per_ha * market_price
        profit_per_ha = revenue_per_ha - total_cost_per_ha
        
        # Calculate total farm metrics
        total_cost = total_cost_per_ha * farm_size
        total_revenue = revenue_per_ha * farm_size
        total_profit = profit_per_ha * farm_size
        
        # Calculate financial ratios
        roi = (profit_per_ha / total_cost_per_ha) * 100 if total_cost_per_ha > 0 else 0
        profit_margin = (profit_per_ha / revenue_per_ha) * 100 if revenue_per_ha > 0 else 0
        
        # Economic score based on profitability
        if roi >= 30:
            economic_score = 100.0
        elif roi >= 20:
            economic_score = 80.0
        elif roi >= 10:
            economic_score = 60.0
        elif roi >= 0:
            economic_score = 40.0
        else:
            economic_score = 20.0
        
        financial_metrics = {
            'cost_per_ha': total_cost_per_ha,
            'revenue_per_ha': revenue_per_ha,
            'profit_per_ha': profit_per_ha,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'roi_percent': roi,
            'profit_margin_percent': profit_margin,
            'breakeven_price': total_cost_per_ha / crop_profile.expected_yield_per_ha
        }
        
        return economic_score, financial_metrics
    
    def calculate_risk_assessment(self, crop_profile: CropProfile,
                                weather_data: WeatherData,
                                soil_data: Dict) -> Tuple[float, List[str]]:
        """
        Calculate risk assessment score (higher score = lower risk)
        
        Args:
            crop_profile: Crop profile with risk factors
            weather_data: Weather information
            soil_data: Soil characteristics
            
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        risk_factors = []
        risk_score = 100.0  # Start with perfect score, deduct for risks
        
        # Pest risk assessment
        pest_risk_map = {"low": 0, "medium": 10, "high": 20}
        pest_deduction = pest_risk_map.get(crop_profile.pest_risk, 15)
        risk_score -= pest_deduction
        if pest_deduction > 0:
            risk_factors.append(f"Pest risk: {crop_profile.pest_risk}")
        
        # Disease risk assessment
        disease_risk_map = {"low": 0, "medium": 10, "high": 20}
        disease_deduction = disease_risk_map.get(crop_profile.disease_risk, 15)
        risk_score -= disease_deduction
        if disease_deduction > 0:
            risk_factors.append(f"Disease risk: {crop_profile.disease_risk}")
        
        # Weather risk assessment
        weather_risk_map = {"low": 0, "medium": 10, "high": 20}
        weather_deduction = weather_risk_map.get(crop_profile.weather_risk, 15)
        risk_score -= weather_deduction
        if weather_deduction > 0:
            risk_factors.append(f"Weather risk: {crop_profile.weather_risk}")
        
        # Soil-specific risks
        soil_ph = soil_data.get('ph_level', 7.0)
        if soil_ph < 5.5 or soil_ph > 8.5:
            risk_score -= 15
            risk_factors.append("Extreme pH levels increase risk")
        
        # Water stress risk
        water_ratio = weather_data.growing_season_rainfall / crop_profile.water_requirement
        if water_ratio < 0.5:
            risk_score -= 20
            risk_factors.append("High water stress risk")
        elif water_ratio < 0.7:
            risk_score -= 10
            risk_factors.append("Moderate water stress risk")
        
        # Temperature stress risk
        if weather_data.max_temperature > crop_profile.max_temperature:
            risk_score -= 15
            risk_factors.append("Heat stress risk")
        if weather_data.min_temperature < crop_profile.min_temperature:
            risk_score -= 15
            risk_factors.append("Cold stress risk")
        
        return max(0.0, risk_score), risk_factors
    
    def generate_management_recommendations(self, crop_profile: CropProfile,
                                          soil_data: Dict,
                                          weather_data: WeatherData,
                                          compatibility_scores: Dict) -> Dict[str, List[str]]:
        """
        Generate specific management recommendations
        
        Returns:
            Dictionary with different types of recommendations
        """
        recommendations = {
            'planting': [],
            'fertilization': [],
            'irrigation': [],
            'pest_management': [],
            'general': []
        }
        
        # Planting recommendations
        if compatibility_scores['soil'] < 60:
            recommendations['planting'].append(
                "Consider soil amendments before planting"
            )
        
        if weather_data.frost_free_days < crop_profile.days_to_maturity + 20:
            recommendations['planting'].append(
                "Plant early in the season to ensure adequate growing time"
            )
        
        recommendations['planting'].append(
            f"Optimal planting window: {self._julian_to_date(crop_profile.planting_window[0])} to "
            f"{self._julian_to_date(crop_profile.planting_window[1])}"
        )
        
        # Fertilization recommendations
        soil_n = soil_data.get('nitrogen_ppm', 0)
        if soil_n < crop_profile.nitrogen_requirement / 10:
            recommendations['fertilization'].append(
                f"Apply {crop_profile.nitrogen_requirement:.0f} kg/ha nitrogen fertilizer"
            )
        
        soil_p = soil_data.get('phosphorus_ppm', 0)
        if soil_p < crop_profile.phosphorus_requirement / 2:
            recommendations['fertilization'].append(
                f"Apply {crop_profile.phosphorus_requirement:.0f} kg/ha phosphorus fertilizer"
            )
        
        # Irrigation recommendations
        water_ratio = weather_data.growing_season_rainfall / crop_profile.water_requirement
        if water_ratio < 0.8:
            irrigation_need = crop_profile.water_requirement - weather_data.growing_season_rainfall
            recommendations['irrigation'].append(
                f"Plan for {irrigation_need:.0f}mm supplemental irrigation"
            )
        
        # Pest management
        if crop_profile.pest_risk in ['medium', 'high']:
            recommendations['pest_management'].append(
                "Implement integrated pest management (IPM) strategies"
            )
            recommendations['pest_management'].append(
                "Monitor regularly for pest activity"
            )
        
        # General recommendations
        if soil_data.get('organic_matter_percent', 2.0) < 2.0:
            recommendations['general'].append(
                "Increase organic matter through compost or cover crops"
            )
        
        return recommendations
    
    def _julian_to_date(self, julian_day: int) -> str:
        """Convert Julian day to MM-DD format"""
        try:
            date = datetime(2024, 1, 1) + timedelta(days=julian_day - 1)
            return date.strftime("%m-%d")
        except:
            return "N/A"
    
    def recommend_crops(self, soil_data: Dict,
                       weather_data: WeatherData,
                       farm_size: float = 10.0,
                       top_n: int = 5,
                       market_conditions: Optional[Dict] = None) -> List[CropRecommendation]:
        """
        Generate crop recommendations based on all factors
        
        Args:
            soil_data: Soil analysis data
            weather_data: Weather and climate data
            farm_size: Farm size in hectares
            top_n: Number of top recommendations to return
            market_conditions: Optional market price multipliers
            
        Returns:
            List of CropRecommendation objects, sorted by suitability
        """
        logger.info("Generating crop recommendations...")
        
        recommendations = []
        
        for crop_profile in self.crop_db.get_all_crops():
            # Calculate compatibility scores
            soil_score, soil_factors = self.calculate_soil_compatibility(
                soil_data, crop_profile
            )
            climate_score, climate_factors = self.calculate_climate_compatibility(
                weather_data, crop_profile
            )
            economic_score, financial_metrics = self.calculate_economic_viability(
                crop_profile, farm_size, market_conditions
            )
            risk_score, risk_factors = self.calculate_risk_assessment(
                crop_profile, weather_data, soil_data
            )
            
            # Calculate overall suitability score (weighted average)
            weights = {'soil': 0.3, 'climate': 0.3, 'economic': 0.25, 'risk': 0.15}
            overall_score = (
                soil_score * weights['soil'] +
                climate_score * weights['climate'] +
                economic_score * weights['economic'] +
                risk_score * weights['risk']
            )
            
            # Determine suitability level
            if overall_score >= 80:
                suitability_level = SuitabilityLevel.EXCELLENT
            elif overall_score >= 65:
                suitability_level = SuitabilityLevel.GOOD
            elif overall_score >= 50:
                suitability_level = SuitabilityLevel.FAIR
            elif overall_score >= 35:
                suitability_level = SuitabilityLevel.POOR
            else:
                suitability_level = SuitabilityLevel.NOT_SUITABLE
            
            # Generate management recommendations
            compatibility_scores = {
                'soil': soil_score,
                'climate': climate_score,
                'economic': economic_score,
                'risk': risk_score
            }
            
            management_recs = self.generate_management_recommendations(
                crop_profile, soil_data, weather_data, compatibility_scores
            )
            
            # Combine all recommendations
            all_recommendations = []
            for rec_type, rec_list in management_recs.items():
                all_recommendations.extend(rec_list)
            
            # Identify positive and negative factors
            positive_factors = []
            negative_factors = []
            limiting_factors = []
            
            # Process soil factors
            for factor in soil_factors:
                if factor.startswith('✓'):
                    positive_factors.append(factor)
                elif factor.startswith('⚠'):
                    negative_factors.append(factor)
            
            # Process climate factors
            for factor in climate_factors:
                if factor.startswith('✓'):
                    positive_factors.append(factor)
                elif factor.startswith('⚠'):
                    if any(word in factor.lower() for word in ['short', 'limited', 'challenging']):
                        limiting_factors.append(factor)
                    else:
                        negative_factors.append(factor)
            
            # Create recommendation object
            recommendation = CropRecommendation(
                crop_profile=crop_profile,
                suitability_score=overall_score,
                suitability_level=suitability_level,
                confidence=min(95.0, overall_score * 1.1),  # Confidence slightly higher than score
                soil_match_score=soil_score,
                climate_match_score=climate_score,
                economic_score=economic_score,
                risk_score=risk_score,
                planting_recommendations=management_recs['planting'],
                management_tips=all_recommendations,
                risk_mitigation=risk_factors,
                expected_roi=financial_metrics.get('roi_percent', 0),
                positive_factors=positive_factors,
                negative_factors=negative_factors,
                limiting_factors=limiting_factors
            )
            
            recommendations.append(recommendation)
        
        # Sort by suitability score and return top N
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations[:top_n]
    
    def train_ml_model(self, training_data: pd.DataFrame):
        """
        Train machine learning model for crop recommendation
        
        Args:
            training_data: DataFrame with features and target crop labels
        """
        logger.info("Training ML model for crop recommendation...")
        
        # Prepare features
        feature_columns = [
            'ph_level', 'nitrogen_ppm', 'phosphorus_ppm', 'potassium_ppm',
            'organic_matter_percent', 'avg_temperature', 'annual_rainfall',
            'growing_season_rainfall', 'frost_free_days'
        ]
        
        X = training_data[feature_columns]
        y = training_data['recommended_crop']
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.ml_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")
    
    def predict_with_ml(self, soil_data: Dict, weather_data: WeatherData) -> List[Tuple[str, float]]:
        """
        Use ML model to predict suitable crops
        
        Args:
            soil_data: Soil characteristics
            weather_data: Weather information
            
        Returns:
            List of (crop_name, probability) tuples
        """
        if not self.ml_model:
            logger.warning("ML model not trained. Use train_ml_model() first.")
            return []
        
        # Prepare features
        features = np.array([[
            soil_data.get('ph_level', 6.5),
            soil_data.get('nitrogen_ppm', 30),
            soil_data.get('phosphorus_ppm', 20),
            soil_data.get('potassium_ppm', 150),
            soil_data.get('organic_matter_percent', 2.5),
            weather_data.avg_temperature,
            weather_data.annual_rainfall,
            weather_data.growing_season_rainfall,
            weather_data.frost_free_days
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions
        probabilities = self.ml_model.predict_proba(features_scaled)[0]
        
        # Convert back to crop names
        crop_predictions = []
        for i, prob in enumerate(probabilities):
            crop_name = self.label_encoder.inverse_transform([i])[0]
            crop_predictions.append((crop_name, prob))
        
        # Sort by probability
        crop_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return crop_predictions[:5]  # Top 5 predictions
    
    def save_model(self, model_path: str):
        """Save trained model and scalers"""
        if self.ml_model:
            joblib.dump({
                'model': self.ml_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }, model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load pre-trained model and scalers"""
        try:
            model_data = joblib.load(model_path)
            self.ml_model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


def create_sample_weather_data(location: str = "Iowa, USA") -> WeatherData:
    """Create sample weather data for testing"""
    return WeatherData(
        location=location,
        avg_temperature=12.0,
        min_temperature=-15.0,
        max_temperature=32.0,
        annual_rainfall=850.0,
        growing_season_rainfall=450.0,
        humidity=70.0,
        wind_speed=15.0,
        frost_free_days=160,
        solar_radiation=14.5,
        photoperiod=13.0,
        last_frost_date="04-15",
        first_frost_date="10-15"
    )


if __name__ == "__main__":
    # Example usage
    from src.soil_analyzer import SoilSample
    
    # Create sample data
    sample_soil = {
        'ph_level': 6.5,
        'nitrogen_ppm': 45,
        'phosphorus_ppm': 28,
        'potassium_ppm': 180,
        'organic_matter_percent': 3.2,
        'texture_class': 'loam'
    }
    
    sample_weather = create_sample_weather_data()
    
    # Initialize recommendation engine
    engine = CropRecommendationEngine()
    
    # Get recommendations
    recommendations = engine.recommend_crops(
        soil_data=sample_soil,
        weather_data=sample_weather,
        farm_size=25.0,
        top_n=3
    )
    
    # Print results
    print("Top Crop Recommendations:")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.crop_profile.name}")
        print(f"   Suitability Score: {rec.suitability_score:.1f}/100")
        print(f"   Suitability Level: {rec.suitability_level.value.title()}")
        print(f"   Expected ROI: {rec.expected_roi:.1f}%")
        print(f"   Confidence: {rec.confidence:.1f}%")
        
        if rec.positive_factors:
            print("   ✓ Positive Factors:")
            for factor in rec.positive_factors[:3]:  # Show top 3
                print(f"     - {factor}")
        
        if rec.limiting_factors:
            print("   ⚠ Limiting Factors:")
            for factor in rec.limiting_factors[:2]:  # Show top 2
                print(f"     - {factor}")
        
        print(f"   Key Recommendations:")
        for tip in rec.management_tips[:3]:  # Show top 3
            print(f"     - {tip}")
                factors.append(f"⚠ pH ({soil_ph:.1f}) is acceptable but not optimal")
        
        score += ph_score
        
        # Nutrient availability (40 points total)
        nutrients = {
            'nitrogen': ('nitrogen_ppm', crop_profile.nitrogen_requirement / 10),  # Convert to ppm
            'phosphorus': ('phosphorus_ppm', crop_profile.phosphorus_requirement / 2),
            'potassium': ('potassium_ppm', crop_profile.potassium_requirement * 5)
        }
        
        for nutrient, (soil_key, requirement) in nutrients.items():
            available = soil_data.get(soil_key, 0)
            if available >= requirement:
                nutrient_score = 13.33  # 40/3 points per nutrient
                factors.append(f"✓ Sufficient {nutrient} available")
            elif available >= requirement * 0.7:
                nutrient_score = 10.0
                factors.append(f"⚠ {nutrient.title()} levels are adequate")
            else:

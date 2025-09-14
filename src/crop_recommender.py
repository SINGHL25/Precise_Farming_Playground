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
# Rule + ML crop suitability
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
                factors.append(f"⚠ Low {nutrient} levels - fertilization needed")
            
            score += nutrient_score
        
        # Soil texture compatibility (20 points)
        soil_texture = soil_data.get('texture_class', 'loam')
        if soil_texture in crop_profile.soil_texture_preference:
            texture_score = 20.0
            factors.append(f"✓ Soil texture ({soil_texture}) is preferred")
        elif soil_texture in ['loam', 'silt_loam', 'clay_loam']:  # Generally good
            texture_score = 15.0
            factors.append(f"⚠ Soil texture ({soil_texture}) is acceptable")
        else:
            texture_score = 8.0
            factors.append(f"⚠ Soil texture ({soil_texture}) may pose challenges")
        
        score += texture_score
        
        # Organic matter consideration (10 points)
        organic_matter = soil_data.get('organic_matter_percent', 2.0)
        if organic_matter >= 3.0:
            om_score = 10.0
            factors.append("✓ Good organic matter content")
        elif organic_matter >= 2.0:
            om_score = 7.0
            factors.append("⚠ Adequate organic matter")
        else:
            om_score = 3.0
            factors.append("⚠ Low organic matter - soil improvement needed")
        
        score += om_score
        
        return min(score, max_score), factors
    
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
        max_score = 100.0
        
        # Temperature compatibility (40 points)
        avg_temp = weather_data.avg_temperature
        min_temp_ok = avg_temp >= crop_profile.min_temperature
        max_temp_ok = avg_temp <= crop_profile.max_temperature
        optimal_min, optimal_max = crop_profile.optimal_temp_range
        
        if optimal_min <= avg_temp <= optimal_max:
            temp_score = 40.0
            factors.append(f"✓ Temperature ({avg_temp:.1f}°C) is optimal")
        elif min_temp_ok and max_temp_ok:
            temp_score = 30.0
            factors.append(f"⚠ Temperature ({avg_temp:.1f}°C) is acceptable")
        else:
            temp_score = 10.0
            factors.append(f"⚠ Temperature ({avg_temp:.1f}°C) may be challenging")
        
        score += temp_score
        
        # Water availability (35 points)
        available_water = weather_data.growing_season_rainfall
        required_water = crop_profile.water_requirement
        
        if available_water >= required_water:
            water_score = 35.0
            factors.append("✓ Sufficient rainfall expected")
        elif available_water >= required_water * 0.8:
            water_score = 25.0
            factors.append("⚠ Rainfall adequate - minor irrigation may be needed")
        elif available_water >= required_water * 0.6:
            water_score = 15.0
            factors.append("⚠ Irrigation will be necessary")
        else:
            water_score = 5.0
            factors.append("⚠ Significant irrigation required")
        
        score += water_score
        
        # Growing season compatibility (15 points)
        frost_free = weather_data.frost_free_days
        maturity_days = crop_profile.days_to_maturity
        
        if frost_free >= maturity_days + 30:  # 30 day buffer
            season_score = 15.0
            factors.append("✓ Sufficient growing season length")
        elif frost_free >= maturity_days:
            season_score = 12.0
            factors.append("⚠ Growing season length is adequate")
        else:
            season_score = 3.0
            factors.append("⚠ Growing season may be too short")
        
        score += season_score
        
        # Solar radiation and photoperiod (10 points)
        solar_rad = weather_data.solar_radiation
        if solar_rad >= 15:  # High solar radiation
            solar_score = 10.0
            factors.append("✓ Excellent solar radiation")
        elif solar_rad >= 10:
            solar_score = 8.0
            factors.append("⚠ Good solar radiation")
        else:
            solar_score = 5.0
            factors.append("⚠ Limited solar radiation")
        
        score += solar_score
        
        return min(score, max_score), factors
    
    def calculate_economic_viability(self, crop_profile: CropProfile,
                                   farm_size_ha: float,
                                   market_factors: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate economic viability score and financial projections
        
        Args:
            crop_profile: Crop requirements profile
            farm_size_ha: Farm size in hectares
            market_factors: Optional market condition adjustments
            
        Returns:
            Tuple of (economic_score, financial_details)
        """
        # Base costs and revenues
        total_costs = (
            crop_profile.seed_cost_per_ha +
            crop_profile.fertilizer_cost_per_ha +
            crop_profile.labor_cost_per_ha +
            crop_profile.machinery_cost_per_ha
        ) * farm_size_ha
        
        gross_revenue = (
            crop_profile.expected_yield_per_ha *
            crop_profile.market_price_per_tonne *
            farm_size_ha
        )
        
        # Apply market factors if provided
        if market_factors:
            price_adjustment = market_factors.get('price_multiplier', 1.0)
            cost_adjustment = market_factors.get('cost_multiplier', 1.0)
            yield_adjustment = market_factors.get('yield_multiplier', 1.0)
            
            gross_revenue *= price_adjustment * yield_adjustment
            total_costs *= cost_adjustment
        
        net_profit = gross_revenue - total_costs
        roi = (net_profit / total_costs * 100) if total_costs > 0 else 0
        profit_per_ha = net_profit / farm_size_ha if farm_size_ha > 0 else 0
        
        # Calculate economic score (0-100)
        if roi >= 50:
            economic_score = 100.0
        elif roi >= 30:
            economic_score = 80.0
        elif roi >= 15:
            economic_score = 60.0
        elif roi >= 5:
            economic_score = 40.0
        elif roi >= 0:
            economic_score = 20.0
        else:
            economic_score = 0.0
        
        financial_details = {
            'total_costs': total_costs,
            'gross_revenue': gross_revenue,
            'net_profit': net_profit,
            'roi_percent': roi,
            'profit_per_hectare': profit_per_ha,
            'break_even_yield': total_costs / (crop_profile.market_price_per_tonne * farm_size_ha),
            'break_even_price': total_costs / (crop_profile.expected_yield_per_ha * farm_size_ha)
        }
        
        return economic_score, financial_details
    
    def assess_risks(self, crop_profile: CropProfile,
                    weather_data: WeatherData,
                    soil_data: Dict) -> Tuple[float, List[str]]:
        """
        Assess various risks associated with crop choice
        
        Args:
            crop_profile: Crop requirements profile
            weather_data: Weather and climate data
            soil_data: Soil characteristics
            
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        risk_factors = []
        total_risk = 0.0
        
        # Weather risks
        weather_risk_value = {"low": 10, "medium": 20, "high": 35}.get(crop_profile.weather_risk, 20)
        if weather_data.annual_rainfall < crop_profile.water_requirement * 0.5:
            weather_risk_value += 15
            risk_factors.append("High drought risk due to low rainfall")
        
        # Pest and disease risks
        pest_risk_value = {"low": 5, "medium": 15, "high": 25}.get(crop_profile.pest_risk, 15)
        disease_risk_value = {"low": 5, "medium": 15, "high": 25}.get(crop_profile.disease_risk, 15)
        
        # Soil-related risks
        soil_risk = 0
        if soil_data.get('ph_level', 7.0) < 5.0 or soil_data.get('ph_level', 7.0) > 8.5:
            soil_risk += 10
            risk_factors.append("pH level may stress plants")
        
        if soil_data.get('organic_matter_percent', 2.0) < 1.0:
            soil_risk += 10
            risk_factors.append("Low organic matter increases production risks")
        
        # Economic risks (market volatility)
        market_risk = 15  # Base market risk
        if crop_profile.market_price_per_tonne > 400:  # High-value crops more volatile
            market_risk += 10
            risk_factors.append("High market price volatility")
        
        total_risk = weather_risk_value + pest_risk_value + disease_risk_value + soil_risk + market_risk
        risk_score = max(0, 100 - total_risk)  # Convert to positive score
        
        return risk_score, risk_factors
    
    def generate_management_recommendations(self, crop_profile: CropProfile,
                                          soil_data: Dict,
                                          weather_data: WeatherData) -> Dict[str, List[str]]:
        """
        Generate specific management recommendations
        
        Args:
            crop_profile: Crop requirements profile
            soil_data: Soil characteristics
            weather_data: Weather and climate data
            
        Returns:
            Dictionary with categorized recommendations
        """
        recommendations = {
            'planting': [],
            'fertilization': [],
            'irrigation': [],
            'pest_management': [],
            'harvest': []
        }
        
        # Planting recommendations
        recommendations['planting'].append(
            f"Plant between day {crop_profile.planting_window[0]} and "
            f"{crop_profile.planting_window[1]} of the year"
        )
        
        if soil_data.get('texture_class') == 'clay':
            recommendations['planting'].append("Ensure proper drainage before planting")
        elif soil_data.get('texture_class') in ['sand', 'loamy_sand']:
            recommendations['planting'].append("Consider deeper planting for better root establishment")
        
        # Fertilization recommendations
        soil_n = soil_data.get('nitrogen_ppm', 30)
        if soil_n < crop_profile.nitrogen_requirement / 10:
            recommendations['fertilization'].append(
                f"Apply {crop_profile.nitrogen_requirement - (soil_n * 10):.0f} kg/ha nitrogen"
            )
        
        soil_p = soil_data.get('phosphorus_ppm', 25)
        if soil_p < crop_profile.phosphorus_requirement / 2:
            recommendations['fertilization'].append(
                f"Apply {crop_profile.phosphorus_requirement - (soil_p * 2):.0f} kg/ha phosphorus"
            )
        
        # Irrigation recommendations
        if weather_data.growing_season_rainfall < crop_profile.water_requirement:
            deficit = crop_profile.water_requirement - weather_data.growing_season_rainfall
            recommendations['irrigation'].append(
                f"Plan for {deficit:.0f}mm supplemental irrigation"
            )
        
        if crop_profile.drought_tolerance == "low":
            recommendations['irrigation'].append("Monitor soil moisture closely")
        
        # Pest management
        if crop_profile.pest_risk == "high":
            recommendations['pest_management'].append("Implement integrated pest management")
            recommendations['pest_management'].append("Regular field scouting recommended")
        
        # Harvest timing
        recommendations['harvest'].append(
            f"Harvest approximately {crop_profile.days_to_maturity} days after planting"
        )
        
        return recommendations
    
    def recommend_crops(self, soil_data: Dict,
                       weather_data: WeatherData,
                       farm_size_ha: float = 10.0,
                       top_n: int = 5,
                       market_factors: Optional[Dict] = None) -> List[CropRecommendation]:
        """
        Generate top crop recommendations based on all factors
        
        Args:
            soil_data: Dictionary with soil characteristics
            weather_data: Weather and climate information
            farm_size_ha: Farm size in hectares
            top_n: Number of top recommendations to return
            market_factors: Optional market condition adjustments
            
        Returns:
            List of CropRecommendation objects, sorted by suitability
        """
        recommendations = []
        
        for crop_profile in self.crop_db.get_all_crops():
            # Calculate individual scores
            soil_score, soil_factors = self.calculate_soil_compatibility(soil_data, crop_profile)
            climate_score, climate_factors = self.calculate_climate_compatibility(weather_data, crop_profile)
            economic_score, financial_details = self.calculate_economic_viability(
                crop_profile, farm_size_ha, market_factors
            )
            risk_score, risk_factors = self.assess_risks(crop_profile, weather_data, soil_data)
            
            # Calculate overall suitability score (weighted average)
            weights = {
                'soil': 0.30,
                'climate': 0.30,
                'economic': 0.25,
                'risk': 0.15
            }
            
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
            management_recs = self.generate_management_recommendations(
                crop_profile, soil_data, weather_data
            )
            
            # Compile positive and negative factors
            positive_factors = [f for f in soil_factors + climate_factors if f.startswith('✓')]
            negative_factors = [f for f in soil_factors + climate_factors + risk_factors 
                              if f.startswith('⚠')]
            
            # Identify limiting factors
            limiting_factors = []
            if soil_score < 50:
                limiting_factors.append("Soil conditions")
            if climate_score < 50:
                limiting_factors.append("Climate conditions")
            if economic_score < 40:
                limiting_factors.append("Economic viability")
            if risk_score < 60:
                limiting_factors.append("High production risks")
            
            # Calculate confidence based on data completeness and score consistency
            confidence = min(95, overall_score * 0.8 + 20)  # Simplified confidence calculation
            
            recommendation = CropRecommendation(
                crop_profile=crop_profile,
                suitability_score=overall_score,
                suitability_level=suitability_level,
                confidence=confidence,
                soil_match_score=soil_score,
                climate_match_score=climate_score,
                economic_score=economic_score,
                risk_score=risk_score,
                planting_recommendations=management_recs['planting'],
                management_tips=(
                    management_recs['fertilization'] +
                    management_recs['irrigation'] +
                    management_recs['pest_management']
                ),
                risk_mitigation=risk_factors,
                expected_roi=financial_details['roi_percent'],
                positive_factors=positive_factors,
                negative_factors=negative_factors,
                limiting_factors=limiting_factors
            )
            
            recommendations.append(recommendation)
        
        # Sort by suitability score and return top N
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        
        logger.info(f"Generated {len(recommendations)} crop recommendations")
        return recommendations[:top_n]
    
    def train_ml_model(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train machine learning model for crop recommendation
        
        Args:
            training_data: DataFrame with historical crop success data
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training ML model for crop recommendations")
        
        # Prepare features
        feature_columns = [
            'ph_level', 'nitrogen_ppm', 'phosphorus_ppm', 'potassium_ppm',
            'organic_matter_percent', 'sand_percent', 'silt_percent', 'clay_percent',
            'avg_temperature', 'annual_rainfall', 'humidity'
        ]
        
        X = training_data[feature_columns]
        y = training_data['best_crop']  # Target variable
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.ml_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, self.ml_model.feature_importances_))
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'n_samples': len(training_data)
        }
    
    def predict_ml_recommendation(self, soil_data: Dict, weather_data: WeatherData) -> Dict:
        """
        Use ML model to predict best crop recommendation
        
        Args:
            soil_data: Dictionary with soil characteristics
            weather_data: Weather and climate information
            
        Returns:
            Dictionary with ML prediction results
        """
        if self.ml_model is None:
            raise ValueError("ML model not trained. Call train_ml_model() first.")
        
        # Prepare input features
        features = np.array([[
            soil_data.get('ph_level', 7.0),
            soil_data.get('nitrogen_ppm', 30),
            soil_data.get('phosphorus_ppm', 25),
            soil_data.get('potassium_ppm', 150),
            soil_data.get('organic_matter_percent', 2.0),
            soil_data.get('sand_percent', 40),
            soil_data.get('silt_percent', 35),
            soil_data.get('clay_percent', 25),
            weather_data.avg_temperature,
            weather_data.annual_rainfall,
            weather_data.humidity
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.ml_model.predict(features_scaled)[0]
        probabilities = self.ml_model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        predicted_crop = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities) * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        
        for idx in top_indices:
            crop_name = self.label_encoder.inverse_transform([idx])[0]
            probability = probabilities[idx] * 100
            top_predictions.append({'crop': crop_name, 'probability': probability})
        
        return {
            'predicted_crop': predicted_crop,
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    
    def save_model(self, model_path: str):
        """Save trained model to file"""
        if self.ml_model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_data = {
            'model': self.ml_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load pre-trained model from file"""
        model_data = joblib.load(model_path)
        
        self.ml_model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        logger.info(f"Model loaded from {model_path}")


def create_sample_training_data() -> pd.DataFrame:
    """Create sample training data for ML model"""
    np.random.seed(42)
    
    crops = ['wheat', 'corn', 'rice', 'soybean']
    n_samples = 1000
    
    data = []
    for _ in range(n_samples):
        # Generate random soil and weather data
        ph = np.random.normal(6.5, 0.8)
        nitrogen = np.random.normal(40, 15)
        phosphorus = np.random.normal(30, 10)
        potassium = np.random.normal(200, 50)
        organic_matter = np.random.normal(3.0, 1.0)
        sand = np.random.uniform(20, 60)
        remaining = 100 - sand
        silt = np.random.uniform(0, remaining)
        clay = 100 - sand - silt
        
        temp = np.random.normal(22, 5)
        rainfall = np.random.normal(600, 200)
        humidity = np.random.normal(65, 10)
        
        # Simple rule-based best crop assignment
        if 6.0 <= ph <= 7.5 and temp >= 20 and rainfall >= 400:
            if nitrogen > 45:
                best_crop = 'corn'
            elif phosphorus > 35:
                best_crop = 'soybean'
            else:
                best_crop = 'wheat'
        elif ph < 6.5 and rainfall > 800:
            best_crop = 'rice'
        else:
            best_crop = np.random.choice(crops)
        
        data.append({
            'ph_level': ph,
            'nitrogen_ppm': nitrogen,
            'phosphorus_ppm': phosphorus,
            'potassium_ppm': potassium,
            'organic_matter_percent': organic_matter,
            'sand_percent': sand,
            'silt_percent': silt,
            'clay_percent': clay,
            'avg_temperature': temp,
            'annual_rainfall': rainfall,
            'humidity': humidity,
            'best_crop': best_crop
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    from soil_analyzer import SoilSample
    
    # Create sample data
    soil_data = {
        'ph_level': 6.5,
        'nitrogen_ppm': 45,
        'phosphorus_ppm': 28,
        'potassium_ppm': 180,
        'organic_matter_percent': 3.2,
        'texture_class': 'loam',
        'sand_percent': 35,
        'silt_percent': 40,
        'clay_percent': 25
    }
    
    weather_data = WeatherData(
        location="Iowa, USA",
        avg_temperature=22,
        min_temperature=5,
        max_temperature=35,
        annual_rainfall=800,
        growing_season_rainfall=450,
        humidity=65,
        wind_speed=15,
        frost_free_days=180,
        solar_radiation=18,
        photoperiod=14,
        last_frost_date="04-15",
        first_frost_date="10-15"
    )
    
    # Initialize recommendation engine
    engine = CropRecommendationEngine()
    
    # Get recommendations
    recommendations = engine.recommend_crops(soil_data, weather_data, farm_size_ha=10)
    
    # Display results
    print(f"\nTop Crop Recommendations:")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.crop_profile.name}")
        print(f"   Suitability Score: {rec.suitability_score:.1f}/100")
        print(f"   Suitability Level: {rec.suitability_level.value}")
        print(f"   Expected ROI: {rec.expected_roi:.1f}%")
        print(f"   Positive Factors: {len(rec.positive_factors)}")
        print(f"   Limiting Factors: {', '.join(rec.limiting_factors) if rec.limiting_factors else 'None'}")

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

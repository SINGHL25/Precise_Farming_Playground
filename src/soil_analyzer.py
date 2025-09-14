"""
Soil Analysis Engine for Precision Farming

This module provides comprehensive soil analysis capabilities including:
- Nutrient content analysis (NPK)
- pH level assessment
- Organic matter evaluation  
- Soil texture classification
- Moisture content analysis
- Health scoring algorithms

Author: Precision Farming Team
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoilTexture(Enum):
    """Soil texture classifications based on USDA standards"""
    CLAY = "clay"
    SANDY_CLAY = "sandy_clay"
    SILTY_CLAY = "silty_clay"
    CLAY_LOAM = "clay_loam"
    SANDY_CLAY_LOAM = "sandy_clay_loam"
    SILTY_CLAY_LOAM = "silty_clay_loam"
    LOAM = "loam"
    SANDY_LOAM = "sandy_loam"
    SILT_LOAM = "silt_loam"
    SAND = "sand"
    LOAMY_SAND = "loamy_sand"
    SILT = "silt"


class SoilHealth(Enum):
    """Soil health categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class SoilSample:
    """
    Data class representing a soil sample with all measured parameters
    """
    sample_id: str
    location: str
    date_collected: str
    
    # Physical properties
    sand_percent: float
    silt_percent: float
    clay_percent: float
    bulk_density: float
    moisture_content: float
    
    # Chemical properties
    ph_level: float
    nitrogen_ppm: float
    phosphorus_ppm: float
    potassium_ppm: float
    organic_matter_percent: float
    cation_exchange_capacity: float
    
    # Additional nutrients (optional)
    calcium_ppm: Optional[float] = None
    magnesium_ppm: Optional[float] = None
    sulfur_ppm: Optional[float] = None
    iron_ppm: Optional[float] = None
    manganese_ppm: Optional[float] = None
    zinc_ppm: Optional[float] = None
    copper_ppm: Optional[float] = None
    boron_ppm: Optional[float] = None
    
    # Environmental factors
    temperature_celsius: Optional[float] = None
    electrical_conductivity: Optional[float] = None
    
    def __post_init__(self):
        """Validate soil sample data after initialization"""
        if not (0 <= self.sand_percent <= 100):
            raise ValueError("Sand percentage must be between 0 and 100")
        if not (0 <= self.silt_percent <= 100):
            raise ValueError("Silt percentage must be between 0 and 100")
        if not (0 <= self.clay_percent <= 100):
            raise ValueError("Clay percentage must be between 0 and 100")
        if abs(self.sand_percent + self.silt_percent + self.clay_percent - 100) > 1:
            raise ValueError("Sand, silt, and clay percentages must sum to 100")
        if not (0 <= self.ph_level <= 14):
            raise ValueError("pH level must be between 0 and 14")


@dataclass
class SoilAnalysisResult:
    """
    Results of comprehensive soil analysis
    """
    sample: SoilSample
    texture_class: SoilTexture
    health_score: float
    health_category: SoilHealth
    nutrient_analysis: Dict[str, float]
    recommendations: List[str]
    limitations: List[str]
    suitability_crops: List[str]


class SoilAnalyzer:
    """
    Main class for soil analysis operations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize soil analyzer with configuration
        
        Args:
            config_path: Path to configuration file (JSON format)
        """
        self.config = self._load_config(config_path)
        self.nutrient_standards = self._load_nutrient_standards()
        self.texture_boundaries = self._load_texture_boundaries()
        logger.info("Soil analyzer initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "ph_optimal_range": [6.0, 7.0],
            "organic_matter_minimum": 2.0,
            "nutrient_weight_factors": {
                "nitrogen": 0.3,
                "phosphorus": 0.25,
                "potassium": 0.25,
                "organic_matter": 0.2
            }
        }
    
    def _load_nutrient_standards(self) -> Dict:
        """Load standard nutrient level classifications"""
        return {
            "nitrogen": {
                "critical": 0, "low": 20, "medium": 40, 
                "high": 60, "very_high": 100
            },
            "phosphorus": {
                "critical": 0, "low": 15, "medium": 30, 
                "high": 50, "very_high": 80
            },
            "potassium": {
                "critical": 0, "low": 100, "medium": 200, 
                "high": 350, "very_high": 500
            }
        }
    
    def _load_texture_boundaries(self) -> Dict:
        """Load soil texture classification boundaries"""
        return {
            # Based on USDA soil texture triangle
            "clay": {"clay_min": 40},
            "sandy_clay": {"clay_min": 35, "clay_max": 55, "sand_min": 45},
            "silty_clay": {"clay_min": 40, "clay_max": 60, "silt_min": 40},
            "clay_loam": {"clay_min": 27, "clay_max": 40, "sand_max": 45},
            "sandy_clay_loam": {"clay_min": 20, "clay_max": 35, "sand_min": 45},
            "silty_clay_loam": {"clay_min": 27, "clay_max": 40, "sand_max": 20},
            "loam": {"clay_min": 7, "clay_max": 27, "sand_min": 23, "sand_max": 52},
            "sandy_loam": {"clay_max": 20, "sand_min": 43, "sand_max": 85},
            "silt_loam": {"clay_max": 27, "silt_min": 50, "sand_max": 50},
            "sand": {"sand_min": 85},
            "loamy_sand": {"sand_min": 70, "sand_max": 90, "clay_max": 15},
            "silt": {"silt_min": 80}
        }
    
    def classify_texture(self, sample: SoilSample) -> SoilTexture:
        """
        Classify soil texture based on sand, silt, clay percentages
        
        Args:
            sample: SoilSample object with texture data
            
        Returns:
            SoilTexture enum value
        """
        sand = sample.sand_percent
        silt = sample.silt_percent  
        clay = sample.clay_percent
        
        # Apply USDA texture classification rules
        if clay >= 40:
            if sand > 45:
                return SoilTexture.SANDY_CLAY
            elif silt >= 40:
                return SoilTexture.SILTY_CLAY
            else:
                return SoilTexture.CLAY
        
        elif clay >= 27:
            if sand > 45:
                return SoilTexture.SANDY_CLAY_LOAM
            elif sand <= 20:
                return SoilTexture.SILTY_CLAY_LOAM
            else:
                return SoilTexture.CLAY_LOAM
        
        elif clay >= 20:
            if sand >= 45:
                return SoilTexture.SANDY_CLAY_LOAM
            else:
                return SoilTexture.LOAM
        
        elif sand >= 85:
            return SoilTexture.SAND
        
        elif sand >= 70:
            return SoilTexture.LOAMY_SAND
        
        elif silt >= 80:
            return SoilTexture.SILT
        
        elif silt >= 50:
            return SoilTexture.SILT_LOAM
        
        elif sand >= 43:
            return SoilTexture.SANDY_LOAM
        
        else:
            return SoilTexture.LOAM
    
    def calculate_health_score(self, sample: SoilSample) -> float:
        """
        Calculate overall soil health score (0-100)
        
        Args:
            sample: SoilSample object
            
        Returns:
            Health score as float between 0-100
        """
        score_components = {}
        
        # pH score (0-25 points)
        ph_optimal = self.config["ph_optimal_range"]
        if ph_optimal[0] <= sample.ph_level <= ph_optimal[1]:
            ph_score = 25
        elif 5.5 <= sample.ph_level <= 8.0:
            # Gradual reduction for sub-optimal pH
            distance = min(abs(sample.ph_level - ph_optimal[0]), 
                         abs(sample.ph_level - ph_optimal[1]))
            ph_score = max(0, 25 - (distance * 5))
        else:
            ph_score = 0
        score_components["ph"] = ph_score
        
        # Nutrient scores (NPK - 0-45 points total)
        nutrient_scores = {}
        for nutrient in ["nitrogen", "phosphorus", "potassium"]:
            value = getattr(sample, f"{nutrient}_ppm")
            standards = self.nutrient_standards[nutrient]
            
            if value >= standards["high"]:
                nutrient_scores[nutrient] = 15
            elif value >= standards["medium"]:
                nutrient_scores[nutrient] = 12
            elif value >= standards["low"]:
                nutrient_scores[nutrient] = 8
            else:
                nutrient_scores[nutrient] = 3
        
        score_components["nutrients"] = sum(nutrient_scores.values())
        
        # Organic matter score (0-20 points)
        if sample.organic_matter_percent >= 4.0:
            om_score = 20
        elif sample.organic_matter_percent >= 3.0:
            om_score = 16
        elif sample.organic_matter_percent >= 2.0:
            om_score = 12
        elif sample.organic_matter_percent >= 1.0:
            om_score = 8
        else:
            om_score = 3
        score_components["organic_matter"] = om_score
        
        # Physical properties score (0-10 points)
        texture = self.classify_texture(sample)
        if texture in [SoilTexture.LOAM, SoilTexture.SILT_LOAM, SoilTexture.CLAY_LOAM]:
            physical_score = 10
        elif texture in [SoilTexture.SANDY_LOAM, SoilTexture.SILTY_CLAY_LOAM]:
            physical_score = 8
        elif texture in [SoilTexture.SANDY_CLAY_LOAM, SoilTexture.CLAY]:
            physical_score = 6
        else:
            physical_score = 4
        score_components["physical"] = physical_score
        
        total_score = sum(score_components.values())
        logger.info(f"Health score components: {score_components}")
        return min(100, total_score)  # Cap at 100
    
    def categorize_health(self, health_score: float) -> SoilHealth:
        """
        Categorize soil health based on numerical score
        
        Args:
            health_score: Numerical health score (0-100)
            
        Returns:
            SoilHealth enum value
        """
        if health_score >= 80:
            return SoilHealth.EXCELLENT
        elif health_score >= 65:
            return SoilHealth.GOOD
        elif health_score >= 50:
            return SoilHealth.FAIR
        elif health_score >= 35:
            return SoilHealth.POOR
        else:
            return SoilHealth.CRITICAL
    
    def generate_recommendations(self, sample: SoilSample, 
                               analysis_result: SoilAnalysisResult) -> List[str]:
        """
        Generate actionable recommendations based on soil analysis
        
        Args:
            sample: SoilSample object
            analysis_result: Analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # pH recommendations
        if sample.ph_level < 6.0:
            recommendations.append(
                f"Soil is acidic (pH {sample.ph_level:.1f}). "
                "Consider applying lime to raise pH to 6.0-7.0 range."
            )
        elif sample.ph_level > 8.0:
            recommendations.append(
                f"Soil is alkaline (pH {sample.ph_level:.1f}). "
                "Consider applying sulfur or organic matter to lower pH."
            )
        
        # Nutrient recommendations
        if sample.nitrogen_ppm < 30:
            recommendations.append(
                "Nitrogen levels are low. Apply nitrogen-rich fertilizer "
                "or compost before planting."
            )
        
        if sample.phosphorus_ppm < 25:
            recommendations.append(
                "Phosphorus levels are low. Consider bone meal or "
                "phosphate fertilizer application."
            )
        
        if sample.potassium_ppm < 150:
            recommendations.append(
                "Potassium levels are low. Apply potassium sulfate or "
                "potash fertilizer."
            )
        
        # Organic matter recommendations
        if sample.organic_matter_percent < 2.0:
            recommendations.append(
                "Organic matter is low. Incorporate compost, manure, or "
                "cover crops to improve soil structure and fertility."
            )
        
        # Texture-based recommendations
        texture = analysis_result.texture_class
        if texture in [SoilTexture.CLAY, SoilTexture.SILTY_CLAY]:
            recommendations.append(
                "Heavy clay soil detected. Improve drainage with organic matter, "
                "avoid working when wet, and consider raised beds."
            )
        elif texture in [SoilTexture.SAND, SoilTexture.LOAMY_SAND]:
            recommendations.append(
                "Sandy soil detected. Increase water retention with organic matter "
                "and consider more frequent, lighter fertilizer applications."
            )
        
        return recommendations
    
    def identify_limitations(self, sample: SoilSample, 
                           analysis_result: SoilAnalysisResult) -> List[str]:
        """
        Identify soil limitations that may affect crop production
        
        Args:
            sample: SoilSample object
            analysis_result: Analysis results
            
        Returns:
            List of limitation strings
        """
        limitations = []
        
        # pH limitations
        if sample.ph_level < 5.5:
            limitations.append("Severe acidity may limit nutrient availability")
        elif sample.ph_level > 8.5:
            limitations.append("High alkalinity may cause iron and zinc deficiency")
        
        # Nutrient limitations
        if sample.nitrogen_ppm < 15:
            limitations.append("Severe nitrogen deficiency")
        if sample.phosphorus_ppm < 10:
            limitations.append("Severe phosphorus deficiency")
        if sample.potassium_ppm < 100:
            limitations.append("Severe potassium deficiency")
        
        # Physical limitations
        texture = analysis_result.texture_class
        if texture == SoilTexture.CLAY:
            limitations.append("Poor drainage and root penetration in heavy clay")
        elif texture == SoilTexture.SAND:
            limitations.append("Poor water and nutrient retention in sandy soil")
        
        # Organic matter limitations
        if sample.organic_matter_percent < 1.0:
            limitations.append("Very low organic matter affects soil structure")
        
        # Salinity limitations (if EC data available)
        if sample.electrical_conductivity and sample.electrical_conductivity > 4.0:
            limitations.append("High salinity may stress plants")
        
        return limitations
    
    def suggest_suitable_crops(self, sample: SoilSample, 
                             analysis_result: SoilAnalysisResult) -> List[str]:
        """
        Suggest crops suitable for current soil conditions
        
        Args:
            sample: SoilSample object
            analysis_result: Analysis results
            
        Returns:
            List of suitable crop names
        """
        suitable_crops = []
        texture = analysis_result.texture_class
        ph = sample.ph_level
        
        # pH-based crop selection
        if 6.0 <= ph <= 7.5:  # Optimal pH range
            suitable_crops.extend(["wheat", "corn", "soybeans", "tomatoes", "lettuce"])
        elif 5.5 <= ph < 6.0:  # Slightly acidic
            suitable_crops.extend(["potatoes", "blueberries", "carrots", "radishes"])
        elif ph < 5.5:  # Acidic
            suitable_crops.extend(["cranberries", "azaleas", "pine trees"])
        elif 7.5 < ph <= 8.0:  # Slightly alkaline
            suitable_crops.extend(["asparagus", "cabbage", "beets"])
        
        # Texture-based refinement
        if texture in [SoilTexture.LOAM, SoilTexture.SILT_LOAM]:
            # Ideal conditions - most crops suitable
            if ph > 6.0:
                suitable_crops.extend(["beans", "peas", "spinach", "broccoli"])
        elif texture in [SoilTexture.CLAY, SoilTexture.CLAY_LOAM]:
            # Good for crops that like moisture retention
            suitable_crops.extend(["rice", "cabbage", "kale"])
        elif texture in [SoilTexture.SANDY_LOAM, SoilTexture.SAND]:
            # Good drainage - suitable for root vegetables
            suitable_crops.extend(["carrots", "onions", "garlic", "herbs"])
        
        # Remove duplicates and return sorted list
        return sorted(list(set(suitable_crops)))
    
    def analyze_sample(self, sample: SoilSample) -> SoilAnalysisResult:
        """
        Perform comprehensive analysis of soil sample
        
        Args:
            sample: SoilSample object to analyze
            
        Returns:
            SoilAnalysisResult with complete analysis
        """
        logger.info(f"Analyzing soil sample: {sample.sample_id}")
        
        # Classify texture
        texture_class = self.classify_texture(sample)
        
        # Calculate health score
        health_score = self.calculate_health_score(sample)
        health_category = self.categorize_health(health_score)
        
        # Analyze individual nutrients
        nutrient_analysis = {
            "nitrogen": self._analyze_nutrient(sample.nitrogen_ppm, "nitrogen"),
            "phosphorus": self._analyze_nutrient(sample.phosphorus_ppm, "phosphorus"),
            "potassium": self._analyze_nutrient(sample.potassium_ppm, "potassium"),
            "pH": self._analyze_ph(sample.ph_level),
            "organic_matter": self._analyze_organic_matter(sample.organic_matter_percent)
        }
        
        # Create initial result for generating recommendations
        temp_result = SoilAnalysisResult(
            sample=sample,
            texture_class=texture_class,
            health_score=health_score,
            health_category=health_category,
            nutrient_analysis=nutrient_analysis,
            recommendations=[],
            limitations=[],
            suitability_crops=[]
        )
        
        # Generate recommendations and limitations
        recommendations = self.generate_recommendations(sample, temp_result)
        limitations = self.identify_limitations(sample, temp_result)
        suitable_crops = self.suggest_suitable_crops(sample, temp_result)
        
        # Create final result
        result = SoilAnalysisResult(
            sample=sample,
            texture_class=texture_class,
            health_score=health_score,
            health_category=health_category,
            nutrient_analysis=nutrient_analysis,
            recommendations=recommendations,
            limitations=limitations,
            suitability_crops=suitable_crops
        )
        
        logger.info(f"Analysis completed. Health score: {health_score:.1f}")
        return result
    
    def _analyze_nutrient(self, value: float, nutrient_type: str) -> Dict[str, Union[str, float]]:
        """Analyze individual nutrient levels"""
        standards = self.nutrient_standards[nutrient_type]
        
        if value >= standards["very_high"]:
            level = "very_high"
        elif value >= standards["high"]:
            level = "high"
        elif value >= standards["medium"]:
            level = "medium"
        elif value >= standards["low"]:
            level = "low"
        else:
            level = "critical"
        
        return {
            "value": value,
            "level": level,
            "status": level.replace("_", " ").title()
        }
    
    def _analyze_ph(self, ph_value: float) -> Dict[str, Union[str, float]]:
        """Analyze pH level"""
        if ph_value < 5.5:
            status = "Very Acidic"
        elif ph_value < 6.0:
            status = "Acidic"
        elif ph_value <= 7.0:
            status = "Optimal"
        elif ph_value <= 7.5:
            status = "Slightly Alkaline"
        elif ph_value <= 8.5:
            status = "Alkaline"
        else:
            status = "Very Alkaline"
        
        return {
            "value": ph_value,
            "status": status
        }
    
    def _analyze_organic_matter(self, om_percent: float) -> Dict[str, Union[str, float]]:
        """Analyze organic matter content"""
        if om_percent >= 4.0:
            status = "Excellent"
        elif om_percent >= 3.0:
            status = "Good"
        elif om_percent >= 2.0:
            status = "Fair"
        elif om_percent >= 1.0:
            status = "Low"
        else:
            status = "Very Low"
        
        return {
            "value": om_percent,
            "status": status
        }
    
    def batch_analyze(self, samples: List[SoilSample]) -> List[SoilAnalysisResult]:
        """
        Analyze multiple soil samples in batch
        
        Args:
            samples: List of SoilSample objects
            
        Returns:
            List of SoilAnalysisResult objects
        """
        logger.info(f"Starting batch analysis of {len(samples)} samples")
        results = []
        
        for sample in samples:
            try:
                result = self.analyze_sample(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing sample {sample.sample_id}: {str(e)}")
                continue
        
        logger.info(f"Batch analysis completed. {len(results)} samples processed successfully")
        return results
    
    def export_results(self, results: List[SoilAnalysisResult], 
                      output_path: str, format: str = "csv") -> None:
        """
        Export analysis results to file
        
        Args:
            results: List of analysis results
            output_path: Path for output file
            format: Output format ('csv', 'json', 'xlsx')
        """
        if format.lower() == "csv":
            self._export_to_csv(results, output_path)
        elif format.lower() == "json":
            self._export_to_json(results, output_path)
        elif format.lower() == "xlsx":
            self._export_to_xlsx(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {output_path}")
    
    def _export_to_csv(self, results: List[SoilAnalysisResult], output_path: str):
        """Export results to CSV format"""
        data = []
        for result in results:
            row = {
                "sample_id": result.sample.sample_id,
                "location": result.sample.location,
                "date_collected": result.sample.date_collected,
                "texture_class": result.texture_class.value,
                "health_score": result.health_score,
                "health_category": result.health_category.value,
                "ph_level": result.sample.ph_level,
                "nitrogen_ppm": result.sample.nitrogen_ppm,
                "phosphorus_ppm": result.sample.phosphorus_ppm,
                "potassium_ppm": result.sample.potassium_ppm,
                "organic_matter_percent": result.sample.organic_matter_percent,
                "recommendations_count": len(result.recommendations),
                "limitations_count": len(result.limitations),
                "suitable_crops_count": len(result.suitability_crops)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def _export_to_json(self, results: List[SoilAnalysisResult], output_path: str):
        """Export results to JSON format"""
        export_data = []
        for result in results:
            result_dict = {
                "sample_id": result.sample.sample_id,
                "location": result.sample.location,
                "analysis": {
                    "texture_class": result.texture_class.value,
                    "health_score": result.health_score,
                    "health_category": result.health_category.value,
                    "nutrient_analysis": result.nutrient_analysis
                },
                "recommendations": result.recommendations,
                "limitations": result.limitations,
                "suitable_crops": result.suitability_crops
            }
            export_data.append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_to_xlsx(self, results: List[SoilAnalysisResult], output_path: str):
        """Export results to Excel format"""
        # This would require openpyxl - implementation similar to CSV but with multiple sheets
        # Summary sheet, detailed analysis sheet, recommendations sheet
        pass


def load_soil_data_from_csv(csv_path: str) -> List[SoilSample]:
    """
    Load soil sample data from CSV file
    
    Args:
        csv_path: Path to CSV file with soil data
        
    Returns:
        List of SoilSample objects
    """
    df = pd.read_csv(csv_path)
    samples = []
    
    required_columns = [
        'sample_id', 'location', 'date_collected',
        'sand_percent', 'silt_percent', 'clay_percent',
        'ph_level', 'nitrogen_ppm', 'phosphorus_ppm', 'potassium_ppm',
        'organic_matter_percent', 'bulk_density', 'moisture_content',
        'cation_exchange_capacity'
    ]
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for _, row in df.iterrows():
        sample = SoilSample(
            sample_id=str(row['sample_id']),
            location=str(row['location']),
            date_collected=str(row['date_collected']),
            sand_percent=float(row['sand_percent']),
            silt_percent=float(row['silt_percent']),
            clay_percent=float(row['clay_percent']),
            bulk_density=float(row['bulk_density']),
            moisture_content=float(row['moisture_content']),
            ph_level=float(row['ph_level']),
            nitrogen_ppm=float(row['nitrogen_ppm']),
            phosphorus_ppm=float(row['phosphorus_ppm']),
            potassium_ppm=float(row['potassium_ppm']),
            organic_matter_percent=float(row['organic_matter_percent']),
            cation_exchange_capacity=float(row['cation_exchange_capacity']),
            # Optional columns
            calcium_ppm=row.get('calcium_ppm', None),
            magnesium_ppm=row.get('magnesium_ppm', None),
            sulfur_ppm=row.get('sulfur_ppm', None),
            temperature_celsius=row.get('temperature_celsius', None),
            electrical_conductivity=row.get('electrical_conductivity', None)
        )
        samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} soil samples from {csv_path}")
    return samples


if __name__ == "__main__":
    # Example usage
    sample_data = SoilSample(
        sample_id="FIELD_A_001",
        location="Farm Field A, Iowa, USA",
        date_collected="2024-03-15",
        sand_percent=35.0,
        silt_percent=40.0,
        clay_percent=25.0,
        bulk_density=1.3,
        moisture_content=22.0,
        ph_level=6.5,
        nitrogen_ppm=45.0,
        phosphorus_ppm=28.0,
        potassium_ppm=180.0,
        organic_matter_percent=3.2,
        cation_exchange_capacity=15.5,
        temperature_celsius=18.0,
        electrical_conductivity=1.2
    )
    
    # Initialize analyzer and analyze sample
    analyzer = SoilAnalyzer()
    result = analyzer.analyze_sample(sample_data)
    
    # Print results
    print(f"Sample: {result.sample.sample_id}")
    print(f"Texture: {result.texture_class.value}")
    print(f"Health Score: {result.health_score:.1f}")
    print(f"Health Category: {result.health_category.value}")
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")
    print(f"\nSuitable Crops: {', '.join(result.suitability_crops)}")

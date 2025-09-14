#!/usr/bin/env python3
"""
Precision Farming CLI Tool

Command-line interface for quick soil analysis and crop recommendations.
Supports batch processing and various output formats.

Usage:
    python cli.py --soil sample.json --output recommendations.csv
    python cli.py --economics --crop wheat --area 25 --location "Iowa, USA"
    python cli.py --batch --input soil_data.csv --format json

Author: Precision Farming Team
Date: 2024
License: MIT
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.soil_analyzer import SoilAnalyzer, SoilSample, load_soil_data_from_csv
from src.crop_recommender import CropRecommendationEngine, WeatherData, create_sample_weather_data
from src.economics_calculator import EconomicsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_soil_from_json(json_path: str) -> SoilSample:
    """Load single soil sample from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return SoilSample(
        sample_id=data.get('sample_id', 'CLI_SAMPLE'),
        location=data.get('location', 'Unknown'),
        date_collected=data.get('date_collected', '2024-01-01'),
        sand_percent=float(data['sand_percent']),
        silt_percent=float(data['silt_percent']),
        clay_percent=float(data['clay_percent']),
        bulk_density=float(data.get('bulk_density', 1.3)),
        moisture_content=float(data.get('moisture_content', 25.0)),
        ph_level=float(data['ph_level']),
        nitrogen_ppm=float(data['nitrogen_ppm']),
        phosphorus_ppm=float(data['phosphorus_ppm']),
        potassium_ppm=float(data['potassium_ppm']),
        organic_matter_percent=float(data['organic_matter_percent']),
        cation_exchange_capacity=float(data.get('cation_exchange_capacity', 15.0)),
        temperature_celsius=data.get('temperature_celsius'),
        electrical_conductivity=data.get('electrical_conductivity')
    )


def soil_to_dict(sample: SoilSample) -> Dict:
    """Convert soil sample to dictionary for crop recommender"""
    return {
        'ph_level': sample.ph_level,
        'nitrogen_ppm': sample.nitrogen_ppm,
        'phosphorus_ppm': sample.phosphorus_ppm,
        'potassium_ppm': sample.potassium_ppm,
        'organic_matter_percent': sample.organic_matter_percent,
        'texture_class': 'loam'  # Would need texture classification
    }


def analyze_soil_command(args):
    """Handle soil analysis command"""
    logger.info(f"Analyzing soil sample: {args.soil}")
    
    # Load soil data
    if args.soil.endswith('.json'):
        soil_sample = load_soil_from_json(args.soil)
        samples = [soil_sample]
    elif args.soil.endswith('.csv'):
        samples = load_soil_data_from_csv(args.soil)
    else:
        raise ValueError("Soil file must be JSON or CSV format")
    
    # Initialize analyzer
    analyzer = SoilAnalyzer()
    
    # Analyze samples
    results = []
    for sample in samples:
        result = analyzer.analyze_sample(sample)
        results.append(result)
        
        # Print summary to console
        print(f"\n{'='*50}")
        print(f"Soil Analysis: {sample.sample_id}")
        print(f"{'='*50}")
        print(f"Location: {sample.location}")
        print(f"Texture Class: {result.texture_class.value.title()}")
        print(f"Health Score: {result.health_score:.1f}/100")
        print(f"Health Category: {result.health_category.value.title()}")
        
        print(f"\nNutrient Analysis:")
        for nutrient, analysis in result.nutrient_analysis.items():
            if isinstance(analysis, dict):
                print(f"  {nutrient.title()}: {analysis.get('status', 'N/A')}")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nSuitable Crops: {', '.join(result.suitability_crops[:5])}")
    
    # Export results if output specified
    if args.output:
        output_format = args.format or args.output.split('.')[-1]
        analyzer.export_results(results, args.output, output_format)
        print(f"\nResults exported to: {args.output}")


def recommend_crops_command(args):
    """Handle crop recommendation command"""
    logger.info("Generating crop recommendations")
    
    # Load soil data
    if args.soil.endswith('.json'):
        soil_sample = load_soil_from_json(args.soil)
    else:
        raise ValueError("Soil file must be JSON format for crop recommendation")
    
    # Convert to dict format
    soil_data = soil_to_dict(soil_sample)
    
    # Get weather data (sample for now)
    weather_data = create_sample_weather_data(args.location or "Iowa, USA")
    
    # Initialize recommendation engine
    engine = CropRecommendationEngine()
    
    # Get recommendations
    recommendations = engine.recommend_crops(
        soil_data=soil_data,
        weather_data=weather_data,
        farm_size=args.area or 10.0,
        top_n=args.top or 5
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"CROP RECOMMENDATIONS - {soil_sample.location}")
    print(f"{'='*60}")
    print(f"Farm Size: {args.area or 10.0} hectares")
    print(f"Analysis Date: {soil_sample.date_collected}")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.crop_profile.name.upper()}")
        print(f"   {'─' * 40}")
        print(f"   Overall Score: {rec.suitability_score:.1f}/100")
        print(f"   Suitability: {rec.suitability_level.value.title()}")
        print(f"   Confidence: {rec.confidence:.1f}%")
        print(f"   Expected ROI: {rec.expected_roi:.1f}%")
        
        print(f"\n   Detailed Scores:")
        print(f"     Soil Compatibility: {rec.soil_match_score:.1f}/100")
        print(f"     Climate Match: {rec.climate_match_score:.1f}/100")
        print(f"     Economic Viability: {rec.economic_score:.1f}/100")
        print(f"     Risk Assessment: {rec.risk_score:.1f}/100")
        
        if rec.positive_factors:
            print(f"\n   ✓ Positive Factors:")
            for factor in rec.positive_factors[:3]:
                print(f"     • {factor}")
        
        if rec.negative_factors:
            print(f"\n   ⚠ Considerations:")
            for factor in rec.negative_factors[:3]:
                print(f"     • {factor}")
        
        if rec.planting_recommendations:
            print(f"\n   📋 Key Recommendations:")
            for tip in rec.planting_recommendations[:3]:
                print(f"     • {tip}")
    
    # Export if requested
    if args.output:
        export_recommendations(recommendations, args.output, args.format)
        print(f"\nRecommendations exported to: {args.output}")


def economics_command(args):
    """Handle economic analysis command"""
    logger.info(f"Calculating economics for {args.crop}")
    
    # This would integrate with the economics calculator
    # For now, show basic calculation
    
    crop_data = {
        'wheat': {'yield': 3.5, 'price': 250, 'cost': 900},
        'corn': {'yield': 8.5, 'price': 200, 'cost': 1250},
        'rice': {'yield': 4.5, 'price': 300, 'cost': 1100},
        'soybean': {'yield': 2.8, 'price': 400, 'cost': 950}
    }
    
    if args.crop.lower() not in crop_data:
        print(f"Error: Crop '{args.crop}' not supported")
        return
    
    crop = crop_data[args.crop.lower()]
    area = args.area
    
    # Calculate economics
    total_cost = crop['cost'] * area
    total_revenue = crop['yield'] * crop['price'] * area
    total_profit = total_revenue - total_cost
    roi = (total_profit / total_cost) * 100
    profit_per_ha = total_profit / area
    
    print(f"\n{'='*50}")
    print(f"ECONOMIC ANALYSIS - {args.crop.title()}")
    print(f"{'='*50}")
    print(f"Location: {args.location}")
    print(f"Farm Size: {area} hectares")
    
    print(f"\nPer Hectare:")
    print(f"  Expected Yield: {crop['yield']} tonnes/ha")
    print(f"  Market Price: ${crop['price']}/tonne")
    print(f"  Production Cost: ${crop['cost']}/ha")
    print(f"  Revenue: ${crop['yield'] * crop['price']:.0f}/ha")
    print(f"  Profit: ${profit_per_ha:.0f}/ha")
    
    print(f"\nTotal Farm ({area} ha):")
    print(f"  Total Cost: ${total_cost:,.0f}")
    print(f"  Total Revenue: ${total_revenue:,.0f}")
    print(f"  Total Profit: ${total_profit:,.0f}")
    print(f"  ROI: {roi:.1f}%")
    
    # Risk assessment
    if roi > 25:
        risk_level = "Low Risk - Excellent Returns"
    elif roi > 15:
        risk_level = "Moderate Risk - Good Returns"
    elif roi > 5:
        risk_level = "Higher Risk - Acceptable Returns"
    else:
        risk_level = "High Risk - Low Returns"
    
    print(f"\nRisk Assessment: {risk_level}")


def batch_analysis_command(args):
    """Handle batch analysis command"""
    logger.info(f"Running batch analysis on: {args.input}")
    
    # Load multiple soil samples
    if args.input.endswith('.csv'):
        samples = load_soil_data_from_csv(args.input)
    else:
        raise ValueError("Batch input must be CSV format")
    
    # Initialize tools
    analyzer = SoilAnalyzer()
    engine = CropRecommendationEngine()
    
    # Process each sample
    batch_results = []
    
    for sample in samples:
        print(f"Processing: {sample.sample_id}")
        
        # Soil analysis
        soil_result = analyzer.analyze_sample(sample)
        
        # Crop recommendations
        soil_dict = soil_to_dict(sample)
        weather_data = create_sample_weather_data(sample.location)
        crop_recs = engine.recommend_crops(soil_dict, weather_data, top_n=3)
        
        # Compile results
        result = {
            'sample_id': sample.sample_id,
            'location': sample.location,
            'health_score': soil_result.health_score,
            'health_category': soil_result.health_category.value,
            'texture_class': soil_result.texture_class.value,
            'top_crop': crop_recs[0].crop_profile.name if crop_recs else 'None',
            'top_crop_score': crop_recs[0].suitability_score if crop_recs else 0,
            'suitable_crops': ', '.join([r.crop_profile.name for r in crop_recs[:3]])
        }
        
        batch_results.append(result)
    
    # Create summary
    df = pd.DataFrame(batch_results)
    
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Samples Processed: {len(batch_results)}")
    print(f"Average Health Score: {df['health_score'].mean():.1f}")
    
    health_dist = df['health_category'].value_counts()
    print(f"\nHealth Distribution:")
    for category, count in health_dist.items():
        print(f"  {category.title()}: {count} samples")
    
    crop_dist = df['top_crop'].value_counts()
    print(f"\nTop Recommended Crops:")
    for crop, count in crop_dist.head().items():
        print(f"  {crop}: {count} samples")
    
    # Export results
    if args.output:
        output_format = args.format or 'csv'
        if output_format == 'csv':
            df.to_csv(args.output, index=False)
        elif output_format == 'json':
            df.to_json(args.output, orient='records', indent=2)
        print(f"\nBatch results exported to: {args.output}")


def export_recommendations(recommendations, output_path: str, format_type: str = None):
    """Export recommendations to file"""
    if not format_type:
        format_type = output_path.split('.')[-1]
    
    export_data = []
    for rec in recommendations:
        data = {
            'crop_name': rec.crop_profile.name,
            'suitability_score': rec.suitability_score,
            'suitability_level': rec.suitability_level.value,
            'confidence': rec.confidence,
            'expected_roi': rec.expected_roi,
            'soil_match': rec.soil_match_score,
            'climate_match': rec.climate_match_score,
            'economic_score': rec.economic_score,
            'risk_score': rec.risk_score,
            'positive_factors': '; '.join(rec.positive_factors),
            'negative_factors': '; '.join(rec.negative_factors),
            'recommendations': '; '.join(rec.planting_recommendations)
        }
        export_data.append(data)
    
    df = pd.DataFrame(export_data)
    
    if format_type.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format_type.lower() == 'json':
        df.to_json(output_path, orient='records', indent=2)
    elif format_type.lower() == 'xlsx':
        df.to_excel(output_path, index=False)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Precision Farming CLI Tool",
        epilog="Examples:\n"
               "  python cli.py --soil sample.json --output analysis.csv\n"
               "  python cli.py --recommend --soil sample.json --area 25\n"
               "  python cli.py --economics --crop wheat --area 50 --location 'Iowa'\n"
               "  python cli.py --batch --input soil_data.csv --output results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Command selection
    parser.add_argument('--soil', type=str, help='Soil data file (JSON or CSV)')
    parser.add_argument('--recommend', action='store_true', 
                       help='Generate crop recommendations')
    parser.add_argument('--economics', action='store_true',
                       help='Perform economic analysis')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process multiple samples')
    
    # Input/Output options
    parser.add_argument('--input', type=str, help='Input file for batch processing')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'xlsx'],
                       help='Output format (auto-detected if not specified)')
    
    # Analysis parameters
    parser.add_argument('--crop', type=str, help='Crop type for economic analysis')
    parser.add_argument('--area', type=float, default=10.0, 
                       help='Farm area in hectares (default: 10.0)')
    parser.add_argument('--location', type=str, default='Iowa, USA',
                       help='Farm location for weather data')
    parser.add_argument('--top', type=int, default=5,
                       help='Number of top recommendations (default: 5)')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--version', action='version', version='Precision Farming CLI 1.0.0')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments and route to appropriate command
    try:
        if args.batch:
            if not args.input:
                parser.error("--batch requires --input")
            batch_analysis_command(args)
        
        elif args.recommend:
            if not args.soil:
                parser.error("--recommend requires --soil")
            recommend_crops_command(args)
        
        elif args.economics:
            if not args.crop:
                parser.error("--economics requires --crop")
            economics_command(args)
        
        elif args.soil:
            analyze_soil_command(args)
        
        else:
            # Show help if no command specified
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

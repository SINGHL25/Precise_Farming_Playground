
"""
Economics Calculator for Precision Farming

Comprehensive economic analysis module for farm profitability assessment,
cost-benefit analysis, ROI calculations, and financial planning.

Features:
- Detailed cost breakdowns
- Revenue projections
- Profitability analysis
- Risk assessment
- Scenario modeling
- Multi-year planning

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
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of farming costs"""
    SEED = "seed"
    FERTILIZER = "fertilizer"
    PESTICIDE = "pesticide"
    LABOR = "labor"
    MACHINERY = "machinery"
    FUEL = "fuel"
    IRRIGATION = "irrigation"
    STORAGE = "storage"
    TRANSPORT = "transport"
    INSURANCE = "insurance"
    LAND_RENT = "land_rent"
    UTILITIES = "utilities"
    OTHER = "other"


class RevenueSource(Enum):
    """Sources of farm revenue"""
    CROP_SALES = "crop_sales"
    GOVERNMENT_SUBSIDIES = "government_subsidies"
    INSURANCE_PAYOUTS = "insurance_payouts"
    CONTRACT_PAYMENTS = "contract_payments"
    VALUE_ADDED_PRODUCTS = "value_added_products"
    OTHER = "other"


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CostItem:
    """Individual cost item"""
    category: CostCategory
    description: str
    amount_per_hectare: float
    fixed_amount: float = 0.0
    is_variable: bool = True
    timing: str = "season_start"  # when cost is incurred
    notes: str = ""


@dataclass
class RevenueItem:
    """Individual revenue item"""
    source: RevenueSource
    description: str
    amount_per_hectare: float = 0.0
    fixed_amount: float = 0.0
    timing: str = "harvest"  # when revenue is received
    certainty: float = 1.0  # probability of receiving (0-1)
    notes: str = ""


@dataclass
class CostBreakdown:
    """Detailed cost breakdown analysis"""
    total_cost_per_hectare: float
    total_fixed_costs: float
    total_variable_costs: float
    cost_categories: Dict[str, float]
    largest_cost_category: str
    cost_efficiency_score: float
    recommendations: List[str]


@dataclass
class RevenueProjection:
    """Revenue projection analysis"""
    total_revenue_per_hectare: float
    expected_revenue: float
    revenue_sources: Dict[str, float]
    revenue_certainty: float
    seasonal_distribution: Dict[str, float]
    growth_projections: List[float]


@dataclass
class RiskAssessment:
    """Risk analysis results"""
    overall_risk_level: RiskLevel
    risk_factors: Dict[str, float]
    probability_of_loss: float
    worst_case_scenario: float
    best_case_scenario: float
    risk_mitigation_suggestions: List[str]
    insurance_recommendations: List[str]


@dataclass
class EconomicAnalysis:
    """Comprehensive economic analysis results"""
    crop_name: str
    farm_size_hectares: float
    analysis_date: str
    
    # Financial metrics
    total_costs: float
    total_revenue: float
    gross_profit: float
    net_profit: float
    profit_margin: float
    roi_percent: float
    
    # Break-even analysis
    break_even_yield: float
    break_even_price: float
    break_even_area: float
    safety_margin: float
    
    # Detailed breakdowns
    cost_breakdown: CostBreakdown
    revenue_projection: RevenueProjection
    risk_assessment: RiskAssessment
    
    # Comparative metrics
    industry_benchmark_profit: float
    profit_vs_benchmark: float
    efficiency_rating: str
    
    # Recommendations
    recommendations: List[str]
    action_items: List[str]


@dataclass
class ScenarioAnalysis:
    """Multi-scenario economic analysis"""
    base_case: EconomicAnalysis
    optimistic_case: EconomicAnalysis
    pessimistic_case: EconomicAnalysis
    scenario_probabilities: Dict[str, float]
    expected_value: float
    value_at_risk: float
    sensitivity_analysis: Dict[str, float]


class EconomicsCalculator:
    """
    Main calculator for farm economics analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize economics calculator
        
        Args:
            config: Configuration dictionary with default values
        """
        self.config = config or self._load_default_config()
        self.cost_database = self._initialize_cost_database()
        self.price_database = self._initialize_price_database()
        logger.info("Economics calculator initialized")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'discount_rate': 8.0,
            'inflation_rate': 2.5,
            'tax_rate': 25.0,
            'insurance_rate': 2.0,
            'currency': 'USD',
            'risk_free_rate': 3.0,
            'market_risk_premium': 5.0,
            'default_planning_horizon': 5
        }
    
    def _initialize_cost_database(self) -> Dict:
        """Initialize default cost database"""
        return {
            'wheat': {
                'seed': 150, 'fertilizer': 200, 'pesticide': 80, 'labor': 300,
                'machinery': 250, 'fuel': 120, 'other': 100
            },
            'corn': {
                'seed': 200, 'fertilizer': 350, 'pesticide': 120, 'labor': 400,
                'machinery': 300, 'fuel': 150, 'other': 150
            },
            'rice': {
                'seed': 180, 'fertilizer': 280, 'pesticide': 150, 'labor': 450,
                'machinery': 200, 'fuel': 100, 'other': 120
            },
            'soybean': {
                'seed': 220, 'fertilizer': 150, 'pesticide': 100, 'labor': 350,
                'machinery': 280, 'fuel': 130, 'other': 100
            }
        }
    
    def _initialize_price_database(self) -> Dict:
        """Initialize default price database"""
        return {
            'wheat': {'price_per_tonne': 250, 'yield_per_hectare': 3.5, 'volatility': 0.15},
            'corn': {'price_per_tonne': 200, 'yield_per_hectare': 8.5, 'volatility': 0.18},
            'rice': {'price_per_tonne': 300, 'yield_per_hectare': 4.5, 'volatility': 0.12},
            'soybean': {'price_per_tonne': 400, 'yield_per_hectare': 2.8, 'volatility': 0.20}
        }
    
    def create_cost_breakdown(self, crop_name: str, farm_size: float, 
                            custom_costs: Optional[Dict] = None) -> CostBreakdown:
        """
        Create detailed cost breakdown analysis
        
        Args:
            crop_name: Name of the crop
            farm_size: Farm size in hectares
            custom_costs: Optional custom cost overrides
            
        Returns:
            CostBreakdown object with detailed analysis
        """
        # Get base costs
        base_costs = self.cost_database.get(crop_name.lower(), 
                                           self.cost_database['wheat'])
        
        # Apply custom costs if provided
        if custom_costs:
            base_costs.update(custom_costs)
        
        # Calculate costs per hectare and total
        cost_per_ha = sum(base_costs.values())
        total_cost = cost_per_ha * farm_size
        
        # Categorize costs
        variable_costs = sum(base_costs[k] for k in ['seed', 'fertilizer', 'pesticide', 'fuel'])
        fixed_costs = sum(base_costs[k] for k in ['labor', 'machinery', 'other'])
        
        # Find largest cost category
        largest_category = max(base_costs.items(), key=lambda x: x[1])
        
        # Calculate efficiency score
        industry_avg_cost = self._get_industry_average_cost(crop_name)
        efficiency_score = min(100, (industry_avg_cost / cost_per_ha) * 100)
        
        # Generate recommendations
        recommendations = self._generate_cost_recommendations(base_costs, efficiency_score)
        
        return CostBreakdown(
            total_cost_per_hectare=cost_per_ha,
            total_fixed_costs=fixed_costs * farm_size,
            total_variable_costs=variable_costs * farm_size,
            cost_categories={k: v * farm_size for k, v in base_costs.items()},
            largest_cost_category=largest_category[0],
            cost_efficiency_score=efficiency_score,
            recommendations=recommendations
        )
    
    def create_revenue_projection(self, crop_name: str, farm_size: float,
                                expected_yield: float, market_price: float,
                                custom_revenue: Optional[Dict] = None) -> RevenueProjection:
        """
        Create revenue projection analysis
        
        Args:
            crop_name: Name of the crop
            farm_size: Farm size in hectares
            expected_yield: Expected yield per hectare
            market_price: Expected market price per tonne
            custom_revenue: Optional additional revenue sources
            
        Returns:
            RevenueProjection object
        """
        # Primary crop revenue
        primary_revenue = expected_yield * market_price
        total_revenue_per_ha = primary_revenue
        
        # Additional revenue sources
        revenue_sources = {'crop_sales': primary_revenue * farm_size}
        
        if custom_revenue:
            for source, amount in custom_revenue.items():
                revenue_sources[source] = amount * farm_size
                total_revenue_per_ha += amount
        
        # Calculate revenue certainty
        crop_data = self.price_database.get(crop_name.lower(), self.price_database['wheat'])
        volatility = crop_data.get('volatility', 0.15)
        revenue_certainty = max(0.5, 1.0 - volatility)
        
        # Expected revenue (accounting for uncertainty)
        expected_revenue = total_revenue_per_ha * revenue_certainty * farm_size
        
        # Seasonal distribution (simplified)
        seasonal_dist = {
            'planting': 0.0,
            'growing': 0.1,  # Some early payments/contracts
            'harvest': 0.7,
            'post_harvest': 0.2
        }
        
        # Growth projections (5-year)
        annual_growth = 0.02  # 2% annual growth
        growth_projections = [
            total_revenue_per_ha * ((1 + annual_growth) ** i) 
            for i in range(1, 6)
        ]
        
        return RevenueProjection(
            total_revenue_per_hectare=total_revenue_per_ha,
            expected_revenue=expected_revenue,
            revenue_sources=revenue_sources,
            revenue_certainty=revenue_certainty,
            seasonal_distribution=seasonal_dist,
            growth_projections=growth_projections
        )
    
    def assess_risk(self, crop_name: str, farm_size: float, 
                   cost_breakdown: CostBreakdown,
                   revenue_projection: RevenueProjection,
                   external_factors: Optional[Dict] = None) -> RiskAssessment:
        """
        Assess financial and operational risks
        
        Args:
            crop_name: Name of the crop
            farm_size: Farm size in hectares
            cost_breakdown: Cost breakdown analysis
            revenue_projection: Revenue projection
            external_factors: Optional external risk factors
            
        Returns:
            RiskAssessment object
        """
        # Calculate baseline metrics
        total_costs = cost_breakdown.total_cost_per_hectare * farm_size
        expected_revenue = revenue_projection.expected_revenue
        
        # Risk factor analysis
        risk_factors = {}
        
        # Market risk
        crop_data = self.price_database.get(crop_name.lower(), self.price_database['wheat'])
        market_volatility = crop_data.get('volatility', 0.15)
        risk_factors['market_risk'] = market_volatility * 100
        
        # Production risk
        yield_variability = 0.20  # Assume 20% yield variability
        risk_factors['production_risk'] = yield_variability * 100
        
        # Financial risk
        debt_to_equity = 0.3  # Assume moderate leverage
        risk_factors['financial_risk'] = debt_to_equity * 100
        
        # Weather risk
        weather_risk = 0.15  # Regional weather variability
        risk_factors['weather_risk'] = weather_risk * 100
        
        # Input cost risk
        input_cost_volatility = 0.10
        risk_factors['input_cost_risk'] = input_cost_volatility * 100
        
        # Add external factors
        if external_factors:
            risk_factors.update(external_factors)
        
        # Overall risk assessment
        avg_risk = np.mean(list(risk_factors.values()))
        
        if avg_risk < 15:
            risk_level = RiskLevel.LOW
        elif avg_risk < 25:
            risk_level = RiskLevel.MODERATE
        elif avg_risk < 35:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        # Scenario analysis
        prob_loss = max(0.05, market_volatility * 0.5)
        worst_case = expected_revenue * (1 - market_volatility - yield_variability)
        best_case = expected_revenue * (1 + market_volatility + yield_variability)
        
        # Risk mitigation suggestions
        mitigation_suggestions = self._generate_risk_mitigation(risk_factors, risk_level)
        
        # Insurance recommendations
        insurance_recs = self._generate_insurance_recommendations(
            crop_name, farm_size, total_costs, risk_level
        )
        
        return RiskAssessment(
            overall_risk_level=risk_level,
            risk_factors=risk_factors,
            probability_of_loss=prob_loss,
            worst_case_scenario=worst_case - total_costs,
            best_case_scenario=best_case - total_costs,
            risk_mitigation_suggestions=mitigation_suggestions,
            insurance_recommendations=insurance_recs
        )
    
    def perform_economic_analysis(self, crop_name: str, farm_size: float,
                                expected_yield: float, market_price: float,
                                custom_costs: Optional[Dict] = None,
                                custom_revenue: Optional[Dict] = None,
                                external_factors: Optional[Dict] = None) -> EconomicAnalysis:
        """
        Perform comprehensive economic analysis
        
        Args:
            crop_name: Name of the crop
            farm_size: Farm size in hectares
            expected_yield: Expected yield per hectare
            market_price: Expected market price per tonne
            custom_costs: Optional custom cost overrides
            custom_revenue: Optional additional revenue sources
            external_factors: Optional external factors
            
        Returns:
            Complete EconomicAnalysis object
        """
        logger.info(f"Performing economic analysis for {crop_name}, {farm_size} ha")
        
        # Create component analyses
        cost_breakdown = self.create_cost_breakdown(crop_name, farm_size, custom_costs)
        revenue_projection = self.create_revenue_projection(
            crop_name, farm_size, expected_yield, market_price, custom_revenue
        )
        risk_assessment = self.assess_risk(
            crop_name, farm_size, cost_breakdown, revenue_projection, external_factors
        )
        
        # Calculate financial metrics
        total_costs = cost_breakdown.total_cost_per_hectare * farm_size
        total_revenue = revenue_projection.total_revenue_per_hectare * farm_size
        gross_profit = total_revenue - total_costs
        
        # Account for taxes
        tax_rate = self.config.get('tax_rate', 25.0) / 100
        net_profit = gross_profit * (1 - tax_rate)
        
        profit_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        roi_percent = (net_profit / total_costs * 100) if total_costs > 0 else 0
        
        # Break-even analysis
        break_even_yield = (cost_breakdown.total_cost_per_hectare / market_price 
                           if market_price > 0 else 0)
        break_even_price = (cost_breakdown.total_cost_per_hectare / expected_yield 
                           if expected_yield > 0 else 0)
        break_even_area = (total_costs / (expected_yield * market_price) 
                          if expected_yield * market_price > 0 else 0)
        
        safety_margin = ((expected_yield - break_even_yield) / expected_yield * 100 
                        if expected_yield > 0 else 0)
        
        # Industry benchmarking
        industry_benchmark = self._get_industry_benchmark_profit(crop_name)
        profit_vs_benchmark = ((roi_percent - industry_benchmark) / industry_benchmark * 100 
                              if industry_benchmark > 0 else 0)
        
        # Efficiency rating
        if roi_percent >= industry_benchmark * 1.2:
            efficiency_rating = "Excellent"
        elif roi_percent >= industry_benchmark * 1.1:
            efficiency_rating = "Good"
        elif roi_percent >= industry_benchmark * 0.9:
            efficiency_rating = "Average"
        else:
            efficiency_rating = "Below Average"
        
        # Generate recommendations
        recommendations = self._generate_economic_recommendations(
            cost_breakdown, revenue_projection, risk_assessment, roi_percent
        )
        
        # Generate action items
        action_items = self._generate_action_items(
            cost_breakdown, revenue_projection, risk_assessment
        )
        
        return EconomicAnalysis(
            crop_name=crop_name.title(),
            farm_size_hectares=farm_size,
            analysis_date=datetime.now().strftime('%Y-%m-%d'),
            total_costs=total_costs,
            total_revenue=total_revenue,
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_margin=profit_margin,
            roi_percent=roi_percent,
            break_even_yield=break_even_yield,
            break_even_price=break_even_price,
            break_even_area=break_even_area,
            safety_margin=safety_margin,
            cost_breakdown=cost_breakdown,
            revenue_projection=revenue_projection,
            risk_assessment=risk_assessment,
            industry_benchmark_profit=industry_benchmark,
            profit_vs_benchmark=profit_vs_benchmark,
            efficiency_rating=efficiency_rating,
            recommendations=recommendations,
            action_items=action_items
        )
    
    def perform_scenario_analysis(self, base_analysis: EconomicAnalysis,
                                scenarios: Optional[Dict] = None) -> ScenarioAnalysis:
        """
        Perform multi-scenario analysis
        
        Args:
            base_analysis: Base case economic analysis
            scenarios: Optional custom scenario parameters
            
        Returns:
            ScenarioAnalysis object
        """
        # Default scenario parameters
        if not scenarios:
            scenarios = {
                'optimistic': {'yield_factor': 1.2, 'price_factor': 1.1, 'cost_factor': 0.95},
                'pessimistic': {'yield_factor': 0.8, 'price_factor': 0.9, 'cost_factor': 1.1}
            }
        
        # Calculate optimistic scenario
        opt_params = scenarios['optimistic']
        optimistic_analysis = self._create_scenario_analysis(base_analysis, opt_params)
        
        # Calculate pessimistic scenario
        pess_params = scenarios['pessimistic']
        pessimistic_analysis = self._create_scenario_analysis(base_analysis, pess_params)
        
        # Scenario probabilities
        probabilities = {'optimistic': 0.25, 'base': 0.5, 'pessimistic': 0.25}
        
        # Expected value calculation
        expected_value = (
            base_analysis.net_profit * probabilities['base'] +
            optimistic_analysis.net_profit * probabilities['optimistic'] +
            pessimistic_analysis.net_profit * probabilities['pessimistic']
        )
        
        # Value at Risk (5% VaR)
        value_at_risk = pessimistic_analysis.net_profit
        
        # Sensitivity analysis
        sensitivity_analysis = self._calculate_sensitivity(base_analysis)
        
        return ScenarioAnalysis(
            base_case=base_analysis,
            optimistic_case=optimistic_analysis,
            pessimistic_case=pessimistic_analysis,
            scenario_probabilities=probabilities,
            expected_value=expected_value,
            value_at_risk=value_at_risk,
            sensitivity_analysis=sensitivity_analysis
        )
    
    def _create_scenario_analysis(self, base_analysis: EconomicAnalysis, 
                                factors: Dict) -> EconomicAnalysis:
        """Create analysis for a specific scenario"""
        # This is a simplified implementation
        # In reality, you would re-run the full analysis with adjusted parameters
        
        adjusted_revenue = (base_analysis.total_revenue * 
                           factors.get('yield_factor', 1.0) * 
                           factors.get('price_factor', 1.0))
        
        adjusted_costs = (base_analysis.total_costs * 
                         factors.get('cost_factor', 1.0))
        
        # Create a copy of the base analysis with adjusted values
        scenario_analysis = EconomicAnalysis(
            crop_name=base_analysis.crop_name,
            farm_size_hectares=base_analysis.farm_size_hectares,
            analysis_date=base_analysis.analysis_date,
            total_costs=adjusted_costs,
            total_revenue=adjusted_revenue,
            gross_profit=adjusted_revenue - adjusted_costs,
            net_profit=(adjusted_revenue - adjusted_costs) * 
                      (1 - self.config.get('tax_rate', 25.0) / 100),
            profit_margin=((adjusted_revenue - adjusted_costs) / adjusted_revenue * 100 
                          if adjusted_revenue > 0 else 0),
            roi_percent=((adjusted_revenue - adjusted_costs) / adjusted_costs * 100 
                        if adjusted_costs > 0 else 0),
            break_even_yield=base_analysis.break_even_yield * factors.get('cost_factor', 1.0) / factors.get('price_factor', 1.0),
            break_even_price=base_analysis.break_even_price * factors.get('cost_factor', 1.0),
            break_even_area=base_analysis.break_even_area / factors.get('yield_factor', 1.0),
            safety_margin=base_analysis.safety_margin,
            cost_breakdown=base_analysis.cost_breakdown,
            revenue_projection=base_analysis.revenue_projection,
            risk_assessment=base_analysis.risk_assessment,
            industry_benchmark_profit=base_analysis.industry_benchmark_profit,
            profit_vs_benchmark=base_analysis.profit_vs_benchmark,
            efficiency_rating=base_analysis.efficiency_rating,
            recommendations=base_analysis.recommendations,
            action_items=base_analysis.action_items
        )
        
        return scenario_analysis
    
    def _calculate_sensitivity(self, analysis: EconomicAnalysis) -> Dict[str, float]:
        """Calculate sensitivity analysis"""
        base_profit = analysis.net_profit
        
        # Test 10% changes in key variables
        sensitivity = {}
        
        # Yield sensitivity
        yield_change = 0.1
        yield_impact = (analysis.total_revenue * yield_change) * 0.75  # 75% flows to profit
        sensitivity['yield'] = (yield_impact / base_profit * 100) if base_profit != 0 else 0
        
        # Price sensitivity  
        price_change = 0.1
        price_impact = (analysis.total_revenue * price_change) * 0.75
        sensitivity['price'] = (price_impact / base_profit * 100) if base_profit != 0 else 0
        
        # Cost sensitivity
        cost_change = 0.1
        cost_impact = analysis.total_costs * cost_change
        sensitivity['costs'] = (cost_impact / base_profit * 100) if base_profit != 0 else 0
        
        return sensitivity
    
    def _get_industry_average_cost(self, crop_name: str) -> float:
        """Get industry average cost for benchmarking"""
        # Simplified industry averages
        industry_costs = {
            'wheat': 1200, 'corn': 1500, 'rice': 1380, 'soybean': 1330
        }
        return industry_costs.get(crop_name.lower(), 1300)
    
    def _get_industry_benchmark_profit(self, crop_name: str) -> float:
        """Get industry benchmark profit margin"""
        # Simplified industry benchmarks (ROI %)
        benchmarks = {
            'wheat': 12.0, 'corn': 15.0, 'rice': 18.0, 'soybean': 14.0
        }
        return benchmarks.get(crop_name.lower(), 13.0)
    
    def _generate_cost_recommendations(self, costs: Dict, efficiency_score: float) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Find highest cost categories
        sorted_costs = sorted(costs.items(), key=lambda x: x[1], reverse=True)
        
        if efficiency_score < 80:
            recommendations.append(
                f"Focus on reducing {sorted_costs[0][0]} costs - your largest expense category"
            )
        
        if costs.get('fertilizer', 0) > 300:
            recommendations.append(
                "Consider soil testing to optimize fertilizer application and reduce costs"
            )
        
        if costs.get('labor', 0) > 400:
            recommendations.append(
                "Evaluate automation opportunities to reduce labor costs"
            )
        
        if costs.get('fuel', 0) > 150:
            recommendations.append(
                "Optimize field operations and equipment efficiency to reduce fuel costs"
            )
        
        return recommendations
    
    def _generate_risk_mitigation(self, risk_factors: Dict, risk_level: RiskLevel) -> List[str]:
        """Generate risk mitigation suggestions"""
        suggestions = []
        
        if risk_factors.get('market_risk', 0) > 20:
            suggestions.append("Consider forward contracts or futures to hedge price risk")
        
        if risk_factors.get('weather_risk', 0) > 15:
            suggestions.append("Invest in weather monitoring and crop insurance")
        
        if risk_factors.get('production_risk', 0) > 25:
            suggestions.append("Diversify crops and implement precision agriculture practices")
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            suggestions.append("Consider reducing farm size or increasing cash reserves")
        
        return suggestions
    
    def _generate_insurance_recommendations(self, crop_name: str, farm_size: float,
                                          total_costs: float, risk_level: RiskLevel) -> List[str]:
        """Generate insurance recommendations"""
        recommendations = []
        
        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Consider crop yield insurance to protect against production losses")
        
        if total_costs > 50000:  # Large investment
            recommendations.append("Evaluate revenue protection insurance for price and yield coverage")
        
        if farm_size > 100:  # Large operation
            recommendations.append("Consider whole farm revenue protection insurance")
        
        recommendations.append(f"Recommended insurance coverage: {int(total_costs * 0.8):,} (80% of costs)")
        
        return recommendations
    
    def _generate_economic_recommendations(self, cost_breakdown: CostBreakdown,
                                         revenue_projection: RevenueProjection,
                                         risk_assessment: RiskAssessment,
                                         roi_percent: float) -> List[str]:
        """Generate overall economic recommendations"""
        recommendations = []
        
        if roi_percent < 10:
            recommendations.append("ROI is below optimal levels. Focus on cost reduction and yield improvement")
        elif roi_percent > 25:
            recommendations.append("Excellent ROI. Consider expanding operations or investing in technology")
        
        if cost_breakdown.cost_efficiency_score < 70:
            recommendations.append("Cost efficiency is below average. Review input costs and optimize operations")
        
        if revenue_projection.revenue_certainty < 0.8:
            recommendations.append("Revenue uncertainty is high. Consider diversification or risk management tools")
        
        if risk_assessment.overall_risk_level == RiskLevel.HIGH:
            recommendations.append("Risk level is high. Implement comprehensive risk management strategies")
        
        return recommendations
    
    def _generate_action_items(self, cost_breakdown: CostBreakdown,
                             revenue_projection: RevenueProjection,
                             risk_assessment: RiskAssessment) -> List[str]:
        """Generate specific action items"""
        actions = []
        
        # Cost-related actions
        if cost_breakdown.largest_cost_category in ['fertilizer', 'pesticide']:
            actions.append("Conduct soil testing to optimize nutrient and pest management programs")
        
        # Revenue-related actions
        if revenue_projection.revenue_certainty < 0.9:
            actions.append("Explore contract farming or forward selling opportunities")
        
        # Risk-related actions
        if risk_assessment.probability_of_loss > 0.2:
            actions.append("Review and update risk management plan within 30 days")
        
        # General actions
        actions.extend([
            "Update financial projections quarterly",
            "Monitor market prices weekly",
            "Review cost performance monthly",
            "Evaluate new technology adoption annually"
        ])
        
        return actions
    
    def export_analysis(self, analysis: Union[EconomicAnalysis, ScenarioAnalysis],
                       output_path: str, format: str = "json") -> None:
        """
        Export economic analysis to file
        
        Args:
            analysis: Analysis results to export
            output_path: Output file path
            format: Export format ('json', 'csv', 'xlsx')
        """
        if format.lower() == "json":
            self._export_to_json(analysis, output_path)
        elif format.lower() == "csv":
            self._export_to_csv(analysis, output_path)
        elif format.lower() == "xlsx":
            self._export_to_xlsx(analysis, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Analysis exported to {output_path}")
    
    def _export_to_json(self, analysis, output_path: str):
        """Export analysis to JSON format"""
        # Convert dataclass to dictionary for JSON serialization
        if isinstance(analysis, EconomicAnalysis):
            export_data = {
                'analysis_type': 'economic_analysis',
                'crop_name': analysis.crop_name,
                'farm_size_hectares': analysis.farm_size_hectares,
                'analysis_date': analysis.analysis_date,
                'financial_metrics': {
                    'total_costs': analysis.total_costs,
                    'total_revenue': analysis.total_revenue,
                    'gross_profit': analysis.gross_profit,
                    'net_profit': analysis.net_profit,
                    'profit_margin': analysis.profit_margin,
                    'roi_percent': analysis.roi_percent
                },
                'break_even_analysis': {
                    'break_even_yield': analysis.break_even_yield,
                    'break_even_price': analysis.break_even_price,
                    'break_even_area': analysis.break_even_area,
                    'safety_margin': analysis.safety_margin
                },
                'recommendations': analysis.recommendations,
                'action_items': analysis.action_items
            }
        elif isinstance(analysis, ScenarioAnalysis):
            export_data = {
                'analysis_type': 'scenario_analysis',
                'base_case_profit': analysis.base_case.net_profit,
                'optimistic_case_profit': analysis.optimistic_case.net_profit,
                'pessimistic_case_profit': analysis.pessimistic_case.net_profit,
                'expected_value': analysis.expected_value,
                'value_at_risk': analysis.value_at_risk,
                'sensitivity_analysis': analysis.sensitivity_analysis
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_to_csv(self, analysis, output_path: str):
        """Export analysis to CSV format"""
        if isinstance(analysis, EconomicAnalysis):
            data = {
                'Metric': [
                    'Crop', 'Farm Size (ha)', 'Total Costs', 'Total Revenue',
                    'Gross Profit', 'Net Profit', 'Profit Margin (%)', 'ROI (%)',
                    'Break-even Yield', 'Break-even Price', 'Safety Margin (%)'
                ],
                'Value': [
                    analysis.crop_name, analysis.farm_size_hectares, 
                    analysis.total_costs, analysis.total_revenue,
                    analysis.gross_profit, analysis.net_profit, 
                    analysis.profit_margin, analysis.roi_percent,
                    analysis.break_even_yield, analysis.break_even_price,
                    analysis.safety_margin
                ]
            }
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
    
    def _export_to_xlsx(self, analysis, output_path: str):
        """Export analysis to Excel format"""
        # This would create a comprehensive Excel report with multiple sheets
        # Implementation would use pandas.ExcelWriter
        pass


# Utility functions
def calculate_loan_payment(principal: float, annual_rate: float, years: int) -> float:
    """Calculate monthly loan payment"""
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    if monthly_rate == 0:
        return principal / num_payments
    
    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
              ((1 + monthly_rate)**num_payments - 1)
    
    return payment


def calculate_net_present_value(cash_flows: List[float], discount_rate: float) -> float:
    """Calculate Net Present Value of cash flows"""
    npv = 0
    for i, cash_flow in enumerate(cash_flows):
        npv += cash_flow / ((1 + discount_rate) ** (i + 1))
    return npv


def calculate_internal_rate_of_return(cash_flows: List[float], 
                                    initial_guess: float = 0.1) -> float:
    """Calculate Internal Rate of Return using iterative method"""
    # Simplified IRR calculation - in practice would use numerical methods
    from scipy.optimize import fsolve
    
    def npv_function(rate):
        return sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
    
    try:
        irr = fsolve(npv_function, initial_guess)[0]
        return irr
    except:
        return None


if __name__ == "__main__":
    # Example usage
    calculator = EconomicsCalculator()
    
    # Perform economic analysis
    analysis = calculator.perform_economic_analysis(
        crop_name="wheat",
        farm_size=25.0,
        expected_yield=3.5,
        market_price=250.0
    )
    
    print(f"Economic Analysis Results:")
    print(f"Crop: {analysis.crop_name}")
    print(f"ROI: {analysis.roi_percent:.1f}%")
    print(f"Net Profit: ${analysis.net_profit:,.0f}")
    print(f"Break-even Yield: {analysis.break_even_yield:.1f} t/ha")
    print(f"Safety Margin: {analysis.safety_margin:.1f}%")
    
    # Perform scenario analysis
    scenario_analysis = calculator.perform_scenario_analysis(analysis)
    
    print(f"\nScenario Analysis:")
    print(f"Expected Value: ${scenario_analysis.expected_value:,.0f}")
    print(f"Value at Risk: ${scenario_analysis.value_at_risk:,.0f}")

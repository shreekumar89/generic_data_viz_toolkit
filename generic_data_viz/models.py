"""
models.py

Data classes and enumerations for the Generic Data Visualization Toolkit.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class AnalysisMode(Enum):
    """Analysis modes for the visualizer."""
    QUICK = "quick"           # Fast overview
    STANDARD = "standard"     # Default analysis
    FULL = "full"            # Comprehensive analysis
    DEEP = "deep"            # In-depth with all features


class ChartCategory(Enum):
    """Categories of auto-generated charts."""
    OVERVIEW = "overview"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    QUALITY = "quality"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    HIERARCHY = "hierarchy"


@dataclass
class VisualizationPlan:
    """Plan for generating visualizations based on data characteristics."""
    distributions: List[str]      # Columns needing distribution charts
    correlations: List[Tuple[str, str]]  # Column pairs for correlation
    categoricals: List[str]       # Categorical columns for bar/pie
    temporals: List[str]          # Date columns for time series
    relationships: List[Tuple[str, str, Optional[str]]]  # x, y, hue
    quality_charts: bool          # Whether to generate quality charts
    total_charts: int


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    profile: Any  # DataProfile
    quality: Any  # QualityReport
    insights: Any  # InsightReport
    dashboard: Any  # DashboardRecommendationReport
    viz_plan: VisualizationPlan
    charts_generated: List[str]
    summary: str
    recommendations: List[str]

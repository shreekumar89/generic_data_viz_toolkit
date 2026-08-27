"""
Generic Data Visualization Toolkit

A domain-agnostic, fully automated data visualization and analysis toolkit
that works with ANY dataset.

Features:
- Automatic data type detection
- Comprehensive data quality analysis
- Insight generation
- Multi-library visualization support
- Dashboard recommendations
- Modular architecture
"""

__version__ = "1.0.0"
__author__ = "Data Visualization Team"

# Main classes
from .visualizer import GenericDataVisualizer
from .type_detector import AutoDataTypeDetector
from .chart_generator import GenericVisualizationGenerator
from .models import (
    AnalysisMode,
    ChartCategory,
    VisualizationPlan,
    AnalysisResult,
)

# Core components (for advanced usage)
from .core import (
    TypeDetector,
    DetectionConfig,
    ColumnDetectionResult,
    DataProfiler,
    DataProfile,
    ColumnRole,
    DataType,
    DataQualityAnalyzer,
    QualityReport,
    InsightEngine,
    InsightReport,
    DashboardRecommender,
)

__all__ = [
    # Main classes
    "GenericDataVisualizer",
    "AutoDataTypeDetector",
    "GenericVisualizationGenerator",
    # Models
    "AnalysisMode",
    "ChartCategory",
    "VisualizationPlan",
    "AnalysisResult",
    # Core components
    "TypeDetector",
    "DetectionConfig",
    "ColumnDetectionResult",
    "DataProfiler",
    "DataProfile",
    "ColumnRole",
    "DataType",
    "DataQualityAnalyzer",
    "QualityReport",
    "InsightEngine",
    "InsightReport",
    "DashboardRecommender",
]

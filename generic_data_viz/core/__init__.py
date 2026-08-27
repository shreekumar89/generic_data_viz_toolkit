"""
Core modules for the Generic Data Visualization Toolkit.

This module contains the foundational components:
- TypeDetector: Shared column type/role classification engine (single
  source of truth used by both DataProfiler and AutoDataTypeDetector)
- DataProfiler: Automatic data understanding and profiling
- DataQualityAnalyzer: Comprehensive data quality analysis
- InsightEngine: Business insight generation
- DashboardRecommender: Dashboard and visualization recommendations
"""

from .type_engine import (
    TypeDetector,
    DetectionConfig,
    ColumnDetectionResult,
    ColumnRole,
    SemanticType,
)
from .data_profiler import DataProfiler, DataProfile, DataType, ColumnProfile
from .data_quality import DataQualityAnalyzer, QualityReport, QualityScore, QualityIssue
from .insight_engine import InsightEngine, InsightReport, BusinessInsight
from .dashboard_recommender import DashboardRecommender

__all__ = [
    # Shared type/role classification engine
    "TypeDetector",
    "DetectionConfig",
    "ColumnDetectionResult",
    "ColumnRole",
    "SemanticType",
    # Profiler
    "DataProfiler",
    "DataProfile",
    "DataType",
    "ColumnProfile",
    # Quality
    "DataQualityAnalyzer",
    "QualityReport",
    "QualityScore",
    "QualityIssue",
    # Insights
    "InsightEngine",
    "InsightReport",
    "BusinessInsight",
    # Dashboard
    "DashboardRecommender",
]

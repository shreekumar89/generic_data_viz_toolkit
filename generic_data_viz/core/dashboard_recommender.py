"""
dashboard_recommender.py

Dashboard Recommendation Engine that suggests KPIs, dashboard types,
and appropriate visualizations based on data characteristics.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class DashboardType(Enum):
    """Types of dashboards."""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    DATA_QUALITY = "data_quality"


class VisualizationType(Enum):
    """Types of visualizations."""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    TIME_SERIES = "time_series"
    KPI_CARD = "kpi_card"
    GAUGE = "gauge"
    TABLE = "table"


class ChartLibrary(Enum):
    """Visualization libraries."""
    MATPLOTLIB = "matplotlib"
    SEABORN = "seaborn"
    PLOTLY = "plotly"
    ALTAIR = "altair"


@dataclass
class KPIRecommendation:
    """Recommended KPI definition."""
    name: str
    description: str
    formula: str
    columns_used: List[str]
    aggregation: str
    visualization: VisualizationType
    trend_comparison: bool
    target_value: Optional[float] = None


@dataclass
class VisualizationRecommendation:
    """Recommended visualization."""
    chart_type: VisualizationType
    title: str
    description: str
    columns: List[str]
    library: ChartLibrary
    priority: int = 5
    rationale: str = ""


@dataclass
class DashboardLayout:
    """Recommended dashboard layout."""
    dashboard_type: DashboardType
    title: str
    description: str
    sections: List[Dict[str, Any]]
    kpis: List[KPIRecommendation]
    visualizations: List[VisualizationRecommendation]
    filters: List[str]
    refresh_frequency: str


@dataclass
class DashboardRecommendationReport:
    """Complete dashboard recommendation report."""
    recommended_dashboards: List[DashboardLayout]
    all_kpis: List[KPIRecommendation]
    visualization_strategy: Dict[str, List[VisualizationRecommendation]]
    library_recommendations: Dict[ChartLibrary, List[str]]
    implementation_notes: List[str]


class DashboardRecommender:
    """Dashboard Recommendation Engine."""
    
    def __init__(self):
        pass
    
    def recommend(
        self,
        df: pd.DataFrame,
        profile: Optional[Any] = None,
        business_context: Optional[Dict[str, Any]] = None
    ) -> DashboardRecommendationReport:
        """Generate dashboard recommendations."""
        # 1. Detect KPIs
        kpis = self._detect_kpis(df, profile)
        
        # 2. Recommend visualizations
        visualizations = self._recommend_visualizations(df, profile, kpis)
        
        # 3. Determine dashboard types
        dashboards = self._recommend_dashboards(df, kpis, visualizations)
        
        # 4. Generate library recommendations
        library_recs = self._recommend_libraries(visualizations)
        
        # 5. Generate implementation notes
        notes = self._generate_implementation_notes(df, dashboards)
        
        viz_strategy = self._create_visualization_strategy(visualizations)
        
        return DashboardRecommendationReport(
            recommended_dashboards=dashboards,
            all_kpis=kpis,
            visualization_strategy=viz_strategy,
            library_recommendations=library_recs,
            implementation_notes=notes
        )
    
    def _detect_kpis(self, df: pd.DataFrame, profile: Optional[Any]) -> List[KPIRecommendation]:
        """Automatically detect potential KPIs."""
        kpis = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            if any(kw in col_lower for kw in ['revenue', 'sales', 'amount', 'value']):
                kpis.extend([
                    KPIRecommendation(
                        name=f"Total {col}",
                        description=f"Sum of all {col} values",
                        formula=f"SUM({col})",
                        columns_used=[col],
                        aggregation="sum",
                        visualization=VisualizationType.KPI_CARD,
                        trend_comparison=True
                    ),
                    KPIRecommendation(
                        name=f"Average {col}",
                        description=f"Average {col} value",
                        formula=f"AVG({col})",
                        columns_used=[col],
                        aggregation="avg",
                        visualization=VisualizationType.GAUGE,
                        trend_comparison=True
                    )
                ])
            
            if any(kw in col_lower for kw in ['score', 'rating', 'grade']):
                kpis.append(KPIRecommendation(
                    name=f"Average {col}",
                    description=f"Average {col}",
                    formula=f"AVG({col})",
                    columns_used=[col],
                    aggregation="avg",
                    visualization=VisualizationType.GAUGE,
                    trend_comparison=True,
                    target_value=df[col].quantile(0.9)
                ))
        
        kpis.append(KPIRecommendation(
            name="Total Records",
            description="Total number of records",
            formula="COUNT(*)",
            columns_used=[],
            aggregation="count",
            visualization=VisualizationType.KPI_CARD,
            trend_comparison=False
        ))
        
        return kpis
    
    def _recommend_visualizations(
        self,
        df: pd.DataFrame,
        profile: Optional[Any],
        kpis: List[KPIRecommendation]
    ) -> List[VisualizationRecommendation]:
        """Recommend appropriate visualizations."""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        
        # Distribution charts
        for col in numeric_cols[:5]:
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.HISTOGRAM,
                title=f"Distribution of {col}",
                description=f"Shows the distribution of {col}",
                columns=[col],
                library=ChartLibrary.SEABORN,
                priority=6,
                rationale="Histogram shows distribution shape"
            ))
        
        # Category comparisons
        for cat_col in cat_cols[:3]:
            if df[cat_col].nunique() <= 15:
                for num_col in numeric_cols[:2]:
                    recommendations.append(VisualizationRecommendation(
                        chart_type=VisualizationType.BAR_CHART,
                        title=f"{num_col} by {cat_col}",
                        description=f"Compare {num_col} across {cat_col}",
                        columns=[cat_col, num_col],
                        library=ChartLibrary.PLOTLY,
                        priority=7,
                        rationale="Bar charts compare values across categories"
                    ))
        
        # Correlation heatmap
        if len(numeric_cols) >= 3:
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.HEATMAP,
                title="Correlation Matrix",
                description="Shows correlations between numeric variables",
                columns=numeric_cols[:10],
                library=ChartLibrary.SEABORN,
                priority=8,
                rationale="Heatmaps reveal relationships"
            ))
        
        # Time series
        if date_cols:
            for num_col in numeric_cols[:2]:
                recommendations.append(VisualizationRecommendation(
                    chart_type=VisualizationType.TIME_SERIES,
                    title=f"{num_col} Over Time",
                    description=f"Trend of {num_col}",
                    columns=[date_cols[0], num_col],
                    library=ChartLibrary.PLOTLY,
                    priority=9,
                    rationale="Time series reveals trends"
                ))
        
        # Scatter plots
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:3]):
                for col2 in numeric_cols[i+1:4]:
                    recommendations.append(VisualizationRecommendation(
                        chart_type=VisualizationType.SCATTER_PLOT,
                        title=f"{col1} vs {col2}",
                        description=f"Relationship between {col1} and {col2}",
                        columns=[col1, col2],
                        library=ChartLibrary.PLOTLY,
                        priority=5,
                        rationale="Scatter plots reveal correlations"
                    ))
        
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def _recommend_dashboards(
        self,
        df: pd.DataFrame,
        kpis: List[KPIRecommendation],
        visualizations: List[VisualizationRecommendation]
    ) -> List[DashboardLayout]:
        """Recommend dashboard layouts."""
        dashboards = []
        
        # Executive Dashboard
        exec_kpis = [k for k in kpis if k.visualization in 
                     (VisualizationType.KPI_CARD, VisualizationType.GAUGE)][:6]
        exec_viz = [v for v in visualizations if v.chart_type in 
                   (VisualizationType.BAR_CHART, VisualizationType.TIME_SERIES, 
                    VisualizationType.PIE_CHART)][:4]
        
        dashboards.append(DashboardLayout(
            dashboard_type=DashboardType.EXECUTIVE,
            title="Executive Dashboard",
            description="High-level overview for decision-making",
            sections=[
                {"name": "Key Metrics", "type": "kpi_row", "items": 4},
                {"name": "Trends", "type": "chart_row", "items": 2},
            ],
            kpis=exec_kpis,
            visualizations=exec_viz,
            filters=df.select_dtypes(include=['object']).columns[:3].tolist(),
            refresh_frequency="daily"
        ))
        
        # Analytical Dashboard
        analytical_viz = [v for v in visualizations if v.chart_type in 
                        (VisualizationType.SCATTER_PLOT, VisualizationType.HEATMAP,
                         VisualizationType.HISTOGRAM, VisualizationType.BOX_PLOT)]
        
        dashboards.append(DashboardLayout(
            dashboard_type=DashboardType.ANALYTICAL,
            title="Analytical Dashboard",
            description="Deep-dive analysis for exploration",
            sections=[
                {"name": "Distribution Analysis", "type": "chart_grid", "items": 4},
                {"name": "Correlation Analysis", "type": "chart_row", "items": 2},
            ],
            kpis=kpis[:4],
            visualizations=analytical_viz,
            filters=df.columns[:5].tolist(),
            refresh_frequency="on-demand"
        ))
        
        return dashboards
    
    def _recommend_libraries(
        self,
        visualizations: List[VisualizationRecommendation]
    ) -> Dict[ChartLibrary, List[str]]:
        """Recommend visualization libraries."""
        library_map = {}
        
        for viz in visualizations:
            lib = viz.library
            if lib not in library_map:
                library_map[lib] = []
            library_map[lib].append(viz.title)
        
        return library_map
    
    def _create_visualization_strategy(
        self,
        visualizations: List[VisualizationRecommendation]
    ) -> Dict[str, List[VisualizationRecommendation]]:
        """Group visualizations by purpose."""
        strategy = {
            'distribution': [],
            'comparison': [],
            'relationship': [],
            'trend': [],
        }
        
        for viz in visualizations:
            if viz.chart_type in (VisualizationType.HISTOGRAM, VisualizationType.BOX_PLOT):
                strategy['distribution'].append(viz)
            elif viz.chart_type in (VisualizationType.BAR_CHART, VisualizationType.PIE_CHART):
                strategy['comparison'].append(viz)
            elif viz.chart_type in (VisualizationType.SCATTER_PLOT, VisualizationType.HEATMAP):
                strategy['relationship'].append(viz)
            elif viz.chart_type in (VisualizationType.LINE_CHART, VisualizationType.TIME_SERIES):
                strategy['trend'].append(viz)
        
        return strategy
    
    def _generate_implementation_notes(
        self,
        df: pd.DataFrame,
        dashboards: List[DashboardLayout]
    ) -> List[str]:
        """Generate implementation notes."""
        notes = []
        
        notes.append(f"Dataset contains {len(df):,} records - consider pagination for tables")
        
        high_cardinality = [c for c in df.select_dtypes(include=['object']).columns 
                          if df[c].nunique() > 50]
        if high_cardinality:
            notes.append(f"High cardinality columns may need search filters: {', '.join(high_cardinality[:3])}")
        
        notes.append("Use caching for aggregations on large datasets")
        notes.append("Implement drill-down capabilities for hierarchical data")
        
        return notes

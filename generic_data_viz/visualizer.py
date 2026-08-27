"""
visualizer.py

Main GenericDataVisualizer class - the primary interface for the toolkit.
"""

import re
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from .models import AnalysisMode, VisualizationPlan, AnalysisResult
from .type_detector import AutoDataTypeDetector
from .chart_generator import GenericVisualizationGenerator
from .core.data_profiler import DataProfiler, DataProfile
from .core.data_quality import DataQualityAnalyzer, QualityReport
from .core.insight_engine import InsightEngine, InsightReport
from .core.dashboard_recommender import DashboardRecommender


class GenericDataVisualizer:
    """
    Generic Data Visualization and Analysis Tool.
    
    Domain-agnostic, fully automated data visualization that works with ANY dataset.
    
    Features:
    - Automatic data type detection
    - Comprehensive data quality analysis
    - Insight generation
    - Multi-library visualization support
    - Dashboard recommendations
    - Modular architecture
    
    Usage:
        visualizer = GenericDataVisualizer(output_dir="./output")
        result = visualizer.analyze("data.csv")
        visualizer.generate_report(result)
    """
    
    def __init__(
        self,
        output_dir: str = "./output",
        engine: str = "plotly",
        interactive: bool = True,
        static: bool = False
    ):
        """
        Initialize the visualizer.

        Args:
            output_dir: Output directory for charts and reports
            engine: Visualization engine (plotly, seaborn, matplotlib)
            interactive: Generate interactive HTML charts (default: True, requires Plotly)
            static: Force static PNG charts instead, regardless of engine/interactive -
                useful for headless environments or when a report will be converted to
                PDF (PDF renderers can't execute the JS an interactive chart needs)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        self.engine = engine
        self.interactive = interactive
        self.static = static

        # Initialize components
        self.profiler = DataProfiler()
        self.quality_analyzer = DataQualityAnalyzer()
        self.insight_engine = InsightEngine()
        self.dashboard_recommender = DashboardRecommender()
        self.viz_generator = GenericVisualizationGenerator(
            output_dir=self.charts_dir,
            engine=engine,
            interactive=interactive,
            static=static
        )
        
        self._df: Optional[pd.DataFrame] = None
        self._column_types: Dict = {}
        self._profile: Optional[DataProfile] = None
        self._quality: Optional[QualityReport] = None
        
    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Load data from file or DataFrame.
        
        Supports: CSV, Excel (.xlsx, .xls), Parquet, JSON
        
        Args:
            source: File path or DataFrame
            
        Returns:
            Loaded DataFrame
        """
        if isinstance(source, pd.DataFrame):
            self._df = source.copy()
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            ext = path.suffix.lower()
            if ext == '.csv':
                self._df = pd.read_csv(path)
            elif ext in ['.xlsx', '.xls']:
                self._df = pd.read_excel(path)
            elif ext == '.parquet':
                self._df = pd.read_parquet(path)
            elif ext == '.json':
                self._df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        
        print(f"✓ Loaded {len(self._df):,} rows × {len(self._df.columns)} columns")
        return self._df
    
    def analyze(
        self,
        source: Optional[Union[str, Path, pd.DataFrame]] = None,
        target_column: Optional[str] = None,
        mode: AnalysisMode = AnalysisMode.STANDARD,
        max_charts: int = 15
    ) -> AnalysisResult:
        """
        Perform complete analysis on the dataset.
        
        Args:
            source: Data source (file path or DataFrame)
            target_column: Optional target column for focused analysis
            mode: Analysis mode (quick, standard, full, deep)
            max_charts: Maximum number of charts to generate
            
        Returns:
            AnalysisResult with all analysis outputs
        """
        # Load data if provided
        if source is not None:
            self.load(source)
        
        if self._df is None:
            raise ValueError("No data loaded. Provide a source or call load() first.")
        
        df = self._df
        
        print("\n" + "=" * 60)
        print("GENERIC DATA VISUALIZATION ACCELERATOR")
        print("=" * 60)
        
        # Step 1: Automatic data type detection
        print("\n[1/7] Detecting data types...")
        self._column_types = AutoDataTypeDetector.analyze_dataset(df)
        
        # Get summary by role
        type_summary = AutoDataTypeDetector.get_summary(self._column_types)
        measures = type_summary['measures']
        dimensions = type_summary['dimensions']
        dates = type_summary['dates']
        identifiers = type_summary['identifiers']
        
        print(f"\n  📊 MEASURES ({len(measures)}):")
        for col in measures:
            info = self._column_types[col]
            print(f"     • {col} ({info['unique_count']:,} unique, {info.get('confidence', 0):.0%} conf)")
        
        print(f"\n  🏷️  DIMENSIONS ({len(dimensions)}):")
        for col in dimensions:
            info = self._column_types[col]
            print(f"     • {col} ({info['unique_count']:,} unique, {info.get('confidence', 0):.0%} conf)")
        
        if dates:
            print(f"\n  📅 DATES ({len(dates)}):")
            for col in dates:
                print(f"     • {col}")
        
        if identifiers:
            print(f"\n  🔑 IDENTIFIERS ({len(identifiers)}):")
            for col in identifiers:
                info = self._column_types[col]
                print(f"     • {col} ({info['semantic_type']})")
        
        # Auto-detect target if not specified
        if target_column is None and measures:
            target_column = measures[0]
            print(f"\n  ✓ Auto-detected target: {target_column}")
        
        # Step 2: Data profiling
        print("\n[2/7] Profiling data...")
        self._profile = self.profiler.profile(df)
        
        # Step 3: Data quality analysis
        print("\n[3/7] Analyzing data quality...")
        self._quality = self.quality_analyzer.analyze(df)
        print(f"  - Quality score: {self._quality.score.overall_score:.1f}/100")
        print(f"  - Issues found: {len(self._quality.issues)}")
        
        # Step 4: Generate insights
        print("\n[4/7] Generating insights...")
        insights = self.insight_engine.generate_insights(df, target_column=target_column)
        print(f"  - Insights discovered: {len(insights.insights)}")
        print(f"  - Action items: {len(insights.action_items)}")

        # Step 5: Recommend dashboards and KPIs
        print("\n[5/7] Recommending dashboards...")
        dashboard = self.dashboard_recommender.recommend(df, profile=self._profile)
        print(f"  - Dashboards recommended: {len(dashboard.recommended_dashboards)}")
        print(f"  - KPIs detected: {len(dashboard.all_kpis)}")

        # Step 6: Create visualization plan
        print("\n[6/7] Planning visualizations...")
        viz_plan = self.viz_generator.create_visualization_plan(
            df, self._column_types, max_charts=max_charts
        )
        print(f"  - Charts planned: {viz_plan.total_charts}")

        # Step 7: Generate charts
        print("\n[7/7] Generating charts...")
        charts = self.viz_generator.generate_all_charts(df, viz_plan, self._profile, self._column_types)

        # Generate summary
        summary = self._generate_summary(df, target_column)
        recommendations = self._generate_recommendations()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        return AnalysisResult(
            profile=self._profile,
            quality=self._quality,
            insights=insights,
            dashboard=dashboard,
            viz_plan=viz_plan,
            charts_generated=charts,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_summary(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """Generate executive summary."""
        type_summary = AutoDataTypeDetector.get_summary(self._column_types)
        measures = type_summary['measures']
        dimensions = type_summary['dimensions']
        dates = type_summary['dates']
        identifiers = type_summary['identifiers']
        
        measures_list = ', '.join(measures[:5]) + ('...' if len(measures) > 5 else '')
        dimensions_list = ', '.join(dimensions[:8]) + ('...' if len(dimensions) > 8 else '')
        
        summary = f"""
DATA ANALYSIS SUMMARY
{'=' * 50}

DATASET OVERVIEW
• Rows: {len(df):,}
• Columns: {len(df.columns)}
• Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

COLUMN CLASSIFICATION
• Measures ({len(measures)}): {measures_list or 'None detected'}
• Dimensions ({len(dimensions)}): {dimensions_list or 'None detected'}
• Dates ({len(dates)}): {', '.join(dates) or 'None detected'}
• Identifiers ({len(identifiers)}): {', '.join(identifiers[:3]) or 'None detected'}
• Target column: {target_column or 'Not specified'}

DATA QUALITY
• Overall score: {self._quality.score.overall_score:.1f}/100
• Completeness: {self._quality.score.completeness_score:.1f}/100
• Consistency: {self._quality.score.consistency_score:.1f}/100
• Validity: {self._quality.score.validity_score:.1f}/100

ISSUES DETECTED
• Total issues: {len(self._quality.issues)}
• Missing values: {df.isnull().sum().sum():,}
• Duplicate rows: {df.duplicated().sum():,}
"""
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data-driven recommendations."""
        recommendations = []
        
        # Quality-based recommendations
        if self._quality.score.completeness_score < 80:
            recommendations.append("⚠️ Consider addressing missing values - completeness is below 80%")
        
        if self._quality.score.consistency_score < 80:
            recommendations.append("⚠️ Review data consistency - some values may be inconsistent")
        
        if self._df.duplicated().sum() > 0:
            dup_pct = self._df.duplicated().sum() / len(self._df) * 100
            recommendations.append(f"🔄 {dup_pct:.1f}% duplicate rows detected - consider deduplication")
        
        # Column-type based recommendations
        high_cardinality = [c for c, info in self._column_types.items() 
                           if info['role'] == 'dimension' and info['unique_ratio'] > 0.5]
        if high_cardinality:
            recommendations.append(f"📊 High cardinality dimensions: {', '.join(high_cardinality[:3])}")
        
        return recommendations
    
    def generate_report(
        self,
        result: AnalysisResult,
        formats: List[str] = ['txt', 'html']
    ):
        """
        Generate analysis reports in multiple formats.
        
        Args:
            result: AnalysisResult from analyze()
            formats: List of formats to generate ('txt', 'html', 'json')
        """
        if 'txt' in formats:
            txt_path = self.output_dir / "analysis_report.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result.summary)
                f.write("\n\nRECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                for rec in result.recommendations:
                    f.write(f"• {rec}\n")
                f.write("\n\nDASHBOARD RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                for dash in result.dashboard.recommended_dashboards:
                    f.write(f"\n• {dash.title} ({dash.dashboard_type.value})\n")
                    f.write(f"  {dash.description}\n")
                    if dash.kpis:
                        f.write(f"  KPIs: {', '.join(k.name for k in dash.kpis)}\n")
                    if dash.visualizations:
                        f.write(f"  Visuals: {', '.join(v.title for v in dash.visualizations)}\n")
                    f.write(f"  Refresh: {dash.refresh_frequency}\n")
                f.write("\n\nDETECTED KPIs\n")
                f.write("-" * 40 + "\n")
                for kpi in result.dashboard.all_kpis:
                    f.write(f"• {kpi.name}: {kpi.formula} ({kpi.aggregation})\n")
                f.write("\n\nCHARTS GENERATED\n")
                f.write("-" * 40 + "\n")
                for chart in result.charts_generated:
                    f.write(f"• {chart}\n")
            print(f"\n✓ Text report saved: {txt_path}")
        
        if 'html' in formats:
            html_path = self.output_dir / "analysis_report.html"
            self._generate_html_report(result, html_path)
            print(f"✓ HTML report saved: {html_path}")
    
    # Matches the single <script src="https://cdn.plot.ly/..."> tag Plotly
    # writes into every standalone chart file it generates.
    _PLOTLY_CDN_SCRIPT_RE = re.compile(
        r'<script[^>]*\bsrc="https://cdn\.plot\.ly/[^"]*"[^>]*></script>', re.IGNORECASE
    )
    _BODY_RE = re.compile(r'<body[^>]*>(.*)</body>', re.DOTALL | re.IGNORECASE)

    def _extract_plotly_fragment(self, html_text: str) -> Optional[str]:
        """
        Pull the embeddable <div>+<script> fragment out of a standalone
        Plotly chart file (written with full_html=True), stripping the
        outer <html>/<head>/<body> wrapper and that file's own CDN
        <script> include.

        This exists because embedding each chart via <iframe src="..."> -
        the previous approach - makes every single chart independently
        fetch and initialize its own ~1.4MB copy of Plotly.js in an
        isolated document. That's fine for one or two charts, but with
        15-20+ charts in one report it means 15-20 full library loads on
        page open, which is exactly what made reports with many charts
        hang. Embedding one shared script tag plus N lightweight
        `Plotly.newPlot(...)` calls fixes that.
        """
        match = self._BODY_RE.search(html_text)
        if not match:
            return None
        return self._PLOTLY_CDN_SCRIPT_RE.sub('', match.group(1))

    def _generate_html_report(self, result: AnalysisResult, path: Path):
        """Generate HTML report with embedded charts."""

        # Taller containers for chart categories whose Plotly figures render taller than the default
        interactive_heights = {'overview': 840, 'quality': 600, 'hierarchy': 650, 'composition': 520}

        charts_html = ""
        plotly_cdn_script = ""
        for chart in result.charts_generated:
            chart_path = self.charts_dir / chart
            if chart_path.exists():
                if chart.endswith('.html'):
                    category = chart.split('_', 2)[1] if chart.count('_') >= 2 else ''
                    height = interactive_heights.get(category, 480)
                    raw_html = chart_path.read_text(encoding='utf-8')

                    if not plotly_cdn_script:
                        cdn_match = self._PLOTLY_CDN_SCRIPT_RE.search(raw_html)
                        if cdn_match:
                            plotly_cdn_script = cdn_match.group(0)

                    fragment = self._extract_plotly_fragment(raw_html)
                    if fragment is not None:
                        charts_html += f"""
                        <div class="chart-container chart-container-interactive" style="height: {height}px;">
                            {fragment}
                            <p class="chart-caption">{chart}</p>
                        </div>
                        """
                    else:
                        # Defensive fallback if a chart file doesn't match the
                        # expected Plotly output shape - an iframe still works,
                        # it's just the slower path for that one chart only.
                        charts_html += f"""
                        <div class="chart-container chart-container-interactive">
                            <iframe src="charts/{chart}" loading="lazy" style="height: {height}px;"></iframe>
                            <p class="chart-caption">{chart}</p>
                        </div>
                        """
                else:
                    charts_html += f"""
                    <div class="chart-container">
                        <img src="charts/{chart}" alt="{chart}">
                        <p class="chart-caption">{chart}</p>
                    </div>
                    """

        recs_html = "\n".join([f"<li>{rec}</li>" for rec in result.recommendations])

        dashboards_html = ""
        for dash in result.dashboard.recommended_dashboards:
            kpi_items = "".join(
                f"<li>{k.name} — <code>{k.formula}</code></li>" for k in dash.kpis
            ) or "<li>None</li>"
            viz_items = "".join(
                f"<li>{v.title}</li>" for v in dash.visualizations
            ) or "<li>None</li>"
            dashboards_html += f"""
            <div class="dashboard-card">
                <h3>{dash.title} <span class="badge">{dash.dashboard_type.value}</span></h3>
                <p>{dash.description}</p>
                <div class="dashboard-columns">
                    <div><strong>KPIs</strong><ul>{kpi_items}</ul></div>
                    <div><strong>Suggested Visuals</strong><ul>{viz_items}</ul></div>
                </div>
                <p class="chart-caption">Refresh frequency: {dash.refresh_frequency}</p>
            </div>
            """

        kpi_rows = "".join(
            f"<tr><td>{k.name}</td><td><code>{k.formula}</code></td>"
            f"<td>{k.aggregation}</td><td>{k.visualization.value}</td></tr>"
            for k in result.dashboard.all_kpis
        )
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #378ADD, #639922); color: white; 
                   padding: 30px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; 
                    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; background: #f8f9fa; padding: 15px 25px; 
                   border-radius: 8px; margin: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #378ADD; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .chart-container {{ display: inline-block; margin: 10px; text-align: center; vertical-align: top; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-container-interactive {{ display: block; width: 100%; margin: 10px 0; overflow: hidden;
                                         border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                         background: white; }}
        .chart-container-interactive iframe {{ width: 100%; height: 480px; border: none;
                                                border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-caption {{ font-size: 12px; color: #666; margin-top: 5px; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
        h2 {{ color: #333; border-bottom: 2px solid #378ADD; padding-bottom: 10px; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        .dashboard-card {{ border: 1px solid #eee; border-radius: 8px; padding: 15px 20px; margin-bottom: 15px; }}
        .dashboard-card h3 {{ margin-top: 0; }}
        .badge {{ background: #378ADD; color: white; font-size: 11px; padding: 2px 10px;
                  border-radius: 10px; vertical-align: middle; }}
        .dashboard-columns {{ display: flex; gap: 30px; flex-wrap: wrap; }}
        .dashboard-columns > div {{ flex: 1; min-width: 200px; }}
        table.kpi-table {{ width: 100%; border-collapse: collapse; }}
        table.kpi-table th, table.kpi-table td {{ text-align: left; padding: 8px 10px;
                                                    border-bottom: 1px solid #eee; }}
        table.kpi-table th {{ color: #666; font-size: 12px; text-transform: uppercase; }}
    </style>
    {plotly_cdn_script}
</head>
<body>
    <div class="header">
        <h1>📊 Data Analysis Report</h1>
        <p>Generated by Generic Data Visualization Toolkit</p>
    </div>

    <div class="section">
        <h2>Dataset Overview</h2>
        <div class="metric">
            <div class="metric-value">{len(self._df):,}</div>
            <div class="metric-label">Rows</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(self._df.columns)}</div>
            <div class="metric-label">Columns</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.quality.score.overall_score:.0f}</div>
            <div class="metric-label">Quality Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(result.insights.insights)}</div>
            <div class="metric-label">Insights</div>
        </div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>{recs_html}</ul>
    </div>

    <div class="section">
        <h2>Dashboard Recommendations</h2>
        {dashboards_html}
    </div>

    <div class="section">
        <h2>Detected KPIs</h2>
        <table class="kpi-table">
            <thead><tr><th>KPI</th><th>Formula</th><th>Aggregation</th><th>Suggested Viz</th></tr></thead>
            <tbody>{kpi_rows}</tbody>
        </table>
    </div>

    <div class="section">
        <h2>Generated Visualizations</h2>
        {charts_html}
    </div>

    <div class="section">
        <h2>Summary</h2>
        <pre>{result.summary}</pre>
    </div>
</body>
</html>
        """
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)

"""
chart_generator.py

Generic visualization generator that creates appropriate charts
based on data characteristics - completely domain-agnostic.

Chart selection and styling follow an "executive report" convention:
horizontal bars for category comparisons (sorted descending, long tails
collapsed into "Other"), donuts for low-cardinality composition, line/area
charts with an automatic average/threshold reference line for trends, a
clean professional palette with functional (red/green) coloring for
alerts vs. on-target data, and an auto-generated micro-insight subtitle
above every interactive chart calling out the key takeaway.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .models import VisualizationPlan


class GenericVisualizationGenerator:
    """
    Automatic visualization generator that creates appropriate charts
    based on data characteristics - completely domain-agnostic.
    """

    # Professional, functional color palette. 'good'/'alert' are reserved
    # for genuinely meaningful signals (on-target vs. below-target trend,
    # low vs. high missing-data severity) - never used as decorative
    # "just another category" colors, so red/green keep their meaning
    # wherever they appear.
    COLORS = {
        'primary': '#2E5EAA',      # standard data / leading category
        'secondary': '#5B7F95',    # standard data / secondary series
        'neutral': '#94A3B8',      # de-emphasized: "Other" bucket, reference lines
        'good': '#2E9E6D',         # on-target / positive trend
        'alert': '#E5484D',        # anomaly / below-target / high-severity
        'warning': '#F5A623',      # caution tier (between good and alert)
        'categorical': ['#2E5EAA', '#5B7F95', '#2E9E6D', '#F5A623', '#8B5CF6', '#0EA5A4'],
        'sequential': 'Blues',
        'diverging': 'RdYlGn_r',
        'gridline': '#E5E7EB',     # light gridlines - low visual noise
        'text_muted': '#6B7280',   # subtitle / annotation text
    }

    # Plotly config applied to every interactive chart: enables scroll-wheel
    # zoom (off by default in Plotly), drops the Plotly logo from the
    # toolbar, and marks the figure responsive so it redraws to fit its
    # container (the report's <iframe>) on browser resize.
    INTERACTIVE_CONFIG = {
        'responsive': True,
        'displaylogo': False,
        'scrollZoom': True,
        'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare'],
    }

    # Used everywhere a chart counts rows/occurrences, instead of a bare
    # "Count" label that doesn't say what's being counted.
    RECORD_COUNT_LABEL = 'Number of Records'

    # Plotly's histogram/box traces embed and bin their *raw* input values
    # client-side in JS rather than pre-aggregating server-side, so an
    # unsampled large column embeds one JSON number per row directly into
    # the chart's HTML file - multiple MB for a column with 100k+ rows,
    # and the actual cause of "many charts -> report hangs on open" for
    # large datasets (as opposed to merely *many* charts, which was a
    # separate, already-fixed issue). A random sample this size reproduces
    # the same distribution shape for visual purposes; insight text and
    # axis-scale decisions below are computed from the full column, not
    # the sample, so reported min/max/median stay exact.
    DISTRIBUTION_SAMPLE_CAP = 20000

    # Axis tick decoration per formatting "kind". Deliberately uses
    # tickprefix/ticksuffix rather than d3's '%' tickformat (which
    # multiplies by 100) since we can't reliably know whether a
    # "percentage-shaped" column is already stored on a 0-100 scale.
    _AXIS_DECORATION = {
        'currency': {'tickprefix': '$', 'tickformat': ',.0f'},
        'percentage': {'ticksuffix': '%', 'tickformat': ',.1f'},
        'count': {'tickformat': ',.0f'},
        'number': {'tickformat': ',.0f'},
    }

    def __init__(
        self,
        output_dir: Path,
        engine: str = 'plotly',
        interactive: bool = True,
        static: bool = False,
        dpi: int = 150
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.interactive = interactive
        self.static = static
        self.dpi = dpi
        self.chart_count = 0
        self.generated_charts = []

        # `static` is an explicit, unconditional override for headless/PDF
        # export workflows: no interactive HTML/JS, only PNG files that any
        # HTML-to-PDF tool (or plain image viewer) can consume. It always
        # wins, regardless of what engine/interactive request.
        requested_plotly = (not static) and (interactive or engine == 'plotly')
        if requested_plotly and not PLOTLY_AVAILABLE:
            print("  ⚠ Plotly not installed - falling back to static charts. Install with: pip install plotly")
        self.use_plotly = requested_plotly and PLOTLY_AVAILABLE

        # Apply styling
        sns.set_theme(style="whitegrid", font_scale=0.9)
        plt.rcParams["figure.dpi"] = dpi
        plt.rcParams["savefig.bbox"] = "tight"

    # ------------------------------------------------------------------
    # Shared, engine-agnostic data-prep / formatting / insight utilities
    # ------------------------------------------------------------------

    def _format_kind(self, column: str, column_types: Optional[Dict[str, Dict]] = None) -> str:
        """
        Classify a measure column as 'currency' | 'percentage' | 'count' |
        'number' for formatting purposes. Prefers the semantic type already
        computed by the type-detection engine (authoritative) and falls
        back to column-name keywords so this still works standalone.
        """
        if column_types and column in column_types:
            semantic = column_types[column].get('semantic_type', '')
            if semantic == 'currency':
                return 'currency'

        name_l = column.lower()
        if any(k in name_l for k in ('percent', 'pct', 'rate', 'ratio', 'margin', 'share')):
            return 'percentage'
        if any(k in name_l for k in ('price', 'cost', 'revenue', 'amount', 'sales',
                                      'income', 'budget', 'spend', 'fee', 'profit')):
            return 'currency'
        if any(k in name_l for k in ('count', 'quantity', 'qty', 'units', 'items', 'orders')):
            return 'count'
        return 'number'

    def _format_value(self, value: Any, kind: str) -> str:
        """Human-readable, kind-aware formatting for insight text and static-chart labels."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 'N/A'
        if kind == 'currency':
            return f'${value:,.0f}'
        if kind == 'percentage':
            return f'{value:,.1f}%'
        if kind == 'count':
            return f'{value:,.0f}'
        return f'{value:,.0f}' if abs(value) >= 1000 else f'{value:,.2f}'

    def _axis_decoration(self, kind: str) -> Dict[str, str]:
        """Plotly update_xaxes/update_yaxes kwargs for a formatting kind."""
        return dict(self._AXIS_DECORATION.get(kind, self._AXIS_DECORATION['number']))

    def _hover_num_token(self, field: str, kind: str) -> str:
        """A `%{field:...}` hovertemplate token, decorated for the given kind."""
        if kind == 'currency':
            return f'$%{{{field}:,.0f}}'
        if kind == 'percentage':
            return f'%{{{field}:,.1f}}%'
        return f'%{{{field}:,.0f}}'

    def _summarize_categorical(self, series: pd.Series, top_n: int = 10) -> pd.Series:
        """
        Value counts sorted descending, with everything outside the Top N
        collapsed into a single 'Other' bucket - the standard executive-
        report convention for keeping category comparisons readable.
        """
        counts = series.value_counts()  # already sorted descending
        if len(counts) <= top_n:
            return counts
        top = counts.iloc[:top_n].copy()
        other_total = counts.iloc[top_n:].sum()
        if other_total > 0:
            top.loc['Other'] = other_total
        return top

    def _titled_text(self, title: str, subtitle: str = '') -> str:
        """Plotly title string with a smaller, muted micro-insight subtitle
        line beneath the main title."""
        if not subtitle:
            return f"<b>{title}</b>"
        return (f"<b>{title}</b><br>"
                f"<span style='font-size:13px;color:{self.COLORS['text_muted']}'>{subtitle}</span>")

    def _categorical_insight(self, counts: pd.Series, column: str) -> str:
        total = counts.sum()
        if total == 0 or len(counts) == 0:
            return ''
        top_label, top_val = counts.index[0], counts.iloc[0]
        share = top_val / total * 100
        return f"<b>{top_label}</b> leads with {share:.0f}% share ({top_val:,.0f} of {total:,.0f})"

    def _distribution_insight(self, data: pd.Series, kind: str) -> str:
        if len(data) == 0:
            return ''
        median_v = data.median()
        return (f"Median {self._format_value(median_v, kind)} · "
                f"Range {self._format_value(data.min(), kind)}–{self._format_value(data.max(), kind)}")

    def _timeseries_insight(self, ts: pd.Series, kind: str) -> Tuple[str, float, str]:
        """
        Returns (subtitle_html, average_value, marker_color).

        The subtitle's up/down arrow reflects the overall period change
        (first value vs. last value) - a separate question from whether
        the chart is currently "on target". `marker_color` instead
        compares the latest point to the average/threshold line the chart
        actually draws, so the color a viewer sees on the latest-point
        marker always matches the reference line it sits next to (green
        at/above average = on target, red/coral below = below target).
        """
        if len(ts) == 0:
            return '', 0.0, self.COLORS['neutral']
        avg_val = float(ts.mean())
        if len(ts) < 2:
            return '', avg_val, self.COLORS['neutral']

        peak_idx = ts.idxmax()
        peak_val = ts.max()
        first_val, last_val = ts.iloc[0], ts.iloc[-1]
        pct_change = ((last_val - first_val) / abs(first_val) * 100) if first_val else 0.0
        change_color = self.COLORS['good'] if pct_change >= 0 else self.COLORS['alert']
        arrow = '▲' if pct_change >= 0 else '▼'
        peak_str = peak_idx.strftime('%b %Y') if hasattr(peak_idx, 'strftime') else str(peak_idx)
        marker_color = self.COLORS['good'] if last_val >= avg_val else self.COLORS['alert']

        subtitle = (f"Peak {self._format_value(peak_val, kind)} in {peak_str} · "
                    f"<span style='color:{change_color}'>{arrow} {abs(pct_change):.0f}%</span> over period")
        return subtitle, avg_val, marker_color

    def _correlation_insight(self, corr: pd.DataFrame) -> str:
        cols = corr.columns.tolist()
        pairs = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                val = corr.loc[c1, c2]
                if pd.notna(val):
                    pairs.append((abs(val), val, c1, c2))
        if not pairs:
            return ''
        pairs.sort(key=lambda t: t[0], reverse=True)
        _, val, c1, c2 = pairs[0]
        direction = 'positive' if val >= 0 else 'negative'
        return f"Strongest relationship: <b>{c1}</b> & <b>{c2}</b> ({direction}, r={val:.2f})"

    def _apply_business_theme(self, fig):
        """
        Clean, professional styling shared by every interactive chart:
        lightened gridlines, a legible sans-serif font, horizontal axis
        labels, and a tidy top-right legend - applied once here rather
        than repeated per chart builder.
        """
        fig.update_layout(
            font=dict(family='-apple-system, "Segoe UI", Roboto, sans-serif', size=13, color='#374151'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        bgcolor='rgba(0,0,0,0)'),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        fig.update_xaxes(gridcolor=self.COLORS['gridline'], zerolinecolor=self.COLORS['gridline'],
                          showline=True, linecolor='#D1D5DB', tickangle=0)
        fig.update_yaxes(gridcolor=self.COLORS['gridline'], zerolinecolor=self.COLORS['gridline'])

    def create_visualization_plan(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, Dict],
        max_charts: int = 15
    ) -> VisualizationPlan:
        """Create a plan for what visualizations to generate."""
        distributions = []
        correlations = []
        categoricals = []
        temporals = []
        relationships = []

        measures = [c for c, info in column_types.items() if info['role'] == 'measure']
        dimensions = [c for c, info in column_types.items() if info['role'] == 'dimension']
        dates = [c for c, info in column_types.items() if info['role'] == 'date']

        # Sort dimensions by cardinality (lowest first)
        dimensions_sorted = sorted(
            dimensions,
            key=lambda c: column_types[c].get('unique_count', 0)
        )

        # Sort measures by confidence (highest first)
        measures_sorted = sorted(
            measures,
            key=lambda c: column_types[c].get('confidence', 0),
            reverse=True
        )

        distributions = measures_sorted[:5]
        categoricals = dimensions_sorted[:8]
        temporals = dates[:2]

        if len(measures_sorted) >= 2:
            import itertools
            pairs = list(itertools.combinations(measures_sorted[:6], 2))[:5]
            correlations = pairs

        if len(measures_sorted) >= 2 and dimensions_sorted:
            best_hue_dims = [d for d in dimensions_sorted if column_types[d].get('unique_count', 0) <= 15][:3]
            for i, (m1, m2) in enumerate(correlations[:3]):
                hue = best_hue_dims[i] if i < len(best_hue_dims) else (best_hue_dims[0] if best_hue_dims else None)
                relationships.append((m1, m2, hue))

        total = len(distributions) + len(categoricals) + len(temporals) + len(relationships) + 3

        return VisualizationPlan(
            distributions=distributions,
            correlations=correlations,
            categoricals=categoricals,
            temporals=temporals,
            relationships=relationships,
            quality_charts=True,
            total_charts=min(total, max_charts)
        )

    def generate_all_charts(
        self,
        df: pd.DataFrame,
        plan: VisualizationPlan,
        profile: Any,
        column_types: Optional[Dict[str, Dict]] = None
    ) -> List[str]:
        """Generate all charts according to the plan."""
        self.chart_count = 0
        self.generated_charts = []

        # 1. Overview chart
        if self.use_plotly:
            self._create_data_overview_plotly(df, profile)
        else:
            self._create_data_overview(df, profile)

        # 2. Distribution charts
        for col in plan.distributions:
            if self.use_plotly:
                self._create_distribution_chart_plotly(df, col, column_types)
            else:
                self._create_distribution_chart(df, col)

        # 3. Categorical charts (bar for >5 categories, donut for composition otherwise)
        for col in plan.categoricals:
            if self.use_plotly:
                self._create_categorical_chart_plotly(df, col)
            else:
                self._create_categorical_chart(df, col)

        # 4. Correlation heatmap
        if plan.correlations:
            cols = list(set([c for pair in plan.correlations for c in pair]))
            if self.use_plotly:
                self._create_correlation_heatmap_plotly(df, cols)
            else:
                self._create_correlation_heatmap(df, cols)

        # 5. Relationship charts
        for x, y, hue in plan.relationships:
            if self.use_plotly:
                self._create_scatter_chart_plotly(df, x, y, hue)
            else:
                self._create_scatter_chart(df, x, y, hue)

        # 6. Quality charts
        if plan.quality_charts:
            if self.use_plotly:
                self._create_missing_values_chart_plotly(df)
            else:
                self._create_missing_values_chart(df)

        # 7. Time series (if applicable)
        for date_col in plan.temporals:
            measures = [c for c in plan.distributions if c in df.columns]
            if measures:
                if self.use_plotly:
                    self._create_time_series_chart_plotly(df, date_col, measures[0], column_types)
                else:
                    self._create_time_series_chart(df, date_col, measures[0])

        # 8. Hierarchy drill-down (interactive only - see docstring on
        # _create_hierarchy_drilldown_chart for why static mode skips this)
        hierarchies = getattr(profile, 'hierarchies', None)
        if self.use_plotly and hierarchies:
            self._create_hierarchy_drilldown_chart(df, hierarchies[0])

        return self.generated_charts

    def _save_chart(self, fig, name: str, category: str):
        """Save static matplotlib chart to file."""
        self.chart_count += 1
        filename = f"{self.chart_count:02d}_{category}_{name}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_charts.append(filename)
        print(f"  ✓ Generated: {filename}")

    def _save_chart_html(self, fig, name: str, category: str):
        """Save interactive Plotly chart as a standalone, responsive HTML file."""
        self.chart_count += 1
        filename = f"{self.chart_count:02d}_{category}_{name}.html"
        filepath = self.output_dir / filename
        self._apply_business_theme(fig)
        fig.update_layout(
            template='plotly_white',
            margin=dict(t=70, l=40, r=20, b=40),
            autosize=True,
        )
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            full_html=True,
            config=self.INTERACTIVE_CONFIG,
        )
        self.generated_charts.append(filename)
        print(f"  ✓ Generated: {filename}")

    def _create_data_overview(self, df: pd.DataFrame, profile: Any):
        """Create overview summary chart."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # 1. Data types breakdown
        type_counts = df.dtypes.astype(str).value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                       colors=self.COLORS['categorical'][:len(type_counts)])
        axes[0, 0].set_title('Column Data Types')

        # 2. Missing values by column
        missing = df.isnull().sum().sort_values(ascending=False).head(10)
        if missing.sum() > 0:
            colors = [self.COLORS['alert'] if v > 0 else self.COLORS['primary'] for v in missing]
            axes[0, 1].barh(missing.index, missing.values, color=colors)
            axes[0, 1].set_title('Top 10 Columns by Missing Values')
            axes[0, 1].set_xlabel('Missing Values (Count of Rows)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            axes[0, 1].set_title('Missing Values')
            axes[0, 1].axis('off')

        # 3. Memory usage
        mem = df.memory_usage(deep=True).drop('Index') / 1024 / 1024
        top_mem = mem.sort_values(ascending=False).head(10)
        axes[1, 0].barh(top_mem.index, top_mem.values, color=self.COLORS['primary'])
        axes[1, 0].set_title('Memory Usage by Column (MB)')
        axes[1, 0].set_xlabel('Memory Usage (MB)')

        # 4. Dataset summary text
        n_measures = len(profile.measures) if hasattr(profile, 'measures') else 'N/A'
        n_dimensions = len(profile.dimensions) if hasattr(profile, 'dimensions') else 'N/A'
        n_dates = len(profile.date_columns) if hasattr(profile, 'date_columns') else 'N/A'

        summary_text = f"""
Dataset Overview
================
Rows: {len(df):,}
Columns: {len(df.columns)}
Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

Measures: {n_measures}
Dimensions: {n_dimensions}
Date Columns: {n_dates}

Missing Values: {df.isnull().sum().sum():,}
Duplicate Rows: {df.duplicated().sum():,}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

        fig.suptitle('Data Overview Dashboard', fontsize=14, fontweight='bold')
        fig.tight_layout()
        self._save_chart(fig, 'overview', 'overview')

    def _create_distribution_chart(self, df: pd.DataFrame, column: str):
        """Create distribution chart for a numeric column."""
        if column not in df.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        data = df[column].dropna()

        if len(data) == 0:
            plt.close(fig)
            return

        # Sample large columns before plotting: KDE estimation is O(n) or
        # worse and gets noticeably slow past tens of thousands of points,
        # with no visible benefit to the rendered PNG. Insight text and
        # scale decisions elsewhere still use the full column.
        plot_data = data if len(data) <= self.DISTRIBUTION_SAMPLE_CAP else data.sample(
            self.DISTRIBUTION_SAMPLE_CAP, random_state=42)

        # Histogram with KDE
        try:
            use_log = data.min() > 0 and (data.max() / data.min()) > 100

            sns.histplot(plot_data, bins=30, kde=True, ax=axes[0], color=self.COLORS['primary'])
            axes[0].set_title(f'Distribution of {column}')
            axes[0].set_xlabel(f'{column} (log scale)' if use_log else column)
            axes[0].set_ylabel(f'{self.RECORD_COUNT_LABEL} ({column})')
            if use_log:
                axes[0].set_xscale('log')
        except Exception:
            pass

        # Box plot
        sns.boxplot(x=plot_data, ax=axes[1], color=self.COLORS['secondary'])
        axes[1].set_title(f'Box Plot: {column}')
        axes[1].set_xlabel(column)

        fig.tight_layout()
        self._save_chart(fig, column.replace(' ', '_')[:20], 'distribution')

    def _create_categorical_chart(self, df: pd.DataFrame, column: str, top_n: int = 10):
        """Create a business-style chart for a categorical column: a donut
        for low-cardinality composition (<=5 categories), otherwise a
        horizontal bar sorted descending with the long tail collapsed
        into 'Other'."""
        if column not in df.columns:
            return

        if df[column].nunique(dropna=True) <= 5:
            self._create_donut_chart(df, column)
            return

        counts = self._summarize_categorical(df[column], top_n=top_n)
        if len(counts) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = []
        for i, label in enumerate(counts.index):
            if label == 'Other':
                colors.append(self.COLORS['neutral'])
            elif i == 0:
                colors.append(self.COLORS['primary'])
            else:
                colors.append(self.COLORS['secondary'])

        ax.barh(counts.index.astype(str), counts.values, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(f'{self.RECORD_COUNT_LABEL} ({column})')
        ax.set_ylabel(column)
        ax.set_title(f'{column}\n{self._categorical_insight(counts, column)}', fontsize=11)

        for i, (idx, val) in enumerate(counts.items()):
            ax.text(val + counts.max() * 0.01, i, f'{val:,}', va='center', fontsize=8)

        fig.tight_layout()
        self._save_chart(fig, column.replace(' ', '_')[:20], 'categorical')

    def _create_donut_chart(self, df: pd.DataFrame, column: str):
        """Static composition donut for a low-cardinality dimension."""
        counts = df[column].value_counts()
        if len(counts) == 0:
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        colors = self.COLORS['categorical'][:len(counts)]
        ax.pie(
            counts.values, labels=counts.index.astype(str), autopct='%1.0f%%',
            colors=colors, wedgeprops=dict(width=0.45, edgecolor='white'),
            pctdistance=0.8
        )
        ax.set_title(f'{column} Composition\n{self._categorical_insight(counts, column)}', fontsize=11)
        fig.tight_layout()
        self._save_chart(fig, column.replace(' ', '_')[:20], 'composition')

    def _create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str]):
        """Create correlation heatmap for numeric columns."""
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            return

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 0.8), max(5, len(numeric_cols) * 0.6)))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                   ax=ax, square=True, mask=mask, linewidths=0.5,
                   cbar_kws={'label': 'Correlation Coefficient (r)'})
        ax.set_title('Correlation Matrix')

        fig.tight_layout()
        self._save_chart(fig, 'correlations', 'correlation')

    def _create_scatter_chart(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None,
        sample_n: int = 5000
    ):
        """Create scatter plot for relationship analysis."""
        if x_col not in df.columns or y_col not in df.columns:
            return

        plot_df = df.dropna(subset=[x_col, y_col])
        if len(plot_df) > sample_n:
            plot_df = plot_df.sample(sample_n, random_state=42)

        if len(plot_df) == 0:
            return

        fig, ax = plt.subplots(figsize=(9, 6))

        try:
            if hue_col and hue_col in df.columns:
                top_cats = df[hue_col].value_counts().head(8).index
                plot_df = plot_df[plot_df[hue_col].isin(top_cats)]
                sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col,
                               alpha=0.5, ax=ax, palette=self.COLORS['categorical'])
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            else:
                sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=0.5, ax=ax,
                               color=self.COLORS['primary'])

            if plot_df[x_col].min() > 0 and (plot_df[x_col].max() / plot_df[x_col].min()) > 100:
                ax.set_xscale('log')
            if plot_df[y_col].min() > 0 and (plot_df[y_col].max() / plot_df[y_col].min()) > 100:
                ax.set_yscale('log')

            ax.set_title(f'{y_col} vs {x_col}')
        except Exception:
            pass

        fig.tight_layout()
        self._save_chart(fig, f'{x_col[:10]}_vs_{y_col[:10]}', 'relationship')

    def _create_missing_values_chart(self, df: pd.DataFrame):
        """Create missing values pattern chart."""
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0].head(20)

        if len(missing_pct) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, max(5, len(missing_pct) * 0.3)))

        colors = []
        for pct in missing_pct.values:
            if pct > 50:
                colors.append(self.COLORS['alert'])
            elif pct > 20:
                colors.append(self.COLORS['warning'])
            else:
                colors.append(self.COLORS['good'])

        ax.barh(missing_pct.index, missing_pct.values, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel('Missing Values (% of Total Rows)')
        ax.set_ylabel('Column')
        ax.set_title('Missing Values by Column')
        ax.axvline(x=50, color=self.COLORS['alert'], linestyle='--', alpha=0.7, label='50% threshold')
        ax.legend()

        for i, (idx, val) in enumerate(missing_pct.items()):
            ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=8)

        fig.tight_layout()
        self._save_chart(fig, 'missing_values', 'quality')

    def _resample_series(self, df_ts: pd.DataFrame, date_col: str, value_col: str, freq: str) -> pd.Series:
        """Resample a time-indexed series, handling pandas' rename of the
        month-end alias from 'M' to 'ME' (pandas >= 2.2) transparently so this
        works across the whole pandas>=1.3 range the toolkit supports."""
        try:
            return df_ts.set_index(date_col).resample(freq)[value_col].mean()
        except ValueError:
            legacy_freq = {'ME': 'M', 'QE': 'Q', 'YE': 'Y'}.get(freq, freq)
            return df_ts.set_index(date_col).resample(legacy_freq)[value_col].mean()

    def _create_time_series_chart(self, df: pd.DataFrame, date_col: str, value_col: str):
        """Create time series chart with an automatic average/threshold reference line."""
        if date_col not in df.columns or value_col not in df.columns:
            return

        try:
            df_ts = df[[date_col, value_col]].dropna()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            df_ts = df_ts.sort_values(date_col)

            date_range = (df_ts[date_col].max() - df_ts[date_col].min()).days

            if date_range > 365:
                freq = 'ME'
                freq_label = 'Monthly'
            elif date_range > 60:
                freq = 'W'
                freq_label = 'Weekly'
            else:
                freq = 'D'
                freq_label = 'Daily'

            ts_agg = self._resample_series(df_ts, date_col, value_col, freq)
            kind = self._format_kind(value_col)
            subtitle, avg_val, marker_color = self._timeseries_insight(ts_agg, kind)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ts_agg.index, ts_agg.values, color=self.COLORS['primary'], linewidth=1.5)
            ax.fill_between(ts_agg.index, ts_agg.values, alpha=0.15, color=self.COLORS['primary'])
            ax.axhline(y=avg_val, color=self.COLORS['neutral'], linestyle='--', linewidth=1,
                       label=f'Average: {self._format_value(avg_val, kind)}')
            if len(ts_agg) > 0:
                ax.scatter([ts_agg.index[-1]], [ts_agg.iloc[-1]], color=marker_color, s=60, zorder=5,
                           edgecolor='white', linewidth=1)
            ax.legend(loc='upper left', fontsize=9)
            title_plain = re.sub(r"<[^>]+>", "", subtitle)
            ax.set_title(f'{value_col} over Time ({freq_label} Average)\n{title_plain}', fontsize=11)
            ax.set_xlabel(date_col)
            ax.set_ylabel(value_col)

            fig.tight_layout()
            self._save_chart(fig, f'timeseries_{value_col[:15]}', 'temporal')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Interactive (Plotly) chart variants
    # ------------------------------------------------------------------

    def _create_data_overview_plotly(self, df: pd.DataFrame, profile: Any):
        """Create interactive overview dashboard using Plotly."""
        type_counts = df.dtypes.astype(str).value_counts()
        missing = df.isnull().sum().sort_values(ascending=False).head(10)
        mem = df.memory_usage(deep=True).drop('Index') / 1024 / 1024
        top_mem = mem.sort_values(ascending=False).head(10)

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'domain'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=('Column Data Types', 'Top 10 Columns by Missing Values',
                             'Memory Usage by Column (MB)', 'Dataset Summary')
        )

        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values,
                   marker=dict(colors=self.COLORS['categorical'])),
            row=1, col=1
        )

        if missing.sum() > 0:
            colors = [self.COLORS['alert'] if v > 0 else self.COLORS['primary'] for v in missing.values]
            fig.add_trace(
                go.Bar(x=missing.values, y=missing.index.astype(str), orientation='h',
                       marker_color=colors, showlegend=False,
                       hovertemplate='<b>%{y}</b><br>Missing: %{x:,.0f}<extra></extra>'),
                row=1, col=2
            )
            fig.update_yaxes(autorange='reversed', row=1, col=2)
            fig.update_xaxes(title_text='Missing Values (Count of Rows)', row=1, col=2)
        else:
            fig.add_annotation(text='No Missing Values', row=1, col=2, showarrow=False, font=dict(size=14))

        fig.add_trace(
            go.Bar(x=top_mem.values, y=top_mem.index.astype(str), orientation='h',
                   marker_color=self.COLORS['primary'], showlegend=False,
                   hovertemplate='<b>%{y}</b><br>%{x:,.2f} MB<extra></extra>'),
            row=2, col=1
        )
        fig.update_yaxes(autorange='reversed', row=2, col=1)
        fig.update_xaxes(title_text='Memory Usage (MB)', row=2, col=1)

        n_measures = len(profile.measures) if hasattr(profile, 'measures') else 'N/A'
        n_dimensions = len(profile.dimensions) if hasattr(profile, 'dimensions') else 'N/A'
        n_dates = len(profile.date_columns) if hasattr(profile, 'date_columns') else 'N/A'
        summary_text = (
            f"Rows: {len(df):,}<br>Columns: {len(df.columns)}<br>"
            f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB<br><br>"
            f"Measures: {n_measures}<br>Dimensions: {n_dimensions}<br>Date Columns: {n_dates}<br><br>"
            f"Missing Values: {df.isnull().sum().sum():,}<br>Duplicate Rows: {df.duplicated().sum():,}"
        )
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', text=[summary_text], textposition='middle center',
                       textfont=dict(size=13, family='monospace'), showlegend=False, hoverinfo='skip'),
            row=2, col=2
        )
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

        fig.update_layout(title_text='Data Overview Dashboard', height=800, showlegend=False)
        self._save_chart_html(fig, 'overview', 'overview')

    def _create_distribution_chart_plotly(self, df: pd.DataFrame, column: str,
                                           column_types: Optional[Dict[str, Dict]] = None):
        """Create interactive distribution chart (histogram + box marginal) using Plotly."""
        if column not in df.columns:
            return

        data = df[column].dropna()
        if len(data) == 0:
            return

        kind = self._format_kind(column, column_types)
        subtitle = self._distribution_insight(data, kind)
        use_log = data.min() > 0 and (data.max() / data.min()) > 100

        plot_data = data if len(data) <= self.DISTRIBUTION_SAMPLE_CAP else data.sample(
            self.DISTRIBUTION_SAMPLE_CAP, random_state=42)

        fig = px.histogram(
            x=plot_data, nbins=30, marginal='box',
            color_discrete_sequence=[self.COLORS['primary']],
        )
        fig.update_traces(
            hovertemplate=(f'{column}: {self._hover_num_token("x", kind)}<br>'
                            f'{self.RECORD_COUNT_LABEL}: %{{y:,.0f}}<extra></extra>'),
            selector=dict(type='histogram')
        )
        fig.update_xaxes(title_text=column, **self._axis_decoration(kind))
        fig.update_yaxes(title_text=f'{self.RECORD_COUNT_LABEL} ({column})')

        if use_log:
            fig.update_xaxes(type='log', title_text=f'{column} (log scale)')

        fig.update_layout(title=dict(text=self._titled_text(f'Distribution of {column}', subtitle)))

        self._save_chart_html(fig, column.replace(' ', '_')[:20], 'distribution')

    def _create_categorical_chart_plotly(self, df: pd.DataFrame, column: str, top_n: int = 10):
        """
        Create a business-style interactive chart for a categorical column:
        a donut for low-cardinality composition (<=5 categories, the
        conventional cutoff where a "share of whole" view stays legible),
        otherwise a horizontal bar sorted descending with the long tail
        collapsed into 'Other'.
        """
        if column not in df.columns:
            return

        if df[column].nunique(dropna=True) <= 5:
            self._create_donut_chart_plotly(df, column)
            return

        counts = self._summarize_categorical(df[column], top_n=top_n)
        if len(counts) == 0:
            return

        colors = []
        for i, label in enumerate(counts.index):
            if label == 'Other':
                colors.append(self.COLORS['neutral'])
            elif i == 0:
                colors.append(self.COLORS['primary'])
            else:
                colors.append(self.COLORS['secondary'])

        subtitle = self._categorical_insight(counts, column)
        labels = [str(i) for i in counts.index]

        count_label = f'{self.RECORD_COUNT_LABEL} ({column})'
        fig = go.Figure(go.Bar(
            x=counts.values, y=labels, orientation='h',
            marker_color=colors, text=[f'{v:,.0f}' for v in counts.values], textposition='outside',
            hovertemplate=f'<b>%{{y}}</b><br>{count_label}: %{{x:,.0f}}<extra></extra>',
        ))
        fig.update_yaxes(autorange='reversed', title_text=column)
        fig.update_xaxes(title_text=count_label, tickformat=',.0f')
        fig.update_layout(title=dict(text=self._titled_text(column, subtitle)))

        self._save_chart_html(fig, column.replace(' ', '_')[:20], 'categorical')

    def _create_donut_chart_plotly(self, df: pd.DataFrame, column: str):
        """
        Composition view for a low-cardinality dimension (<=5 categories):
        a donut chart is the standard executive-report choice for showing
        how a whole breaks down into a handful of parts, which a bar chart
        (built for ranking many items) doesn't communicate as clearly.
        """
        counts = df[column].value_counts()
        if len(counts) == 0:
            return

        subtitle = self._categorical_insight(counts, column)
        colors = self.COLORS['categorical'][:len(counts)]

        fig = go.Figure(go.Pie(
            labels=[str(i) for i in counts.index], values=counts.values, hole=0.55,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent', textposition='outside',
            hovertemplate=(f'<b>%{{label}}</b><br>{self.RECORD_COUNT_LABEL} ({column}): '
                            '%{value:,.0f}<br>Share: %{percent}<extra></extra>'),
        ))
        fig.update_layout(title=dict(text=self._titled_text(f'{column} Composition', subtitle)),
                           showlegend=False)

        self._save_chart_html(fig, column.replace(' ', '_')[:20], 'composition')

    def _create_correlation_heatmap_plotly(self, df: pd.DataFrame, columns: List[str]):
        """Create interactive correlation heatmap using Plotly."""
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            return

        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_display = corr.mask(mask)
        subtitle = self._correlation_insight(corr)

        fig = px.imshow(
            corr_display, text_auto='.2f', color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1, aspect='auto',
            labels=dict(color='Correlation Coefficient (r)'),
        )
        fig.update_xaxes(side='bottom', title_text='')
        fig.update_yaxes(title_text='')
        fig.update_traces(hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation (r): %{z:.2f}<extra></extra>')
        fig.update_layout(title=dict(text=self._titled_text('Correlation Matrix', subtitle)))

        self._save_chart_html(fig, 'correlations', 'correlation')

    def _create_scatter_chart_plotly(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None,
        sample_n: int = 5000
    ):
        """Create interactive scatter plot for relationship analysis using Plotly."""
        if x_col not in df.columns or y_col not in df.columns:
            return

        plot_df = df.dropna(subset=[x_col, y_col])
        if len(plot_df) > sample_n:
            plot_df = plot_df.sample(sample_n, random_state=42)
        if len(plot_df) == 0:
            return

        try:
            r = plot_df[x_col].corr(plot_df[y_col])
            subtitle = f'Correlation r = {r:.2f}' if pd.notna(r) else ''
        except Exception:
            subtitle = ''

        color_col = None
        if hue_col and hue_col in df.columns:
            top_cats = df[hue_col].value_counts().head(8).index
            plot_df = plot_df[plot_df[hue_col].isin(top_cats)]
            color_col = hue_col

        fig = px.scatter(
            plot_df, x=x_col, y=y_col, color=color_col, opacity=0.5,
            color_discrete_sequence=self.COLORS['categorical'],
        )
        if color_col is None:
            fig.update_traces(marker=dict(color=self.COLORS['primary']))

        if plot_df[x_col].min() > 0 and (plot_df[x_col].max() / plot_df[x_col].min()) > 100:
            fig.update_xaxes(type='log')
        if plot_df[y_col].min() > 0 and (plot_df[y_col].max() / plot_df[y_col].min()) > 100:
            fig.update_yaxes(type='log')

        fig.update_layout(title=dict(text=self._titled_text(f'{y_col} vs {x_col}', subtitle)))

        self._save_chart_html(fig, f'{x_col[:10]}_vs_{y_col[:10]}', 'relationship')

    def _create_missing_values_chart_plotly(self, df: pd.DataFrame):
        """Create interactive missing values pattern chart using Plotly."""
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0].head(20)

        if len(missing_pct) == 0:
            return

        colors = []
        for pct in missing_pct.values:
            if pct > 50:
                colors.append(self.COLORS['alert'])
            elif pct > 20:
                colors.append(self.COLORS['warning'])
            else:
                colors.append(self.COLORS['good'])

        worst_col, worst_val = missing_pct.index[0], missing_pct.iloc[0]
        subtitle = f"Highest gap: <b>{worst_col}</b> at {worst_val:.0f}% missing"

        fig = go.Figure(go.Bar(
            x=missing_pct.values, y=missing_pct.index.astype(str), orientation='h',
            marker_color=colors, text=[f'{v:.1f}%' for v in missing_pct.values], textposition='outside',
            hovertemplate='<b>%{y}</b><br>Missing Values: %{x:,.1f}% of rows<extra></extra>',
        ))
        fig.add_vline(x=50, line_dash='dash', line_color=self.COLORS['alert'], annotation_text='50% threshold')
        fig.update_yaxes(autorange='reversed', title_text='Column')
        fig.update_xaxes(title_text='Missing Values (% of Total Rows)', ticksuffix='%')
        fig.update_layout(title=dict(text=self._titled_text('Missing Values by Column', subtitle)),
                           height=max(400, len(missing_pct) * 28))

        self._save_chart_html(fig, 'missing_values', 'quality')

    def _create_time_series_chart_plotly(self, df: pd.DataFrame, date_col: str, value_col: str,
                                          column_types: Optional[Dict[str, Dict]] = None):
        """
        Create an interactive time-series chart with an automatic
        average/threshold reference line: the dashed neutral line marks
        the period average, and the latest data point is marked green if
        it's at or above that average ("on target") or red/coral if below
        ("below target") - functional coloring, not decoration.
        """
        if date_col not in df.columns or value_col not in df.columns:
            return

        try:
            df_ts = df[[date_col, value_col]].dropna()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            df_ts = df_ts.sort_values(date_col)

            date_range = (df_ts[date_col].max() - df_ts[date_col].min()).days

            if date_range > 365:
                freq, freq_label = 'ME', 'Monthly'
            elif date_range > 60:
                freq, freq_label = 'W', 'Weekly'
            else:
                freq, freq_label = 'D', 'Daily'

            ts_agg = self._resample_series(df_ts, date_col, value_col, freq)
            kind = self._format_kind(value_col, column_types)
            subtitle, avg_val, marker_color = self._timeseries_insight(ts_agg, kind)

            fig = px.area(x=ts_agg.index, y=ts_agg.values)
            fig.update_traces(
                line=dict(color=self.COLORS['primary'], width=2),
                fillcolor='rgba(46, 94, 170, 0.15)',
                hovertemplate=f'%{{x|%b %d, %Y}}<br>{value_col}: {self._hover_num_token("y", kind)}<extra></extra>',
            )
            fig.add_hline(
                y=avg_val, line_dash='dash', line_color=self.COLORS['neutral'],
                annotation_text=f'Average: {self._format_value(avg_val, kind)}',
                annotation_position='top left', annotation_font_color=self.COLORS['text_muted'],
            )
            if len(ts_agg) > 0:
                fig.add_trace(go.Scatter(
                    x=[ts_agg.index[-1]], y=[ts_agg.iloc[-1]], mode='markers',
                    marker=dict(color=marker_color, size=11, line=dict(color='white', width=1.5)),
                    showlegend=False, hoverinfo='skip',
                ))

            fig.update_xaxes(title_text=date_col)
            fig.update_yaxes(title_text=value_col, **self._axis_decoration(kind))
            fig.update_layout(title=dict(text=self._titled_text(f'{value_col} Over Time ({freq_label})', subtitle)))

            self._save_chart_html(fig, f'timeseries_{value_col[:15]}', 'temporal')
        except Exception:
            pass

    def _create_hierarchy_drilldown_chart(self, df: pd.DataFrame, hierarchy: Tuple[str, ...]):
        """
        Create an interactive sunburst chart for drill-down through a
        detected dimension hierarchy (parent -> child, e.g. Category ->
        Subcategory). Sunburst segments are natively click-to-zoom in
        Plotly - clicking a parent segment drills into its children with
        no extra code or backend required, which is what makes this
        genuinely interactive rather than just another static picture.

        Interactive-only: a non-interactive sunburst is just a pie chart
        with extra steps, so this is skipped entirely in static/PDF mode.
        """
        path_cols = [c for c in hierarchy if c in df.columns]
        if len(path_cols) < 2:
            return

        plot_df = df.dropna(subset=path_cols)
        if len(plot_df) == 0:
            return

        counts = plot_df.groupby(path_cols).size().reset_index(name='count')

        # Defense in depth: DataProfiler's hierarchy detection already
        # caps the child level's cardinality (MAX_HIERARCHY_CHILD_UNIQUE),
        # but this guards independently in case a hierarchy tuple ever
        # reaches this method some other way. A sunburst with hundreds of
        # leaf segments is neither readable nor cheap to render/embed.
        if len(counts) > 150:
            return

        try:
            fig = px.sunburst(
                counts, path=path_cols, values='count',
                color_discrete_sequence=self.COLORS['categorical'],
                title=f"Drill-down: {' → '.join(path_cols)}"
            )
            fig.update_traces(insidetextorientation='radial')
            self._save_chart_html(fig, '_'.join(path_cols)[:25], 'hierarchy')
        except Exception:
            pass

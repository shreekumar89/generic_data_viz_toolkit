"""
test_business_charts.py

Tests for the executive-report chart styling upgrade: business-friendly
chart-type selection (horizontal bars sorted descending with an "Other"
bucket, donuts for low-cardinality composition), the semantic color
palette (functional red/green, not decorative), auto-generated
micro-insight subtitles, and currency/percentage-aware hover/axis
formatting - across both the interactive (Plotly) and static
(Matplotlib) chart-builder paths.

Run with:
    python -m unittest discover -s tests -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generic_data_viz import GenericDataVisualizer
from generic_data_viz.chart_generator import GenericVisualizationGenerator


class TempOutputMixin:
    def make_tempdir(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix='gdv_biz_test_'))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d


def _make_generator(tmp_path, **kwargs) -> GenericVisualizationGenerator:
    return GenericVisualizationGenerator(output_dir=tmp_path, **kwargs)


class TestCategoricalSummarization(unittest.TestCase, TempOutputMixin):
    """_summarize_categorical: sort-descending + Top-N/Other bucketing,
    the data-prep step shared by both chart engines."""

    def setUp(self):
        self.gen = _make_generator(self.make_tempdir())

    def test_collapses_long_tail_into_other(self):
        # 15 categories, sizes 15..1 (Category_0 has 15 rows, ..., Category_14 has 1 row)
        rows = []
        for i in range(15):
            rows += [f'Category_{i}'] * (15 - i)
        series = pd.Series(rows)

        summary = self.gen._summarize_categorical(series, top_n=10)

        self.assertEqual(len(summary), 11)  # top 10 + Other
        self.assertEqual(summary.index[-1], 'Other')
        # Other = sum of categories 10..14 = 5+4+3+2+1 = 15
        self.assertEqual(summary.loc['Other'], 15)
        # Sorted descending except Other, which is always last regardless of magnitude
        self.assertTrue((summary.iloc[:10].values == sorted(summary.iloc[:10].values, reverse=True)).all())

    def test_no_other_bucket_when_within_top_n(self):
        series = pd.Series(['A'] * 5 + ['B'] * 3 + ['C'] * 1)
        summary = self.gen._summarize_categorical(series, top_n=10)
        self.assertNotIn('Other', summary.index)
        self.assertEqual(len(summary), 3)
        self.assertEqual(list(summary.index), ['A', 'B', 'C'])  # sorted descending


class TestFormattingHelpers(unittest.TestCase, TempOutputMixin):
    """Kind inference and kind-aware value/axis/hover formatting."""

    def setUp(self):
        self.gen = _make_generator(self.make_tempdir())

    def test_format_kind_from_name(self):
        self.assertEqual(self.gen._format_kind('Revenue'), 'currency')
        self.assertEqual(self.gen._format_kind('ConversionRate'), 'percentage')
        self.assertEqual(self.gen._format_kind('OrderCount'), 'count')
        self.assertEqual(self.gen._format_kind('SomeRandomMetric'), 'number')

    def test_format_kind_prefers_semantic_type(self):
        column_types = {'Value': {'semantic_type': 'currency'}}
        self.assertEqual(self.gen._format_kind('Value', column_types), 'currency')

    def test_format_value(self):
        self.assertEqual(self.gen._format_value(1234.5, 'currency'), '$1,234')
        self.assertEqual(self.gen._format_value(12.34, 'percentage'), '12.3%')
        self.assertEqual(self.gen._format_value(42, 'count'), '42')
        self.assertEqual(self.gen._format_value(None, 'currency'), 'N/A')

    def test_axis_decoration_uses_prefix_suffix_not_percent_multiply(self):
        # Regression guard: must NOT use d3's '%' tickformat, which would
        # multiply an already-0-100-scale value by 100 again.
        currency_deco = self.gen._axis_decoration('currency')
        pct_deco = self.gen._axis_decoration('percentage')
        self.assertEqual(currency_deco.get('tickprefix'), '$')
        self.assertEqual(pct_deco.get('ticksuffix'), '%')
        self.assertNotIn('%', pct_deco.get('tickformat', ''))

    def test_hover_num_token(self):
        self.assertEqual(self.gen._hover_num_token('y', 'currency'), '$%{y:,.0f}')
        self.assertEqual(self.gen._hover_num_token('y', 'percentage'), '%{y:,.1f}%')
        self.assertEqual(self.gen._hover_num_token('x', 'count'), '%{x:,.0f}')


class TestTimeseriesInsightMarkerColor(unittest.TestCase, TempOutputMixin):
    """The latest-point marker color must match the average/threshold
    line it's drawn against - a viewer reads the dot's color relative to
    the visible reference line, not some invisible first-period value."""

    def setUp(self):
        self.gen = _make_generator(self.make_tempdir())

    def test_marker_green_when_latest_at_or_above_average(self):
        ts = pd.Series([10, 20, 30, 100])  # avg=40, last=100 >= avg
        _, avg, marker_color = self.gen._timeseries_insight(ts, 'number')
        self.assertEqual(marker_color, self.gen.COLORS['good'])

    def test_marker_red_when_latest_below_average(self):
        ts = pd.Series([100, 100, 100, 10])  # avg=77.5, last=10 < avg
        _, avg, marker_color = self.gen._timeseries_insight(ts, 'number')
        self.assertEqual(marker_color, self.gen.COLORS['alert'])

    def test_subtitle_arrow_reflects_first_vs_last_independent_of_marker(self):
        # last (10) < average (77.5) -> marker red, but last (10) > first (5)
        # -> subtitle should still show an upward arrow/green span, since
        # it answers a different question (overall period change).
        ts = pd.Series([5, 100, 100, 10])
        subtitle, avg, marker_color = self.gen._timeseries_insight(ts, 'number')
        self.assertEqual(marker_color, self.gen.COLORS['alert'])
        self.assertIn(self.gen.COLORS['good'], subtitle)
        self.assertIn('▲', subtitle)


class TestChartTypeDispatch(unittest.TestCase, TempOutputMixin):
    """Donut for <=5 categories, horizontal bar (+Other) beyond that -
    both the interactive and static paths must agree on this."""

    def _low_and_high_cardinality_df(self):
        rng = np.random.default_rng(7)
        n = 300
        return pd.DataFrame({
            'Region': rng.choice(['North', 'South', 'East', 'West'], n),           # 4 -> donut
            'Product': rng.choice([f'P{i}' for i in range(20)], n),                # 20 -> bar+Other
        })

    def test_interactive_dispatch(self):
        df = self._low_and_high_cardinality_df()
        out = self.make_tempdir()
        gen = _make_generator(out, engine='plotly', interactive=True)
        gen._create_categorical_chart_plotly(df, 'Region')
        gen._create_categorical_chart_plotly(df, 'Product')

        categories = [c.split('_', 2)[1] for c in gen.generated_charts]
        self.assertIn('composition', categories)  # Region -> donut
        self.assertIn('categorical', categories)  # Product -> bar

    def test_static_dispatch(self):
        df = self._low_and_high_cardinality_df()
        out = self.make_tempdir()
        gen = _make_generator(out, static=True)
        gen._create_categorical_chart(df, 'Region')
        gen._create_categorical_chart(df, 'Product')

        categories = [c.split('_', 2)[1] for c in gen.generated_charts]
        self.assertIn('composition', categories)
        self.assertIn('categorical', categories)


class TestInteractiveChartContent(unittest.TestCase, TempOutputMixin):
    """Generated chart HTML actually contains the micro-insight subtitle,
    the average/threshold reference line, and formatted (not raw-float)
    hover values."""

    def setUp(self):
        rng = np.random.default_rng(11)
        n = 400
        self.df = pd.DataFrame({
            'OrderDate': pd.date_range('2023-01-01', periods=n, freq='D').astype(str),
            'Region': rng.choice(['North', 'South', 'East', 'West'], n),
            'Product': rng.choice([f'P{i}' for i in range(20)], n),
            'Revenue': np.round(rng.uniform(100, 5000, n), 2),
        })
        self.out = self.make_tempdir()
        self.gen = _make_generator(self.out, engine='plotly', interactive=True)

    def _content(self, filename: str) -> str:
        return (self.out / filename).read_text(encoding='utf-8')

    def test_categorical_chart_has_insight_subtitle(self):
        self.gen._create_categorical_chart_plotly(self.df, 'Product')
        content = self._content(self.gen.generated_charts[-1])
        self.assertIn('leads with', content)

    def test_donut_chart_has_insight_subtitle(self):
        self.gen._create_categorical_chart_plotly(self.df, 'Region')
        content = self._content(self.gen.generated_charts[-1])
        self.assertIn('leads with', content)

    def test_timeseries_chart_has_average_line_and_insight(self):
        self.gen._create_time_series_chart_plotly(self.df, 'OrderDate', 'Revenue')
        content = self._content(self.gen.generated_charts[-1])
        self.assertIn('Average:', content)
        self.assertIn('Peak', content)

    def test_timeseries_chart_hover_uses_currency_formatting_not_raw_float(self):
        self.gen._create_time_series_chart_plotly(self.df, 'OrderDate', 'Revenue')
        content = self._content(self.gen.generated_charts[-1])
        # The hovertemplate embeds a literal '$' plus a d3 thousands-format
        # spec, not a bare '%{y}' that would render an unformatted float.
        self.assertIn('$%{y:,.0f}', content)


class TestAxisLabelsNameWhatTheyMeasure(unittest.TestCase, TempOutputMixin):
    """
    Regression guard for a real usability complaint: a bare "Count" axis
    label doesn't say what's being counted. Every count-shaped axis must
    name the column it's counting rows of, not just say "Count"/"MB"/
    "Missing %" in isolation - checked across both chart engines.
    """

    def setUp(self):
        rng = np.random.default_rng(13)
        n = 300
        self.df = pd.DataFrame({
            'Units Sold': rng.integers(200, 4500, n),
            'Product': rng.choice([f'P{i}' for i in range(15)], n),
            'Region': rng.choice(['North', 'South', 'East', 'West'], n),
        })

    def _content(self, out_dir: Path, filename: str) -> str:
        return (out_dir / filename).read_text(encoding='utf-8')

    def test_interactive_histogram_ylabel_names_the_column(self):
        out = self.make_tempdir()
        gen = _make_generator(out, engine='plotly', interactive=True)
        gen._create_distribution_chart_plotly(self.df, 'Units Sold')
        content = self._content(out, gen.generated_charts[-1])
        self.assertIn('Number of Records (Units Sold)', content)
        self.assertNotIn('"Count"', content)

    def test_interactive_bar_xlabel_names_the_column(self):
        out = self.make_tempdir()
        gen = _make_generator(out, engine='plotly', interactive=True)
        gen._create_categorical_chart_plotly(self.df, 'Product')
        content = self._content(out, gen.generated_charts[-1])
        self.assertIn('Number of Records (Product)', content)

    def test_interactive_overview_labels_units(self):
        out = self.make_tempdir()
        gen = _make_generator(out, engine='plotly', interactive=True)
        from generic_data_viz.core.data_profiler import DataProfiler
        profile = DataProfiler().profile(self.df)
        gen._create_data_overview_plotly(self.df, profile)
        content = self._content(out, gen.generated_charts[-1])
        self.assertIn('Memory Usage (MB)', content)

    def _capture_static_axes(self, gen, build_fn):
        """
        Static chart builders save-and-close their figure as a side
        effect and don't return it, so intercept `_save_chart` to grab
        the figure's axes before it's closed, without needing to touch
        the file the PNG would otherwise be written to.
        """
        import matplotlib.pyplot as plt
        captured = {}

        def fake_save(fig, name, category):
            captured['axes'] = fig.axes
            plt.close(fig)

        with mock.patch.object(gen, '_save_chart', side_effect=fake_save):
            build_fn()
        return captured['axes']

    def test_static_histogram_labels_are_specific(self):
        out = self.make_tempdir()
        gen = _make_generator(out, static=True)
        axes = self._capture_static_axes(
            gen, lambda: gen._create_distribution_chart(self.df, 'Units Sold')
        )
        hist_ax = axes[0]
        self.assertEqual(hist_ax.get_ylabel(), 'Number of Records (Units Sold)')
        self.assertEqual(hist_ax.get_xlabel(), 'Units Sold')

    def test_static_categorical_chart_labels_are_specific(self):
        out = self.make_tempdir()
        gen = _make_generator(out, static=True)
        axes = self._capture_static_axes(
            gen, lambda: gen._create_categorical_chart(self.df, 'Product')
        )
        ax = axes[0]
        self.assertEqual(ax.get_xlabel(), 'Number of Records (Product)')
        self.assertEqual(ax.get_ylabel(), 'Product')


class TestEndToEndBusinessReport(unittest.TestCase, TempOutputMixin):
    """Full pipeline still succeeds and produces the new chart types /
    insight content when run through GenericDataVisualizer end-to-end."""

    def test_report_generation_succeeds_with_business_charts(self):
        rng = np.random.default_rng(5)
        n = 350
        df = pd.DataFrame({
            'OrderID': range(1, n + 1),
            'OrderDate': pd.date_range('2023-01-01', periods=n, freq='D').astype(str),
            'Region': rng.choice(['North', 'South', 'East', 'West'], n),
            'Product': rng.choice([f'Product_{i}' for i in range(20)], n),
            'Revenue': np.round(rng.uniform(100, 5000, n), 2),
            'Quantity': rng.integers(1, 50, n),
        })
        out = self.make_tempdir()
        viz = GenericDataVisualizer(output_dir=str(out))
        result = viz.analyze(df, max_charts=15)
        viz.generate_report(result, formats=['html'])

        self.assertTrue((out / 'analysis_report.html').exists())
        self.assertTrue(all(c.endswith('.html') for c in result.charts_generated))
        categories = [c.split('_', 2)[1] for c in result.charts_generated]
        self.assertIn('composition', categories)  # Region, 4 categories -> donut
        self.assertIn('categorical', categories)  # Product, 20 categories -> bar+Other


if __name__ == '__main__':
    unittest.main()

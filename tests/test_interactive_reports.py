"""
test_interactive_reports.py

Tests for the interactive (Plotly) chart generation and HTML report
embedding introduced to replace static PNG charts as the default reporting
experience, while preserving a static/PNG fallback for headless or
PDF-export workflows.

Covers:
- GenericVisualizationGenerator's engine/interactive/static precedence
  rules (static always wins; an explicit engine="seaborn"/"matplotlib"
  still yields static charts even without --static, for backward
  compatibility with existing callers).
- End-to-end: GenericDataVisualizer with default construction produces
  interactive .html chart files, embeds them via <iframe> in the HTML
  report, and configures responsiveness/pan-zoom.
- End-to-end: static=True produces .png chart files embedded via <img>,
  with zero JS/iframe dependency (safe for PDF conversion).
- The hierarchy drill-down (sunburst) chart: generated only in
  interactive mode when a dimension hierarchy is detected.

Run with:
    python -m unittest discover -s tests -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generic_data_viz import GenericDataVisualizer
from generic_data_viz.chart_generator import GenericVisualizationGenerator
from generic_data_viz.core.data_profiler import DataProfiler


def _sample_dataframe_with_hierarchy(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    cat_map = {
        'Electronics': ['Phones', 'Laptops', 'Cameras'],
        'Clothing': ['Shirts', 'Pants'],
    }
    categories = rng.choice(list(cat_map.keys()), n)
    subcats = [rng.choice(cat_map[c]) for c in categories]
    return pd.DataFrame({
        'OrderID': range(1, n + 1),
        'OrderDate': pd.date_range('2023-01-01', periods=n, freq='D').astype(str),
        'Category': categories,
        'Subcategory': subcats,
        'Region': rng.choice(['North', 'South', 'East', 'West'], n),
        'Revenue': np.round(rng.uniform(100, 5000, n), 2),
        'Quantity': rng.integers(1, 50, n),
    })


class TempOutputMixin:
    def make_tempdir(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix='gdv_test_'))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d


class TestEngineSelectionPrecedence(unittest.TestCase, TempOutputMixin):
    """GenericVisualizationGenerator's static/engine/interactive precedence."""

    def _use_plotly(self, **kwargs) -> bool:
        out = self.make_tempdir()
        gen = GenericVisualizationGenerator(output_dir=out, **kwargs)
        return gen.use_plotly

    def test_defaults_are_interactive(self):
        self.assertTrue(self._use_plotly())

    def test_static_overrides_everything(self):
        self.assertFalse(self._use_plotly(static=True))
        # Even a contradictory explicit engine="plotly" + static=True: static wins.
        self.assertFalse(self._use_plotly(engine='plotly', interactive=True, static=True))

    def test_explicit_static_engine_without_static_flag_is_still_static(self):
        # Backward compatibility: existing callers that explicitly pass
        # engine="seaborn"/"matplotlib" (the old defaults) and don't pass
        # interactive=True keep getting static charts.
        self.assertFalse(self._use_plotly(engine='seaborn', interactive=False))
        self.assertFalse(self._use_plotly(engine='matplotlib', interactive=False))

    def test_explicit_interactive_true_forces_plotly_regardless_of_engine(self):
        self.assertTrue(self._use_plotly(engine='seaborn', interactive=True))


class TestInteractiveReportGeneration(unittest.TestCase, TempOutputMixin):
    """End-to-end: default (interactive) report generation."""

    @classmethod
    def setUpClass(cls):
        cls.df = _sample_dataframe_with_hierarchy()

    def setUp(self):
        self.out_dir = self.make_tempdir()
        self.viz = GenericDataVisualizer(output_dir=str(self.out_dir))  # all defaults
        self.result = self.viz.analyze(self.df, max_charts=15)
        self.viz.generate_report(self.result, formats=['html'])
        self.report_html = (self.out_dir / 'analysis_report.html').read_text(encoding='utf-8')

    def test_defaults_produce_html_charts_not_png(self):
        self.assertTrue(all(c.endswith('.html') for c in self.result.charts_generated))

    def test_report_embeds_charts_inline_not_via_iframe(self):
        # Charts are embedded as inline <div>+<script> fragments sharing
        # one Plotly.js load, not one <iframe src="..."> per chart - the
        # old approach made every chart independently fetch and
        # initialize its own copy of the library, which is what made
        # reports with many charts hang on open.
        self.assertNotIn('<iframe src="charts/', self.report_html)
        self.assertNotIn('<img src="charts/', self.report_html)
        self.assertIn('Plotly.newPlot', self.report_html)

    def test_report_loads_plotly_js_exactly_once(self):
        # The actual regression guard for the "many charts -> page hangs"
        # bug: regardless of how many interactive charts are in the
        # report, the CDN script tag must appear exactly once.
        self.assertGreaterEqual(len(self.result.charts_generated), 5)
        self.assertEqual(self.report_html.count('cdn.plot.ly'), 1)

    def test_hierarchy_drilldown_chart_generated(self):
        hierarchy_charts = [c for c in self.result.charts_generated if '_hierarchy_' in c]
        self.assertEqual(len(hierarchy_charts), 1)

    def test_chart_files_configure_responsiveness_and_scroll_zoom(self):
        # Plotly serializes the config dict into the page's JS; check the
        # keys we configured (INTERACTIVE_CONFIG in chart_generator.py)
        # actually made it into the written file.
        chart_path = self.out_dir / 'charts' / self.result.charts_generated[0]
        content = chart_path.read_text(encoding='utf-8')
        self.assertIn('responsive', content)
        self.assertIn('scrollZoom', content)
        self.assertIn('displaylogo', content)

    def test_dashboard_and_kpi_sections_still_present(self):
        self.assertIn('Dashboard Recommendations', self.report_html)
        self.assertIn('Detected KPIs', self.report_html)


class TestStaticFallbackReportGeneration(unittest.TestCase, TempOutputMixin):
    """End-to-end: static=True produces the old PNG/<img> behavior, for
    headless environments or PDF conversion (no JS/iframe dependency)."""

    @classmethod
    def setUpClass(cls):
        cls.df = _sample_dataframe_with_hierarchy(n=120)

    def setUp(self):
        self.out_dir = self.make_tempdir()
        self.viz = GenericDataVisualizer(output_dir=str(self.out_dir), static=True)
        self.result = self.viz.analyze(self.df, max_charts=10)
        self.viz.generate_report(self.result, formats=['html'])
        self.report_html = (self.out_dir / 'analysis_report.html').read_text(encoding='utf-8')

    def test_static_produces_png_charts(self):
        self.assertTrue(all(c.endswith('.png') for c in self.result.charts_generated))
        for c in self.result.charts_generated:
            self.assertTrue((self.out_dir / 'charts' / c).exists())

    def test_report_embeds_charts_via_img_not_iframe(self):
        self.assertIn('<img src="charts/', self.report_html)
        self.assertNotIn('<iframe', self.report_html)

    def test_hierarchy_drilldown_skipped_in_static_mode(self):
        # Sunburst-style drill-down is only meaningful interactively; a
        # static image of it is just a pie chart with extra steps, so it's
        # deliberately not generated in static/PDF mode.
        hierarchy_charts = [c for c in self.result.charts_generated if '_hierarchy_' in c]
        self.assertEqual(hierarchy_charts, [])


class TestBackwardCompatibleDataPipeline(unittest.TestCase, TempOutputMixin):
    """Existing callers that constructed GenericDataVisualizer with the old
    explicit defaults (engine='seaborn', interactive=False) must still get
    the exact old static-PNG behavior - the upgrade only changes what
    happens when engine/interactive are left unspecified."""

    def test_explicit_old_defaults_still_produce_static_png(self):
        out_dir = self.make_tempdir()
        df = pd.DataFrame({
            'ID': range(50),
            'Category': np.random.choice(['A', 'B'], 50),
            'Value': np.random.uniform(0, 100, 50),
        })
        viz = GenericDataVisualizer(output_dir=str(out_dir), engine='seaborn', interactive=False)
        result = viz.analyze(df, max_charts=5)
        self.assertTrue(all(c.endswith('.png') for c in result.charts_generated))

    def test_hierarchy_detection_still_populates_data_profile(self):
        # DataProfiler.profile() (a pipeline entry point unrelated to chart
        # rendering) must keep working exactly as before.
        df = _sample_dataframe_with_hierarchy(n=100)
        profile = DataProfiler().profile(df)
        self.assertIn(('Category', 'Subcategory'), profile.hierarchies)


if __name__ == '__main__':
    unittest.main()

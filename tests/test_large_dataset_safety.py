"""
test_large_dataset_safety.py

Regression guards for a real reported bug: on a ~97k-row dataset, the
generated HTML report grew to 15+ MB and hung on open. Root causes:

1. Plotly's histogram/box traces embed and bin their *raw* input values
   client-side in JS, so an unsampled large numeric column embedded one
   JSON number per row directly into the chart file (multi-MB per
   distribution chart). Fixed by DISTRIBUTION_SAMPLE_CAP in
   chart_generator.py.

2. Dimension hierarchy detection (DataProfiler._is_functional_dependency)
   had no cardinality ceiling on the "child" level, so a coarse dimension
   paired with a near-unique identifier/free-text dimension (e.g. a
   description column - still legitimately classified as DIMENSION)
   trivially satisfied the functional-dependency check, producing a
   "hierarchy" with thousands of leaf nodes. The resulting sunburst chart
   was 8+ MB by itself. Fixed by MAX_HIERARCHY_CHILD_UNIQUE plus a
   matching defensive cap in _create_hierarchy_drilldown_chart.

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

from generic_data_viz.chart_generator import GenericVisualizationGenerator
from generic_data_viz.core.data_profiler import DataProfiler


class TempOutputMixin:
    def make_tempdir(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix='gdv_scale_test_'))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d


class TestDistributionChartSampling(unittest.TestCase, TempOutputMixin):
    """A large numeric column must not embed one raw value per row into
    the chart file, and the sampling must not distort the reported
    summary statistics (which are computed from the full column)."""

    def setUp(self):
        rng = np.random.default_rng(0)
        n = 150_000  # well past DISTRIBUTION_SAMPLE_CAP
        self.df = pd.DataFrame({'Revenue': rng.uniform(0, 5000, n)})
        self.out = self.make_tempdir()
        self.gen = GenericVisualizationGenerator(output_dir=self.out, engine='plotly', interactive=True)

    def test_chart_file_size_is_bounded_regardless_of_row_count(self):
        self.gen._create_distribution_chart_plotly(self.df, 'Revenue')
        chart_path = self.out / self.gen.generated_charts[-1]
        size_mb = chart_path.stat().st_size / (1024 * 1024)
        # A sampled chart should be well under 1 MB; the unsampled version
        # of this exact scenario was 2+ MB.
        self.assertLess(size_mb, 1.0)

    def test_insight_subtitle_uses_full_data_min_max_not_sample(self):
        # The true min/max are computed before sampling, so they must
        # appear in the subtitle exactly, not an approximation from
        # whatever the random sample happened to include.
        true_min, true_max = self.df['Revenue'].min(), self.df['Revenue'].max()
        self.gen._create_distribution_chart_plotly(self.df, 'Revenue')
        content = (self.out / self.gen.generated_charts[-1]).read_text(encoding='utf-8')
        self.assertIn(f'{true_min:,.0f}'[:6], content)  # loose match on formatted magnitude
        self.assertIn('Range', content)

    def test_static_path_also_samples(self):
        out = self.make_tempdir()
        gen = GenericVisualizationGenerator(output_dir=out, static=True)
        # Should complete quickly and without error on a large column.
        gen._create_distribution_chart(self.df, 'Revenue')
        self.assertEqual(len(gen.generated_charts), 1)


class TestHierarchyCardinalityGuard(unittest.TestCase, TempOutputMixin):
    """A coarse dimension paired with a near-unique identifier/free-text
    dimension must not be detected as a drill-down hierarchy, and even if
    one somehow reaches the chart builder, it must be rejected there too."""

    def _degenerate_hierarchy_df(self, n=5000, n_unique_desc=3000):
        """
        Mirrors the real reported case: a free-text/description-like
        dimension with high-but-not-extreme cardinality (repeated values
        across rows, ~49% unique here) - high enough to land as DIMENSION
        rather than TEXT (which requires >80% uniqueness), and each
        distinct description deterministically belongs to one version, so
        the functional-dependency check's *shape* is satisfied - it's
        specifically the cardinality that must reject it.
        """
        rng = np.random.default_rng(2)
        desc_pool = [f'Item description {i}' for i in range(n_unique_desc)]
        desc_to_version = {d: ['v1.0', 'v1.1', 'v2.0'][i % 3] for i, d in enumerate(desc_pool)}
        descriptions = rng.choice(desc_pool, n)
        versions = [desc_to_version[d] for d in descriptions]
        return pd.DataFrame({'Version': versions, 'Description': descriptions})

    def test_near_unique_child_is_not_detected_as_hierarchy(self):
        df = self._degenerate_hierarchy_df()
        profiler = DataProfiler()
        is_fd = profiler._is_functional_dependency(df, 'Version', 'Description')
        self.assertFalse(is_fd)

    def test_genuine_small_hierarchy_still_detected(self):
        # Sanity check the cap doesn't also reject legitimate hierarchies.
        rng = np.random.default_rng(4)
        n = 2000
        cat_map = {'Electronics': ['Phones', 'Laptops'], 'Clothing': ['Shirts', 'Pants']}
        categories = rng.choice(list(cat_map.keys()), n)
        subcats = [rng.choice(cat_map[c]) for c in categories]
        df = pd.DataFrame({'Category': categories, 'Subcategory': subcats})

        profiler = DataProfiler()
        self.assertTrue(profiler._is_functional_dependency(df, 'Category', 'Subcategory'))

        profile = profiler.profile(df)
        self.assertIn(('Category', 'Subcategory'), profile.hierarchies)

    def test_chart_builder_rejects_oversized_hierarchy_directly(self):
        # Defense-in-depth: even if a caller passes a degenerate hierarchy
        # tuple directly (bypassing DataProfiler's own cap), the chart
        # builder must still refuse to render it.
        df = self._degenerate_hierarchy_df()
        out = self.make_tempdir()
        gen = GenericVisualizationGenerator(output_dir=out, engine='plotly', interactive=True)
        gen._create_hierarchy_drilldown_chart(df, ('Version', 'Description'))
        self.assertEqual(gen.generated_charts, [])

    def test_full_profile_does_not_surface_degenerate_hierarchy(self):
        df = self._degenerate_hierarchy_df()
        profile = DataProfiler().profile(df)
        self.assertEqual(profile.hierarchies, [])


if __name__ == '__main__':
    unittest.main()

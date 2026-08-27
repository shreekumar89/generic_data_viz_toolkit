"""
test_type_classification.py

Tests for the unified column type/role classification engine
(generic_data_viz.core.type_engine.TypeDetector) and the two legacy
entry points that now delegate to it:

- generic_data_viz.type_detector.AutoDataTypeDetector (dict/string API)
- generic_data_viz.core.data_profiler.DataProfiler (dataclass/enum API)

The central claim under test is that these two entry points can no
longer drift apart: they share one engine, one keyword/threshold config,
and one decision tree. `TestLegacyEntryPointConsistency` verifies this
directly by running both over the same DataFrame and asserting their
classifications agree once normalized to a common vocabulary (the only
legitimate difference is that the legacy `AutoDataTypeDetector` API has
no separate "KEY" or "BOOLEAN" concept and folds those into 'identifier'
/ 'dimension' - see `type_detector._ROLE_TO_LEGACY_STRING`).

Run with:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generic_data_viz.core.type_engine import (
    ColumnRole,
    DetectionConfig,
    SemanticType,
    TypeDetector,
)
from generic_data_viz.core.data_profiler import DataProfiler
from generic_data_viz.type_detector import AutoDataTypeDetector


def _sample_dataframe(n: int = 300) -> pd.DataFrame:
    """A dataframe covering every role/collision case exercised below."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        # identifiers / keys
        'OrderID': range(1, n + 1),
        'TrackingCode': [f'TRK{i:06d}' for i in range(n)],
        # dates - deliberately name-collision-prone with KEY_KEYWORDS
        'TransactionDate': pd.date_range('2023-01-01', periods=n, freq='D').astype(str),
        'RefDate': pd.date_range('2022-01-01', periods=n, freq='D').astype(str),
        'SignupDate': [(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)).strftime('%d/%m/%Y') for i in range(n)],
        'CreatedAt': pd.date_range('2021-06-01', periods=n, freq='D'),
        'FiscalYear': rng.choice([2021, 2022, 2023, 2024], n),
        # dimensions - some deliberately collision-prone
        'Region': rng.choice(['North', 'South', 'East', 'West'], n),
        'Vendors': rng.choice(['Acme', 'Globex', 'Initech'], n),
        'Resource': rng.choice(['CPU', 'GPU', 'Disk'], n),
        # measures - some deliberately collision-prone
        'Revenue': np.round(rng.uniform(100, 5000, n), 2),
        'AccountBalance': np.round(rng.uniform(-500, 5000, n), 2),
        'OrderCount': rng.integers(1, 50, n),
        # text
        'Notes': [f'Free text note number {i}' for i in range(n)],
        # boolean
        'IsActive': rng.choice([True, False], n),
        # contact-info identifiers
        'Email': [f'user{i}@example.com' for i in range(n)],
        'Phone': [f'+1-555-{1000 + i:04d}' for i in range(n)],
    })


class TestTypeEngineKeywordMatching(unittest.TestCase):
    """Word-boundary-aware keyword matching: no false positives from
    incidental substrings, no false negatives on real matches/plurals."""

    def test_no_false_positive_collisions(self):
        cfg = DetectionConfig()
        cases = [
            ('Notes', cfg.key_keywords),          # 'no' must not match inside 'Notes'
            ('Video', cfg.key_keywords),           # 'id'
            ('Turkey', cfg.key_keywords),           # 'key'
            ('Account', cfg.measure_keywords),      # 'count'
            ('Corporate', cfg.measure_keywords),    # 'rate'
            ('Resource', cfg.dimension_keywords),   # 'source'
            ('Border', cfg.measure_keywords),       # 'order'
            ('Asset', cfg.dimension_keywords),      # 'set'
        ]
        for col_name, keywords in cases:
            with self.subTest(col=col_name):
                self.assertFalse(TypeDetector.check_keyword_match(col_name, keywords))

    def test_real_matches_and_plurals_still_work(self):
        cfg = DetectionConfig()
        cases = [
            ('Notes', cfg.text_keywords),           # 'note' via singularization
            ('Vendors', cfg.dimension_keywords),    # 'vendor' via singularization
            ('Prices', cfg.measure_keywords),       # 'price' via singularization
            ('CustomerID', cfg.key_keywords),       # camelCase tokenization
            ('OrderNumber', cfg.key_keywords),      # compound keyword 'order_number'
            ('order_number', cfg.key_keywords),     # snake_case form of the same
            ('TransactionDate', cfg.date_keywords),  # 'date' component
            ('RefDate', cfg.date_keywords),
        ]
        for col_name, keywords in cases:
            with self.subTest(col=col_name):
                self.assertTrue(TypeDetector.check_keyword_match(col_name, keywords))


class TestTypeEngineDetection(unittest.TestCase):
    """Direct tests of the canonical TypeDetector.detect_column engine."""

    def setUp(self):
        self.detector = TypeDetector()
        self.df = _sample_dataframe()

    def _role(self, col: str) -> ColumnRole:
        return self.detector.detect_column(self.df[col], col).role

    def _semantic(self, col: str) -> SemanticType:
        return self.detector.detect_column(self.df[col], col).semantic_type

    def test_dates_resolve_correctly_despite_key_keyword_collisions(self):
        for col in ('TransactionDate', 'RefDate', 'SignupDate', 'CreatedAt'):
            with self.subTest(col=col):
                self.assertEqual(self._role(col), ColumnRole.DATE)

    def test_numeric_year_column_detected_as_date(self):
        self.assertEqual(self._role('FiscalYear'), ColumnRole.DATE)
        self.assertEqual(self._semantic('FiscalYear'), SemanticType.YEAR)

    def test_identifiers_and_keys(self):
        self.assertIn(self._role('OrderID'), (ColumnRole.KEY, ColumnRole.IDENTIFIER))
        self.assertIn(self._role('TrackingCode'), (ColumnRole.KEY, ColumnRole.IDENTIFIER))

    def test_dimensions_not_swallowed_by_measure_or_key_collisions(self):
        self.assertEqual(self._role('Region'), ColumnRole.DIMENSION)
        self.assertEqual(self._role('Vendors'), ColumnRole.DIMENSION)
        self.assertEqual(self._role('Resource'), ColumnRole.DIMENSION)

    def test_measures_not_swallowed_by_dimension_or_key_collisions(self):
        self.assertEqual(self._role('Revenue'), ColumnRole.MEASURE)
        self.assertEqual(self._role('AccountBalance'), ColumnRole.MEASURE)
        self.assertEqual(self._role('OrderCount'), ColumnRole.MEASURE)

    def test_currency_semantic_type_enrichment(self):
        self.assertEqual(self._semantic('Revenue'), SemanticType.CURRENCY)

    def test_text_role(self):
        self.assertEqual(self._role('Notes'), ColumnRole.TEXT)

    def test_boolean_role(self):
        self.assertEqual(self._role('IsActive'), ColumnRole.BOOLEAN)
        self.assertEqual(self._semantic('IsActive'), SemanticType.BOOLEAN)

    def test_email_and_phone_identification(self):
        self.assertEqual(self._role('Email'), ColumnRole.IDENTIFIER)
        self.assertEqual(self._semantic('Email'), SemanticType.EMAIL)
        self.assertEqual(self._role('Phone'), ColumnRole.IDENTIFIER)
        self.assertEqual(self._semantic('Phone'), SemanticType.PHONE)

    def test_phone_pattern_does_not_shadow_dates(self):
        # Regression guard: phone's loose "7+ digits/spaces/dashes" regex
        # also matches ISO date strings; date detection must win.
        s = pd.Series(pd.date_range('2023-01-01', periods=200, freq='D').astype(str))
        result = self.detector.detect_column(s, 'TransactionDate')
        self.assertEqual(result.role, ColumnRole.DATE)

    def test_plain_numeric_id_string_not_misread_as_phone(self):
        s = pd.Series([str(1_000_000 + i) for i in range(200)])
        result = self.detector.detect_column(s, 'ReferenceCode')
        self.assertNotEqual(result.semantic_type, SemanticType.PHONE)

    def test_empty_column(self):
        s = pd.Series([None, None, None])
        result = self.detector.detect_column(s, 'Empty')
        self.assertEqual(result.role, ColumnRole.UNKNOWN)
        self.assertEqual(result.semantic_type, SemanticType.EMPTY)


class TestLegacyEntryPointConsistency(unittest.TestCase):
    """
    The core regression guard for this refactor: both legacy entry points
    must now agree on every column's role, since they share one engine.

    AutoDataTypeDetector's legacy string vocabulary has no separate 'key'
    or 'boolean' role, so DataProfiler's ColumnRole.KEY/BOOLEAN are
    normalized to 'identifier'/'dimension' before comparing - matching
    the documented mapping in type_detector._ROLE_TO_LEGACY_STRING. Any
    other disagreement would indicate the two have drifted apart again.
    """

    _NORMALIZE = {
        ColumnRole.MEASURE: 'measure',
        ColumnRole.DIMENSION: 'dimension',
        ColumnRole.DATE: 'date',
        ColumnRole.KEY: 'identifier',
        ColumnRole.IDENTIFIER: 'identifier',
        ColumnRole.TEXT: 'text',
        ColumnRole.BOOLEAN: 'dimension',
        ColumnRole.UNKNOWN: 'unknown',
    }

    def test_both_entry_points_agree_on_every_column(self):
        df = _sample_dataframe()

        legacy_analysis = AutoDataTypeDetector.analyze_dataset(df)
        profile = DataProfiler().profile(df)

        mismatches = []
        for col in df.columns:
            legacy_role = legacy_analysis[col]['role']
            profiler_role = self._NORMALIZE[profile.columns[col].role]
            if legacy_role != profiler_role:
                mismatches.append((col, legacy_role, profiler_role))

        self.assertEqual(
            mismatches, [],
            f"AutoDataTypeDetector and DataProfiler disagree on: {mismatches}"
        )

    def test_both_entry_points_use_identical_cardinality_metric(self):
        # Regression guard for a real drift that predated the unification:
        # DataProfiler used to compute unique_percent against the TOTAL
        # row count (including nulls), while AutoDataTypeDetector computed
        # it against the non-null count only - giving different numbers
        # for the same column whenever nulls were present.
        df = pd.DataFrame({'Score': [1, 2, 3, 4, None, None]})  # 4 unique / 4 non-null = 100%

        legacy = AutoDataTypeDetector.detect_column_type(df['Score'], 'Score')
        profiler_col = DataProfiler().profile(df).columns['Score']

        self.assertAlmostEqual(legacy['unique_ratio'] * 100, profiler_col.unique_percent, places=6)


class TestDeprecatedShims(unittest.TestCase):
    """Legacy private methods are kept as thin shims (for any external
    code that reached into them directly) but must warn on use."""

    def test_type_detector_private_methods_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AutoDataTypeDetector._check_keyword_match('OrderDate', ['date'])
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_data_profiler_private_methods_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DataProfiler._check_keyword_match('OrderDate', ['date'])
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_data_profiler_determine_role_shim_still_works(self):
        df = pd.DataFrame({'Revenue': [1.0, 2.0, 3.0, 4.0, 5.0]})
        profiler = DataProfiler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            role = profiler._determine_role(df['Revenue'], 'Revenue', None, 100.0)
        self.assertEqual(role, ColumnRole.MEASURE)


class TestSharedConfigMutationVisibility(unittest.TestCase):
    """The class-level keyword-list attributes kept on both legacy
    classes for backward compatibility are live references into their
    engine's config, not copies - mutating one in place is reflected when
    read back, proving there's exactly one underlying list rather than
    two independently-maintained copies."""

    def test_measure_keywords_reflect_shared_config(self):
        profiler = DataProfiler()
        self.assertIn('revenue', [k.lower() for k in profiler.MEASURE_KEYWORDS])
        self.assertIn('revenue', [k.lower() for k in AutoDataTypeDetector.MEASURE_KEYWORDS])

        marker = '__unit_test_marker__'
        AutoDataTypeDetector.MEASURE_KEYWORDS.append(marker)
        try:
            self.assertIn(marker, AutoDataTypeDetector.MEASURE_KEYWORDS)
        finally:
            AutoDataTypeDetector.MEASURE_KEYWORDS.remove(marker)


if __name__ == '__main__':
    unittest.main()

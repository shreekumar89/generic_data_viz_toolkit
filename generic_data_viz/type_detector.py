"""
type_detector.py

Legacy, dict-based facade over the shared column-classification engine in
`generic_data_viz.core.type_engine`.

All keyword lists, thresholds, and the actual decision tree used to live
directly on `AutoDataTypeDetector` - and a near-identical copy lived on
`DataProfiler` in `core/data_profiler.py`, with different keyword lists
and thresholds that could (and did) silently drift apart. Both now
delegate to the same `TypeDetector` engine; this class exists only to
keep the historical dict-shaped public API
(`detect_column_type`/`analyze_dataset`/`get_summary`/`print_analysis`)
working unchanged for existing callers (`GenericDataVisualizer`,
`GenericVisualizationGenerator`, `examples/usage_example.py`).

The canonical engine distinguishes a couple of things this legacy API
never did (KEY vs IDENTIFIER, and BOOLEAN as its own role); those are
folded back into this class's historical vocabulary via
`_ROLE_TO_LEGACY_STRING` so `detect_column_type()`'s output shape and the
`'role'` values it has always produced are unchanged.
"""

import warnings
from typing import Any, Dict, List

import pandas as pd

from .core.type_engine import ColumnRole, DetectionConfig, SemanticType, TypeDetector

_ENGINE = TypeDetector()

# The legacy API only ever exposed these six role strings. The canonical
# engine's KEY and BOOLEAN roles are mapped down onto the closest of the
# six so every existing caller (chart planning, summaries) keeps working
# exactly as before.
_ROLE_TO_LEGACY_STRING = {
    ColumnRole.MEASURE: 'measure',
    ColumnRole.DIMENSION: 'dimension',
    ColumnRole.DATE: 'date',
    ColumnRole.KEY: 'identifier',
    ColumnRole.IDENTIFIER: 'identifier',
    ColumnRole.TEXT: 'text',
    ColumnRole.BOOLEAN: 'dimension',
    ColumnRole.UNKNOWN: 'unknown',
}


class AutoDataTypeDetector:
    """
    Enhanced automatic data type detection engine (legacy facade).

    Uses a combination of:
    1. Column name keyword matching (extensive keyword lists)
    2. Data distribution analysis (cardinality, unique ratios)
    3. Value pattern matching (regex for URLs, emails, dates, etc.)
    4. Data type inference (numeric vs categorical)

    Detects semantic data types beyond basic pandas dtypes:
    - Measures (numeric values to aggregate)
    - Dimensions (categorical grouping fields)
    - Dates/Times (temporal fields)
    - Identifiers/Keys (unique IDs)
    - Text (free-form text fields)
    - URLs, Emails, Phone numbers

    NOTE: The actual classification logic now lives in
    `generic_data_viz.core.type_engine.TypeDetector`, shared with
    `DataProfiler`, so the two can no longer drift apart. The class
    attributes below (MEASURE_KEYWORDS, etc.) are kept for backward
    compatibility: reading them, or mutating them in place (e.g.
    `.append(...)`), still works since they're the same list objects the
    shared engine's default `DetectionConfig` holds. Reassigning them
    (`AutoDataTypeDetector.MEASURE_KEYWORDS = [...]`) will NOT affect
    detection - construct `TypeDetector(DetectionConfig(...))` directly
    for custom rules instead.
    """

    # Backward-compatible references into the shared engine's config.
    PATTERNS = _ENGINE.config.patterns
    MEASURE_KEYWORDS = _ENGINE.config.measure_keywords
    DIMENSION_KEYWORDS = _ENGINE.config.dimension_keywords
    DATE_KEYWORDS = _ENGINE.config.date_keywords
    KEY_KEYWORDS = _ENGINE.config.key_keywords
    URL_KEYWORDS = _ENGINE.config.url_keywords
    TEXT_KEYWORDS = _ENGINE.config.text_keywords
    MAX_DIMENSION_UNIQUE_COUNT = _ENGINE.config.max_dimension_unique_count
    MAX_DIMENSION_UNIQUE_RATIO = _ENGINE.config.max_dimension_unique_ratio
    MIN_IDENTIFIER_UNIQUE_RATIO = _ENGINE.config.min_identifier_unique_ratio

    # ------------------------------------------------------------------
    # Deprecated private helpers - kept as thin, warning-emitting shims
    # over the shared engine for any external code that reached into
    # these "private" methods directly. Scheduled for removal; new code
    # should use generic_data_viz.core.type_engine.TypeDetector.
    # ------------------------------------------------------------------

    @classmethod
    def _tokenize_column_name(cls, col_name: str) -> List[str]:
        warnings.warn(
            "AutoDataTypeDetector._tokenize_column_name is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.tokenize_column_name.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.tokenize_column_name(col_name)

    @classmethod
    def _singularize(cls, token: str) -> str:
        warnings.warn(
            "AutoDataTypeDetector._singularize is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.singularize.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.singularize(token)

    @classmethod
    def _check_keyword_match(cls, col_name: str, keywords: List[str]) -> bool:
        warnings.warn(
            "AutoDataTypeDetector._check_keyword_match is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.check_keyword_match.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.check_keyword_match(col_name, keywords)

    @classmethod
    def _detect_value_patterns(cls, series: pd.Series, sample_size: int = 100) -> Dict[str, float]:
        warnings.warn(
            "AutoDataTypeDetector._detect_value_patterns is deprecated; use "
            "TypeDetector().detect_value_patterns.",
            DeprecationWarning, stacklevel=2,
        )
        return _ENGINE.detect_value_patterns(series, sample_size)

    @classmethod
    def _is_string_or_object_dtype(cls, series: pd.Series) -> bool:
        warnings.warn(
            "AutoDataTypeDetector._is_string_or_object_dtype is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.is_string_or_object_dtype.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.is_string_or_object_dtype(series)

    @classmethod
    def _date_parse_success_ratio(cls, series: pd.Series, sample_size: int = 100) -> float:
        warnings.warn(
            "AutoDataTypeDetector._date_parse_success_ratio is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.date_parse_success_ratio.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.date_parse_success_ratio(series, sample_size)

    @classmethod
    def _detect_date(
        cls,
        non_null: pd.Series,
        is_string_dtype: bool,
        is_numeric_dtype: bool,
        patterns: Dict[str, float],
        is_date_keyword: bool
    ):
        warnings.warn(
            "AutoDataTypeDetector._detect_date is deprecated; use "
            "TypeDetector().detect_date.",
            DeprecationWarning, stacklevel=2,
        )
        return _ENGINE.detect_date(non_null, is_string_dtype, is_numeric_dtype, patterns, is_date_keyword)

    # ------------------------------------------------------------------
    # Public API (unchanged signatures/shapes)
    # ------------------------------------------------------------------

    @classmethod
    def detect_column_type(cls, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Detect semantic type for a single column.

        Returns:
            Dict with role, semantic_type, unique_count, unique_ratio, confidence
        """
        r = _ENGINE.detect_column(series, col_name)
        return {
            'role': _ROLE_TO_LEGACY_STRING[r.role],
            'semantic_type': r.semantic_type.value,
            'confidence': r.confidence,
            'reason': r.reason,
            'unique_count': r.unique_count,
            'unique_ratio': r.unique_ratio,
            'null_count': r.null_count,
            'null_ratio': r.null_ratio,
        }

    @classmethod
    def analyze_dataset(cls, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze all columns in a dataset."""
        return {col: cls.detect_column_type(df[col], col) for col in df.columns}

    @classmethod
    def get_summary(cls, analysis: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Get a summary of columns by role."""
        summary = {
            'measures': [],
            'dimensions': [],
            'dates': [],
            'identifiers': [],
            'text': [],
            'unknown': []
        }

        for col, info in analysis.items():
            role = info.get('role', 'unknown')
            if role == 'measure':
                summary['measures'].append(col)
            elif role == 'dimension':
                summary['dimensions'].append(col)
            elif role == 'date':
                summary['dates'].append(col)
            elif role == 'identifier':
                summary['identifiers'].append(col)
            elif role == 'text':
                summary['text'].append(col)
            else:
                summary['unknown'].append(col)

        return summary

    @classmethod
    def print_analysis(cls, analysis: Dict[str, Dict]):
        """Pretty print the analysis results."""
        print("\n" + "=" * 70)
        print("COLUMN TYPE DETECTION RESULTS")
        print("=" * 70)

        for col, info in analysis.items():
            role = info.get('role', 'unknown').upper()
            sem_type = info.get('semantic_type', 'unknown')
            unique = info.get('unique_count', 0)
            ratio = info.get('unique_ratio', 0)
            conf = info.get('confidence', 0)
            reason = info.get('reason', '')

            role_emoji = {
                'measure': '📊',
                'dimension': '🏷️',
                'date': '📅',
                'identifier': '🔑',
                'text': '📝',
                'unknown': '❓'
            }.get(info.get('role', 'unknown'), '❓')

            print(f"\n{role_emoji} {col}")
            print(f"   Role: {role} | Type: {sem_type}")
            print(f"   Unique: {unique:,} ({ratio:.1%}) | Confidence: {conf:.0%}")
            print(f"   Reason: {reason}")

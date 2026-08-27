"""
data_profiler.py

Automatic data understanding and profiling module that detects data types,
measures, dimensions, keys, hierarchies, and generates business-friendly overviews.

Column role/semantic-type classification is delegated to the shared
`generic_data_viz.core.type_engine.TypeDetector` engine - the same one
`AutoDataTypeDetector` (in `type_detector.py`) uses - so the two can no
longer carry independent keyword lists/thresholds that silently drift
apart. `ColumnRole` and `DataType` below are re-exports of the engine's
canonical `ColumnRole`/`SemanticType` enums, kept under their historical
names so existing imports (`from generic_data_viz.core.data_profiler
import ColumnRole, DataType`) keep working unchanged.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .type_engine import ColumnRole, SemanticType as DataType, TypeDetector


@dataclass
class ColumnProfile:
    """Detailed profile of a single column."""
    name: str
    dtype: str
    role: ColumnRole
    semantic_type: DataType
    null_count: int
    null_percent: float
    unique_count: int
    unique_percent: float
    sample_values: List[Any]
    statistics: Dict[str, Any] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    is_potential_key: bool = False
    is_hierarchical: bool = False
    parent_column: Optional[str] = None
    business_description: str = ""


@dataclass
class DataProfile:
    """Complete profile of a dataset."""
    row_count: int
    column_count: int
    memory_usage_mb: float
    columns: Dict[str, ColumnProfile]
    measures: List[str]
    dimensions: List[str]
    date_columns: List[str]
    key_columns: List[str]
    hierarchies: List[Tuple[str, ...]]
    data_dictionary: Dict[str, str]
    business_summary: str


class DataProfiler:
    """
    Automatic data understanding and profiling engine.

    Column role/semantic-type classification is delegated to a shared
    `generic_data_viz.core.type_engine.TypeDetector` instance (see
    `_detector` below) rather than duplicating keyword lists and a
    decision tree here. The class attributes below (MEASURE_KEYWORDS,
    etc.) are kept for backward compatibility - reading them, or mutating
    them in place, still reflects this instance's config - but reassigning
    them won't affect detection; pass a custom `TypeDetector` to
    `__init__` for that.
    """

    # A hierarchy's child level needs to be small enough to actually serve
    # as a drill-down level (e.g. a handful of categories -> a few dozen
    # subcategories). Without this cap, any near-unique identifier or
    # free-text dimension column (still classified as DIMENSION - that
    # role tolerates up to 500 unique values) would trivially satisfy the
    # functional-dependency check below when paired with any coarser
    # column, producing a "hierarchy" with thousands of leaf nodes that's
    # neither readable nor practical to render (a sunburst chart built
    # from one is enormous and slow to load - see chart_generator.py's
    # own matching guard on the render side).
    MAX_HIERARCHY_CHILD_UNIQUE = 100

    def __init__(self, sample_size: int = 1000, detector: Optional[TypeDetector] = None):
        self.sample_size = sample_size
        self._detector = detector or TypeDetector()

        # Backward-compatible references into this instance's engine config.
        self.PATTERNS = self._detector.config.patterns
        self.MEASURE_KEYWORDS = self._detector.config.measure_keywords
        self.DIMENSION_KEYWORDS = self._detector.config.dimension_keywords
        self.DATE_KEYWORDS = self._detector.config.date_keywords
        self.KEY_KEYWORDS = self._detector.config.key_keywords

    # ------------------------------------------------------------------
    # Deprecated private helpers - kept as thin, warning-emitting shims
    # for any external code that reached into these "private" methods
    # directly. Scheduled for removal; new code should use
    # generic_data_viz.core.type_engine.TypeDetector.
    # ------------------------------------------------------------------

    @classmethod
    def _tokenize_column_name(cls, col_name: str) -> List[str]:
        warnings.warn(
            "DataProfiler._tokenize_column_name is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.tokenize_column_name.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.tokenize_column_name(col_name)

    @classmethod
    def _singularize(cls, token: str) -> str:
        warnings.warn(
            "DataProfiler._singularize is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.singularize.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.singularize(token)

    @classmethod
    def _check_keyword_match(cls, col_name: str, keywords: List[str]) -> bool:
        warnings.warn(
            "DataProfiler._check_keyword_match is deprecated; use "
            "generic_data_viz.core.type_engine.TypeDetector.check_keyword_match.",
            DeprecationWarning, stacklevel=2,
        )
        return TypeDetector.check_keyword_match(col_name, keywords)

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Generate a complete profile of the dataset."""
        columns = {}
        measures = []
        dimensions = []
        date_columns = []
        key_columns = []
        
        for col in df.columns:
            profile = self._profile_column(df[col], col)
            columns[col] = profile
            
            if profile.role == ColumnRole.MEASURE:
                measures.append(col)
            elif profile.role == ColumnRole.DIMENSION:
                dimensions.append(col)
            elif profile.role == ColumnRole.DATE:
                date_columns.append(col)
            elif profile.role == ColumnRole.KEY:
                key_columns.append(col)
        
        hierarchies = self._detect_hierarchies(df, dimensions)
        data_dict = self._generate_data_dictionary(columns)
        summary = self._generate_business_summary(df, measures, dimensions, date_columns, key_columns)
        
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            columns=columns,
            measures=measures,
            dimensions=dimensions,
            date_columns=date_columns,
            key_columns=key_columns,
            hierarchies=hierarchies,
            data_dictionary=data_dict,
            business_summary=summary
        )
    
    def _profile_column(self, series: pd.Series, name: str) -> ColumnProfile:
        """Generate detailed profile for a single column."""
        dtype = str(series.dtype)

        detection = self._detector.detect_column(series, name)
        role = detection.role
        semantic_type = detection.semantic_type
        null_count = detection.null_count
        null_percent = detection.null_ratio * 100
        unique_count = detection.unique_count
        unique_percent = detection.unique_ratio * 100

        valid_values = series.dropna()
        sample_values = valid_values.head(5).tolist() if len(valid_values) > 0 else []

        statistics = self._calculate_statistics(series, role)
        patterns = self._detect_patterns(series)
        is_potential_key = unique_percent > 95 and null_percent < 1
        description = self._generate_column_description(name, role, semantic_type, statistics)

        return ColumnProfile(
            name=name,
            dtype=dtype,
            role=role,
            semantic_type=semantic_type,
            null_count=null_count,
            null_percent=null_percent,
            unique_count=unique_count,
            unique_percent=unique_percent,
            sample_values=sample_values,
            statistics=statistics,
            patterns=patterns,
            is_potential_key=is_potential_key,
            business_description=description
        )

    def _detect_semantic_type(self, series: pd.Series, name: str) -> DataType:
        warnings.warn(
            "DataProfiler._detect_semantic_type is deprecated; use "
            "TypeDetector().detect_column(series, name).semantic_type.",
            DeprecationWarning, stacklevel=2,
        )
        return self._detector.detect_column(series, name).semantic_type

    def _determine_role(self, series: pd.Series, name: str, semantic_type: DataType, unique_percent: float) -> ColumnRole:
        warnings.warn(
            "DataProfiler._determine_role is deprecated (and its semantic_type/"
            "unique_percent arguments are now ignored - the shared engine "
            "recomputes everything from `series`); use "
            "TypeDetector().detect_column(series, name).role.",
            DeprecationWarning, stacklevel=2,
        )
        return self._detector.detect_column(series, name).role
    
    def _calculate_statistics(self, series: pd.Series, role: ColumnRole) -> Dict[str, Any]:
        """Calculate relevant statistics based on column role."""
        stats = {}
        
        if role == ColumnRole.MEASURE:
            valid = series.dropna()
            if len(valid) > 0:
                stats = {
                    'min': float(valid.min()),
                    'max': float(valid.max()),
                    'mean': float(valid.mean()),
                    'median': float(valid.median()),
                    'std': float(valid.std()),
                    'q25': float(valid.quantile(0.25)),
                    'q75': float(valid.quantile(0.75)),
                }
        
        elif role == ColumnRole.DIMENSION:
            value_counts = series.value_counts()
            stats = {
                'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'mode_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_5': value_counts.head(5).to_dict(),
            }
        
        elif role == ColumnRole.DATE:
            valid = pd.to_datetime(series, errors='coerce').dropna()
            if len(valid) > 0:
                stats = {
                    'min_date': str(valid.min()),
                    'max_date': str(valid.max()),
                    'date_range_days': (valid.max() - valid.min()).days,
                }
        
        return stats
    
    def _detect_patterns(self, series: pd.Series) -> List[str]:
        """Detect common patterns in the data."""
        patterns = []
        
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if len(valid) > 0:
                if (valid >= 0).all():
                    patterns.append("all_positive")
                if (valid == valid.astype(int)).all():
                    patterns.append("all_integers")
        
        return patterns
    
    def _detect_hierarchies(self, df: pd.DataFrame, dimensions: List[str]) -> List[Tuple[str, ...]]:
        """Detect hierarchical relationships between dimensions."""
        hierarchies = []
        
        for i, col1 in enumerate(dimensions):
            for col2 in dimensions[i+1:]:
                if self._is_functional_dependency(df, col1, col2):
                    hierarchies.append((col1, col2))
                elif self._is_functional_dependency(df, col2, col1):
                    hierarchies.append((col2, col1))
        
        return hierarchies
    
    def _is_functional_dependency(self, df: pd.DataFrame, parent: str, child: str) -> bool:
        """
        Check whether `child` is a finer-grained sub-level of `parent` in a
        hierarchy sense (e.g. parent='Category', child='Subcategory'):
        `parent` must have fewer distinct values than `child`, and every
        value of `child` must map to exactly one value of `parent`.

        Note the grouping direction: to test "does Category have a fixed
        Subcategory" would be backwards (one Category has MANY
        Subcategories) - the constant relationship runs the other way,
        from child up to parent, so the group-by is on `child`.
        """
        child_unique = df[child].nunique()
        if df[parent].nunique() >= child_unique:
            return False
        if child_unique > self.MAX_HIERARCHY_CHILD_UNIQUE:
            return False
        grouped = df.groupby(child)[parent].nunique()
        return (grouped == 1).all()
    
    def _generate_data_dictionary(self, columns: Dict[str, ColumnProfile]) -> Dict[str, str]:
        """Generate an automatic data dictionary."""
        return {name: profile.business_description for name, profile in columns.items()}
    
    def _generate_column_description(self, name: str, role: ColumnRole, semantic_type: DataType, statistics: Dict[str, Any]) -> str:
        """Generate a business-friendly description for a column."""
        role_desc = {
            ColumnRole.MEASURE: "numeric measure for aggregation",
            ColumnRole.DIMENSION: "categorical dimension for grouping/filtering",
            ColumnRole.DATE: "date/time field for temporal analysis",
            ColumnRole.KEY: "unique identifier/key field",
            ColumnRole.TEXT: "free-form text field",
            ColumnRole.BOOLEAN: "true/false flag field",
            ColumnRole.IDENTIFIER: "reference identifier",
        }
        
        desc = f"{name}: {role_desc.get(role, 'general field')}"
        
        if role == ColumnRole.MEASURE and statistics:
            desc += f". Range: {statistics.get('min', 'N/A'):.2f} to {statistics.get('max', 'N/A'):.2f}"
        elif role == ColumnRole.DIMENSION and statistics.get('mode'):
            desc += f". Most common: {statistics['mode']}"
        
        return desc
    
    def _generate_business_summary(self, df: pd.DataFrame, measures: List[str], dimensions: List[str], date_columns: List[str], key_columns: List[str]) -> str:
        """Generate a business-friendly summary of the dataset."""
        lines = [
            f"Dataset Overview",
            f"================",
            f"",
            f"This dataset contains {len(df):,} records with {len(df.columns)} columns.",
            f"",
            f"Column Classification:",
            f"  - Measures (numeric values for analysis): {len(measures)}",
            f"  - Dimensions (categorical fields for grouping): {len(dimensions)}",
            f"  - Date/Time fields: {len(date_columns)}",
            f"  - Key/Identifier fields: {len(key_columns)}",
            f"",
        ]
        
        if measures:
            lines.append(f"Key Measures: {', '.join(measures[:5])}")
        if dimensions:
            lines.append(f"Key Dimensions: {', '.join(dimensions[:5])}")
        if date_columns:
            lines.append(f"Time Dimensions: {', '.join(date_columns)}")
        
        return "\n".join(lines)

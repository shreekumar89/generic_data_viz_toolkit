"""
type_engine.py

Single source of truth for column type/role classification.

Historically, `type_detector.AutoDataTypeDetector` and
`core.data_profiler.DataProfiler` each carried their own copy of the
keyword lists, cardinality thresholds, and decision tree used to classify
a column as a measure, dimension, date, identifier, etc. They drifted:
different MEASURE/DIMENSION/DATE/KEY keyword lists, different regex
pattern sets, and even a different definition of "uniqueness" (one
divided by non-null count, the other by total row count) - so the same
column could be classified differently depending on which module analyzed
it, and a fix applied to one side (e.g. word-boundary-aware keyword
matching, or date-detection priority) silently failed to reach the other.

This module is the fix: one `TypeDetector` engine, configured by one
`DetectionConfig`, returning one canonical `ColumnDetectionResult` built
from a single `ColumnRole` / `SemanticType` vocabulary. Both
`AutoDataTypeDetector` and `DataProfiler` now delegate to it; see the
"legacy facade" docstrings in those modules for how their historical
APIs map onto the canonical result.
"""

import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class ColumnRole(Enum):
    """Canonical analytical role of a column."""
    MEASURE = "measure"
    DIMENSION = "dimension"
    DATE = "date"
    KEY = "key"
    IDENTIFIER = "identifier"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class SemanticType(Enum):
    """
    Canonical semantic subtype, finer-grained than ColumnRole.

    This is a superset of the two vocabularies it replaces. Members like
    ADDRESS, GEO_COORDINATE, NUMERIC_DISCRETE, CATEGORICAL_NOMINAL, and
    CATEGORICAL_ORDINAL are preserved for name compatibility with the old
    `data_profiler.DataType` enum but are not currently emitted by the
    detection logic (they weren't reliably emitted before unification
    either - ADDRESS/GEO_COORDINATE were already unused, and the
    nominal/ordinal split duplicated the cardinality-tier distinction
    CATEGORICAL/CATEGORICAL_HIGH already makes). They're kept as reserved
    values rather than dropped, in case external code references them.
    """
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"          # reserved, see docstring
    NUMERIC_CATEGORICAL = "numeric_categorical"
    CATEGORICAL = "categorical"
    CATEGORICAL_HIGH = "categorical_high"
    CATEGORICAL_NOMINAL = "categorical_nominal"    # reserved, see docstring
    CATEGORICAL_ORDINAL = "categorical_ordinal"    # reserved, see docstring
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DATETIME_STRING = "datetime_string"
    YEAR = "year"
    UNIX_TIMESTAMP_SECONDS = "unix_timestamp_seconds"
    UNIX_TIMESTAMP_MILLIS = "unix_timestamp_millis"
    TEXT = "text"
    TEXT_SHORT = "text_short"
    TEXT_LONG = "text_long"
    TEXT_DESCRIPTION = "text_description"
    BOOLEAN = "boolean"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    ADDRESS = "address"                            # reserved, see docstring
    GEO_COORDINATE = "geo_coordinate"               # reserved, see docstring
    ID = "id"
    ID_NUMERIC = "id_numeric"
    ID_TEXT = "id_text"
    ID_ALPHANUMERIC = "id_alphanumeric"             # reserved, see docstring
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass
class DetectionConfig:
    """
    All keyword lists, regex patterns, and thresholds used for
    classification, gathered in one place. Both legacy entry points share
    an instance of this (or their own default-constructed copy - see the
    module docstring), so there is exactly one place left to edit the
    actual rules.
    """

    measure_keywords: List[str] = field(default_factory=lambda: [
        # Financial/Monetary
        'amount', 'price', 'cost', 'revenue', 'quantity', 'total', 'sum',
        'sales', 'profit', 'margin', 'balance', 'fee', 'tax', 'discount',
        'payment', 'income', 'expense', 'budget', 'spend', 'earning',
        'invoice', 'billing', 'charge', 'credit', 'debit', 'interest',
        'value',
        # Metrics/KPIs
        'score', 'rating', 'rate', 'percent', 'percentage', 'ratio', 'index',
        'avg', 'average', 'mean', 'median', 'count', 'frequency', 'occurrence',
        'conversion', 'retention', 'churn', 'growth', 'decline', 'change',
        # Physical measurements
        'weight', 'height', 'width', 'length', 'depth', 'area', 'volume',
        'distance', 'speed', 'velocity', 'temperature', 'pressure', 'density',
        'mass', 'capacity', 'power', 'energy', 'force', 'duration', 'age',
        'size',
        # Quantities
        'qty', 'units', 'items', 'pieces', 'stock', 'inventory', 'supply',
        'demand', 'order', 'shipment', 'delivery', 'returns', 'refund',
        # Performance
        'performance', 'efficiency', 'productivity', 'utilization', 'coverage',
        'accuracy', 'precision', 'recall', 'f1', 'auc', 'rmse', 'mae',
        # Impact/Sustainability
        'impact', 'footprint', 'emission', 'consumption', 'usage', 'waste',
    ])

    dimension_keywords: List[str] = field(default_factory=lambda: [
        # Entity names
        'name', 'title', 'label', 'description', 'desc', 'display',
        # Classifications/Hierarchies
        'category', 'subcategory', 'type', 'subtype', 'kind', 'variety',
        'class', 'subclass', 'classification', 'taxonomy', 'hierarchy',
        'group', 'subgroup', 'grouping', 'cluster', 'segment', 'subsegment',
        'sector', 'subsector', 'industry', 'vertical', 'domain', 'area',
        'division', 'subdivision', 'department', 'unit', 'team',
        'tier', 'level', 'grade', 'rank', 'priority', 'severity',
        # Geographic/Location
        'country', 'nation', 'state', 'province', 'region', 'subregion',
        'city', 'town', 'village', 'district', 'zone', 'territory',
        'location', 'place', 'site', 'facility', 'branch', 'office',
        'address', 'street', 'postal', 'zip', 'continent', 'hemisphere',
        'market', 'geography', 'geo', 'locale', 'jurisdiction',
        # Business entities
        'brand', 'manufacturer', 'maker', 'producer', 'supplier',
        'vendor', 'seller', 'merchant', 'retailer', 'distributor',
        'customer', 'client', 'account', 'subscriber', 'member', 'user',
        'partner', 'affiliate', 'reseller', 'dealer', 'agent',
        'company', 'organization', 'org', 'enterprise', 'business',
        'store', 'shop', 'outlet', 'warehouse', 'depot',
        # Product attributes
        'product', 'item', 'article', 'sku', 'model', 'variant',
        'color', 'colour', 'size', 'style', 'design', 'pattern',
        'material', 'fabric', 'composition', 'ingredient', 'component',
        'flavor', 'flavour', 'scent', 'fragrance', 'texture',
        'shape', 'form', 'format', 'edition', 'version', 'release',
        'collection', 'line', 'series', 'family', 'range',
        'packaging', 'package', 'pack', 'bundle', 'set', 'kit',
        # Status/State
        'status', 'state', 'condition', 'stage', 'phase', 'step',
        'mode', 'method', 'approach', 'technique', 'process',
        'flag', 'indicator', 'marker', 'tag', 'badge',
        # Channel/Source
        'channel', 'source', 'origin', 'medium', 'platform', 'device',
        'campaign', 'promotion', 'initiative', 'program', 'project',
        'referrer', 'referral', 'acquisition', 'touchpoint',
        # Demographics
        'gender', 'sex', 'age_group', 'generation', 'cohort',
        'occupation', 'profession', 'role', 'position', 'function',
        'education', 'degree', 'qualification', 'certification',
        'income_bracket', 'wealth', 'socioeconomic',
        # Language/Culture
        'language', 'lang', 'locale', 'currency', 'timezone',
        # Technical
        'environment', 'env', 'instance', 'server', 'host', 'node',
        'application', 'app', 'service', 'module', 'component',
        'feature', 'capability', 'option', 'setting', 'config',
    ])

    date_keywords: List[str] = field(default_factory=lambda: [
        'date', 'time', 'datetime', 'timestamp', 'ts',
        'year', 'month', 'day', 'week', 'quarter', 'qtr',
        'hour', 'minute', 'second', 'millisecond',
        'created', 'updated', 'modified', 'changed', 'edited',
        'start', 'end', 'begin', 'finish', 'due', 'deadline',
        'birth', 'death', 'expiry', 'expiration', 'valid',
        'effective', 'termination', 'cancellation',
        'opened', 'closed', 'completed', 'submitted', 'approved',
        'published', 'released', 'launched', 'deployed',
        'registered', 'enrolled', 'subscribed', 'activated',
        'first', 'last', 'recent', 'latest', 'earliest',
        'period', 'interval', 'duration', 'tenure',
        'scored_at', 'fetched_at', 'processed_at', 'synced_at',
    ])

    key_keywords: List[str] = field(default_factory=lambda: [
        'id', 'ids', 'key', 'keys', 'pk', 'fk', 'sk',
        'code', 'codes', 'number', 'num', 'no', 'nbr',
        'identifier', 'guid', 'uuid', 'uid', 'hash',
        'ref', 'reference', 'index', 'idx', 'seq', 'sequence',
        'serial', 'barcode', 'ean', 'upc', 'isbn', 'sku',
        'account_number', 'order_number', 'invoice_number',
        'tracking', 'confirmation', 'transaction', 'txn',
    ])

    url_keywords: List[str] = field(default_factory=lambda: [
        'url', 'link', 'href', 'uri', 'endpoint', 'path', 'route'
    ])

    text_keywords: List[str] = field(default_factory=lambda: [
        'description', 'desc', 'comment', 'note', 'remark',
        'detail', 'explanation', 'summary', 'abstract', 'body',
        'content', 'text', 'message', 'memo', 'narrative'
    ])

    currency_keywords: List[str] = field(default_factory=lambda: [
        'price', 'cost', 'amount', 'revenue', 'value'
    ])

    # Order matters only in the sense that more specific patterns should be
    # checked before looser ones; the engine checks each by name rather than
    # walking this dict in order, but the ordering is kept meaningful for
    # readability and for any external code that iterates it directly.
    patterns: Dict[str, str] = field(default_factory=lambda: {
        'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
        'url': r'^https?://[\w\.-]+',
        'datetime': r'^\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}',
        'date_iso': r'^\d{4}[-/]\d{2}[-/]\d{2}',
        'date_eu': r'^\d{2}[-/]\d{2}[-/]\d{4}',
        'date_month_name': (
            r'^\d{1,2}[\s\-]?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
            r'[a-z]*[\s\-,]+\d{2,4}$'
            r'|^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}$'
        ),
        'time': r'^\d{2}:\d{2}(:\d{2})?',
        'id_numeric': r'^\d+$',
        'id_alphanumeric': r'^[A-Z]{2,4}[\d]+$',
        # Deliberately checked after id_numeric/id_alphanumeric: this regex
        # is just "7+ digits/spaces/dashes/parens", which would otherwise
        # also match - and misclassify - plain numeric ID strings.
        'phone': r'^[\+]?[\d\s\-\(\)]{7,}$',
        'currency': r'^\$?€?£?[\d,]+\.?\d{0,2}$',
        'percentage': r'^[\d\.]+\s*%$',
        'version': r'^\d+\.\d+(\.\d+)?$',
    })

    max_dimension_unique_count: int = 500
    max_dimension_unique_ratio: float = 0.10
    min_identifier_unique_ratio: float = 0.80
    key_unique_percent_threshold: float = 90.0
    high_uniqueness_numeric_min_ratio: float = 0.7
    high_uniqueness_numeric_min_count: int = 1000
    low_cardinality_numeric_max_unique: int = 50
    low_cardinality_numeric_max_ratio: float = 0.01
    default_measure_min_ratio: float = 0.01
    date_pattern_score_threshold: float = 0.5
    date_parse_success_threshold: float = 0.7
    email_pattern_threshold: float = 0.7
    phone_pattern_threshold: float = 0.7
    year_min: float = 1900
    year_max: float = 2100
    unix_ms_min: float = 1e11
    unix_ms_max: float = 2.5e12
    unix_sec_min: float = 1e8
    unix_sec_max: float = 2.5e9


@dataclass
class ColumnDetectionResult:
    """Canonical result of classifying one column."""
    role: ColumnRole
    semantic_type: SemanticType
    confidence: float
    reason: str
    unique_count: int
    unique_ratio: float
    null_count: int
    null_ratio: float


class TypeDetector:
    """
    Single source of truth for column type/role detection.

    Combines:
    1. Column name keyword matching (word-boundary aware, tokenized -
       snake_case/kebab-case/camelCase/PascalCase all handled, with naive
       singularization so plurals still match).
    2. Data distribution analysis (cardinality, unique ratios - always
       computed against the non-null count).
    3. Value pattern matching (regex for URLs, emails, phone numbers,
       dates, etc.).
    4. Dtype inference (boolean vs datetime vs numeric vs categorical).

    Usage:
        detector = TypeDetector()                       # default rules
        detector = TypeDetector(DetectionConfig(...))    # custom rules
        result = detector.detect_column(series, "OrderDate")
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    # ------------------------------------------------------------------
    # Pure utilities (no config dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize_column_name(col_name: str) -> List[str]:
        """
        Split a column name into lowercase tokens, handling snake_case,
        kebab-case, space-separated, and camelCase/PascalCase names alike.
        e.g. 'TransactionDate' -> ['transaction', 'date']
             'order_number'    -> ['order', 'number']
             'ECO_SCORE'       -> ['eco', 'score']
             'vendorID'        -> ['vendor', 'id']
        """
        s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', col_name)
        s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s)
        s = re.sub(r'[^0-9a-zA-Z]+', '_', s)
        return [t.lower() for t in s.split('_') if t]

    @staticmethod
    def singularize(token: str) -> str:
        """Naive singularization so 'Vendors'/'Notes' still match keywords
        stored as 'vendor'/'note' (and vice versa) without resorting to
        free substring matching."""
        if len(token) <= 3:
            return token
        if token.endswith('ies'):
            return token[:-3] + 'y'
        if token.endswith(('sses', 'xes', 'zes', 'ches', 'shes')):
            return token[:-2]
        if token.endswith('s') and not token.endswith('ss'):
            return token[:-1]
        return token

    @classmethod
    def check_keyword_match(cls, col_name: str, keywords: List[str]) -> bool:
        """
        Word-boundary-aware keyword match against a tokenized column name.

        Avoids false positives like 'no' matching inside 'Notes', 'id'
        inside 'Video', 'key' inside 'Turkey', 'age' inside 'Average',
        'order' inside 'Border', 'count' inside 'Account', 'rate' inside
        'Corporate', or 'source' inside 'Resource' - all names that happen
        to contain a keyword as a fragment but have nothing to do with it.

        Tokens are compared both as-is and singularized so real matches
        (e.g. 'Vendors' -> keyword 'vendor', or 'Unit' -> keyword 'units')
        still work. Multi-word keywords like 'order_number' are matched as
        a contiguous run of tokens.
        """
        tokens = cls.tokenize_column_name(col_name)
        if not tokens:
            return False
        norm_tokens = [cls.singularize(t) for t in tokens]

        for kw in keywords:
            kw_tokens = [t for t in re.split(r'[\s_-]+', kw.lower()) if t]
            if not kw_tokens:
                continue
            norm_kw_tokens = [cls.singularize(t) for t in kw_tokens]
            n = len(kw_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == kw_tokens or norm_tokens[i:i + n] == norm_kw_tokens:
                    return True
        return False

    @staticmethod
    def is_string_or_object_dtype(series: pd.Series) -> bool:
        """Check if series is string, object, or categorical dtype."""
        dtype_str = str(series.dtype).lower()
        return (
            pd.api.types.is_object_dtype(series) or
            isinstance(series.dtype, pd.CategoricalDtype) or
            'string' in dtype_str or
            'str' in dtype_str or
            dtype_str == 'object'
        )

    @staticmethod
    def date_parse_success_ratio(series: pd.Series, sample_size: int = 100) -> float:
        """Fraction of a sampled, non-null series that pandas can parse as a date."""
        values = series.astype(str)
        if len(values) == 0:
            return 0.0
        sample = values.sample(min(sample_size, len(values)), random_state=42)
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
        except (ValueError, TypeError):
            return 0.0
        return parsed.notna().mean()

    # ------------------------------------------------------------------
    # Config-dependent detection
    # ------------------------------------------------------------------

    def detect_value_patterns(self, series: pd.Series, sample_size: int = 100) -> Dict[str, float]:
        """Detect value patterns in a column by sampling; returns the match
        ratio for every configured pattern (not just the first match)."""
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return {}

        sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)
        pattern_matches = {}

        for pattern_name, pattern in self.config.patterns.items():
            matches = sample.str.match(pattern, flags=re.IGNORECASE, na=False).sum()
            pattern_matches[pattern_name] = matches / len(sample)

        return pattern_matches

    def detect_date(
        self,
        non_null: pd.Series,
        is_string_dtype: bool,
        is_numeric_dtype: bool,
        patterns: Dict[str, float],
        is_date_keyword: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Robust date/time detection that runs ahead of numeric and identifier
        classification. A column like "TransactionDate" or "RefDate" would
        otherwise get swallowed by identifier/key keyword matching (which
        also matches substrings like "transaction" or "ref"), and a column
        of daily order dates stored as strings would get swallowed by the
        high-cardinality "looks like an ID" check - both because the date
        check ran too late. Combines value-pattern matching (works
        regardless of column name) with keyword-gated parse confirmation
        (catches formats the regexes don't cover, e.g. 'January 5, 2024'),
        plus numeric epoch/year detection when the name hints at it.
        """
        cfg = self.config

        if is_string_dtype:
            pattern_score = max(
                patterns.get('date_iso', 0),
                patterns.get('date_eu', 0),
                patterns.get('datetime', 0),
                patterns.get('date_month_name', 0),
            )
            if pattern_score > cfg.date_pattern_score_threshold:
                return {
                    'role': ColumnRole.DATE,
                    'semantic_type': SemanticType.DATETIME_STRING,
                    'confidence': min(0.95, 0.6 + pattern_score * 0.35),
                    'reason': f'{pattern_score:.0%} of sampled values match a date pattern'
                }

            if is_date_keyword:
                ratio = self.date_parse_success_ratio(non_null)
                if ratio > cfg.date_parse_success_threshold:
                    return {
                        'role': ColumnRole.DATE,
                        'semantic_type': SemanticType.DATETIME_STRING,
                        'confidence': min(0.9, 0.5 + ratio * 0.4),
                        'reason': f'Date keyword + {ratio:.0%} of values parse as valid dates'
                    }

        elif is_numeric_dtype and is_date_keyword and len(non_null) > 0:
            if non_null.between(cfg.year_min, cfg.year_max).mean() > 0.9:
                return {
                    'role': ColumnRole.DATE,
                    'semantic_type': SemanticType.YEAR,
                    'confidence': 0.8,
                    'reason': 'Date keyword + values fall within a plausible year range'
                }
            if non_null.between(cfg.unix_ms_min, cfg.unix_ms_max).mean() > 0.9:
                return {
                    'role': ColumnRole.DATE,
                    'semantic_type': SemanticType.UNIX_TIMESTAMP_MILLIS,
                    'confidence': 0.75,
                    'reason': 'Date keyword + values fall within a plausible Unix-ms range'
                }
            if non_null.between(cfg.unix_sec_min, cfg.unix_sec_max).mean() > 0.9:
                return {
                    'role': ColumnRole.DATE,
                    'semantic_type': SemanticType.UNIX_TIMESTAMP_SECONDS,
                    'confidence': 0.75,
                    'reason': 'Date keyword + values fall within a plausible Unix-seconds range'
                }

        return None

    def detect_column(self, series: pd.Series, col_name: str) -> ColumnDetectionResult:
        """
        Detect the canonical role and semantic type of a single column.

        This is the merged decision tree: it uses `type_detector`'s more
        nuanced numeric/string cardinality logic as the base (it was the
        more thoroughly hardened of the two originals - tokenized keyword
        matching, priority-ordered date detection), and folds in
        capabilities that only `data_profiler` used to have: boolean-dtype
        detection, the KEY-vs-IDENTIFIER >90%-unique distinction, and
        email/phone pattern recognition (previously computed but never
        actually used by `type_detector`).

        One behavior change from the old `DataProfiler` path is worth
        calling out: `DataProfiler` used a flat "unique_percent > 20 ->
        measure, else dimension" rule for numeric columns with no keyword
        match. That threshold is replaced here by `type_detector`'s finer
        cardinality tiers (a numeric column with, say, 15% unique values
        is now a measure rather than a dimension) - a deliberate
        consolidation onto the more discriminating rule, not an accident.
        """
        cfg = self.config
        non_null = series.dropna()
        total_count = len(series)
        null_count = int(series.isnull().sum())
        null_ratio = null_count / total_count if total_count > 0 else 0.0

        if len(non_null) == 0:
            return ColumnDetectionResult(
                role=ColumnRole.UNKNOWN,
                semantic_type=SemanticType.EMPTY,
                confidence=0.0,
                reason='All values are null',
                unique_count=0,
                unique_ratio=0.0,
                null_count=null_count,
                null_ratio=null_ratio,
            )

        unique_count = int(non_null.nunique())
        unique_ratio = unique_count / len(non_null)

        def make(role: ColumnRole, semantic_type: SemanticType, confidence: float, reason: str) -> ColumnDetectionResult:
            return ColumnDetectionResult(
                role=role, semantic_type=semantic_type, confidence=confidence, reason=reason,
                unique_count=unique_count, unique_ratio=unique_ratio,
                null_count=null_count, null_ratio=null_ratio,
            )

        def key_or_identifier(ratio: float) -> ColumnRole:
            return ColumnRole.KEY if ratio * 100 > cfg.key_unique_percent_threshold else ColumnRole.IDENTIFIER

        # Keyword-based flags
        is_date_kw = self.check_keyword_match(col_name, cfg.date_keywords)
        is_key_kw = self.check_keyword_match(col_name, cfg.key_keywords)
        is_measure_kw = self.check_keyword_match(col_name, cfg.measure_keywords)
        is_dimension_kw = self.check_keyword_match(col_name, cfg.dimension_keywords)
        is_url_kw = self.check_keyword_match(col_name, cfg.url_keywords)
        is_text_kw = self.check_keyword_match(col_name, cfg.text_keywords)

        # Dtype flags
        is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(series)
        is_bool_dtype = pd.api.types.is_bool_dtype(series)
        is_numeric_dtype = pd.api.types.is_numeric_dtype(series) and not is_bool_dtype
        is_string_dtype = self.is_string_or_object_dtype(series)

        patterns: Dict[str, float] = {}
        if is_string_dtype:
            patterns = self.detect_value_patterns(series)

        # === DETECTION LOGIC (priority-based) ===

        # 1. Datetime dtype - unambiguous ground truth, checked first.
        if is_datetime_dtype:
            return make(ColumnRole.DATE, SemanticType.DATETIME, 0.95, 'Datetime dtype detected')

        # 2. Boolean dtype (pandas treats bool as numeric, so this must run
        # before the numeric branch or booleans get swallowed by it).
        if is_bool_dtype:
            return make(ColumnRole.BOOLEAN, SemanticType.BOOLEAN, 0.95, 'Boolean dtype detected')

        # 3. URL detection
        if is_url_kw or patterns.get('url', 0) > 0.5:
            return make(ColumnRole.IDENTIFIER, SemanticType.URL, 0.9, 'URL pattern detected')

        # 4. Email pattern detection (can't collide with dates - requires '@').
        if patterns.get('email', 0) > cfg.email_pattern_threshold:
            return make(ColumnRole.IDENTIFIER, SemanticType.EMAIL, 0.9, 'Email pattern detected')

        # 5. Robust date detection - runs BEFORE the phone check (phone's
        # loose "7+ digits/spaces/dashes/parens" regex would otherwise also
        # match ISO date strings like "2023-01-01" and misclassify them),
        # and BEFORE numeric/identifier-key logic so dates are never
        # miscategorized just because their name also contains an
        # identifier-like token (e.g. "TransactionDate", "RefDate" both
        # match KEY_KEYWORDS) or because they have high cardinality as
        # strings (e.g. daily order dates almost all unique).
        date_result = self.detect_date(non_null, is_string_dtype, is_numeric_dtype, patterns, is_date_kw)
        if date_result:
            return make(date_result['role'], date_result['semantic_type'],
                        date_result['confidence'], date_result['reason'])

        # 6. Phone pattern detection. Requires the value NOT to also look
        # like a plain numeric ID (a bare digit string like "1234567"
        # matches phone's loose regex too; id_numeric is the more
        # specific/correct read in that case).
        if patterns.get('phone', 0) > cfg.phone_pattern_threshold and patterns.get('id_numeric', 0) < 0.5:
            return make(ColumnRole.IDENTIFIER, SemanticType.PHONE, 0.85, 'Phone number pattern detected')

        # 7. NUMERIC columns - checked before generic keyword matching
        if is_numeric_dtype:
            if unique_ratio > cfg.min_identifier_unique_ratio and is_key_kw:
                return make(key_or_identifier(unique_ratio), SemanticType.ID_NUMERIC, 0.9,
                            f'Numeric key with {unique_ratio:.1%} unique')

            if unique_ratio > cfg.high_uniqueness_numeric_min_ratio and unique_count > cfg.high_uniqueness_numeric_min_count:
                return make(key_or_identifier(unique_ratio), SemanticType.ID_NUMERIC, 0.8,
                            f'High uniqueness numeric ({unique_ratio:.1%})')

            if unique_count <= cfg.low_cardinality_numeric_max_unique and unique_ratio < cfg.low_cardinality_numeric_max_ratio:
                return make(ColumnRole.DIMENSION, SemanticType.NUMERIC_CATEGORICAL, 0.85,
                            f'Low cardinality numeric ({unique_count} unique values)')

            if is_measure_kw:
                is_currency = self.check_keyword_match(col_name, cfg.currency_keywords)
                semantic = SemanticType.CURRENCY if is_currency else SemanticType.NUMERIC_CONTINUOUS
                return make(ColumnRole.MEASURE, semantic, 0.9, f'Numeric with measure keyword "{col_name}"')

            if unique_ratio > cfg.default_measure_min_ratio:
                return make(ColumnRole.MEASURE, SemanticType.NUMERIC_CONTINUOUS, 0.75,
                            'Numeric with variance (default measure)')

            return make(ColumnRole.DIMENSION, SemanticType.NUMERIC_CATEGORICAL, 0.7,
                        f'Low variance numeric ({unique_count} unique)')

        # 8. Identifier/Key detection for non-numeric
        if is_key_kw and unique_ratio > 0.5:
            return make(key_or_identifier(unique_ratio), SemanticType.ID, 0.9,
                        f'Key keyword + high uniqueness ({unique_ratio:.1%})')

        # 9. String/categorical columns
        if is_string_dtype:
            if unique_ratio > cfg.min_identifier_unique_ratio and total_count > 100:
                if is_text_kw:
                    return make(ColumnRole.TEXT, SemanticType.TEXT_DESCRIPTION, 0.85,
                                'High uniqueness text field')
                return make(key_or_identifier(unique_ratio), SemanticType.ID_TEXT, 0.8,
                            f'High uniqueness ({unique_ratio:.1%})')

            if is_dimension_kw:
                return make(ColumnRole.DIMENSION, SemanticType.CATEGORICAL, 0.9,
                            f'Dimension keyword match in "{col_name}"')

            if unique_count <= cfg.max_dimension_unique_count and unique_ratio <= cfg.max_dimension_unique_ratio:
                return make(ColumnRole.DIMENSION, SemanticType.CATEGORICAL, 0.85,
                            f'Low cardinality string ({unique_count} unique, {unique_ratio:.1%})')

            if unique_count <= cfg.max_dimension_unique_count:
                return make(ColumnRole.DIMENSION, SemanticType.CATEGORICAL_HIGH, 0.7,
                            f'Medium cardinality ({unique_count} unique)')

            avg_len = non_null.astype(str).str.len().mean()
            semantic = SemanticType.TEXT_LONG if avg_len and avg_len > 100 else SemanticType.TEXT
            return make(ColumnRole.TEXT, semantic, 0.6, 'High cardinality text')

        # 10. Final fallback: low-cardinality non-string/numeric columns
        # (rare exotic dtypes) read as a dimension; otherwise unknown.
        if unique_ratio < 0.5:
            return make(ColumnRole.DIMENSION, SemanticType.UNKNOWN, 0.5, 'Low cardinality fallback')

        return make(ColumnRole.UNKNOWN, SemanticType.UNKNOWN, 0.3, 'Could not determine type')


# A ready-to-use engine instance with default rules, for callers (and
# legacy facades) that don't need custom configuration.
DEFAULT_DETECTOR = TypeDetector()


def _deprecated(old_name: str, new_reference: str):
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version; "
        f"use {new_reference} instead.",
        DeprecationWarning,
        stacklevel=3,
    )

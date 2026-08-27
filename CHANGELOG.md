# Changelog

Audit log of all changes made to this toolkit across Claude Code sessions.
Newest entries first. Each entry names the files touched and the specific
bugs fixed, so a later session can tell *why* something looks the way it
does before changing it again.

There is no git history backing this (see "Recommendation" at the bottom
of this file), so this document is currently the only record of intent
behind the code. Keep it updated when you make further changes.

---

## 10. Large-dataset report-hang fix (payload size)

**Symptom:** on a ~97k-row dataset, `analysis_report.html` grew to 15.6MB
and hung on open, even after fix #9 below.

**Root causes found by inspecting the actual generated output directory:**
- Plotly's histogram/box traces embed *raw* column values and bin them
  client-side in JS. An unsampled 97k-row numeric column produced a
  2.1-2.2MB chart file, ×3 distribution charts.
- Dimension-hierarchy detection had no cardinality ceiling on the "child"
  column. A coarse column paired with a legitimately-DIMENSION-classified
  but high-cardinality column (e.g. `Article Description`, ~49% unique)
  was detected as a valid hierarchy, producing an 8.78MB sunburst chart
  with thousands of leaf segments.

**Fixes:**
- `generic_data_viz/chart_generator.py`: added `DISTRIBUTION_SAMPLE_CAP =
  20000`. Distribution charts (both Plotly and Matplotlib paths) now plot
  a capped random sample; insight subtitle stats (median/min/max) and the
  log-scale decision still use the *full* column so they stay accurate.
- `generic_data_viz/core/data_profiler.py`: added `MAX_HIERARCHY_CHILD_UNIQUE
  = 100` and applied it in `_is_functional_dependency` - a "hierarchy"
  whose child has more than 100 distinct values is no longer detected,
  regardless of whether the functional-dependency math holds.
- `generic_data_viz/chart_generator.py`: added a matching defensive cap
  (150 leaf rows) directly in `_create_hierarchy_drilldown_chart`, in case
  a hierarchy tuple ever reaches it some other way.
- New `tests/test_large_dataset_safety.py` (7 tests) - reproduces the
  exact bug pattern (large numeric column; a coarse dimension paired with
  a ~49%-unique description-like dimension) and asserts both fixes hold,
  plus a sanity check that a genuine small hierarchy (Category ->
  Subcategory) is still detected.

Verified: reproduced the reporter's exact scenario (96,920 rows, a
description-like high-cardinality dimension) - report size dropped from a
projected 15MB+ to 1.1MB, loaded in ~1s, no console errors, hierarchy
chart correctly no longer generated.

---

## 9. Report-hang fix: N charts = N copies of Plotly.js

**Symptom:** with ~19 charts, opening `analysis_report.html` was slow and
eventually became unresponsive.

**Root cause:** each chart was embedded via `<iframe src="charts/x.html">`.
Every iframe is a fully separate document, so each one independently
downloaded and initialized its own ~1.4MB copy of Plotly.js in an isolated
JS realm - 19 charts meant 19 full library loads on page open.

**Fix (`generic_data_viz/visualizer.py`):**
- `_generate_html_report` no longer embeds charts via `<iframe>`. It reads
  each standalone chart's saved HTML, extracts the embeddable
  `<div>+<script>` fragment (`_extract_plotly_fragment`, using
  `_BODY_RE`/`_PLOTLY_CDN_SCRIPT_RE`), strips that file's own CDN
  `<script>` tag, and inlines the fragment directly into the report.
  Exactly one shared CDN `<script>` tag is emitted once in the report's
  `<head>`.
- Falls back to the old `<iframe>` embedding per-chart only if fragment
  extraction fails for that specific file (defensive, not expected to
  trigger in normal operation).
- Individual standalone chart files (`charts/*.html`) are unchanged and
  still fully self-contained if opened directly.
- Updated `tests/test_interactive_reports.py`: replaced the outdated
  iframe-embedding assertion with `test_report_loads_plotly_js_exactly_once`
  (checks `cdn.plot.ly` appears exactly once regardless of chart count)
  and `test_report_embeds_charts_inline_not_via_iframe`.

---

## 8. Axis labels must name what they measure

**Symptom (user-reported):** a distribution chart's Y axis just said
"Count" - ambiguous, doesn't say count of what.

**Fix (`generic_data_viz/chart_generator.py`):** added
`RECORD_COUNT_LABEL = 'Number of Records'` and used it as
`f'{RECORD_COUNT_LABEL} ({column})'` everywhere a row-count axis appears:
- Distribution histogram Y axis (Plotly + static): was `Count`, now e.g.
  `Number of Records (Units Sold)`. Also fixed the static histogram/box
  X axis, which was unlabeled outside the log-scale branch.
- Categorical bar chart X axis and hover, donut hover: same pattern.
- Overview dashboard: `Missing Count` -> `Missing Values (Count of Rows)`,
  `MB` -> `Memory Usage (MB)`.
- Missing-values chart: `Missing %` -> `Missing Values (% of Total Rows)`,
  added a `Column` Y-axis title.
- Correlation heatmap: added a colorbar label (`Correlation Coefficient
  (r)`) and a proper hover template - previously unlabeled entirely.
- New `TestAxisLabelsNameWhatTheyMeasure` (6 tests) in
  `tests/test_business_charts.py`, including Matplotlib-axis inspection
  via mocking `_save_chart` to intercept the figure before it's closed.

---

## 7. Executive/business-report chart styling upgrade

Rewrote `generic_data_viz/chart_generator.py` around modular, reusable
styling/insight helpers, applied to both the Plotly and Matplotlib chart
builders:

- **New color palette** (`COLORS` dict): `good`/`alert`/`neutral`/
  `warning` are reserved exclusively for functional meaning (on-target vs.
  below-target trend, missing-data severity tiers, "Other" bucket) - never
  used decoratively.
- **Chart-type selection by cardinality:** categorical dimensions with
  <=5 unique values now render as a **donut** (composition view);
  >5 renders as a **horizontal bar, sorted descending, with everything
  past the Top 10 collapsed into an "Other" bucket**
  (`_summarize_categorical`). Both engines.
- **Time series:** gained an automatic average/threshold reference line
  (`add_hline`) plus a marker on the latest point colored green (at/above
  average) or red (below). *Bug caught during my own visual QA and fixed*:
  the marker color was initially computed from first-vs-last trend, which
  didn't match what the average line visually implied - it now compares
  the latest value to the same average the line draws, while the
  subtitle's up/down arrow independently reports overall period change.
- **Micro-insight subtitles** above every chart title
  (`_categorical_insight`, `_distribution_insight`, `_timeseries_insight`,
  `_correlation_insight`) - e.g. "'South' leads with 29% share",
  "Peak $3,066 in Dec 2023 - up 11% over period".
- **Kind-aware formatting** (`_format_kind`, `_format_value`,
  `_axis_decoration`, `_hover_num_token`): currency/percentage/count
  detection (prefers the type-engine's `semantic_type` when available,
  falls back to column-name keywords), applied to hover tooltips and axis
  ticks. Deliberately uses `tickprefix`/`ticksuffix` decoration rather than
  d3's `%` tickformat, since that would incorrectly multiply an
  already-0-100-scale percentage by 100 again.
- **Centralized theme** (`_apply_business_theme`, called once from
  `_save_chart_html`): lightened gridlines, legible font, horizontal tick
  labels, clean legend placement.
- New `tests/test_business_charts.py` (17 tests at the time, later +6 for
  axis labels = 22).

---

## 6. Interactive charts became the default; explicit static/PDF fallback added

- `GenericDataVisualizer` and `GenericVisualizationGenerator` defaults
  changed: `engine="plotly"`, `interactive=True` (previously `"seaborn"` /
  `False`). Explicitly passing the old values still produces the exact old
  static-PNG behavior - only callers relying on the *defaults* are
  affected, which is the intended upgrade.
- Added an explicit `static: bool = False` constructor parameter with
  **unconditional veto power** over `engine`/`interactive`. This fixes a
  real design bug: the previous `requested_plotly = interactive or engine
  == 'plotly'` OR-coupling meant there was no way to force static output
  if either flag leaned interactive.
- CLI (`generic_data_viz/__main__.py`): `--engine` default is now `None`,
  resolved at runtime to `plotly` (or `matplotlib` if `--static` is set);
  added `--static` flag; `--interactive` kept registered for backward
  compatibility (now a no-op, since interactive is already the default).
- Added `_create_hierarchy_drilldown_chart` (sunburst, Plotly-only) driven
  by `DataProfile.hierarchies` - data the profiler already computed but
  that nothing had ever consumed before this.
- **Bug found while wiring this up:** `DataProfiler._is_functional_dependency`
  /`_detect_hierarchies` had the grouping direction inverted, so real
  hierarchies (e.g. Category -> Subcategory) were never actually detected
  before this fix. See entry 10 above for the *second* hierarchy bug
  (cardinality) found later.
- `plotly` promoted from optional to a core dependency in `pyproject.toml`
  and `requirements.txt` (with a documented runtime fallback to Matplotlib
  if it's ever missing anyway).
- New `tests/test_interactive_reports.py`.

---

## 5. Unified type-classification engine (major refactor)

Eliminated a maintenance-risk duplication: `type_detector.py` and
`core/data_profiler.py` each carried their own keyword lists, thresholds,
and decision tree for classifying a column as measure/dimension/date/
identifier/etc., and had silently drifted apart (different keyword
coverage, and a real bug: one used non-null count for "uniqueness", the
other used total row count).

- **New `generic_data_viz/core/type_engine.py`** - single source of truth:
  - `DetectionConfig` (dataclass): all keyword lists, regex patterns,
    thresholds in one place.
  - `ColumnRole` / `SemanticType` (enums): unified vocabulary, superset of
    both originals' concepts (some members like `ADDRESS`,
    `CATEGORICAL_ORDINAL` are reserved/unused, kept for name compatibility
    with the old `data_profiler.DataType`).
  - `TypeDetector`: the actual decision tree, merging both originals'
    logic and adding capabilities neither had alone (boolean-dtype
    detection, email/phone pattern recognition - previously computed by
    `type_detector.py` but never used, currency semantic-type detection,
    a KEY-vs-IDENTIFIER >90%-unique split).
  - `ColumnDetectionResult` (dataclass): canonical per-column result.
- `type_detector.py`'s `AutoDataTypeDetector` is now a thin, backward-
  compatible dict-shaped facade over `TypeDetector`
  (`detect_column_type`/`analyze_dataset`/`get_summary`/`print_analysis`
  signatures unchanged). Canonical `KEY`/`BOOLEAN` roles are folded back
  to `'identifier'`/`'dimension'` strings to match the historical 6-value
  vocabulary.
- `core/data_profiler.py`'s `DataProfiler` delegates to the same
  `TypeDetector`; `ColumnRole`/`DataType` are now re-exports of the
  engine's enums (same objects, not copies).
- Old private methods on both classes (`_check_keyword_match`,
  `_tokenize_column_name`, `_detect_semantic_type`, `_determine_role`,
  etc.) kept as thin shims that emit `DeprecationWarning` and delegate to
  the shared engine, per explicit instruction to warn rather than silently
  remove.
- **Bugs found and fixed during this refactor:**
  - The null-denominator drift mentioned above (now one consistent
    definition: unique_ratio = unique_count / non-null count).
  - A phone-vs-date pattern collision *introduced by this refactor itself*
    (phone's loose "7+ digits/spaces/dashes" regex also matched ISO date
    strings) - caught by the test suite, fixed by re-ordering detection so
    date-checks run before the phone check.
- `core/__init__.py` and top-level `generic_data_viz/__init__.py` updated
  to export `TypeDetector`, `DetectionConfig`, `ColumnDetectionResult`.
- New `tests/test_type_classification.py` (20 tests) - includes
  `TestLegacyEntryPointConsistency`, which runs both legacy entry points
  over the same DataFrame and asserts they agree on every column (the
  core claim of the refactor).
- **Disclosed, intentional behavior change:** `DataProfiler` used to use a
  flat "unique_percent > 20 -> measure, else dimension" rule for numeric
  columns with no keyword match. That's replaced by `type_detector`'s
  finer cardinality tiers (deliberate consolidation onto the more
  discriminating rule).
- **Bonus fix:** `DataProfiler` previously had no `TEXT_KEYWORDS` concept,
  so free-text columns like `Notes` landed as `UNKNOWN`. Now shares
  `type_detector`'s keyword set, so they correctly land as `TEXT`.

---

## 4. Word-boundary-aware keyword matching (fixes false-positive column-type collisions)

**Symptom:** naive substring matching (`kw in col_lower`) caused real
misclassifications: `'no'` matched inside `'Notes'` (-> wrongly flagged as
an identifier), `'id'` inside `'Video'`, `'key'` inside `'Turkey'`,
`'count'` inside `'Account'`, `'rate'` inside `'Corporate'`, `'source'`
inside `'Resource'`, `'age'` inside `'Average'`, `'order'` inside `'Border'`,
`'set'` inside `'Asset'`.

**Fix:** replaced substring matching with proper tokenization (splits
snake_case, kebab-case, spaces, camelCase/PascalCase) plus naive
singularization, matching whole tokens only (including contiguous-token
matching for compound keywords like `order_number`). Applied to both
`type_detector.py` and `data_profiler.py` (independently at the time;
later unified into one implementation - see entry 5).

---

## 3. Reliable date detection

**Symptom:** date columns stored as strings (`OrderDate`, `TransactionDate`,
`RefDate`) were being misclassified as `identifier` instead of `date`.

**Root causes:**
- High-cardinality-string -> identifier check ran *before* the date check.
- Identifier/key-keyword matching ran before the date check too, and some
  date-ish column names (e.g. `TransactionDate`) incidentally also matched
  identifier keywords (`transaction`, `ref`, etc.) via substring matching
  (see entry 4 - this compounded the problem until tokenization landed).

**Fixes (`generic_data_viz/type_detector.py`):**
- Reordered `detect_column_type`'s priority list so date detection runs
  before both the numeric-ID branch and the string-identifier branch.
- Added pattern-based date detection (ISO, EU `DD/MM/YYYY`, month-name
  formats like "Jan 5, 2024") that works regardless of column name.
- Added a keyword-gated `pd.to_datetime` parse-success fallback for
  formats the regexes miss.
- Added numeric epoch/year detection (a `Year` column of 2020-2023, or
  Unix ms/seconds timestamps) when the column name hints at it.

**Bugs found as a side effect of this fix:**
- `data_profiler.py`'s semantic-type pattern loop picked the first regex
  match by dict order, and the loose `phone` pattern was checked before
  the date patterns, mislabeling every date-like string as `PHONE`.
  Reordered the pattern dict.
- Once dates started actually reaching the time-series chart code (they
  never had before, always being misfiled as identifiers), it turned out
  `resample('M')` is broken on modern pandas (`'M'` deprecated in favor of
  `'ME'`). Fixed with a version-tolerant fallback
  (`_resample_series` in `chart_generator.py`) across the pandas range the
  toolkit supports.

---

## 2. Interactive Plotly charts wired in + Dashboard Recommender wired in

(Two related asks handled together early in the project.)

- **Dashboard Recommender:** `DashboardRecommender` was instantiated in
  `GenericDataVisualizer.__init__` but `.recommend()` was never called -
  dead code. Wired into `analyze()` as pipeline step `[5/7]`. Added
  `dashboard` field to `AnalysisResult` (`models.py`). HTML/TXT reports
  now include "Dashboard Recommendations" and "Detected KPIs" sections.
  CLI prints a dashboard summary.
- **Interactive charts:** added Plotly chart-building methods in
  `chart_generator.py` parallel to the existing Matplotlib ones (overview,
  distribution, categorical, correlation heatmap, scatter, missing
  values, time series), gated by a `use_plotly` flag computed from
  `engine`/`interactive` at construction time. This is the foundation
  entries 6-10 above built on and fixed further.

---

## 1. Initial codebase analysis

No code changes - read through the whole toolkit and reported its
architecture, strengths, and gaps (dead code paths, the type-classification
duplication that became entry 5, missing tests, `max_charts` not actually
enforced as a hard cap in `generate_all_charts` - **this one is still
unaddressed, see "Known open items" below**).

---

## Known open items (flagged, not yet fixed - deliberately out of scope so far)

- `GenericVisualizationGenerator.generate_all_charts` does not actually
  enforce `VisualizationPlan.total_charts`/`max_charts` as a hard cap; it
  generates the full quota from every phase (distributions, categoricals,
  correlation, relationships, quality, temporals, hierarchy) regardless.
  `max_charts` only affects planning math today, not actual output count.
- `AutoDataTypeDetector`'s identifier-keyword collision risk is much
  reduced (entry 4) but not exhaustively audited for every possible short
  token.
- A `Pandas4Warning` about `select_dtypes(include='object')` in
  `generic_data_viz/core/data_quality.py` (unrelated file, never touched)
  - pre-existing, cosmetic only.
- `examples/usage_example.py` has its own `freq='H'` deprecated pandas
  alias bug (same class of issue as the `'M'`->`'ME'` fix in entry 3, but
  in example code, not the library) - never fixed, out of scope.
- `numpy` import in `data_profiler.py` is unused (pre-existing, predates
  all sessions above) - left alone.
- No automated check that `DISTRIBUTION_SAMPLE_CAP`/
  `MAX_HIERARCHY_CHILD_UNIQUE` are well-chosen for every possible dataset
  shape - they're reasonable defaults verified against the one large
  real-world dataset reported, not exhaustively tuned.

## Current defaults worth knowing before touching anything

- `GenericDataVisualizer(output_dir, engine="plotly", interactive=True,
  static=False)` - interactive HTML charts by default; pass
  `static=True` for the old static-PNG behavior (headless/PDF workflows).
- CLI: `python -m generic_data_viz --input ... --output ...` defaults to
  interactive; add `--static` to opt out.
- Test suite: `python -m unittest discover -s tests -v` (64 tests as of
  entry 10, all passing). Run this before and after any change.
- No git repository yet - see recommendation in the handoff message this
  changelog was created alongside.

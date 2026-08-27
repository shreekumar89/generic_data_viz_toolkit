# generic_data_viz_toolkit

Domain-agnostic, automated data visualization/analysis toolkit. Point it
at any CSV/Excel/Parquet/JSON file; it auto-detects column types
(measure/dimension/date/identifier), profiles data quality, generates
business insights, recommends dashboards/KPIs, and produces an HTML
report with interactive Plotly charts (or static Matplotlib/Seaborn PNGs
for headless/PDF workflows).

**Before changing anything, read `CHANGELOG.md`** - it has a detailed,
chronological record of every fix and why it was made. Several past bugs
were subtle (inverted comparison direction, OR-coupled flags, dict
iteration order determining regex priority) and easy to reintroduce by
"simplifying" code that looks redundant but isn't. Check the changelog
before assuming something is dead code or an oversight.

## Architecture

```
generic_data_viz/
├── visualizer.py          # GenericDataVisualizer - main orchestrator, HTML/TXT report generation
├── chart_generator.py      # GenericVisualizationGenerator - all chart building (Plotly + Matplotlib)
├── type_detector.py        # AutoDataTypeDetector - legacy dict-API facade over core/type_engine.py
├── models.py                # AnalysisMode, ChartCategory, VisualizationPlan, AnalysisResult
├── __main__.py               # CLI
└── core/
    ├── type_engine.py       # TypeDetector/DetectionConfig - SINGLE SOURCE OF TRUTH for column
    │                         # role/semantic-type classification. Both type_detector.py and
    │                         # data_profiler.py delegate here. Do not re-add keyword lists or
    │                         # a decision tree to either of those two files - that duplication
    │                         # is exactly what caused past drift bugs (see CHANGELOG entry 5).
    ├── data_profiler.py      # DataProfiler - delegates classification to type_engine.py;
    │                         # also does statistics, hierarchy detection, data dictionary
    ├── data_quality.py       # DataQualityAnalyzer - missing/duplicate/outlier scoring
    ├── insight_engine.py     # InsightEngine - natural-language business insights
    └── dashboard_recommender.py  # DashboardRecommender - KPI/dashboard suggestions

tests/
├── test_type_classification.py   # type_engine.py + both legacy facades agree
├── test_interactive_reports.py   # Plotly chart generation, single shared CDN script
├── test_business_charts.py       # chart styling, insight subtitles, axis labels
└── test_large_dataset_safety.py  # payload-size guards for large datasets
```

## Current defaults (changed from the project's original design - see CHANGELOG entry 6)

- `GenericDataVisualizer(output_dir, engine="plotly", interactive=True, static=False)`
  - Interactive HTML charts are the default. `static=True` forces the old
    static-PNG behavior (for headless environments or PDF conversion) and
    **unconditionally overrides** `engine`/`interactive` - don't reintroduce
    the old `interactive or engine == 'plotly'` OR-only coupling.
- CLI mirrors this: `--static` opts out of interactive; `--interactive` is
  kept registered but is a no-op (interactive is already default).
- `python -m generic_data_viz --input data.csv --output ./results`

## Before making changes

1. Run the test suite first to confirm a clean baseline:
   ```
   python -m unittest discover -s tests -v
   ```
2. Read `CHANGELOG.md` for context on the area you're about to touch.
3. After changes, re-run the full suite, and if you touched chart
   generation or the HTML report, actually generate a report from a
   dataset with a few thousand+ rows and open it - several past bugs
   (report-hang, 15MB payloads) only showed up at real data scale, not in
   small unit-test fixtures.
4. Update `CHANGELOG.md` with a new dated/numbered entry for whatever you
   change, following the existing entries' format (symptom/root cause/fix/
   files touched). This file is the project's only change history (see
   "No git repository" below) - keep it current.

## Known open items (see CHANGELOG.md bottom for the full list)

- `max_charts` is not actually enforced as a hard cap in
  `GenericVisualizationGenerator.generate_all_charts` - it only affects
  the planning math today.
- No automated tuning of `DISTRIBUTION_SAMPLE_CAP` /
  `MAX_HIERARCHY_CHILD_UNIQUE` beyond the one real dataset that surfaced
  the bugs they fix.

## Git

This project is now a git repository. `CHANGELOG.md` is still the primary
record of *intent* behind each change (symptom/root cause/fix), while git
provides the diff/rollback safety net. Keep updating `CHANGELOG.md` on every
change as described above - the commit log complements it, it doesn't
replace it.

# Generic Data Visualization Toolkit

A **domain-agnostic**, fully automated data visualization and analysis toolkit that works with **ANY dataset**.

## Features

- ✅ **Automatic Data Type Detection** - Semantic type inference (measures, dimensions, dates, identifiers)
- ✅ **Comprehensive Data Quality Analysis** - Multi-dimensional scoring (completeness, consistency, validity)
- ✅ **Automated Insight Generation** - Business insights with natural language explanations
- ✅ **Interactive HTML Charts by Default** - Plotly-powered hover tooltips, pan/zoom, legend toggling, and click-to-drill-down (with static Matplotlib/Seaborn PNG export available for headless or PDF-conversion workflows)
- ✅ **Dashboard Recommendations** - KPI and chart suggestions
- ✅ **Modular Architecture** - Clean separation of concerns
- ✅ **Portable** - Self-contained package ready for deployment

## Installation

### Option 1: Quick Install (Recommended)

```bash
# Navigate to the toolkit folder
cd generic_data_viz_toolkit

# Install in editable mode
pip install -e .
```

### Option 2: Install Dependencies Only

```bash
pip install -r requirements.txt
```

### Option 3: System-wide Install

```bash
pip install .
```

## Quick Start

### Python API

```python
from generic_data_viz import GenericDataVisualizer

# Initialize - interactive Plotly charts by default
visualizer = GenericDataVisualizer(output_dir="./output")

# Analyze ANY dataset
result = visualizer.analyze("your_data.csv")

# Generate reports (TXT + HTML, with interactive charts embedded)
visualizer.generate_report(result, formats=['txt', 'html'])

# Access results
print(result.summary)
print(result.recommendations)

# Need static PNGs instead (headless environment, or the report will be
# converted to PDF, which can't run the JS an interactive chart needs)?
static_visualizer = GenericDataVisualizer(output_dir="./output_static", static=True)
```

### Command Line Interface

```bash
# Basic analysis - interactive HTML charts by default
python -m generic_data_viz --input data.csv --output ./results

# Full analysis with all features
python -m generic_data_viz --input data.xlsx --output ./results --full

# Static PNG charts instead (headless environments, or PDF conversion)
python -m generic_data_viz --input data.csv --output ./results --static

# Specify target column for focused analysis
python -m generic_data_viz --input data.csv --target revenue --output ./results

# Limit charts generated
python -m generic_data_viz --input data.csv --output ./results --max-charts 10
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input data file (CSV, Excel, Parquet, JSON) | Required |
| `--output` | `-o` | Output directory | `./output` |
| `--target` | `-t` | Target column for focused analysis | Auto-detected |
| `--mode` | `-m` | Analysis mode (quick, standard, full, deep) | `standard` |
| `--engine` | `-e` | Visualization engine (plotly, seaborn, matplotlib) | `plotly` (or `matplotlib` if `--static` is set) |
| `--interactive` | | Generate interactive HTML charts (this is the default now; kept for backward compatibility) | True |
| `--static` | | Generate static PNG charts instead - for headless environments or PDF conversion | False |
| `--max-charts` | | Maximum number of charts | 15 |
| `--full` | | Run full analysis with all features | False |

### Interactive charts

By default, every chart in the HTML report is an interactive Plotly figure
embedded via `<iframe>`, not a static image:

- **Hover** any data point for its exact values.
- **Pan/zoom** - scroll-wheel zoom and box-select zoom are enabled; double-click to reset.
- **Legend toggle** - click a legend entry to show/hide that series.
- **Drill-down** - when the profiler detects a dimension hierarchy (e.g. Category → Subcategory), a sunburst chart is generated where clicking a segment zooms into its children.
- Charts resize responsively with the browser window.

Pass `static=True` (Python API) or `--static` (CLI) to fall back to plain
PNG charts - useful for headless environments or when the HTML report will
be converted to PDF, since PDF renderers can't execute the JS an
interactive chart needs. Plotly falls back to static charts automatically
if it isn't installed, with a console warning.

## Package Structure

```
generic_data_viz_toolkit/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Installation script
├── pyproject.toml               # Modern packaging config
├── generic_data_viz/            # Main package
│   ├── __init__.py              # Package exports
│   ├── __main__.py              # CLI entry point
│   ├── accelerator.py           # Main visualization tool
│   ├── models.py                # Data classes and enums
│   ├── type_detector.py         # Auto data type detection
│   ├── chart_generator.py       # Visualization generation
│   ├── visualizer.py            # Main orchestrator
│   └── core/                    # Core framework modules
│       ├── __init__.py
│       ├── type_engine.py       # Shared column type/role classification engine
│       ├── data_profiler.py     # Automatic data profiling (delegates to type_engine)
│       ├── data_quality.py      # Quality analysis & scoring
│       ├── insight_engine.py    # Business insight generation
│       └── dashboard_recommender.py  # Dashboard suggestions
├── examples/
│   └── usage_example.py         # Example usage script
└── tests/
    └── test_type_classification.py  # Type/role classification tests
```

### Column type/role classification

`type_detector.AutoDataTypeDetector` and `core.data_profiler.DataProfiler`
are two independent entry points, but both delegate to a single shared
engine - `core.type_engine.TypeDetector`, configured by
`core.type_engine.DetectionConfig` - so there's exactly one place that
holds the keyword lists, cardinality thresholds, and decision tree used
to classify a column as a measure, dimension, date, identifier/key, text,
or boolean field. This eliminates the risk of the two entry points
silently drifting apart. For advanced/custom use:

```python
from generic_data_viz import TypeDetector, DetectionConfig

detector = TypeDetector(DetectionConfig(measure_keywords=[...], ...))
result = detector.detect_column(df["OrderDate"], "OrderDate")
print(result.role, result.semantic_type, result.confidence)
```

## Core Components

### 1. AutoDataTypeDetector

Automatically detects semantic data types beyond basic pandas dtypes:

- **Measures**: Numeric values to aggregate (price, quantity, score)
- **Dimensions**: Categorical grouping fields (category, region, brand)
- **Dates**: Date/time fields
- **Identifiers**: Unique IDs and keys

```python
from generic_data_viz import AutoDataTypeDetector

analysis = AutoDataTypeDetector.analyze_dataset(df)
summary = AutoDataTypeDetector.get_summary(analysis)

print(f"Measures: {summary['measures']}")
print(f"Dimensions: {summary['dimensions']}")
```

### 2. DataQualityAnalyzer

Multi-dimensional quality scoring:

- **Completeness**: Missing value analysis
- **Uniqueness**: Duplicate detection
- **Consistency**: Format and value consistency
- **Validity**: Range and format validation

```python
from generic_data_viz.core import DataQualityAnalyzer

analyzer = DataQualityAnalyzer()
report = analyzer.analyze(df)

print(f"Quality Score: {report.score.overall_score}/100")
print(f"Issues Found: {len(report.issues)}")
```

### 3. InsightEngine

Automated business insight generation:

```python
from generic_data_viz.core import InsightEngine

engine = InsightEngine()
insights = engine.generate_insights(df, target_column="revenue")

for insight in insights.insights:
    print(f"📊 {insight.headline}")
    print(f"   {insight.summary}")
```

## Output Files

The toolkit generates:

1. **Charts** (`output/charts/`):
   - Overview dashboard
   - Distribution charts for measures
   - Categorical charts for dimensions
   - Correlation heatmap
   - Relationship scatter plots
   - Missing values visualization

2. **Reports**:
   - `analysis_report.txt` - Plain text summary
   - `analysis_report.html` - Interactive HTML report with embedded charts

## Example Output

```
============================================================
GENERIC DATA VISUALIZATION ACCELERATOR
============================================================

[1/6] Detecting data types...

  📊 MEASURES (3):
     • Article Weight (409 unique, 90% conf)
     • Article Price (Euro) (56 unique, 90% conf)
     • ECO_SCORE (3,506 unique, 90% conf)

  🏷️  DIMENSIONS (10):
     • Brand (4 unique, 90% conf)
     • Category (7 unique, 90% conf)
     • Sector (3 unique, 90% conf)

[2/6] Profiling data...
[3/6] Analyzing data quality...
  - Quality score: 97.5/100
  - Issues found: 7

[4/6] Generating insights...
  - Insights discovered: 8
  - Action items: 5

[5/6] Planning visualizations...
  - Charts planned: 15

[6/6] Generating charts...

============================================================
ANALYSIS COMPLETE
============================================================
```

## Migration to Another Machine

1. **Copy the entire `generic_data_viz_toolkit` folder** to the target machine

2. **Install dependencies**:
   ```bash
   cd generic_data_viz_toolkit
   pip install -r requirements.txt
   ```

3. **Or install as a package**:
   ```bash
   pip install -e .
   ```

4. **Run**:
   ```bash
   python -m generic_data_viz --input your_data.csv --output ./results
   ```

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- plotly (default interactive-chart engine; falls back to matplotlib/seaborn PNGs with a console warning if not installed)
- squarify (optional, for treemaps)
- altair (optional, not currently used by any code path)

## License

MIT License

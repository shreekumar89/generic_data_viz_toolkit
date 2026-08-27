"""
Quick-run script for exploring a new dataset with the toolkit.

Usage:
    python run_analysis.py path/to/your_data.csv [output_dir]

Or edit DATA_PATH / OUTPUT_DIR below and run with no arguments (in VS Code:
the "Run Python File" > button, top right, or press F5).
"""

import sys
from pathlib import Path

from generic_data_viz import GenericDataVisualizer

# Defaults used only when no command-line arguments are given.
DATA_PATH = "your_data.csv"          # <-- change this, or pass a path as arg 1
OUTPUT_DIR = "./Output Analysis"     # <-- or pass an output dir as arg 2

data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR

if not Path(data_path).exists():
    sys.exit(
        f"Data file not found: {data_path}\n"
        "Pass a path as the first argument, or edit DATA_PATH in this script."
    )

viz = GenericDataVisualizer(output_dir=output_dir, engine="plotly", interactive=True)
result = viz.analyze(data_path, max_charts=15)
viz.generate_report(result, formats=["txt", "html"])

print(result.summary)
print("\nRecommendations:")
for rec in result.recommendations:
    print(f"  {rec}")

print("\nDashboard recommendations:")
for dash in result.dashboard.recommended_dashboards:
    print(f"  • {dash.title} ({len(dash.kpis)} KPIs, {len(dash.visualizations)} visuals)")

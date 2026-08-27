"""
Quick-run script for exploring a new dataset with the toolkit in VS Code.
Edit DATA_PATH below, then click the "Run Python File" ▷ button (top right)
or press F5.
"""

from generic_data_viz import GenericDataVisualizer

DATA_PATH = "C:\\Users\\shree\\Desktop\\generic_data_viz_toolkit\\generic_data_viz_toolkit\\ecoscores_full_N.xlsx"   # <-- change this to your file
OUTPUT_DIR = "C:\\Users\\shree\\Desktop\\generic_data_viz_toolkit\\generic_data_viz_toolkit\\Output Analysis\\New"

viz = GenericDataVisualizer(output_dir=OUTPUT_DIR, engine="plotly", interactive=True)
result = viz.analyze(DATA_PATH, max_charts=15)
viz.generate_report(result, formats=["txt", "html"])

print(result.summary)
print("\nRecommendations:")
for rec in result.recommendations:
    print(f"  {rec}")

print("\nDashboard recommendations:")
for dash in result.dashboard.recommended_dashboards:
    print(f"  • {dash.title} ({len(dash.kpis)} KPIs, {len(dash.visualizations)} visuals)")

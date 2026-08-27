"""
__main__.py

Command-line interface for the Generic Data Visualization Toolkit.

Usage:
    python -m generic_data_viz --input data.csv --output ./results
    python -m generic_data_viz --input data.xlsx --output ./results --full
"""

import argparse
import sys
from pathlib import Path

from .visualizer import GenericDataVisualizer
from .models import AnalysisMode


def main():
    """Command-line interface for the Generic Data Visualization Toolkit."""
    parser = argparse.ArgumentParser(
        description="Generic Data Visualization and Analysis Toolkit - Works with ANY dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data.csv --output ./results
  %(prog)s --input data.xlsx --output ./results --full
  %(prog)s --input data.csv --target revenue --output ./results
  %(prog)s --input data.csv --static --output ./results   # static PNGs for PDF/headless export
        """
    )
    
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input data file (CSV, Excel, Parquet, JSON)"
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--target", "-t",
        help="Target column for focused analysis (auto-detected if not specified)"
    )
    parser.add_argument(
        "--mode", "-m", choices=["quick", "standard", "full", "deep"],
        default="standard",
        help="Analysis mode (default: standard)"
    )
    parser.add_argument(
        "--engine", "-e", choices=["seaborn", "matplotlib", "plotly"],
        default=None,
        help="Visualization engine (default: plotly, or matplotlib if --static is set)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Generate interactive HTML charts (this is now the default; kept for "
             "backward compatibility with existing scripts)"
    )
    parser.add_argument(
        "--static", action="store_true",
        help="Generate static PNG charts instead of interactive HTML. Use this for "
             "headless environments or when the report will be converted to PDF, "
             "since PDF renderers can't execute the JS an interactive chart needs."
    )
    parser.add_argument(
        "--max-charts", type=int, default=15,
        help="Maximum number of charts to generate (default: 15)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full analysis with all features"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine mode
    mode = AnalysisMode.FULL if args.full else AnalysisMode[args.mode.upper()]

    # --static always wins, regardless of --engine/--interactive; otherwise
    # interactive Plotly charts are the default. An explicit --engine choice
    # (seaborn/matplotlib) still forces static charts even without --static,
    # matching this toolkit's historical CLI behavior.
    engine = args.engine or ("matplotlib" if args.static else "plotly")
    interactive = (not args.static) and (args.interactive or engine == "plotly")

    # Initialize visualizer
    visualizer = GenericDataVisualizer(
        output_dir=args.output,
        engine=engine,
        interactive=interactive,
        static=args.static
    )
    
    print(f"\nLoading: {input_path}")
    
    # Run analysis
    result = visualizer.analyze(
        source=input_path,
        target_column=args.target,
        mode=mode,
        max_charts=args.max_charts
    )
    
    # Generate reports
    visualizer.generate_report(result, formats=['txt', 'html'])
    
    # Print summary
    print("\n" + result.summary)
    
    print("\nRECOMMENDATIONS:")
    for rec in result.recommendations:
        print(f"  {rec}")

    print("\nDASHBOARD RECOMMENDATIONS:")
    for dash in result.dashboard.recommended_dashboards:
        print(f"  • {dash.title} ({len(dash.kpis)} KPIs, {len(dash.visualizations)} visuals)")

    print(f"\n✓ Analysis complete!")
    print(f"  Output directory: {Path(args.output).resolve()}")
    print(f"  Charts generated: {len(result.charts_generated)}")


if __name__ == "__main__":
    main()

"""
usage_example.py

Example usage of the Generic Data Visualization Toolkit.
"""

import pandas as pd
import numpy as np

# Import from the toolkit
from generic_data_viz import GenericDataVisualizer, AutoDataTypeDetector, AnalysisMode


def example_with_sample_data():
    """Example using generated sample data."""
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'CustomerID': range(1, n + 1),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n),
        'Product': np.random.choice([f'Product_{i}' for i in range(1, 20)], n),
        'Quantity': np.random.randint(1, 50, n),
        'Price': np.round(np.random.uniform(10, 500, n), 2),
        'Revenue': np.round(np.random.uniform(100, 5000, n), 2),
        'Rating': np.random.randint(1, 6, n),
        'Date': pd.date_range('2024-01-01', periods=n, freq='H'),
    })
    
    # Add some missing values
    df.loc[np.random.choice(n, 50), 'Rating'] = np.nan
    
    print("Sample Data Created:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    # 1. Automatic Type Detection
    print("\n" + "=" * 60)
    print("1. AUTOMATIC TYPE DETECTION")
    print("=" * 60)
    
    analysis = AutoDataTypeDetector.analyze_dataset(df)
    summary = AutoDataTypeDetector.get_summary(analysis)
    
    print(f"\nMeasures: {summary['measures']}")
    print(f"Dimensions: {summary['dimensions']}")
    print(f"Dates: {summary['dates']}")
    print(f"Identifiers: {summary['identifiers']}")
    
    # 2. Full Analysis
    print("\n" + "=" * 60)
    print("2. FULL ANALYSIS")
    print("=" * 60)
    
    visualizer = GenericDataVisualizer(output_dir="./example_output")
    result = visualizer.analyze(df, target_column="Revenue", max_charts=10)
    
    # 3. Generate Reports
    visualizer.generate_report(result, formats=['txt', 'html'])
    
    # 4. Print Summary
    print(result.summary)
    
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"  {rec}")
    
    print(f"\n✓ Charts generated: {len(result.charts_generated)}")
    print("✓ Reports saved to ./example_output/")


def example_with_file():
    """Example using a file."""
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("Usage: python usage_example.py <path_to_data_file>")
        print("Supported formats: CSV, Excel (.xlsx), Parquet, JSON")
        return
    
    visualizer = GenericDataVisualizer(output_dir="./file_output")
    result = visualizer.analyze(file_path)
    visualizer.generate_report(result)
    
    print(result.summary)


if __name__ == "__main__":
    example_with_sample_data()

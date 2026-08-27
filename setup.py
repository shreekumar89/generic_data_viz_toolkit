"""
Setup script for Generic Data Visualization Toolkit.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="generic-data-viz",
    version="1.0.0",
    author="Data Visualization Team",
    author_email="",
    description="A domain-agnostic, automated data visualization and analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "interactive": [
            "plotly>=5.0.0",
            "altair>=4.0.0",
        ],
        "full": [
            "plotly>=5.0.0",
            "altair>=4.0.0",
            "squarify>=0.4.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-viz=generic_data_viz.__main__:main",
        ],
    },
    include_package_data=True,
    keywords="data visualization, analytics, EDA, data quality, insights, dashboard",
)

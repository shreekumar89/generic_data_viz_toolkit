"""
data_quality.py

Comprehensive data quality analysis module with scoring, duplicate detection,
outlier identification, consistency checks, and missing value analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats


class QualityIssueType(Enum):
    """Types of data quality issues."""
    MISSING_VALUE = "missing_value"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    INVALID_FORMAT = "invalid_format"
    INVALID_RANGE = "invalid_range"
    REFERENTIAL = "referential"
    COMPLETENESS = "completeness"


class Severity(Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityIssue:
    """Represents a single data quality issue."""
    issue_type: QualityIssueType
    severity: Severity
    column: Optional[str]
    description: str
    affected_rows: int
    affected_percent: float
    sample_values: List[Any] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class QualityScore:
    """Data quality score breakdown."""
    overall_score: float
    completeness_score: float
    uniqueness_score: float
    consistency_score: float
    validity_score: float
    accuracy_score: float
    timeliness_score: float


@dataclass
class QualityReport:
    """Complete data quality report."""
    score: QualityScore
    issues: List[QualityIssue]
    column_scores: Dict[str, float]
    duplicate_count: int
    duplicate_rows: Optional[pd.DataFrame]
    outliers: Dict[str, List[int]]
    missing_patterns: Dict[str, float]
    recommendations: List[str]
    summary: str


class DataQualityAnalyzer:
    """Comprehensive data quality analysis engine."""
    
    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        duplicate_subset: Optional[List[str]] = None
    ):
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.duplicate_subset = duplicate_subset
    
    def analyze(self, df: pd.DataFrame) -> QualityReport:
        """Perform comprehensive data quality analysis."""
        issues = []
        column_scores = {}
        
        # 1. Missing value analysis
        missing_patterns = self._analyze_missing_values(df)
        missing_issues = self._create_missing_issues(df, missing_patterns)
        issues.extend(missing_issues)
        
        # 2. Duplicate detection
        duplicate_count, duplicate_rows = self._detect_duplicates(df)
        if duplicate_count > 0:
            issues.append(self._create_duplicate_issue(df, duplicate_count))
        
        # 3. Outlier detection
        outliers = self._detect_outliers(df)
        outlier_issues = self._create_outlier_issues(df, outliers)
        issues.extend(outlier_issues)
        
        # 4. Consistency checks
        consistency_issues = self._check_consistency(df)
        issues.extend(consistency_issues)
        
        # 5. Validity checks
        validity_issues = self._check_validity(df)
        issues.extend(validity_issues)
        
        # 6. Calculate column-level scores
        for col in df.columns:
            column_scores[col] = self._calculate_column_score(df[col], col, issues)
        
        # 7. Calculate overall quality score
        score = self._calculate_quality_score(df, issues, missing_patterns, duplicate_count, outliers)
        
        # 8. Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        # 9. Generate summary
        summary = self._generate_summary(df, score, issues)
        
        return QualityReport(
            score=score,
            issues=issues,
            column_scores=column_scores,
            duplicate_count=duplicate_count,
            duplicate_rows=duplicate_rows,
            outliers=outliers,
            missing_patterns=missing_patterns,
            recommendations=recommendations,
            summary=summary
        )
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze missing value patterns."""
        return (df.isna().sum() / len(df) * 100).to_dict()
    
    def _create_missing_issues(self, df: pd.DataFrame, missing_patterns: Dict[str, float]) -> List[QualityIssue]:
        """Create quality issues for missing values."""
        issues = []
        
        for col, pct in missing_patterns.items():
            if pct > 0:
                if pct > 50:
                    severity = Severity.CRITICAL
                    rec = f"Consider dropping column '{col}' or investigate data source"
                elif pct > 20:
                    severity = Severity.HIGH
                    rec = f"Implement imputation strategy for '{col}'"
                elif pct > 5:
                    severity = Severity.MEDIUM
                    rec = f"Review missing value handling for '{col}'"
                else:
                    severity = Severity.LOW
                    rec = f"Minor missing values in '{col}' - may be acceptable"
                
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.MISSING_VALUE,
                    severity=severity,
                    column=col,
                    description=f"{pct:.1f}% missing values in '{col}'",
                    affected_rows=int(df[col].isna().sum()),
                    affected_percent=pct,
                    recommendation=rec
                ))
        
        return issues
    
    def _detect_duplicates(self, df: pd.DataFrame) -> Tuple[int, Optional[pd.DataFrame]]:
        """Detect duplicate rows."""
        subset = self.duplicate_subset
        duplicates = df[df.duplicated(subset=subset, keep='first')]
        return len(duplicates), duplicates if len(duplicates) > 0 else None
    
    def _create_duplicate_issue(self, df: pd.DataFrame, duplicate_count: int) -> QualityIssue:
        """Create quality issue for duplicates."""
        pct = duplicate_count / len(df) * 100
        
        if pct > 20:
            severity = Severity.CRITICAL
        elif pct > 10:
            severity = Severity.HIGH
        elif pct > 5:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
        return QualityIssue(
            issue_type=QualityIssueType.DUPLICATE,
            severity=severity,
            column=None,
            description=f"{duplicate_count:,} duplicate rows ({pct:.1f}%)",
            affected_rows=duplicate_count,
            affected_percent=pct,
            recommendation="Review and deduplicate records"
        )
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers in numeric columns."""
        outliers = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            outlier_indices = self._iqr_outliers(df, col)
            
            if outlier_indices:
                outliers[col] = outlier_indices
        
        return outliers
    
    def _iqr_outliers(self, df: pd.DataFrame, col: str) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - self.outlier_threshold * IQR
        upper = Q3 + self.outlier_threshold * IQR
        
        mask = (df[col] < lower) | (df[col] > upper)
        return df[mask].index.tolist()
    
    def _create_outlier_issues(self, df: pd.DataFrame, outliers: Dict[str, List[int]]) -> List[QualityIssue]:
        """Create quality issues for outliers."""
        issues = []
        
        for col, indices in outliers.items():
            pct = len(indices) / len(df) * 100
            
            if pct > 10:
                severity = Severity.HIGH
            elif pct > 5:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            sample = df.loc[indices[:5], col].tolist()
            
            issues.append(QualityIssue(
                issue_type=QualityIssueType.OUTLIER,
                severity=severity,
                column=col,
                description=f"{len(indices)} outliers detected in '{col}' ({pct:.1f}%)",
                affected_rows=len(indices),
                affected_percent=pct,
                sample_values=sample,
                recommendation=f"Review outliers in '{col}' - may be data errors or valid extremes"
            ))
        
        return issues
    
    def _check_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for data consistency issues."""
        issues = []
        
        # Check for mixed types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            types = df[col].dropna().apply(type).unique()
            if len(types) > 1:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INCONSISTENT,
                    severity=Severity.MEDIUM,
                    column=col,
                    description=f"Mixed data types in '{col}'",
                    affected_rows=len(df),
                    affected_percent=100,
                    recommendation=f"Standardize data types in '{col}'"
                ))
        
        return issues
    
    def _check_validity(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for validity issues."""
        issues = []
        
        positive_keywords = ['price', 'amount', 'quantity', 'count', 'age', 'weight']
        
        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in positive_keywords):
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.INVALID_RANGE,
                        severity=Severity.HIGH,
                        column=col,
                        description=f"Negative values in '{col}' (expected positive)",
                        affected_rows=neg_count,
                        affected_percent=neg_count / len(df) * 100,
                        recommendation=f"Review negative values in '{col}'"
                    ))
        
        return issues
    
    def _calculate_column_score(self, series: pd.Series, col: str, issues: List[QualityIssue]) -> float:
        """Calculate quality score for a single column."""
        score = 100.0
        
        missing_pct = series.isna().mean() * 100
        score -= min(missing_pct, 50)
        
        col_issues = [i for i in issues if i.column == col]
        for issue in col_issues:
            if issue.severity == Severity.CRITICAL:
                score -= 20
            elif issue.severity == Severity.HIGH:
                score -= 10
            elif issue.severity == Severity.MEDIUM:
                score -= 5
            elif issue.severity == Severity.LOW:
                score -= 2
        
        return max(0, score)
    
    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue],
        missing_patterns: Dict[str, float],
        duplicate_count: int,
        outliers: Dict[str, List[int]]
    ) -> QualityScore:
        """Calculate multi-dimensional quality score."""
        avg_missing = np.mean(list(missing_patterns.values()))
        completeness = 100 - min(avg_missing, 100)
        
        dup_pct = duplicate_count / len(df) * 100 if len(df) > 0 else 0
        uniqueness = 100 - min(dup_pct * 2, 100)
        
        consistency_issues = [i for i in issues if i.issue_type == QualityIssueType.INCONSISTENT]
        consistency = 100 - len(consistency_issues) * 10
        consistency = max(0, consistency)
        
        validity_issues = [i for i in issues if i.issue_type in (
            QualityIssueType.INVALID_FORMAT, 
            QualityIssueType.INVALID_RANGE
        )]
        validity = 100 - len(validity_issues) * 15
        validity = max(0, validity)
        
        total_outliers = sum(len(v) for v in outliers.values())
        outlier_pct = total_outliers / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0
        accuracy = 100 - min(outlier_pct * 5, 50)
        
        timeliness = 100.0
        
        overall = (
            completeness * 0.25 +
            uniqueness * 0.20 +
            consistency * 0.15 +
            validity * 0.20 +
            accuracy * 0.15 +
            timeliness * 0.05
        )
        
        return QualityScore(
            overall_score=overall,
            completeness_score=completeness,
            uniqueness_score=uniqueness,
            consistency_score=consistency,
            validity_score=validity,
            accuracy_score=accuracy,
            timeliness_score=timeliness
        )
    
    def _generate_recommendations(self, issues: List[QualityIssue]) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        high = [i for i in issues if i.severity == Severity.HIGH]
        
        if critical:
            recommendations.append("🔴 CRITICAL: Address these issues immediately:")
            for issue in critical:
                recommendations.append(f"   - {issue.recommendation}")
        
        if high:
            recommendations.append("🟠 HIGH PRIORITY:")
            for issue in high:
                recommendations.append(f"   - {issue.recommendation}")
        
        return recommendations
    
    def _generate_summary(self, df: pd.DataFrame, score: QualityScore, issues: List[QualityIssue]) -> str:
        """Generate executive summary."""
        grade = "A" if score.overall_score >= 90 else \
                "B" if score.overall_score >= 80 else \
                "C" if score.overall_score >= 70 else \
                "D" if score.overall_score >= 60 else "F"
        
        critical_count = sum(1 for i in issues if i.severity == Severity.CRITICAL)
        high_count = sum(1 for i in issues if i.severity == Severity.HIGH)
        
        return f"""
Data Quality Summary
====================
Overall Score: {score.overall_score:.1f}/100 (Grade: {grade})

Dimension Scores:
  - Completeness: {score.completeness_score:.1f}/100
  - Uniqueness: {score.uniqueness_score:.1f}/100
  - Consistency: {score.consistency_score:.1f}/100
  - Validity: {score.validity_score:.1f}/100
  - Accuracy: {score.accuracy_score:.1f}/100

Issues Found: {len(issues)}
  - Critical: {critical_count}
  - High: {high_count}
  - Other: {len(issues) - critical_count - high_count}
"""

"""
insight_engine.py

Business Insight Engine that generates natural-language insights,
identifies key drivers, detects anomalies, and provides recommendations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats


class InsightType(Enum):
    """Types of business insights."""
    KEY_DRIVER = "key_driver"
    ANOMALY = "anomaly"
    TREND = "trend"
    COMPARISON = "comparison"
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    PATTERN = "pattern"
    SUMMARY = "summary"


class AudienceLevel(Enum):
    """Target audience for insights."""
    EXECUTIVE = "executive"
    MANAGER = "manager"
    ANALYST = "analyst"
    TECHNICAL = "technical"


@dataclass
class BusinessInsight:
    """A business insight with context and recommendations."""
    insight_type: InsightType
    headline: str
    summary: str
    detailed_explanation: str
    business_impact: str
    recommendations: List[str]
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    audience_level: AudienceLevel = AudienceLevel.ANALYST
    priority: int = 5


@dataclass
class DriverAnalysis:
    """Analysis of key drivers for a target variable."""
    target_variable: str
    top_drivers: List[Tuple[str, float, str]]
    driver_interactions: List[Dict[str, Any]]
    unexplained_variance: float


@dataclass
class AnomalyReport:
    """Report of detected anomalies."""
    anomalies: List[Dict[str, Any]]
    anomaly_score: float
    clusters: List[Dict[str, Any]]


@dataclass
class InsightReport:
    """Complete insight report."""
    insights: List[BusinessInsight]
    driver_analysis: Optional[DriverAnalysis]
    anomalies: AnomalyReport
    executive_summary: str
    key_metrics: Dict[str, Any]
    action_items: List[str]


class InsightEngine:
    """Business Insight Generation Engine."""
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        max_insights: int = 20,
        anomaly_threshold: float = 0.05
    ):
        self.min_confidence = min_confidence
        self.max_insights = max_insights
        self.anomaly_threshold = anomaly_threshold
    
    def generate_insights(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> InsightReport:
        """Generate comprehensive business insights."""
        insights = []
        
        # 1. Generate summary insights
        summary_insights = self._generate_summary_insights(df)
        insights.extend(summary_insights)
        
        # 2. Identify key drivers (if target specified)
        driver_analysis = None
        if target_column and target_column in df.columns:
            driver_analysis = self._analyze_drivers(df, target_column)
            driver_insights = self._generate_driver_insights(driver_analysis)
            insights.extend(driver_insights)
        
        # 3. Detect and explain anomalies
        anomaly_report = self._detect_anomalies(df)
        anomaly_insights = self._generate_anomaly_insights(anomaly_report)
        insights.extend(anomaly_insights)
        
        # 4. Generate comparison insights
        comparison_insights = self._generate_comparison_insights(df)
        insights.extend(comparison_insights)
        
        # 5. Generate opportunity and risk insights
        opportunity_insights = self._identify_opportunities(df, target_column)
        insights.extend(opportunity_insights)
        
        # 6. Prioritize and filter insights
        insights = self._prioritize_insights(insights)[:self.max_insights]
        
        # 7. Generate executive summary
        executive_summary = self._generate_executive_summary(df, insights, driver_analysis)
        
        # 8. Extract key metrics
        key_metrics = self._extract_key_metrics(df, target_column)
        
        # 9. Generate action items
        action_items = self._generate_action_items(insights)
        
        return InsightReport(
            insights=insights,
            driver_analysis=driver_analysis,
            anomalies=anomaly_report,
            executive_summary=executive_summary,
            key_metrics=key_metrics,
            action_items=action_items
        )
    
    def _generate_summary_insights(self, df: pd.DataFrame) -> List[BusinessInsight]:
        """Generate high-level summary insights."""
        insights = []
        
        insights.append(BusinessInsight(
            insight_type=InsightType.SUMMARY,
            headline="Dataset Overview",
            summary=f"Analyzing {len(df):,} records across {len(df.columns)} variables",
            detailed_explanation=f"The dataset contains {len(df):,} rows and {len(df.columns)} columns. "
                                f"Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}, "
                                f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}.",
            business_impact="Understanding the data scope helps set appropriate expectations.",
            recommendations=["Validate data completeness", "Check for required fields"],
            confidence=1.0,
            audience_level=AudienceLevel.EXECUTIVE,
            priority=3
        ))
        
        missing_pct = df.isna().mean().mean() * 100
        if missing_pct > 5:
            insights.append(BusinessInsight(
                insight_type=InsightType.RISK,
                headline="Data Completeness Alert",
                summary=f"Average missing data rate: {missing_pct:.1f}%",
                detailed_explanation=f"The dataset has an average missing rate of {missing_pct:.1f}%.",
                business_impact="Missing data can lead to biased analysis.",
                recommendations=["Identify root cause", "Implement imputation strategy"],
                confidence=0.95,
                audience_level=AudienceLevel.ANALYST,
                priority=7 if missing_pct > 20 else 5
            ))
        
        return insights
    
    def _analyze_drivers(self, df: pd.DataFrame, target: str) -> DriverAnalysis:
        """Analyze key drivers of a target variable."""
        drivers = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        
        target_series = df[target].dropna()
        
        for col in numeric_cols:
            valid_mask = df[col].notna() & df[target].notna()
            if valid_mask.sum() < 10:
                continue
            
            try:
                corr, p_value = stats.pearsonr(df.loc[valid_mask, col], df.loc[valid_mask, target])
                if p_value < 0.05 and abs(corr) > 0.1:
                    direction = "positive" if corr > 0 else "negative"
                    drivers.append((col, abs(corr), direction))
            except Exception:
                continue
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if df[col].nunique() > 50:
                continue
            
            try:
                groups = df.groupby(col)[target].mean()
                variance = groups.var() / target_series.var() if target_series.var() > 0 else 0
                if variance > 0.1:
                    drivers.append((col, float(variance), "categorical"))
            except Exception:
                continue
        
        drivers.sort(key=lambda x: x[1], reverse=True)
        
        explained = sum(d[1] for d in drivers[:3]) if drivers else 0
        unexplained = max(0, 1 - explained)
        
        return DriverAnalysis(
            target_variable=target,
            top_drivers=drivers[:10],
            driver_interactions=[],
            unexplained_variance=unexplained
        )
    
    def _generate_driver_insights(self, analysis: DriverAnalysis) -> List[BusinessInsight]:
        """Generate insights from driver analysis."""
        insights = []
        
        if not analysis.top_drivers:
            return insights
        
        top_driver, importance, direction = analysis.top_drivers[0]
        insights.append(BusinessInsight(
            insight_type=InsightType.KEY_DRIVER,
            headline=f"Primary Driver: {top_driver}",
            summary=f"'{top_driver}' is the strongest predictor of '{analysis.target_variable}'",
            detailed_explanation=f"'{top_driver}' has correlation {importance:.2f} with the target.",
            business_impact=f"Focus on '{top_driver}' for maximum impact.",
            recommendations=[f"Monitor '{top_driver}' closely", "Validate causal relationship"],
            confidence=min(importance + 0.3, 0.95),
            supporting_data={'correlation': importance, 'direction': direction},
            audience_level=AudienceLevel.MANAGER,
            priority=9
        ))
        
        return insights
    
    def _detect_anomalies(self, df: pd.DataFrame) -> AnomalyReport:
        """Detect anomalies in the dataset."""
        anomalies = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            Q1, Q3 = series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if len(outliers) > 0:
                anomalies.append({
                    'column': col,
                    'type': 'outlier',
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                })
        
        total_anomalies = sum(a['count'] for a in anomalies)
        anomaly_score = min(total_anomalies / len(df), 1.0) if len(df) > 0 else 0
        
        return AnomalyReport(
            anomalies=anomalies,
            anomaly_score=anomaly_score,
            clusters=[]
        )
    
    def _generate_anomaly_insights(self, report: AnomalyReport) -> List[BusinessInsight]:
        """Generate insights from anomaly detection."""
        insights = []
        
        if not report.anomalies:
            return insights
        
        high_anomaly = [a for a in report.anomalies if a['percentage'] > 5]
        if high_anomaly:
            cols = [a['column'] for a in high_anomaly]
            insights.append(BusinessInsight(
                insight_type=InsightType.ANOMALY,
                headline="Significant Outliers Detected",
                summary=f"Columns with high outlier rates: {', '.join(cols[:3])}",
                detailed_explanation="Multiple columns show outlier rates above 5%.",
                business_impact="Outliers may skew analysis results.",
                recommendations=["Investigate root cause", "Consider robust statistics"],
                confidence=0.9,
                audience_level=AudienceLevel.ANALYST,
                priority=8
            ))
        
        return insights
    
    def _generate_comparison_insights(self, df: pd.DataFrame) -> List[BusinessInsight]:
        """Generate comparative insights between groups."""
        insights = []
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for cat_col in cat_cols[:3]:
            if df[cat_col].nunique() > 20 or df[cat_col].nunique() < 2:
                continue
            
            for num_col in num_cols[:3]:
                try:
                    group_means = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                    if len(group_means) < 2:
                        continue
                    
                    top, bottom = group_means.index[0], group_means.index[-1]
                    top_val, bottom_val = group_means.iloc[0], group_means.iloc[-1]
                    
                    if bottom_val == 0:
                        continue
                    
                    diff_pct = (top_val - bottom_val) / abs(bottom_val) * 100
                    
                    if abs(diff_pct) > 20:
                        insights.append(BusinessInsight(
                            insight_type=InsightType.COMPARISON,
                            headline=f"Performance Gap: {cat_col}",
                            summary=f"'{top}' outperforms '{bottom}' by {diff_pct:.0f}% on {num_col}",
                            detailed_explanation=f"'{top}' averages {top_val:.2f} vs '{bottom}' {bottom_val:.2f}",
                            business_impact=f"Understand why '{top}' performs better.",
                            recommendations=["Analyze success factors", "Apply best practices"],
                            confidence=0.85,
                            audience_level=AudienceLevel.MANAGER,
                            priority=6
                        ))
                        break
                except Exception:
                    continue
        
        return insights
    
    def _identify_opportunities(self, df: pd.DataFrame, target: Optional[str]) -> List[BusinessInsight]:
        """Identify business opportunities and risks."""
        insights = []
        
        for col in df.select_dtypes(include=['object', 'category']).columns[:5]:
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) == 0:
                continue
            
            top_pct = value_counts.iloc[0] * 100
            if top_pct > 70:
                insights.append(BusinessInsight(
                    insight_type=InsightType.RISK,
                    headline=f"Concentration Risk: {col}",
                    summary=f"'{value_counts.index[0]}' represents {top_pct:.1f}% of {col}",
                    detailed_explanation="High concentration can indicate business risk.",
                    business_impact="Over-reliance creates vulnerability.",
                    recommendations=["Develop diversification strategy", "Monitor trends"],
                    confidence=0.9,
                    audience_level=AudienceLevel.EXECUTIVE,
                    priority=7
                ))
        
        return insights
    
    def _prioritize_insights(self, insights: List[BusinessInsight]) -> List[BusinessInsight]:
        """Prioritize insights by importance and confidence."""
        for insight in insights:
            type_boost = {
                InsightType.KEY_DRIVER: 2,
                InsightType.RISK: 2,
                InsightType.ANOMALY: 1,
                InsightType.OPPORTUNITY: 1,
            }.get(insight.insight_type, 0)
            
            insight.priority = min(10, insight.priority + type_boost)
        
        return sorted(insights, key=lambda x: (x.priority, x.confidence), reverse=True)
    
    def _generate_executive_summary(
        self,
        df: pd.DataFrame,
        insights: List[BusinessInsight],
        driver_analysis: Optional[DriverAnalysis]
    ) -> str:
        """Generate executive-level summary."""
        lines = [
            "EXECUTIVE SUMMARY",
            "=" * 50,
            "",
            f"Analysis of {len(df):,} records yielded {len(insights)} key insights.",
            "",
            "KEY FINDINGS:",
        ]
        
        for i, insight in enumerate(insights[:3], 1):
            lines.append(f"{i}. {insight.headline}")
            lines.append(f"   {insight.summary}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _extract_key_metrics(self, df: pd.DataFrame, target: Optional[str]) -> Dict[str, Any]:
        """Extract key metrics from the dataset."""
        metrics = {
            'record_count': len(df),
            'column_count': len(df.columns),
            'completeness': (1 - df.isna().mean().mean()) * 100,
        }
        
        if target and target in df.columns:
            target_series = df[target].dropna()
            if pd.api.types.is_numeric_dtype(target_series):
                metrics['target_mean'] = float(target_series.mean())
                metrics['target_median'] = float(target_series.median())
        
        return metrics
    
    def _generate_action_items(self, insights: List[BusinessInsight]) -> List[str]:
        """Generate prioritized action items."""
        actions = []
        
        for insight in insights:
            if insight.priority >= 7 and insight.recommendations:
                actions.append(f"[{insight.insight_type.value.upper()}] {insight.recommendations[0]}")
        
        return actions[:10]

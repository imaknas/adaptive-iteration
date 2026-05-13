"""core/analyzer.py — Pattern detection from ledger records.

Finds top / bottom performers and common features, computes per-dimension variance,
and produces a structured AnalysisResult for use by HypothesisEngine.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from .ledger import Ledger


@dataclass
class DimensionStats:
    """Variance and mean for a single metric dimension."""
    name: str
    mean: float
    variance: float
    stdev: float
    count: int


@dataclass
class AnalysisResult:
    """Output of Analyzer.analyze()."""
    domain: str
    total_records: int
    top_performers: list[dict[str, Any]]        # records with highest primary_metric
    bottom_performers: list[dict[str, Any]]      # records with lowest primary_metric
    top_features: dict[str, Any]                 # common variable→variant among top
    bottom_features: dict[str, Any]              # common variable→variant among bottom
    dimension_stats: list[DimensionStats]        # per-metric variance across all records
    winning_patterns: dict[str, str]             # variable → variant that won most often


class Analyzer:
    """Derives insights from a Ledger.

    Parameters
    ----------
    ledger       : Ledger instance to analyse
    primary_metric : the metric key to rank performers by (e.g. "avg_view_pct")
    top_n        : how many top/bottom performers to surface
    """

    def __init__(
        self,
        ledger: Ledger,
        primary_metric: str = "primary_metric",
        top_n: int = 5,
    ) -> None:
        self.ledger = ledger
        self.primary_metric = primary_metric
        self.top_n = top_n

    # ── Public ─────────────────────────────────────────────────────────────────

    def analyze(self, domain: Optional[str] = None) -> AnalysisResult:
        """Run full analysis for *domain* (or all domains if None)."""
        records = self.ledger.query(domain=domain) if domain else self.ledger.records

        if not records:
            return AnalysisResult(
                domain=domain or "all",
                total_records=0,
                top_performers=[],
                bottom_performers=[],
                top_features={},
                bottom_features={},
                dimension_stats=[],
                winning_patterns={},
            )

        # Sort by primary metric (records without it go to the bottom)
        def _primary(r: dict) -> float:
            return float(r.get("metric_values", {}).get(self.primary_metric) or 0.0)

        ranked = sorted(records, key=_primary, reverse=True)
        top = ranked[: self.top_n]
        bottom = ranked[-self.top_n:]

        return AnalysisResult(
            domain=domain or "all",
            total_records=len(records),
            top_performers=top,
            bottom_performers=bottom,
            top_features=self._common_features(top),
            bottom_features=self._common_features(bottom),
            dimension_stats=self._dimension_stats(records),
            winning_patterns=self._winning_patterns(records),
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _common_features(self, records: list[dict]) -> dict[str, Any]:
        """Return {variable: variant} pairs that appear in >50% of records."""
        if not records:
            return {}
        counts: dict[str, dict[str, int]] = {}
        for r in records:
            var = r.get("variable", "")
            val = r.get("variant", "")
            if var and val:
                counts.setdefault(var, {}).setdefault(val, 0)
                counts[var][val] += 1

        threshold = len(records) * 0.5
        result: dict[str, Any] = {}
        for var, val_counts in counts.items():
            best_val, best_count = max(val_counts.items(), key=lambda x: x[1])
            if best_count >= threshold:
                result[var] = best_val
        return result

    def _dimension_stats(self, records: list[dict]) -> list[DimensionStats]:
        """Compute mean / variance for every metric key found in records."""
        all_metrics: dict[str, list[float]] = {}
        for r in records:
            for k, v in r.get("metric_values", {}).items():
                if isinstance(v, (int, float)) and v is not None:
                    all_metrics.setdefault(k, []).append(float(v))

        stats: list[DimensionStats] = []
        for name, values in all_metrics.items():
            if len(values) < 2:
                continue
            mean = statistics.mean(values)
            var  = statistics.variance(values)
            std  = statistics.stdev(values)
            stats.append(DimensionStats(
                name=name, mean=round(mean, 4), variance=round(var, 4),
                stdev=round(std, 4), count=len(values),
            ))
        return sorted(stats, key=lambda s: s.variance, reverse=True)

    def _winning_patterns(self, records: list[dict]) -> dict[str, str]:
        """For each variable, find the variant that won most frequently."""
        win_counts: dict[str, dict[str, int]] = {}
        for r in records:
            if not r.get("winner"):
                continue
            var = r.get("variable", "")
            val = r.get("variant", "")
            if var and val:
                win_counts.setdefault(var, {}).setdefault(val, 0)
                win_counts[var][val] += 1

        return {
            var: max(val_counts, key=val_counts.get)
            for var, val_counts in win_counts.items()
        }

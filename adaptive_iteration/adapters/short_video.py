"""adapters/short_video.py — Example ShortVideoAdapter.

This is a reference implementation showing how to build a DomainAdapter
for short-form video platforms (e.g. YouTube Shorts, Instagram Reels).

In a real deployment, replace `collect_metrics` and `get_signals` with
calls to your actual analytics backend.

Usage
-----
    from adaptive_iteration.adapters.short_video import ShortVideoAdapter

    adapter = ShortVideoAdapter(platform="youtube")
    metrics = adapter.collect_metrics(["video_001", "video_002"])
    signals = adapter.get_signals(metrics)
"""
from __future__ import annotations

import random
from typing import Any

from .base import DomainAdapter


class ShortVideoAdapter(DomainAdapter):
    """Example DomainAdapter for short-video platforms.

    Supports "youtube" and "instagram" as platform targets.
    Metrics are simulated — replace with your real analytics calls.

    Parameters
    ----------
    platform : "youtube" or "instagram"
    """

    SUPPORTED_PLATFORMS = ("youtube", "instagram")

    def __init__(self, platform: str) -> None:
        if platform not in self.SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Unsupported platform {platform!r}. "
                f"Choose from {self.SUPPORTED_PLATFORMS}."
            )
        self.platform = platform

    # ── DomainAdapter interface ────────────────────────────────────────────────

    def collect_metrics(self, item_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch metrics for each item ID.

        Replace this with real API calls to your analytics provider.
        This implementation returns simulated data for illustration.
        """
        results = []
        for item_id in item_ids:
            if self.platform == "youtube":
                results.append({
                    "id": item_id,
                    "views": random.randint(500, 50_000),
                    "avg_view_pct": round(random.uniform(30, 90), 1),
                    "avg_view_sec": round(random.uniform(15, 75), 1),
                    "like_rate": round(random.uniform(0.5, 8.0), 2),
                    "subscribers_gained": random.randint(0, 50),
                })
            else:  # instagram
                reach = random.randint(200, 20_000)
                likes = random.randint(10, int(reach * 0.15))
                results.append({
                    "id": item_id,
                    "reach": reach,
                    "likes": likes,
                    "saves": random.randint(0, int(reach * 0.05)),
                    "shares": random.randint(0, int(reach * 0.03)),
                    "comments": random.randint(0, int(reach * 0.02)),
                    "avg_watch_time_ms": random.randint(3_000, 30_000),
                    "total_interactions": likes + random.randint(5, 100),
                })
        return results

    def get_signals(self, metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalise raw metrics into framework signals.

        YouTube  → primary_metric = avg_view_pct (0–100)
        Instagram → primary_metric = engagement_score
                    = (saves×3 + shares×2 + comments×2 + likes) / reach × 100
        """
        if not metrics:
            return {"primary_metric": 0.0, "secondary_metrics": {}}

        if self.platform == "youtube":
            return self._yt_signals(metrics)
        return self._ig_signals(metrics)

    def format_context(
        self,
        top_items: list[dict[str, Any]],
        bottom_items: list[dict[str, Any]],
    ) -> str:
        lines = [f"Platform: {self.platform}"]
        lines.append("\nTop performers:")
        for item in top_items:
            lines.append(self._fmt(item))
        lines.append("\nBottom performers:")
        for item in bottom_items:
            lines.append(self._fmt(item))
        return "\n".join(lines)

    def describe(self) -> str:
        return f"ShortVideoAdapter(platform={self.platform!r})"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _yt_signals(self, metrics: list[dict]) -> dict[str, Any]:
        valid = [m for m in metrics if m.get("avg_view_pct") is not None]
        if not valid:
            return {"primary_metric": 0.0, "secondary_metrics": {}}
        n = len(metrics)
        return {
            "primary_metric": round(
                sum(m["avg_view_pct"] for m in valid) / len(valid), 2
            ),
            "secondary_metrics": {
                "views":              round(sum(m.get("views", 0) for m in metrics) / n, 1),
                "like_rate":          round(sum(m.get("like_rate", 0) for m in metrics) / n, 3),
                "avg_view_sec":       round(sum(m.get("avg_view_sec", 0) for m in metrics) / n, 1),
                "subscribers_gained": round(sum(m.get("subscribers_gained", 0) for m in metrics) / n, 2),
            },
        }

    def _ig_signals(self, metrics: list[dict]) -> dict[str, Any]:
        valid = [m for m in metrics if m.get("reach", 0) > 0]
        if not valid:
            return {"primary_metric": 0.0, "secondary_metrics": {}}
        n = len(metrics)

        def _eng(m: dict) -> float:
            reach = m.get("reach", 1)
            return (
                m.get("saves", 0) * 3
                + m.get("shares", 0) * 2
                + m.get("comments", 0) * 2
                + m.get("likes", 0)
            ) / reach * 100

        return {
            "primary_metric": round(sum(_eng(m) for m in valid) / len(valid), 4),
            "secondary_metrics": {
                "reach":            round(sum(m.get("reach", 0) for m in metrics) / n, 1),
                "like_rate":        round(
                    sum(m.get("likes", 0) / max(m.get("reach", 1), 1) * 100 for m in metrics) / n, 3
                ),
                "avg_watch_time_s": round(
                    sum(m.get("avg_watch_time_ms", 0) for m in metrics) / n / 1000, 2
                ),
                "shares":           round(sum(m.get("shares", 0) for m in metrics) / n, 2),
                "saved":            round(sum(m.get("saves", 0) for m in metrics) / n, 2),
                "total_interactions": round(sum(m.get("total_interactions", 0) for m in metrics) / n, 2),
            },
        }

    def _fmt(self, item: dict) -> str:
        item_id = item.get("id", "?")
        if self.platform == "youtube":
            return (
                f"  [{item_id}] "
                f"avg_view_pct={item.get('avg_view_pct')}%  "
                f"views={item.get('views')}  "
                f"like_rate={item.get('like_rate')}%"
            )
        reach = item.get("reach", 0)
        likes = item.get("likes", 0)
        lr = round(likes / reach * 100, 2) if reach > 0 else 0.0
        return (
            f"  [{item_id}] "
            f"like_rate={lr}%  "
            f"reach={reach}  "
            f"avg_watch={item.get('avg_watch_time_ms', 0) // 1000}s"
        )

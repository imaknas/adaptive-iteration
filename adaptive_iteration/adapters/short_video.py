"""adapters/short_video.py — Example adapter for short-video platforms (simulated data).

Shows the shape of a real adapter: each unit listed in Experiment.items becomes one
Observation. Replace _fetch() with calls to your analytics backend.

    adapter = ShortVideoAdapter(platform="youtube")
    ledger.record_observations(adapter.collect_observations(experiment))
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Optional

from ..core.experiment import Experiment
from ..core.metrics import Observation


class ShortVideoAdapter:
    SUPPORTED_PLATFORMS = ("youtube", "instagram")

    def __init__(self, platform: str, seed: Optional[int] = None) -> None:
        if platform not in self.SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported platform {platform!r}. "
                             f"Choose from {self.SUPPORTED_PLATFORMS}.")
        self.platform = platform
        self._rng = random.Random(seed)

    def collect_observations(self, experiment: Experiment) -> list[Observation]:
        now = datetime.now(timezone.utc).isoformat()
        observations = []
        for item in experiment.items:
            observations.append(Observation(
                experiment_id=experiment.id,
                variant=item["variant"],
                unit_id=item["item_id"],
                produced_at=item["produced_at"],
                observed_at=now,
                metrics=self._fetch(item["item_id"]),
                pair_id=item.get("pair_id"),
            ))
        return observations

    def _fetch(self, video_id: str) -> dict[str, Optional[float]]:
        """Simulated analytics. A real platform sometimes has no data yet → None, not 0."""
        if self._rng.random() < 0.05:
            return {"views": None, "avg_view_pct": None, "like_rate": None}
        if self.platform == "youtube":
            return {
                "views": float(self._rng.randint(500, 50_000)),
                # Shorts loop, so average percentage viewed can exceed 100.
                "avg_view_pct": round(self._rng.uniform(30, 130), 1),
                "like_rate": round(self._rng.uniform(1, 8), 2),
            }
        return {
            "views": float(self._rng.randint(200, 20_000)),
            "avg_view_pct": None,
            "like_rate": round(self._rng.uniform(2, 12), 2),
        }

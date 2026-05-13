"""adapters/base.py — DomainAdapter ABC.

Every domain (short video, blog, email, ...) must implement these three methods.
core/ never imports from here; HypothesisEngine receives the outputs as plain dicts/strings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DomainAdapter(ABC):
    """Bridge between adaptive_iteration.core and a concrete external system.

    Three required methods
    ----------------------
    collect_metrics(item_ids)
        Pull raw metric data for the given item IDs.
        Returns a list of dicts, one per item. Each dict must include at least
        the item's ID under some consistent key (e.g. "id").

    get_signals(metrics)
        Normalise raw metrics into framework-friendly signals:
          - "primary_metric": float  (the single most-important KPI for ranking)
          - "secondary_metrics": dict  (any additional useful metrics)
        Returns a dict.

    format_context(top_items, bottom_items)
        Produce a human-readable text block describing top and bottom performers.
        This is injected verbatim into the HypothesisEngine prompt.
        Returns a str.
    """

    # ── Required ───────────────────────────────────────────────────────────────

    @abstractmethod
    def collect_metrics(self, item_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch raw metrics for *item_ids* from the external system.

        Parameters
        ----------
        item_ids : list of platform-specific IDs (video IDs, post IDs, …)

        Returns
        -------
        list[dict]
            One dict per item. Shape is domain-specific; must be consistent with
            what get_signals() expects.
        """
        ...

    @abstractmethod
    def get_signals(self, metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalise raw metrics into a standard signals dict.

        Expected output schema
        ----------------------
        {
            "primary_metric": <float>,          # main ranking KPI
            "secondary_metrics": {<str>: <float>, ...},
        }
        """
        ...

    @abstractmethod
    def format_context(
        self,
        top_items: list[dict[str, Any]],
        bottom_items: list[dict[str, Any]],
    ) -> str:
        """Return a text block describing top vs bottom performers.

        Used verbatim in the HypothesisEngine LLM prompt.
        """
        ...

    # ── Optional convenience ───────────────────────────────────────────────────

    def describe(self) -> str:
        """Short human-readable description of this adapter (for logs/docs)."""
        return self.__class__.__name__

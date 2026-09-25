"""core/registry.py — Per-domain variable registry, stored in the ledger.

Evidence accumulates by variable name, so the same idea under three names splits
its evidence three ways. The registry gives every variable one canonical name;
other spellings become aliases. It is a view over ledger events ("variable",
"variable_alias"), so the ledger stays the single source of truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .ledger import Ledger


@dataclass(frozen=True)
class VariableDef:
    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    execution: Optional[str] = None   # how the pipeline actually realises this variable

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "aliases": list(self.aliases), "execution": self.execution}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VariableDef":
        return cls(name=d["name"], description=d.get("description", ""),
                   aliases=tuple(d.get("aliases", ())), execution=d.get("execution"))


class VariableRegistry:
    def __init__(self, ledger: Ledger, domain: str) -> None:
        self.ledger = ledger
        self.domain = domain

    def _state(self) -> tuple[dict[str, VariableDef], dict[str, str]]:
        defs: dict[str, VariableDef] = {}
        alias_to_name: dict[str, str] = {}
        for e in self.ledger.events:
            if e.get("domain") != self.domain:
                continue
            if e.get("kind") == "variable":
                d = VariableDef.from_dict(e)
                defs[d.name] = d
                for al in d.aliases:
                    alias_to_name[al] = d.name
            elif e.get("kind") == "variable_alias":
                alias_to_name[e["alias"]] = e["name"]
        # follow alias chains (a merged into b, later b merged into c)
        for al in list(alias_to_name):
            seen = {al}
            target = alias_to_name[al]
            while target in alias_to_name and target not in seen:
                seen.add(target)
                target = alias_to_name[target]
            alias_to_name[al] = target
        return defs, alias_to_name

    def variables(self) -> list[VariableDef]:
        defs, alias_to_name = self._state()
        merged_away = set(alias_to_name)
        result = []
        for d in defs.values():
            if d.name in merged_away:
                continue
            aliases = tuple(sorted(a for a, n in alias_to_name.items() if n == d.name))
            result.append(VariableDef(d.name, d.description, aliases, d.execution))
        return sorted(result, key=lambda d: d.name)

    def resolve(self, name: str) -> Optional[str]:
        """Canonical name for *name* (itself or an alias), or None if unregistered."""
        defs, alias_to_name = self._state()
        if name in alias_to_name:
            return alias_to_name[name]
        return name if name in defs else None

    def get(self, name: str) -> Optional[VariableDef]:
        canonical = self.resolve(name)
        if canonical is None:
            return None
        return next((d for d in self.variables() if d.name == canonical), None)

    def register(self, definition: VariableDef) -> None:
        if self.resolve(definition.name) is not None:
            raise ValueError(f"variable {definition.name!r} already registered "
                             f"(as {self.resolve(definition.name)!r})")
        self.ledger.append_raw("variable", {"domain": self.domain, **definition.to_dict()})

    def merge(self, name: str, into: str) -> None:
        """Make *name* an alias of *into*; their evidence is counted together from now on."""
        src, dst = self.resolve(name), self.resolve(into)
        if src is None or dst is None:
            raise KeyError(f"both variables must be registered: {name!r}, {into!r}")
        if src == dst:
            return
        self.ledger.append_raw("variable_alias", {"domain": self.domain,
                                                  "alias": src, "name": dst})


class DuplicateDetector(Protocol):
    def find_match(self, candidate: VariableDef, registry: VariableRegistry) -> Optional[str]:
        """Return the canonical name of an existing variable *candidate* duplicates, or None."""
        ...


def _tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", name.lower()) if t)


class TokenSetDetector:
    """Same set of name tokens = same variable ("visual_opening_style" == "opening_visual_style").
    Semantic near-duplicates ("intro_…" vs "opening_…") are out of reach for a string rule —
    merge them by hand or inject a smarter DuplicateDetector."""

    def find_match(self, candidate: VariableDef, registry: VariableRegistry) -> Optional[str]:
        wanted = {_tokens(n) for n in (candidate.name, *candidate.aliases)}
        for d in registry.variables():
            if any(_tokens(n) in wanted for n in (d.name, *d.aliases)):
                return d.name
        return None

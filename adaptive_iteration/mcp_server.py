"""mcp_server.py — The service operations as MCP tools (stdio).

Requires the optional extra:  pip install "adaptive-iteration[mcp]"
Run:                           adaptive-iteration --ledger /abs/path/ledger.jsonl mcp

One server serves one ledger file. Tools are thin wrappers over service.py.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar

from . import service

F = TypeVar("F", bound=Callable[..., Any])

INSTRUCTIONS = """\
adaptive-iteration judges A/B experiments on units you produce (posts, emails,
videos, proposals, model outputs) so that an automated loop does not fool itself.

Workflow:
1. configure(domain, metric, min_effect, rule) once per domain. min_effect is the
   smallest difference worth acting on; ask the user if you don't know it.
   rule="proportion" for 0/1 metrics (clicked, replied), otherwise "welch".
2. register_variable for each thing you intend to vary.
3. accept_proposal to start an experiment (A vs B on one variable). Give every
   proposal an expected_effect (the B − A you expect, in metric units) and a
   proposed_by name: proposals that could not be detected in time are rejected,
   and evidence() keeps each proposer's track record. Use review_proposals first to
   check several ideas without writing anything.
4. Before producing each unit, call assign_variant and use the variant it returns.
5. record_observations: one row per produced unit, with the variant it got.
6. evaluate(domain=...) whenever you like — verdicts only happen at checkpoints.
   Follow decision.next_action.
7. pending(domain) lists closed verdicts not yet in effect. Put each into effect in
   the pipeline (ask the user first if it changes the pipeline), then mark_applied.
8. evidence(domain) before proposing what to test next.

Rules you must follow:
- Act only on closed verdicts (b_better, a_better, equivalent, no_detectable_diff).
  "insufficient" means keep collecting; never act on the interim numbers.
- Missing metric values are null, never 0.
- Record every unit, one row each; never pre-aggregate or drop unfavourable units.
- Get every unit's variant from assign_variant; never choose it yourself.
- Change nothing else about the pipeline while an experiment runs; if something
  else changes, tell the user — results spanning the change are confounded.
- configure() does not change running experiments; don't use it to rescue one.
"""


def build_server(ledger: str) -> Any:
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as e:
        raise service.ServiceError(
            'the MCP server needs the optional extra: pip install "adaptive-iteration[mcp]"'
        ) from e
    from . import __version__

    server = MCPServer(name="adaptive-iteration", instructions=INSTRUCTIONS,
                       version=__version__)
    read_only = ToolAnnotations(readOnlyHint=True)

    def guard(fn: F) -> F:
        """Report ServiceError messages to the agent (they say how to fix the call)."""
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except service.ServiceError as e:
                raise ToolError(str(e)) from e
        return wrapper  # type: ignore[return-value]

    @server.tool()
    @guard
    def configure(domain: str, metric: str, min_effect: float, rule: str = "welch",
                  higher_is_better: bool = True, window_days: float = 7.0,
                  maturity_hours: float = 72.0, max_windows: int = 4,
                  valid_min: Optional[float] = None, valid_max: Optional[float] = None
                  ) -> dict[str, Any]:
        """Set how a domain's experiments are judged. rule: "welch" (numbers) or
        "proportion" (0/1 outcomes). Applies only to experiments started afterwards."""
        return service.configure(ledger, domain, metric, min_effect,
                                 higher_is_better=higher_is_better, rule=rule,
                                 window_days=window_days, maturity_hours=maturity_hours,
                                 max_windows=max_windows, valid_min=valid_min,
                                 valid_max=valid_max)

    @server.tool()
    @guard
    def register_variable(domain: str, name: str, description: str = "",
                          execution: Optional[str] = None) -> dict[str, Any]:
        """Register a variable you will test. execution: how the pipeline applies it."""
        return service.register_variable(ledger, domain, name, description, execution)

    @server.tool()
    @guard
    def merge_variables(domain: str, name: str, into: str) -> dict[str, Any]:
        """Declare two registered variables the same thing; evidence is pooled."""
        return service.merge_variables(ledger, domain, name, into)

    @server.tool(annotations=read_only)
    @guard
    def review_proposals(domain: str, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        """Check proposed experiments without writing anything. Each proposal:
        {"variable", "variant_a", "variant_b", "expected_effect", "proposed_by",
         "description"?, "mode"?, "new_variable"?: {"description", "execution"}}.
        "screening" says whether the expected effect is detectable in time. Variants are label strings or
        {"label", "hint"}. Status is known / merged / new / rejected."""
        return {"reviewed": service.review_proposals(ledger, domain, proposals)}

    @server.tool()
    @guard
    def accept_proposal(domain: str, proposal: dict[str, Any], start: bool = True
                        ) -> dict[str, Any]:
        """Review one proposal and, unless rejected, add and start the experiment.
        Returns its id; tag every unit you produce with that id and its variant."""
        return service.accept_proposal(ledger, domain, proposal, start=start)

    @server.tool()
    @guard
    def record_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
        """Record one row per produced unit: {"experiment_id", "variant", "unit_id",
        "produced_at" (ISO), "metrics": {name: number|null}, "observed_at"?,
        "stratum"?, "pair_id"?}. Recording a unit again later updates it.
        All-or-nothing: if any row is invalid, nothing is written."""
        return service.record_observations(ledger, observations)

    @server.tool()
    @guard
    def evaluate(experiment_id: Optional[str] = None, domain: Optional[str] = None
                 ) -> dict[str, Any]:
        """Judge one experiment, or every running one in a domain. Safe to call often;
        verdicts are taken only at checkpoints. Follow decision.next_action."""
        return {"results": service.evaluate(ledger, experiment_id, domain=domain)}

    @server.tool()
    @guard
    def assign_variant(experiment_id: str, unit_id: str, stratum: Optional[str] = None
                       ) -> dict[str, Any]:
        """Which variant this unit gets (label, params, hint). Balanced within the
        stratum, recorded, and stable for the same unit_id. Call before producing it."""
        return service.assign_variant(ledger, experiment_id, unit_id, stratum)

    @server.tool(annotations=read_only)
    @guard
    def pending(domain: str) -> dict[str, Any]:
        """Closed verdicts not yet put into effect. changes_pipeline=true means B won."""
        return {"pending": service.pending(ledger, domain)}

    @server.tool()
    @guard
    def mark_applied(experiment_id: str) -> dict[str, Any]:
        """Record that a closed experiment's verdict is now in effect in the pipeline."""
        return service.mark_applied(ledger, experiment_id)

    @server.tool(annotations=read_only)
    @guard
    def evidence(domain: str, markdown: bool = False) -> Any:
        """Per-variable status, best variant, effect and cost so far."""
        return service.evidence(ledger, domain, "markdown" if markdown else "json")

    @server.tool(annotations=read_only)
    @guard
    def status(domain: Optional[str] = None) -> dict[str, Any]:
        """Experiments per domain, with state and verdict."""
        return service.status(ledger, domain)

    @server.tool(annotations=read_only)
    @guard
    def calibrate(domain: str, effect: float, per_window: int,
                  pool: Optional[list[float]] = None, sims: int = 1000) -> dict[str, Any]:
        """On this domain's own data: effect=0 gives how often a winner is declared when
        nothing differs; effect>0 gives how often (and how fast) a real effect is caught."""
        return service.calibrate(ledger, domain, effect=effect, per_window=per_window,
                                 pool=pool, sims=sims)

    return server


def serve(ledger: Optional[str]) -> None:
    if not ledger:
        raise service.ServiceError("no ledger: pass --ledger PATH or set "
                                   "ADAPTIVE_ITERATION_LEDGER")
    build_server(ledger).run("stdio")

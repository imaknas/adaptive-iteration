"""cli.py — `adaptive-iteration` command line, built for scripts and agents.

Every command prints JSON on stdout (evidence --markdown prints text). Errors print
{"error": "..."} and exit 1. The ledger path comes from --ledger or the
ADAPTIVE_ITERATION_LEDGER environment variable.

    adaptive-iteration --ledger data/ledger.jsonl configure --domain newsletter \\
        --metric clicked --min-effect 0.05 --rule proportion
    adaptive-iteration register-variable --domain newsletter --name subject_style
    adaptive-iteration accept --domain newsletter --json '{"variable": "subject_style",
        "variant_a": "statement", "variant_b": "question"}'
    cat rows.jsonl | adaptive-iteration record
    adaptive-iteration evaluate --domain newsletter
    adaptive-iteration mcp            # serve the same operations over MCP (stdio)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from . import service


def _load_json(text: str) -> Any:
    text = text.strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]  # JSONL


def _payload(args: argparse.Namespace) -> Any:
    if getattr(args, "json", None):
        return _load_json(args.json)
    source = getattr(args, "file", None)
    if source and source != "-":
        with open(source, encoding="utf-8") as f:
            return _load_json(f.read())
    return _load_json(sys.stdin.read())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adaptive-iteration",
                                description="Judge A/B experiments honestly; JSON in, JSON out.")
    p.add_argument("--ledger", default=os.environ.get("ADAPTIVE_ITERATION_LEDGER"),
                   help="ledger file (default: $ADAPTIVE_ITERATION_LEDGER)")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("configure", help="set how a domain's experiments are judged")
    c.add_argument("--domain", required=True)
    c.add_argument("--metric", required=True)
    c.add_argument("--min-effect", type=float, required=True,
                   help="smallest difference worth acting on")
    c.add_argument("--lower-is-better", action="store_true")
    c.add_argument("--rule", choices=["welch", "proportion"], default="welch",
                   help="proportion for 0/1 metrics")
    c.add_argument("--superiority", choices=["significance", "margin"], default="significance")
    c.add_argument("--window-days", type=float, default=7.0)
    c.add_argument("--maturity-hours", type=float, default=72.0)
    c.add_argument("--max-windows", type=int, default=4)
    c.add_argument("--valid-min", type=float)
    c.add_argument("--valid-max", type=float)

    r = sub.add_parser("register-variable", help="name a variable you will test")
    r.add_argument("--domain", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--description", default="")
    r.add_argument("--execution", help="how the pipeline applies this variable")

    m = sub.add_parser("merge-variables", help="make one variable an alias of another")
    m.add_argument("--domain", required=True)
    m.add_argument("--name", required=True)
    m.add_argument("--into", required=True)

    for name, help_ in (("review", "check proposals (JSON list) without writing"),
                        ("accept", "review one proposal (JSON object) and start it")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--domain", required=True)
        s.add_argument("--json", help="inline JSON (otherwise --file or stdin)")
        s.add_argument("--file", help="JSON file, or - for stdin")
        if name == "accept":
            s.add_argument("--no-start", action="store_true")
            s.add_argument("--started-at")

    st = sub.add_parser("start", help="start an experiment added with --no-start")
    st.add_argument("--experiment", required=True)
    st.add_argument("--started-at")

    rec = sub.add_parser("record", help="record observations: JSON list or JSONL")
    rec.add_argument("--json")
    rec.add_argument("--file")

    e = sub.add_parser("evaluate", help="judge one experiment or all running in a domain")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment")
    g.add_argument("--domain")
    e.add_argument("--now", help="ISO time to evaluate at (default: now)")

    ev = sub.add_parser("evidence", help="what the ledger supports, per variable")
    ev.add_argument("--domain", required=True)
    ev.add_argument("--markdown", action="store_true")

    s = sub.add_parser("status", help="experiments per domain")
    s.add_argument("--domain")

    ca = sub.add_parser("calibrate", help="false-winner rate / detection rate on your data")
    ca.add_argument("--domain", required=True)
    ca.add_argument("--effect", type=float, required=True, help="0 for the false-winner rate")
    ca.add_argument("--per-window", type=int, required=True, help="units per variant per window")
    ca.add_argument("--pool-file", help="JSON list of historical values (default: ledger data)")
    ca.add_argument("--sims", type=int, default=1000)

    asg = sub.add_parser("assign", help="which variant a unit should get")
    asg.add_argument("--experiment", required=True)
    asg.add_argument("--unit", required=True)
    asg.add_argument("--stratum")

    pe = sub.add_parser("pending", help="closed verdicts not yet put into effect")
    pe.add_argument("--domain", required=True)

    ma = sub.add_parser("mark-applied", help="record that a verdict is now in effect")
    ma.add_argument("--experiment", required=True)

    ab = sub.add_parser("abandon", help="close an experiment with no verdict")
    ab.add_argument("--experiment", required=True)
    ab.add_argument("--reason", required=True)

    rs = sub.add_parser("restart", help="abandon an experiment and start it again")
    rs.add_argument("--experiment", required=True)
    rs.add_argument("--reason", required=True)
    rs.add_argument("--at", help="new start time (ISO; default now)")
    rs.add_argument("--no-start", action="store_true")

    sub.add_parser("mcp", help="serve these operations over MCP on stdio")
    return p


def run(args: argparse.Namespace) -> Any:
    if args.command == "mcp":
        from .mcp_server import serve
        serve(args.ledger)
        return None
    if not args.ledger:
        raise service.ServiceError("no ledger: pass --ledger PATH or set "
                                   "ADAPTIVE_ITERATION_LEDGER")
    L = args.ledger
    if args.command == "configure":
        return service.configure(
            L, args.domain, args.metric, args.min_effect,
            higher_is_better=not args.lower_is_better, rule=args.rule,
            superiority=args.superiority, window_days=args.window_days,
            maturity_hours=args.maturity_hours, max_windows=args.max_windows,
            valid_min=args.valid_min, valid_max=args.valid_max)
    if args.command == "register-variable":
        return service.register_variable(L, args.domain, args.name, args.description,
                                         args.execution)
    if args.command == "merge-variables":
        return service.merge_variables(L, args.domain, args.name, args.into)
    if args.command == "review":
        data = _payload(args)
        return service.review_proposals(L, args.domain, data if isinstance(data, list)
                                        else [data])
    if args.command == "accept":
        data = _payload(args)
        if isinstance(data, list):
            if len(data) != 1:
                raise service.ServiceError("accept takes exactly one proposal")
            data = data[0]
        return service.accept_proposal(L, args.domain, data, start=not args.no_start,
                                       started_at=args.started_at)
    if args.command == "start":
        return service.start_experiment(L, args.experiment, args.started_at)
    if args.command == "record":
        data = _payload(args)
        return service.record_observations(L, data if isinstance(data, list) else [data])
    if args.command == "evaluate":
        return service.evaluate(L, args.experiment, domain=args.domain, now=args.now)
    if args.command == "evidence":
        return service.evidence(L, args.domain, "markdown" if args.markdown else "json")
    if args.command == "status":
        return service.status(L, args.domain)
    if args.command == "calibrate":
        pool = None
        if args.pool_file:
            with open(args.pool_file, encoding="utf-8") as f:
                pool = [float(v) for v in json.load(f)]
        return service.calibrate(L, args.domain, effect=args.effect,
                                 per_window=args.per_window, pool=pool, sims=args.sims)
    if args.command == "assign":
        return service.assign_variant(L, args.experiment, args.unit, args.stratum)
    if args.command == "pending":
        return service.pending(L, args.domain)
    if args.command == "mark-applied":
        return service.mark_applied(L, args.experiment)
    if args.command == "abandon":
        return service.abandon_experiment(L, args.experiment, args.reason)
    if args.command == "restart":
        return service.restart_experiment(L, args.experiment, args.reason, at=args.at,
                                          start=not args.no_start)
    raise service.ServiceError(f"unknown command {args.command}")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (service.ServiceError, OSError, json.JSONDecodeError) as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1
    if result is None:
        return 0
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

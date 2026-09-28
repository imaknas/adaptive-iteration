"""Every service operation, and every parameter it takes, is reachable from the CLI
and from MCP — except where leaving it out is deliberate (listed below with why)."""
import asyncio
import inspect

import pytest

from ordal import service
from ordal.cli import build_parser

OPS = sorted(n for n, f in inspect.getmembers(service, inspect.isfunction)
             if f.__module__ == "ordal.service" and not n.startswith("_"))

CLI_NAMES = {"review_proposals": "review", "accept_proposal": "accept",
             "start_experiment": "start", "record_observations": "record",
             "assign_variant": "assign", "abandon_experiment": "abandon",
             "restart_experiment": "restart", "migrate_ledger": "migrate"}
# service parameter -> CLI dest, where they differ
CLI_PARAMS = {"experiment_id": "experiment", "unit_id": "unit", "observations": "json",
              "proposals": "json", "proposal": "json", "fmt": "markdown",
              "higher_is_better": "lower_is_better", "start": "no_start",
              "pool": "pool_file", "pool_b": "pool_b_file", "at": "at"}

MCP_LEFT_OUT = {
    # converting an old ledger file is a one-time job for a person, not an agent tool
    "migrate_ledger": "one-time file conversion",
}
MCP_PARAMS_LEFT_OUT = {
    # letting an agent pick the judging time would let it jump to a checkpoint on
    # demand; over MCP a verdict is always taken at the real current time
    ("evaluate", "now"),
}
MCP_PARAMS = {("evidence", "fmt"): "markdown"}


def params(op):
    return [p for p in inspect.signature(getattr(service, op)).parameters if p != "ledger"]


def test_every_operation_has_a_cli_command_taking_all_its_parameters():
    subs = build_parser()._subparsers._group_actions[0].choices
    for op in OPS:
        name = CLI_NAMES.get(op, op.replace("_", "-"))
        assert name in subs, f"no CLI command for service.{op}"
        dests = {a.dest for a in subs[name]._actions} | {"src", "dst"}
        for p in params(op):
            assert p in dests or CLI_PARAMS.get(p) in dests, f"CLI {name} lacks {p}"


def test_every_operation_has_an_mcp_tool_taking_all_its_parameters():
    pytest.importorskip("mcp")
    from mcp import Client

    from ordal.mcp_server import build_server

    async def schemas():
        async with Client(build_server("/tmp/unused.jsonl")) as c:
            return {t.name: set(t.input_schema.get("properties", {}))
                    for t in (await c.list_tools()).tools}

    tools = asyncio.run(schemas())
    for op in OPS:
        if op in MCP_LEFT_OUT:
            assert op not in tools
            continue
        assert op in tools, f"no MCP tool for service.{op}"
        for p in params(op):
            if (op, p) in MCP_PARAMS_LEFT_OUT:
                assert p not in tools[op]
                continue
            assert MCP_PARAMS.get((op, p), p) in tools[op], f"MCP {op} lacks {p}"
    assert set(tools) <= set(OPS), f"MCP tools with no service op: {set(tools) - set(OPS)}"

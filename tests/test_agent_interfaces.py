"""service.py, the CLI and the MCP server: the interfaces agents use."""
import asyncio
import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from adaptive_iteration import service
from adaptive_iteration.cli import main as cli

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat()


def rows(exp_id, n, p_a, p_b, seed=0, start=T0):
    """n emails per variant, 0/1 clicked, produced over the first 5 days."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        for label, p in (("statement", p_a), ("question", p_b)):
            produced = start + timedelta(hours=i * 120 / n)
            out.append({"experiment_id": exp_id, "variant": label, "unit_id": f"{label}-{i}",
                        "produced_at": iso(produced),
                        "observed_at": iso(produced + timedelta(days=3)),
                        "metrics": {"clicked": 1.0 if rng.random() < p else 0.0}})
    return out


@pytest.fixture
def ledger(tmp_path):
    return str(tmp_path / "ledger.jsonl")


def setup_domain(ledger, **kw):
    service.configure(ledger, "nl", "clicked", 0.05, rule="proportion", **kw)
    service.register_variable(ledger, "nl", "subject_style", "how the subject is phrased")
    res = service.accept_proposal(ledger, "nl", {"variable": "subject_style",
                                                 "variant_a": "statement",
                                                 "variant_b": "question"},
                                  started_at=iso(T0))
    return res["experiment"]["id"]


# ── service ───────────────────────────────────────────────────────────────────

def test_full_loop_through_service(ledger):
    exp_id = setup_domain(ledger)
    service.record_observations(ledger, rows(exp_id, 400, 0.10, 0.25))
    [res] = service.evaluate(ledger, domain="nl", now=iso(T0 + timedelta(days=8)))
    d = res["decision"]
    assert d["outcome"] == "b_better" and d["closed"]
    assert "adopt variant B (question)" in d["next_action"]
    assert res["experiment"]["status"] == "closed"        # state after the verdict
    assert service.status(ledger, "nl")["nl"]["experiments"][0]["outcome"] == "b_better"
    ev = service.evidence(ledger, "nl")
    assert ev["variables"][0]["status"] == "concluded"
    assert "subject_style" in service.evidence(ledger, "nl", "markdown")


def test_interim_verdict_tells_agent_not_to_act(ledger):
    exp_id = setup_domain(ledger)
    service.record_observations(ledger, rows(exp_id, 20, 0.10, 0.12))
    [res] = service.evaluate(ledger, exp_id, now=iso(T0 + timedelta(days=8)))
    d = res["decision"]
    assert d["outcome"] == "insufficient" and not d["closed"]
    assert "do not act" in d["next_action"]
    assert d["next_checkpoint"].startswith("2026-09-15")


def test_goalposts_cannot_move_for_a_running_experiment(ledger):
    exp_id = setup_domain(ledger)
    service.record_observations(ledger, rows(exp_id, 150, 0.10, 0.14, seed=3))
    # an agent tries to rescue the experiment by making min_effect tiny
    out = service.configure(ledger, "nl", "clicked", 0.001, rule="proportion")
    assert "keep their original settings" in out["note"]
    [res] = service.evaluate(ledger, exp_id, now=iso(T0 + timedelta(days=8)))
    assert res["decision"]["settings"]["metric"]["min_effect"] == 0.05


def test_record_is_all_or_nothing_and_rejects_bad_rows(ledger):
    exp_id = setup_domain(ledger)
    good = rows(exp_id, 2, 0.1, 0.1)
    bad = [{**good[0], "variant": "shouty"}, {**good[1], "metrics": {"clicked": "yes"}}]
    with pytest.raises(service.ServiceError) as e:
        service.record_observations(ledger, good + bad)
    assert "row 4" in str(e.value) and "row 5" in str(e.value)
    assert service.status(ledger, "nl")["nl"]["experiments"][0]["observations"] == 0


def test_unconfigured_domain_and_rejections_explain_themselves(ledger):
    with pytest.raises(service.ServiceError, match="configure"):
        service.review_proposals(ledger, "nope", [{"variable": "x", "variant_a": "a",
                                                   "variant_b": "b"}])
    exp_id = setup_domain(ledger)
    with pytest.raises(service.ServiceError, match="already has an open experiment"):
        service.accept_proposal(ledger, "nl", {"variable": "subject_style",
                                               "variant_a": "x", "variant_b": "y"})
    assert exp_id


def test_calibrate_from_ledger_data_including_proportions(ledger):
    exp_id = setup_domain(ledger)
    service.record_observations(ledger, rows(exp_id, 100, 0.12, 0.12))
    null = service.calibrate(ledger, "nl", effect=0, per_window=100, sims=200)
    assert null["false_winner_rate"] <= 0.05 and null["pool_size"] == 200
    real = service.calibrate(ledger, "nl", effect=0.15, per_window=100, sims=200)
    assert real["detection_rate"] > 0.9


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_cli(capsys, *args):
    code = cli(list(args))
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip().startswith(("{", "[")) else out)


def test_cli_loop_and_errors(ledger, capsys, tmp_path):
    L = ("--ledger", ledger)
    assert run_cli(capsys, *L, "configure", "--domain", "nl", "--metric", "clicked",
                   "--min-effect", "0.05", "--rule", "proportion")[0] == 0
    run_cli(capsys, *L, "register-variable", "--domain", "nl", "--name", "subject_style")
    code, out = run_cli(capsys, *L, "accept", "--domain", "nl", "--started-at", iso(T0),
                        "--json", '{"variable": "subject_style", "variant_a": "statement",'
                                  ' "variant_b": "question"}')
    assert code == 0
    exp_id = out["experiment"]["id"]
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows(exp_id, 400, 0.1, 0.25)))
    code, out = run_cli(capsys, *L, "record", "--file", str(f))
    assert out == {"recorded": 800}
    code, out = run_cli(capsys, *L, "evaluate", "--domain", "nl",
                        "--now", iso(T0 + timedelta(days=8)))
    assert out[0]["decision"]["outcome"] == "b_better"
    code, out = run_cli(capsys, *L, "evidence", "--domain", "nl", "--markdown")
    assert "subject_style" in out
    code, out = run_cli(capsys, *L, "record", "--json", '[{"experiment_id": "zzz"}]')
    assert code == 1 and "unknown experiment" in out["error"]


def test_cli_needs_a_ledger(capsys, monkeypatch):
    monkeypatch.delenv("ADAPTIVE_ITERATION_LEDGER", raising=False)
    code, out = run_cli(capsys, "status")
    assert code == 1 and "--ledger" in out["error"]


# ── MCP ───────────────────────────────────────────────────────────────────────

def test_mcp_tools_end_to_end(ledger):
    pytest.importorskip("mcp")
    from mcp import Client

    from adaptive_iteration.mcp_server import build_server

    async def scenario():
        async with Client(build_server(ledger)) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert {"configure", "accept_proposal", "record_observations", "evaluate",
                    "evidence", "calibrate"} <= set(tools)
            assert tools["evidence"].annotations.read_only_hint is True
            assert "never act on the interim numbers" in c.instructions

            async def call(tool, **args):
                r = await c.call_tool(tool, args)
                return r

            await call("configure", domain="nl", metric="clicked", min_effect=0.05,
                       rule="proportion")
            bad = await call("accept_proposal", domain="nl",
                             proposal={"variable": "subject_style", "variant_a": "s",
                                       "variant_b": "q"})
            assert bad.is_error and "not registered" in bad.content[0].text
            await call("register_variable", domain="nl", name="subject_style")
            ok = await call("accept_proposal", domain="nl",
                            proposal={"variable": "subject_style", "variant_a": "statement",
                                      "variant_b": "question"})
            assert not ok.is_error
            return json.loads(ok.content[0].text)["experiment"]["id"]

    exp_id = asyncio.run(scenario())
    assert service.status(ledger, "nl")["nl"]["experiments"][0]["id"] == exp_id

"""replay_experiment: an experiment's checkpoints under other judging settings."""
import asyncio
import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from ordal import service
from ordal.cli import main as cli

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture
def ledger(tmp_path):
    L = str(tmp_path / "l.jsonl")
    service.configure(L, "d", "m", 3.0)
    service.register_variable(L, "d", "v")
    eid = service.accept_proposal(L, "d", {"variable": "v", "variant_a": "a",
                                           "variant_b": "b"},
                                  started_at=T0.isoformat())["experiment"]["id"]
    rng = random.Random(1)
    rows = []
    for i in range(60):
        for label, mean in (("a", 50), ("b", 56)):
            t = T0 + timedelta(hours=i * 4)
            x = rng.gauss(0, 1)
            rows.append({"experiment_id": eid, "variant": label, "unit_id": f"{label}{i}",
                         "produced_at": t.isoformat(),
                         "observed_at": (t + timedelta(days=4)).isoformat(),
                         "metrics": {"m": mean + 6 * x + rng.gauss(0, 8)},
                         "stratum": "x" if i % 2 else "y", "covariate": x})
    service.record_observations(L, rows)
    return L, eid


def events(L):
    with open(L, encoding="utf-8") as f:
        return f.read()


def test_defaults_to_the_locked_settings_and_writes_nothing(ledger):
    L, eid = ledger
    before = events(L)
    service.configure(L, "d", "m", 50.0)            # later edit must not leak into replay
    r = service.replay_experiment(L, eid)
    assert r["settings_used"]["metric"]["min_effect"] == 3.0 and r["overridden"] == []
    assert [c["checkpoint"] for c in r["checkpoints"]][0] == 1
    assert all("next_action" not in c for c in r["checkpoints"])   # hypothetical only
    assert events(L).count("\n") == before.count("\n") + 1          # only the configure call
    last = r["checkpoints"][-1]
    assert last["stratified"] is True


def test_overrides_change_only_what_is_given(ledger):
    L, eid = ledger
    base = service.replay_experiment(L, eid)
    strict = service.replay_experiment(L, eid, superiority="margin", max_windows=2)
    assert strict["overridden"] == ["max_windows", "superiority"]
    assert strict["settings_used"]["window_days"] == base["settings_used"]["window_days"]
    assert len(strict["checkpoints"]) <= 2


def test_recorded_verdict_is_shown_for_comparison(ledger):
    L, eid = ledger
    assert service.replay_experiment(L, eid)["recorded_verdict"] is None
    service.evaluate(L, eid, now=(T0 + timedelta(days=30)).isoformat())
    assert service.replay_experiment(L, eid)["recorded_verdict"] is not None


def test_errors_explain_themselves(ledger, tmp_path):
    L, eid = ledger
    with pytest.raises(service.ServiceError, match="unknown experiment"):
        service.replay_experiment(L, "nope")
    with pytest.raises(service.ServiceError, match="rule must be"):
        service.replay_experiment(L, eid, rule="bayes")
    waiting = service.accept_proposal(L, "d", {"variable": "w", "variant_a": "a",
                                               "variant_b": "b",
                                               "new_variable": {"description": "w"}},
                                      start=False)["experiment"]["id"]
    with pytest.raises(service.ServiceError, match="not started"):
        service.replay_experiment(L, waiting)


def test_cli_replay(ledger, capsys):
    L, eid = ledger
    assert cli(["--ledger", L, "replay", "--experiment", eid, "--superiority", "margin"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["settings_used"]["superiority"] == "margin"


def test_mcp_replay_is_read_only(ledger):
    pytest.importorskip("mcp")
    from mcp import Client

    from ordal.mcp_server import build_server

    L, eid = ledger

    async def scenario():
        async with Client(build_server(L)) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert tools["replay_experiment"].annotations.read_only_hint is True
            assert "never for choosing settings" in c.instructions
            r = await c.call_tool("replay_experiment", {"experiment_id": eid, "max_windows": 2})
            return json.loads(r.content[0].text)

    out = asyncio.run(scenario())
    assert out["overridden"] == ["max_windows"]

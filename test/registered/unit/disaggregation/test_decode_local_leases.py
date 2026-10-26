"""CPU ownership/state-machine tests; GPU inference is a separate requirement."""

import importlib.util
import sys
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/disaggregation/decode_local.py"
)
spec = importlib.util.spec_from_file_location("decode_local_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def ident(turn=0, request="a", predecessor=None, op="pd", **extra):
    return {
        "run": "run",
        "generation": "gen",
        "session": "s",
        "request": request,
        "turn": turn,
        "predecessor": predecessor,
        "op": op,
        **extra,
    }


@pytest.fixture
def store():
    clock = [100]
    releases, events = [], []
    config = {
        "run": "run",
        "generation": "gen",
        "ttl_ns": 10,
        "max_leases": 2,
        "max_pin_bytes": 100,
        "max_requests": 100,
    }
    result = module.LeaseStore(
        config,
        releases.append,
        lambda event, **row: events.append((event, row)),
        lambda: clock[0],
    )
    result.clock_value, result.releases, result.events = clock, releases, events
    return result


def seed(store, tokens=(1, 2), size=10):
    initial = ident()
    assert store.admit(initial, [1])[1] is None
    owner = object()
    assert store.pin(owner, initial, tokens, size)
    return owner


def test_atomic_handoff_and_single_consumption(store):
    owner = seed(store)
    execution = []
    second = ident(1, "b", "a", "local")
    lease, reason, _ = store.admit(
        second, [1, 2, 3], restore=lambda lease: execution.append(lease.owner)
    )
    assert reason is None and execution == [owner] and lease.owner is owner
    store.clock_value[0] = 111
    store.expire()
    assert store.releases == []  # TTL cannot free the execution reference.
    assert store.admit(second, [1, 2, 3])[1:] == ("duplicate", False)
    assert store.bytes == 0
    assert store.pin(owner, second, [1, 2, 3], 10)
    assert store.admit(ident(2, "c", "b", "local"), [1, 2, 3, 4])[1] is None


def test_prefix_miss_keeps_original_reference_until_pd_replacement(store):
    owner = seed(store)
    second = ident(1, "b", "a", "local")
    assert store.admit(second, [1, 9, 3])[1:] == ("prefix_mismatch", True)
    assert store.leases[("run", "s")].owner is owner
    second["op"] = "pd"
    assert store.admit(second, [1, 9, 3])[1] is None
    assert store.releases == [owner]


def test_no_refresh_and_expiry_boundary(store):
    owner = seed(store)
    expires = store.leases[("run", "s")].expires_ns
    store.clock_value[0] = 109
    assert not store.pin(owner, ident(), [1, 2], 10)
    assert store.leases[("run", "s")].expires_ns == expires
    store.clock_value[0] = 110
    assert store.admit(ident(1, "b", "a", "local"), [1, 2, 3])[1:] == ("no_lease", True)
    assert store.releases == [owner] and store.bytes == 0
    store.expire()
    assert store.releases == [owner]


def test_capacity_and_restore_failure_do_not_consume(store):
    owner = seed(store)
    request = ident(1, "b", "a", "local")
    assert store.admit(request, [1, 2, 3], capacity_ok=lambda _: False)[1:] == (
        "capacity",
        True,
    )

    def broken(_):
        raise RuntimeError("invalid state")

    with pytest.raises(RuntimeError):
        store.admit(request, [1, 2, 3], restore=broken)
    assert store.leases[("run", "s")].owner is owner
    assert ("run", "s", "b") not in store.seen


def test_run_generation_old_round_close_and_shared_independence(store):
    a = seed(store)
    bident = ident(session="other")
    assert store.admit(bident, [1])[1] is None
    b = object()
    assert store.pin(b, bident, [1, 2], 20)
    assert store.admit(ident(1, "b", "a", "local", generation="old"), [1, 2, 3])[
        1:
    ] == ("generation", False)
    assert store.admit(ident(0, "z", None, "local"), [1, 2, 3])[1:] == (
        "stale_or_predecessor",
        False,
    )
    store.close(ident())
    store.close(ident())
    assert store.releases == [a] and store.bytes == 20
    assert store.leases[("run", "other")].owner is b
    assert store.admit(ident(1, "b", "a", "local"), [1, 2, 3])[1:] == ("closed", False)


def test_pin_budget(store):
    assert store.admit(ident(), [1])[1] is None
    assert not store.pin(object(), ident(), [1], 101)
    assert store.bytes == 0


def test_restart_invalidates_old_generation(store):
    seed(store)
    store.config["generation"] = "new"
    assert store.admit(ident(1, "b", "a", "local"), [1, 2, 3])[1:] == (
        "generation",
        False,
    )

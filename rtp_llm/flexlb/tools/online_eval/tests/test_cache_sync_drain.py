"""Long request tails must not hide health observations or drain failures."""

from types import SimpleNamespace

import pytest

from adapters.flow_control import FlowObservation, FlowStatus
from adapters.java_flow import JavaFlowGroup
from cases.cache_sync_recovery.actions import drain_flow
from runtime.deadline import Deadline, StageTimeout


@pytest.fixture
def draining_flow(tmp_path, monkeypatch):
    now = [0.0]
    flow = JavaFlowGroup(None, tmp_path, run_id="run", group_id="g", phase_id="p", poll_s=.2,
                         collection_profile="aggregate")
    flow.control.mkdir()
    fault = {"exit_early": False, "bad_accounting": False}

    def status():
        terminal = now[0] >= 63
        row = dict(flow.identity, schema_version=1, revision=1, recorded_epoch_ms=1,
                    state="SENDING" if now[0] < 3 else "DRAINED" if terminal else "DRAINING",
                    submitted=1, started=1, terminal=int(terminal), outstanding=int(not terminal),
                    process_returncode=0 if now[0] >= 65 or fault["exit_early"] else None,
                    observed_started=1, observed_terminal=0 if fault["bad_accounting"] else int(terminal),
                    unfinished_ids=[])
        if now[0] >= 3:
            row["sending_end_reason"] = "natural"
        return row

    flow.status = status
    flow.observe_control = lambda: FlowObservation(FlowStatus.decode(status()), status()["process_returncode"])
    monkeypatch.setattr("adapters.java_flow.time.sleep", lambda seconds: now.__setitem__(0, now[0]+seconds))
    return SimpleNamespace(flow=flow, now=now, fault=fault,
                           deadline=Deadline(80, clock=lambda: now[0]), samples=[], events=[])


def run_cycle(state, sample=None):
    def observe(phase):
        state.samples.append((phase, state.now[0]))
        if sample is not None:
            sample()

    drain_flow(state.flow, state.deadline, observe, 2,
               lambda name: state.events.append((name, state.now[0])))


def test_sixty_second_tail_samples_through_stop_acknowledgement_and_process_exit(draining_flow):
    state = draining_flow
    run_cycle(state)
    assert state.events[0][0] == "sending_stopped"
    assert 3 <= state.events[0][1] < 4
    assert state.events[-1][0] == "drain_finished"
    assert state.events[-1][1] >= 65
    assert len(state.samples) >= 32
    assert all(phase == "drain" for phase, _ in state.samples)
    # Include both adjacent phases; resetting the clock at phase boundaries is not a fix.
    times = [0] + [t for _, t in state.samples] + [state.now[0]+2]
    assert max(b-a for a, b in zip(times, times[1:])) < 2.3
    assert any(63 <= t < 65 for _, t in state.samples)  # terminal counters alone do not prove exit
    assert state.flow.evidence_snapshot()["complete"] is True


def test_sampling_failure_propagates_and_cleanup_can_retry_without_the_observer(draining_flow):
    state = draining_flow

    def unavailable():
        if state.now[0] >= 12:
            raise ValueError("diagnostic snapshot incomplete")

    with pytest.raises(ValueError, match="snapshot incomplete"):
        run_cycle(state, unavailable)
    assert state.flow._drain_result is None
    assert not any(name == "drain_finished" for name, _ in state.events)
    # Resource cleanup uses the original observer-free API and retains final accounting checks.
    state.now[0] = 65
    assert state.flow.drain(state.deadline)["terminal"] == 1


@pytest.mark.parametrize("fault", ["deadline", "exit_early", "bad_accounting"])
def test_observation_cannot_turn_incomplete_drain_into_success(draining_flow, fault):
    state = draining_flow
    if fault == "deadline":
        state.deadline = Deadline(12, clock=lambda: state.now[0])
    else:
        state.fault[fault] = True
    with pytest.raises((StageTimeout, RuntimeError)):
        run_cycle(state)
    assert state.flow._drain_result is None
    assert not any(name == "drain_finished" for name, _ in state.events)

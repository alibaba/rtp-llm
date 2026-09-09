"""Exercise evidence prerequisites and raw red/green verdicts with simulated I/O."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import late_completion as action
from flexlb_test_framework.scenario.catalog import handlers


class LateCompletionCaseTest(unittest.TestCase):
    def run_case(self, healthy=False, delivered=True, client_ok=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now, state = [0.0], dict(missing=False, fetched=False)

            def event(
                version, running=(), hidden=False, finished=(), released=0, requested=0
            ):
                with (root / "engine_events.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            dict(
                                event="worker_status_delivery_fault",
                                port=2,
                                status_version=version,
                                requested_version=requested,
                                targets=[
                                    dict(
                                        rid=1,
                                        running=running,
                                        hidden=hidden,
                                        finished=finished,
                                        released_version=released,
                                    )
                                ],
                            )
                        )
                        + "\n"
                    )

            def http(ctx, server, path, deadline, body=None):
                if path == "inject":
                    if body["type"] == "status_completion_delay":
                        event(1, ["KV_ALLOCATED"])
                    else:
                        state["missing"] = True
                        event(2, hidden=True)
                        event(3, ["KV_ALLOCATED"])
                    return 200, {"ok": True}
                return 200, dict(
                    scheduler_inflight=0 if healthy and state["fetched"] else 1,
                    decode_endpoints=[
                        dict(
                            confirmed_accepted=int(not state["missing"]),
                            confirmed_running=0,
                        )
                    ],
                )

            class Stream:
                def __iter__(self):
                    state["fetched"] = True
                    if delivered:
                        event(4, finished=[dict(error_code=0)], released=1)
                        event(5, released=1, requested=1)
                    yield NS(
                        flatten_output=NS(finished=[client_ok]),
                        HasField=lambda key: False,
                    )

                def done(self):
                    return True

            ops = NS(
                next_request_id=lambda: 1,
                schedule=lambda *a, **k: NS(
                    code=200, success=True, enqueued_by_master=True
                ),
                role_addr=lambda *a: "engine:2",
                prefill_addr=lambda *a: "engine:1",
                _channel=lambda a: a,
                pb2=NS(FetchRequestPB=lambda **kw: kw),
                pb2_grpc=NS(
                    RpcServiceStub=lambda channel: NS(
                        FetchResponse=lambda *a, **kw: Stream()
                    )
                ),
            )
            ctx = NS(
                env_epoch=1,
                artifact_dir=root,
                env=NS(run_dir=root),
                ops=ops,
                clock=lambda: now[0],
            )
            deadline = NS(
                check=lambda: None,
                remaining=lambda: 90 - now[0],
                sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            )
            with patch.object(action, "_http", http):
                result = action.execute(
                    ctx,
                    dict(missing_rounds=1, delay_ms=300, clean_window_s=30),
                    deadline,
                )
            return {c.id: c.status for c in result.checks}, json.loads(
                (root / "late-completion.json").read_text()
            )

    def test_delivered_terminal_with_zombie_is_raw_fail_after_full_window(self):
        checks, evidence = self.run_case()
        self.assertEqual(checks, dict(construction="PASS", scheduler_clean="FAIL"))
        self.assertGreaterEqual(
            evidence["clean_finished_s"] - evidence["clean_started_s"], 30
        )

    def test_correct_scheduler_can_pass_same_case(self):
        checks, _ = self.run_case(healthy=True)
        self.assertEqual(set(checks.values()), {"PASS"})

    def test_missing_delivery_or_failed_client_is_invalid_construction(self):
        for kw in (dict(delivered=False), dict(client_ok=False)):
            checks, _ = self.run_case(**kw)
            self.assertEqual(set(checks.values()), {"ERROR"})

    def test_exactly_one_instance_and_no_finding_wrapper(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scenarios/status/late_completion.yaml"
        )
        instances = compile_scenarios(load_scenarios(source), handlers=handlers())
        self.assertEqual(
            [i["id"] for i in instances],
            ["late_completion::after_missing::single-batch"],
        )
        self.assertFalse(instances[0].get("findings"))


if __name__ == "__main__":
    unittest.main()

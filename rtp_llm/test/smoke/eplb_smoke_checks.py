"""Verify real, completed expert migration before the final smoke request."""

import json
import os
import re
from pathlib import Path


def check_eplb_and_probe(runner, server, roles, state):
    if os.environ.get("SMOKE_ASSERT_EPLB_APPLIED", "0").lower() not in ("1", "true"):
        return
    evidence = {}
    try:
        for role, manager, ep_size in roles:
            evidence[role] = check_eplb_applied(
                Path(manager.log_file_path).parent, ep_size
            )
        probe = runner.task_info.model_copy(deep=True)
        probes = [
            q
            for q in probe.query_result
            if q.get("case_name") == "eplb_post_rebalance_json_probe"
        ]
        if len(probes) != 1:
            raise AssertionError(
                "EPLB smoke requires exactly one eplb_post_rebalance_json_probe"
            )
        probe.query_result = probes
        probe.taskinfo_rel_path += ".post_eplb"
        result = runner._curl_server_impl(server, probe)
        state.query_status.extend(result.query_status)
        state.total_count += result.total_count
        state.ret = state.ret and result.ret
        evidence["post_rebalance_request_passed"] = result.ret
    except Exception as error:
        state.ret = False
        state.err_msg += f" EPLB migration/probe failed: {error}"
        evidence["error"] = str(error)
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    (output / "eplb_migration_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


_APPLIED = re.compile(
    r"\[EPLB\] applied model_id=(\d+) layers=(\[[\d, ]+\]) update=(\d+) "
    r"changed_slots=(\[[\d, ]+\]) ep_rank=(\d+)"
)


def check_eplb_applied(log_dir: Path, ep_size: int) -> list:
    """Require target-only, nonzero migration records from every EP rank.

    A completed group update has identical layers and changed_slots on every rank.
    stdout and worker logs may contain the same record; duplicates are harmless.
    """
    updates = {}
    for path in log_dir.rglob("*.log*"):
        if not path.is_file():
            continue
        for match in _APPLIED.finditer(path.read_text(errors="replace")):
            model, layers, update, changed, rank = match.groups()
            model, update, rank = int(model), int(update), int(rank)
            layers, changed = tuple(json.loads(layers)), tuple(json.loads(changed))
            if model != 0:
                raise AssertionError(
                    f"Unexpected draft EPLB migration for model_id={model} in {path}"
                )
            if len(layers) != len(changed):
                raise AssertionError(f"EPLB layer/changed-slot counts differ in {path}")
            if any(changed):
                updates.setdefault((model, layers, update), {})[rank] = changed

    completed = [
        (key, ranks)
        for key, ranks in sorted(updates.items())
        if set(ranks) == set(range(ep_size)) and len(set(ranks.values())) == 1
    ]
    if not completed:
        raise AssertionError(
            f"No completed EPLB migration for model_id=0 on all "
            f"{ep_size} ranks in {log_dir}; observed updates: {updates}"
        )
    (model, layers, update), ranks = completed[0]
    return [
        {
            "model_id": model,
            "layer_ids": list(layers),
            "update": update,
            "changed_slots": list(ranks[0]),
            "ep_ranks": sorted(ranks),
        }
    ]

"""SSH placement and previews from canonical config, with no network effects."""

from __future__ import annotations

from typing import Any, Mapping

from rtp_llm.test.perf_test.decode.scheduler import requires_distributed_decode
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import SshResourcePlan, derive_gang_topology, freeze


def plan_resources(
    *, mode: str, resolved_config: Mapping[str, Any], world_size: int, case_count: int
) -> SshResourcePlan:
    execution = resolved_config["execution"]
    if execution["backend"] != "ssh":
        raise PlanningError("SSH resource planning requires backend=ssh")
    layout = derive_gang_topology(world_size, execution["gpus_per_machine"])
    local = layout.local_world_size
    count = layout.machine_count
    hosts = execution["hosts"]
    if len(hosts) != count:
        raise PlanningError(
            f"execution.hosts requires exactly {count} members; got {len(hosts)}"
        )
    if mode == "decode":
        requires_distributed_decode(resolved_config["model"]["topology"], local)
    return SshResourcePlan(
        "ssh", world_size, count, local, tuple(freeze(host) for host in hosts)
    )


def ssh_preview(plan):
    execution = plan.resolved_config["execution"]
    resource = plan.resource_plan
    commands = []
    for index, host in enumerate(resource.hosts):
        base = ["ssh"]
        if execution.get("run_user"):
            base += ["-l", execution["run_user"]]
        if execution.get("ssh_port"):
            base += ["-p", str(execution["ssh_port"])]
        commands.append(
            {
                "stage": "member",
                "member_id": str(index),
                "argv": [
                    *base,
                    host["ssh_target"],
                    execution["python_bin"],
                    "<versioned-worker>",
                    "--descriptor",
                    "<generated-member-descriptor>",
                ],
            }
        )
    return {
        "commands": commands,
        "gang": {
            "machine_count": resource.machine_count,
            "local_world_size": resource.local_world_size,
            "world_size": resource.world_size,
            "members": [
                {
                    "member_id": str(index),
                    "ssh_target": host["ssh_target"],
                    "advertise_host": host["advertise_host"],
                    "world_rank": index * resource.local_world_size,
                }
                for index, host in enumerate(resource.hosts)
            ],
        },
    }

"""Freeze observed launch inputs before reports are assembled."""

import json
from pathlib import Path


def collect(directory, environments, metadata=None, *, required=()):
    epochs = {}
    for epoch, location in environments.items():
        root = Path(location)
        data = {"directory": str(root), "topology": (metadata or {}).get(epoch)}
        for field, filename in (("master_artifact", "master-artifact.json"),
                                ("actual_master_config", "actual-master-config.json"),
                                ("performance", "perf.json"), ("master_config", "master_config.json")):
            path = root / filename
            if path.is_file():
                data[field] = json.loads(path.read_text())
            elif field in required:
                raise ValueError(f"required provenance file missing: {path}")
        epochs[epoch] = data
    flows = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("**/flow-input.json"))]
    from runtime.paths import MOCK_JAR
    from traffic.traffic_source import sha256_file
    mock_artifact = dict(path=str(MOCK_JAR), sha256=sha256_file(MOCK_JAR)) if Path(MOCK_JAR).is_file() else None
    return dict(environments=epochs, flows=flows, mock_artifact=mock_artifact)


def gate_provenance(ctx, flow, *, source_files):
    """Freeze the same observed launch inputs for all gates, before collection."""
    from runtime.paths import API_JAR, MOCK_JAR
    from runtime.java_flow import evidence_environment
    from traffic.traffic_source import sha256_file
    from workload.gate_evidence import trace_workload_sha

    epoch = str(ctx.env_epoch)
    observed = collect(ctx.artifact_dir, {epoch: ctx.env.run_dir}, required={
        "master_artifact", "actual_master_config", "master_config", "performance"})
    launch = observed["environments"][epoch]
    trace = dict(flow.trace_manifest)
    trace["workload_sha256"] = trace_workload_sha(trace["path"])
    client = json.loads((flow.directory / "flow-input.json").read_text())
    files = {str(path): sha256_file(path) for path in (API_JAR, MOCK_JAR, *source_files)}
    return dict(instance=ctx.instance["id"],
        configuration_sha256=ctx.instance["implementation"]["configuration_sha256"],
        topology=dict(prefill=ctx.env.spec.n_prefill, decode=ctx.env.spec.n_decode),
        capacity=dict(prefill_cache_blocks=ctx.env.spec.prefill_cache_blocks,
                      decode_cache_blocks=ctx.env.spec.decode_cache_blocks,
                      mock_extra_args=ctx.env.spec.mock_extra_args),
        master_artifact=launch["master_artifact"], actual_master_config=launch["actual_master_config"],
        master_config=launch["master_config"], performance=launch["performance"],
        trace=trace, files=files, client_environment=evidence_environment(client["environment"]))

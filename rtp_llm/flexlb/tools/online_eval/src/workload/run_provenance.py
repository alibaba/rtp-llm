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


def gate_provenance(ctx, flow, *, source_files, analyzer_file):
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
    from runtime import observation
    from cases import windows, metric_inputs, check_inputs, numeric_parameters
    import input_contract
    from analysis import statistics, time_buckets
    common = (Path(__file__), observation.__file__, windows.__file__, metric_inputs.__file__, input_contract.__file__,
              statistics.__file__, time_buckets.__file__, check_inputs.__file__, numeric_parameters.__file__)
    files = {str(path): sha256_file(path) for path in (API_JAR, MOCK_JAR, *source_files, *common)}
    value = dict(instance=ctx.instance["id"], env_epoch=ctx.env_epoch,
        configuration_sha256=ctx.instance["implementation"]["configuration_sha256"],
        topology=dict(prefill=ctx.env.spec.n_prefill, decode=ctx.env.spec.n_decode),
        capacity=dict(prefill_cache_blocks=ctx.env.spec.prefill_cache_blocks,
                      decode_cache_blocks=ctx.env.spec.decode_cache_blocks,
                      mock_extra_args=ctx.env.spec.mock_extra_args),
        master_artifact=launch["master_artifact"], actual_master_config=launch["actual_master_config"],
        master_config=launch["master_config"], performance=launch["performance"],
        trace=trace, files=files, client_environment=evidence_environment(client["environment"]),
        mock_jar_sha256=files[str(MOCK_JAR)], analyzer_sha256=sha256_file(analyzer_file))
    return validate_gate_provenance(value)


def validate_gate_provenance(value):
    """Validate the shared identity/artifact contract; case policies stay with cases."""
    if not isinstance(value, dict):
        raise ValueError("provenance must be an object")
    identity = value.get("instance")
    if type(identity) is not str or not identity.strip():
        raise ValueError("missing/invalid instance identity")
    for name in ("master_artifact", "actual_master_config", "performance",
                 "topology", "capacity", "trace", "client_environment"):
        if not isinstance(value.get(name), dict) or not value[name]:
            raise ValueError("missing/invalid provenance mapping: " + name)
    hashes = {
        "configuration_sha256": value.get("configuration_sha256"),
        "mock_jar_sha256": value.get("mock_jar_sha256"),
        "analyzer_sha256": value.get("analyzer_sha256"),
        "master_artifact.jar_sha256": value["master_artifact"].get("jar_sha256"),
        "trace.sha256": value["trace"].get("sha256"),
        "trace.workload_sha256": value["trace"].get("workload_sha256"),
    }
    for name, digest in hashes.items():
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing/invalid artifact or workload SHA256: " + name)
    for group, fields in (("topology", ("prefill", "decode")),
                          ("capacity", ("prefill_cache_blocks", "decode_cache_blocks"))):
        for name in fields:
            number = value[group].get(name)
            if type(number) is not int or number < 0:
                raise ValueError("missing/invalid provenance count: " + group + "." + name)
    return value

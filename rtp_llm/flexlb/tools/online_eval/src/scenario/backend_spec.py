"""Convert frozen environment plans into controlled Java startup specifications."""

def make_env_spec(plan, profile, lease):
    """Controlled environment rendering; no JVM is started by this function."""
    from flexlb_cfg import OMIT, ConfigOverride
    from adapters.environment_config import EnvSpec, MasterSpec

    kwargs = {
        k: OMIT if v == {"omit": True} else v
        for k, v in plan["config_overrides"].items()
    }
    from scenario.environment_snapshot import EnvironmentSnapshot
    snapshot = EnvironmentSnapshot.read(plan["rendered"])
    preset_perf, preset_runtime = snapshot.performance, snapshot.runtime
    from adapters.zookeeper_config import ZooKeeperConfig
    elected = plan.get('master_layout') == 'dual_zookeeper'
    spec = EnvSpec(
        label="scenario",
        runtime_mode="scenario",
        n_prefill=plan["n_prefill"],
        n_decode=plan["n_decode"],
        master_profile=profile,
        config_overrides=ConfigOverride(**kwargs),
        raw_config=snapshot.master_json,
        discovery=plan["discovery"],
        master_stable_window_s=plan.get("master_stable_window_s", 3),
        masters=(
            [
                MasterSpec(name="A", http_port=lease["master_base"], advertised_ip="127.0.0.1" if elected else None),
                MasterSpec(name="B", http_port=lease["master_base"] if elected else lease["master_base"] + 3,
                           bind_ip="127.0.0.2" if elected else "127.0.0.1",
                           advertised_ip="127.0.0.2" if elected else None),
            ]
            if plan.get("master_layout", "single") in ("dual_standalone", "dual_zookeeper")
            else []
        ),
        zk_consistency=ZooKeeperConfig.read(plan["zookeeper"]) if elected else None,
        perf=preset_perf,
        master_env=({"FLEXLB_DEBUG_ENABLED": "true"} if plan["debug_enabled"] else {}),
        master_debug_log=plan.get("master_debug_log", False),
    )
    if plan.get("master_test_control", False):
        # A per-environment token opts the isolated Master into test controls.
        import secrets
        spec.master_env["FLEXLB_TEST_CONTROL_TOKEN"] = secrets.token_hex(32)
    if "mock_heap" in preset_runtime:
        spec.mock_heap = preset_runtime["mock_heap"]
    spec.mock_extra_args.extend(preset_runtime.get("mock_extra_args", []))
    for key in (
        "prefill_cache_blocks",
        "decode_cache_blocks",
        "mock_auto_fetch",
        "mock_fetch_attach_timeout_ms",
    ):
        if key in plan:
            setattr(spec, key, plan[key])
    if plan["debug_enabled"]:
        spec.master_extra_args.append("--flexlb.debug.enabled=true")
    if "metric_whitelist" in plan:
        spec.master_extra_args.append(
            "--flexlb.monitor.metric-whitelist=" + plan["metric_whitelist"]
        )
    return spec


def configure_master_sync_log(spec, artifact_dir, env_epoch):
    """Choose a private per-setup log directory, with no arbitrary path input."""
    from pathlib import Path

    if spec.masters or type(env_epoch) is not int or env_epoch < 1:
        raise ValueError(
            "private sync log requires one master and a positive environment epoch"
        )
    log_dir = Path(artifact_dir).resolve() / f"master-sync-{env_epoch}"
    log_dir.mkdir(parents=True, exist_ok=False)
    spec.master_extra_args = [*spec.master_extra_args, f"--flexlb.log.path={log_dir}"]
    return log_dir / "sync.log"



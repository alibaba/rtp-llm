load(":defs.bzl", "custom_smoke_test")

def timeout_recovery_suites():
    """Full model-service E2E; short production budgets, no delay injection."""
    tests = []
    for scenario, tp, legacy in [
        ("local_queue", 1, False),
        ("local_queue", 1, True),
        ("local_rpc", 2, False),
        ("local_d2h_rpc", 2, False),
        ("kvcm_queue", 1, False),
        ("kvcm_rpc", 2, False),
        ("kvcm_sdk", 1, False),
    ]:
        name = "cache_timeout_" + scenario + ("_legacy" if legacy else "")
        data = ["//rtp_llm:sdk", "timeout_recovery_checks.py"]
        if scenario.startswith("kvcm"):
            data.append("//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin")
        tests.append(custom_smoke_test(
            name = name,
            main = "cache_timeout_recovery_e2e_test.py",
            smoke_args = "--tp_size %d --world_size %d" % (tp, tp),
            args = ["--scenario", scenario] + (["--legacy"] if legacy else []),
            gpu_type = ["L20_CU13"],
            tags = ["no-remote", "cache_timeout_recovery"],
            data = data,
            deps = [":pace_smoke_fixture", "//rtp_llm:requests", "//rtp_llm:transformers"],
        ))
    native.test_suite(
        name = "smoke_cache_timeout_recovery",
        tests = tests,
        tags = ["manual"],
    )

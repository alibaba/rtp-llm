# Static CI targets keep this default; generated packages pass the frozen
# execution.test_timeout_seconds to both this watchdog and Bazel --test_timeout.
_PERF_TEST_WATCHDOG_SECONDS = "7000"

def _gpu_exec_properties(gpu_type, gpu_count, extra):
    properties = dict(extra)
    for key, value in {"gpu": gpu_type, "gpu_count": str(gpu_count)}.items():
        if key in properties and properties[key] != value:
            fail("exec_properties.%s conflicts with planned value %s" % (key, value))
        properties[key] = value
    return properties

def unified_prefill_ci_test(name, config, gpu_type, gpu_count, data = [], deps = [], watchdog_seconds = _PERF_TEST_WATCHDOG_SECONDS):
    """Run one strict unified prefill config in an AOne Bazel test action."""
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:local_main.py",
        srcs = ["//rtp_llm/test/perf_test:local_main.py"],
        timeout = "eternal",
        env = {
            "GPU_COUNT": str(gpu_count),
            "RTP_TEST_WATCHDOG_SECONDS": str(watchdog_seconds),
        },
        deps = [
            "//rtp_llm/test/perf_test:perf_test_entry_lib",
            "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ] + deps,
        data = [
            "//rtp_llm:sdk",
            config,
        ] + data,
        args = [
            "--mode=prefill",
            "--action=run",
            "--config=$(location " + config + ")",
        ],
        tags = ["manual", gpu_type],
        exec_properties = {
            "gpu": gpu_type,
            "gpu_count": str(gpu_count),
        },
    )
    return name


def perf_test_aone_shard(name, resolved_config, resolved_workload, shard_plan, shard_id, gpu_type, gpu_count, watchdog_seconds = _PERF_TEST_WATCHDOG_SECONDS, exec_properties = {}):
    """Define one unified prefill shard using only declared runfiles."""
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:aone/worker.py",
        srcs = ["//rtp_llm/test/perf_test:aone/worker.py"],
        timeout = "eternal",
        deps = ["//rtp_llm/test/perf_test:aone_lib"],
        data = [
            "//rtp_llm:sdk",
            resolved_config,
            resolved_workload,
            shard_plan,
        ],
        args = [
            "--resolved-config=$(location " + resolved_config + ")",
            "--resolved-workload=$(location " + resolved_workload + ")",
            "--shard-plan=$(location " + shard_plan + ")",
            "--shard-id=" + shard_id,
        ],
        tags = ["manual", gpu_type],
        exec_properties = _gpu_exec_properties(gpu_type, gpu_count, exec_properties),
        env = {
            "WORLD_SIZE": str(gpu_count),
            "RTP_TEST_WATCHDOG_SECONDS": str(watchdog_seconds),
        },
    )


def perf_test_gang_shard(name, resolved_config, resolved_workload, gang_plan, gpu_type, gpus_per_host, machine_count, watchdog_seconds = _PERF_TEST_WATCHDOG_SECONDS, exec_properties = {}):
    """Define one multi-machine gang run as sharded remote test actions.

    The single target expands to ``machine_count`` test shards and every
    shard requests the full registered per-host GPU count so each member
    occupies a whole machine. No world-size env is set here: the gang
    worker computes it from the gang plan. The submit side runs these
    targets with the frozen ``execution.test_timeout_seconds`` budget.
    """
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:aone/gang_worker.py",
        srcs = ["//rtp_llm/test/perf_test:aone/gang_worker.py"],
        timeout = "eternal",
        shard_count = machine_count,
        env = {
            "GPU_COUNT": str(gpus_per_host),
            "RTP_TEST_WATCHDOG_SECONDS": str(watchdog_seconds),
        },
        deps = ["//rtp_llm/test/perf_test:aone_lib"],
        data = [
            "//rtp_llm:sdk",
            resolved_config,
            resolved_workload,
            gang_plan,
        ],
        args = [
            "--resolved-config=$(location " + resolved_config + ")",
            "--resolved-workload=$(location " + resolved_workload + ")",
            "--gang-plan=$(location " + gang_plan + ")",
        ],
        tags = ["manual", gpu_type],
        exec_properties = _gpu_exec_properties(gpu_type, gpus_per_host, exec_properties),
        env_inherit = [
            "PERF_TEST_BROKER_TOKEN",
        ],
    )


def unified_decode_remote_test(name, config, gpu_type, gpu_count, data = [], deps = [], watchdog_seconds = _PERF_TEST_WATCHDOG_SECONDS):
    """Run all local-backend decode cases in one ordinary GPU test action.

    Config and optional workload inputs are declared runfiles. Model and
    tokenizer paths in config must be readable on the selected worker.
    Results are retained through TEST_UNDECLARED_OUTPUTS_DIR.
    """
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:decode_remote_test.py",
        srcs = ["//rtp_llm/test/perf_test:decode_remote_test.py"],
        timeout = "eternal",
        env = {
            "GPU_COUNT": str(gpu_count),
            "RTP_TEST_WATCHDOG_SECONDS": str(watchdog_seconds),
        },
        deps = [
            "//rtp_llm/test/perf_test:perf_test_entry_lib",
            "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ] + deps,
        data = ["//rtp_llm:sdk", config] + data,
        args = ["--config=$(location " + config + ")"],
        tags = ["manual", gpu_type],
        exec_properties = {
            "gpu": gpu_type,
            "gpu_count": str(gpu_count),
        },
    )

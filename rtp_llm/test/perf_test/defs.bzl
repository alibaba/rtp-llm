load("//rtp_llm/test/smoke:defs.bzl", "get_world_size_from_smoke_args")

def tp_tuple(gpu_type, tp_size):
    out = []
    for gpu, max_count in gpu_type:
        for tp in tp_size:
            if max_count >= tp:
                out.append((gpu, tp))
    return out

def perf_test(name, model_type, ckpt_path, gpu_type,
              tokenizer_path = None,
              compare_test_result = "",
              envs = {},
              args = [], data = [], deps = []):
    if tokenizer_path == None:
        tokenizer_path = ckpt_path

    world_size = get_world_size_from_smoke_args(" ".join(args))

    test_args = [
        "--model_type", model_type,
        "--checkpoint_path", ckpt_path,
        "--tokenizer_path", tokenizer_path,
        "--perf_test_name", name,
    ]

    extra_data = []
    if compare_test_result:
        test_args += ["--compare_test_result", "$(location " + compare_test_result + ")"]
        extra_data.append(compare_test_result)

    for k, v in envs.items():
        test_args += ["--test_env", k + "=" + v]

    test_args += args

    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:test_entry.py",
        srcs = ["//rtp_llm/test/perf_test:test_entry.py"],
        timeout = "eternal",
        deps = [
            "//rtp_llm:pyodps",
            "//rtp_llm:testlib",
            "//rtp_llm/test/perf_test:perf_test_lib",
        ] + deps,
        data = [
            "//rtp_llm:sdk",
        ] + extra_data + data,
        args = test_args,
        tags = ["manual", gpu_type],
        exec_properties = {
            "gpu": gpu_type,
            "gpu_count": str(world_size),
        },
        env = {
            "WORLD_SIZE": str(world_size),
        }
    )
    return name

def cache_grid_ci_test(name, profile, grid, gpu_type, gpu_count, envs = {}, measure_runs = None):
    """Define one independently schedulable cache-grid shard."""
    test_args = [
        "--profile=$(location " + profile + ")",
        "--cache_grid_json=$(location " + grid + ")",
        "--partial=2",
        "--decode_test_length=1",
        "--batch_size=1",
        "--cache_request_transport=dashsc_input_ids",
    ]
    if measure_runs != None:
        test_args.append("--cache_measure_runs=" + str(measure_runs))

    test_env = {}
    test_env.update(envs)
    test_env["WORLD_SIZE"] = str(gpu_count)
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:batch_decode_test.py",
        srcs = ["//rtp_llm/test/perf_test:batch_decode_test.py"],
        timeout = "eternal",
        deps = [
            "//rtp_llm:pyodps",
            "//rtp_llm:testlib",
            "//rtp_llm/test/perf_test:perf_test_lib",
        ],
        data = [
            "//rtp_llm:sdk",
            profile,
            grid,
        ],
        args = test_args,
        tags = ["manual", gpu_type],
        exec_properties = {
            "gpu": gpu_type,
            "gpu_count": str(gpu_count),
        },
        env = test_env,
    )

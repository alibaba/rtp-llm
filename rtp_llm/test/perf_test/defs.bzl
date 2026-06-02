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

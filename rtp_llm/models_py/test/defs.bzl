"""Models-py tests: explicit device reservation and a shared unittest entrypoint."""

def models_py_test(name, src, gpu = "", gpus = 1, deps = [], data = [], args = [], env = {}, tags = [], testlib = "//rtp_llm:testlib"):
    runner = "//rtp_llm/models_py/test:test_main.py"
    test_env = {
        "GPU_COUNT": str(gpus if gpu else 0),
        "MODELS_PY_TEST_GPU_TYPE": gpu,
    }
    test_env.update(env)
    native.py_test(
        name = name,
        srcs = [src, runner],
        main = runner,
        args = [native.package_name().replace("/", ".") + "." + src[:-3].replace("/", ".")] + args,
        deps = [
            testlib,
            "//rtp_llm/models_py/test:test_runner",
        ] + deps,
        data = data,
        env = test_env,
        tags = tags + ([gpu] if gpu else []),
        exec_properties = {"gpu": gpu, "gpu_count": str(gpus)} if gpu else {},
    )

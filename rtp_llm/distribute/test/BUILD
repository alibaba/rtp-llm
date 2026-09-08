filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
        "testdata/tokenizer",
        "testdata/cpt",
    ]),
)

# Needs 2 visible devices, no interconnect and in fact no GPU compute: the test
# stubs torch.cuda.set_device out (distributed_server_test.py:18) and registers a
# FakeModel, so the 2 ranks only exercise gang/world-info registration over
# localhost sockets. This target carries no "H20" tag -- only the exec_properties
# pin was routing it to H20 -- so dropping the 'gpu' key lets the ut-sm8x job
# default (--remote_default_exec_properties=gpu=A10) take effect. Passed on sm86
# (8x RTX 3090). A10 CI workers expose 4 GPUs, so gpu_count 2 is satisfiable.
py_test(
    name = "distributed_server_test",
    srcs = [
        "distributed_server_test.py",
        "fake_model.py"
    ],
    data = [
        ":testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    timeout = 'short',
    flaky = True,
    imports = ["aios/kmonitor/python_client"],
    exec_properties = {'gpu':'A10', 'gpu_count':'2'},
    env = {"GPU_COUNT": "2"},
)

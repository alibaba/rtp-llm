# Dynamic decode attention backend dispatch unit tests.
#
# decode_gate / selector do no GPU *compute* (only torch + math), but they still
# import the dispatch modules, which transitively pulls in librtp_compute_ops
# (via rtp_llm.models_py.modules.__init__ -> modules.base -> device -> ops).
# That .so links libcuda.so.1 (the NVIDIA *driver* lib), so these tests fail at
# import time on a driver-less CPU node. In CI the untagged-test stage
# (--test_tag_filters=...,-H20,...) schedules onto the gpu=gpu_cuda12 remote pool,
# which is heterogeneous and includes cuda12_9-cpu nodes with no driver -> flaky
# ImportError: libcuda.so.1: cannot open shared object file.
#
# Making them truly CPU-safe would require decoupling the modules package
# __init__ from compute_ops (broad, risky). Instead pin them to the H20 SKU like
# the sibling GPU tests below, so CI runs them in the H20 stage on driver-equipped
# nodes. They don't need real GPU compute -- H20 is just the available tag that
# routes to a node that has the CUDA driver.

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_decode_gate",
    srcs = ["test_decode_gate.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_selector",
    srcs = ["test_selector.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# backend_bench exercises real CUDA (clone/graph/eager timing), so it needs a GPU.
py_test(
    name = "test_backend_bench",
    srcs = ["test_backend_bench.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

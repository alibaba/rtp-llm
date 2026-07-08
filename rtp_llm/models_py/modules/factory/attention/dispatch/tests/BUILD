# Dynamic decode attention backend dispatch unit tests.
#
# selector does no GPU *compute* (only torch + math), but it still imports the
# dispatch modules, which transitively pulls in librtp_compute_ops (via
# rtp_llm.models_py.modules.__init__ -> modules.base -> device -> ops). That .so
# links libcuda.so.1 (the NVIDIA *driver* lib), so these tests fail at import
# time on a driver-less CPU node. Pin to H20 SKU so CI runs them on
# driver-equipped nodes.

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

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

# backend_selector uses only mocks (no GPU compute), but importing the dispatch
# package transitively pulls in librtp_compute_ops (links libcuda.so.1), so keep it
# pinned to a driver-equipped node like the other dispatch tests.
py_test(
    name = "test_backend_selector",
    srcs = ["test_backend_selector.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

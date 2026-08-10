
load(
    "//bazel:cuda13_packages.bzl",
    "CUDA13_EXPECTED_DEPENDENCY_VERSIONS",
    "CUDA13_UNAVAILABLE_REQUIREMENTS",
)

test_envs = {}

cuda13_dependency_import_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flash-mla",
    "//rtp_llm:flashinfer-python",
    "//rtp_llm:nvidia-cutlass-dsl",
    "//rtp_llm:rtp-kernel",
    "//rtp_llm:tilelang",
    "@rules_python//python/runfiles",
] + [
    "//rtp_llm:" + target
    for target in CUDA13_UNAVAILABLE_REQUIREMENTS.values()
]

py_test(
    name = "cuda13_dependency_import_test",
    srcs = ["cuda13_dependency_import_test.py"],
    deps = cuda13_dependency_import_deps,
    env = {
        "CUDA13_PIP_REPO_PREFIX": "pip_gpu_cuda13_torch",
        "CUDA13_UNAVAILABLE_DISTRIBUTIONS": ",".join(CUDA13_UNAVAILABLE_REQUIREMENTS.keys()),
        "EXPECTED_APACHE_TVM_FFI_VERSION": CUDA13_EXPECTED_DEPENDENCY_VERSIONS["x86"][
            "apache-tvm-ffi"
        ],
        "EXPECTED_NVIDIA_CUTLASS_DSL_VERSION": CUDA13_EXPECTED_DEPENDENCY_VERSIONS["x86"][
            "nvidia-cutlass-dsl"
        ],
    },
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "cuda13_arm_dependency_import_test",
    srcs = ["cuda13_dependency_import_test.py"],
    main = "cuda13_dependency_import_test.py",
    deps = cuda13_dependency_import_deps,
    env = {
        "CUDA13_PIP_REPO_PREFIX": "pip_cuda13_arm_torch",
        "CUDA13_UNAVAILABLE_DISTRIBUTIONS": ",".join(CUDA13_UNAVAILABLE_REQUIREMENTS.keys()),
        "EXPECTED_APACHE_TVM_FFI_VERSION": CUDA13_EXPECTED_DEPENDENCY_VERSIONS["arm"][
            "apache-tvm-ffi"
        ],
        "EXPECTED_NVIDIA_CUTLASS_DSL_VERSION": CUDA13_EXPECTED_DEPENDENCY_VERSIONS["arm"][
            "nvidia-cutlass-dsl"
        ],
    },
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test (
    name = "qwen3_test",
    srcs = ["qwen3_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:auto_model",
    ],
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

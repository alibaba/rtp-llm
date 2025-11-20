package(default_visibility = ["//visibility:public"])
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
# make different torch for different device when in compiling
load("//bazel:defs.bzl", "upload_pkg", "copy_target_to", "upload_wheel", "rename_wheel", "rename_wheel_aarch64")
load("//bazel:arch_select.bzl", "requirement", "whl_deps", "internal_deps", "jit_deps", "triton_deps")
load("//open_source/bazel:arch_select.bzl", "platform_deps")
load("//bazel:bundle.bzl", "bundle_files", "bundle_tar")

load("@release_version//:defs.bzl", "RELEASE_VERSION")

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
    _if_rocm = "if_rocm",
)

if_rocm = _if_rocm

tensorrt = [
    "tensorrt",
    "tensorrt-cu12-bindings",
    "tensorrt-cu12-libs",
]

flashinfer = [
    "flashinfer-python",
]

xft_dep = select({
    "@//:using_arm": [],
    "//:xft_use_icx": [
        # "xfastertransformer_devel_icx",
    ],
    "//conditions:default": [
        # "xfastertransformer_devel",
    ],
})

arch_dep = select({
    "@//:using_arm": [],
    "@//:using_cuda12_arm": [],
    "//conditions:default": [":decord"],
})

arch_with_version_dep = select({
    "@//:using_arm": [],
    "@//:using_cuda12_arm": [],
    "//conditions:default": ["decord==0.6.0"],
})

requirement([
    "sentencepiece",
    "transformers",
    "pynvml",
    "tiktoken",
    "protobuf",
    "grpcio-tools",
    "setuptools",
    "Pillow",
    "pillow-heif",
    "pillow-avif-plugin",
    "lru-dict",
    "cpm_kernels",
    "uvicorn",
    "fastapi",
    "psutil",
    "pyodps",
    "thrift",
    "torch",
    "torchvision",
    "numpy",
    "safetensors",
    "einops",
    "prettytable",
    "timm",
    "aiohttp",
    "onnx",
    "sentence-transformers",
    "orjson",
    "xfastertransformer_devel",
    "xfastertransformer_devel_icx",
    "decord",
    # add qwen agent package
    "pydantic",
    "json5",
    "dashscope",
    "jieba",
    "openai",
    "oss2",
    "pyOpenSSL",
    "nest_asyncio",
    "blobfile",
    "partial_json_parser",
    "librosa",
    "matplotlib", # required by qwen vl tokenizer
    "flash_attn",
    "av",
    "pyrsmi",
    "amdsmi",
    "fast-safetensors",
    "setproctitle",
    "bitsandbytes",
    "portalocker",
    "concurrent_log_handler",
    "aiter",
    "fastsafetensors",
] + tensorrt + flashinfer)

filegroup(
    name = "cutlass_config",
    srcs = glob(["utils/gemm_utils/luts/*"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "empty_target",
    srcs = [],
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = glob([
        "utils/**/*.py",
    ]),
    data = [":cutlass_config"],
    deps = [
        ":torch",
        ":safetensors",
        #":decord",
        ":lru-dict",
        ":cpm_kernels",
        ":prettytable",
        ":psutil",
        "//rtp_llm/aios/kmonitor:kmonitor_py",
    ] + arch_dep + internal_deps()
)

py_library(
    name = "eplb",
    srcs = glob([
        "eplb/*.py",
    ]),
    deps = [
        ":utils",
    ],
)

py_library(
    name = "gang",
    srcs = glob([
        "gang/*.py",
    ])
)

py_library(
    name = "_ft_pickler",
    srcs = ["_ft_pickler.py"],
)

py_library(
    name = "ops",
    srcs = glob([
        "ops/**/*.py",
    ]),
    deps = [
        ":torch",
        ":utils",
    ],
)

py_library(
    name = "pipeline",
    srcs = glob([
        "pipeline/**/*.py",
    ]),
)

py_library(
    name = "device",
    srcs = glob([
        "device/**/*.py",
    ]),
)

py_library(
    name = "models",
    srcs = glob([
        "models/*.py",
        "models/**/*.py",
    ], exclude=["models/test/*.py"]),
    deps = [
        ":sentencepiece",
        ":sentence-transformers",
        ":transformers",
        ":prettytable",
        ":pynvml",
        ":tiktoken",
        ":protobuf",
        ":Pillow",
        ":pillow-heif",
        ":pillow-avif-plugin",
        ":torch",
        ":torchvision",
        ":pyOpenSSL",
        ":einops",
        ":utils",
        ":ops",
        ":timm",
        ":onnx",
        #":decord",
        ":nest_asyncio",
        ":matplotlib",
        ":av",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/models_py:models",
    ] + arch_dep + select({
        "@//:using_cuda12_9_x86": tensorrt + flashinfer,
        "@//:cuda_pre_12_9": tensorrt + flashinfer,
        "//conditions:default": []
    }) + select({
        "@//:using_arm": [],
        "@//:using_cuda12_arm": [],
        # "//:xft_use_icx": [
        #     "xfastertransformer_devel_icx",
        # ],
        "//conditions:default": [
            "xfastertransformer_devel",
        ],
    }) + select({
        "@//:using_arm": [],
	    "@//:using_rocm": ["pyrsmi", "amdsmi"],
        "@//:using_cpu": [],
        "@//:using_cuda12_arm": [],
        "//conditions:default": ["flash_attn"]
    }) + triton_deps(["triton"]),
)

py_library(
    name = "vipserver",
    srcs = glob([
        "vipserver/*.py",
    ]),
)

filegroup(
    name = "alog_config",
    srcs = ["config/alog.conf"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "release_version",
    srcs = ["release_version.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "config",
    srcs = glob([
        "config/*.py",
        "config/**/*.py",
    ]),
    deps = [
        "//rtp_llm/distribute:distribute"
    ],
    data = [":alog_config"]
)

py_library(
    name = "config_ops",
    deps = [
        ":config",
        ":ops"
    ],
)

py_library(
    name = "structure",
    srcs = glob([
        "structure/*.py",
    ])
)

py_library(
    name = "cli",
    deps = [
        ":config",
        ":release_version"
    ],
    srcs = glob([
        "cli/*.py",
    ])
)

filegroup(
    name = "async_model_files",
    srcs = glob(["async_decoder_engine/**/*.py"]),
)

py_library(
    name = "async_model",
    srcs = [
        ":async_model_files"
    ],
    deps = [
        ":utils",
        ":ops",
        ":config",
        ":structure",
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
    ],
)

py_library(
    name = "openai_api",
    srcs = glob([
        "openai/*.py",
        "openai/**/*.py",
    ]),
    deps = [
        ":utils",
        ":ops",
        ":config",
        ":structure",
    ],
    data = [
        "openai/renderers/qwen_agent/utils/qwen.tiktoken"
    ],
)

py_library(
    name = "frontend",
    srcs = glob([
        "frontend/*.py",
        "frontend/**/*.py",
    ]),
    deps = [
        ":openai_api"
    ]
)

py_library(
    name = "sdk",
    srcs = [
        '__init__.py',
        'model_factory.py',
        'start_server.py',
        'start_frontend_server.py',
        'start_backend_server.py',
        '_ft_pickler.py',
        'model_factory_register.py',
    ],
    deps = [
        "//rtp_llm/server:server",
        ":uvicorn",
        ":fastapi",
        ":psutil",
        ":oss2",
        ":orjson",
        # add qwen agent package
        ":pydantic",
        ":json5",
        ":dashscope",
        ":jieba",
        ":openai",
        ":librosa",
        ":setproctitle",
        ":portalocker",
        ":concurrent_log_handler",
        ":blobfile",
        ":partial_json_parser",
    ],
    data = jit_deps(),
    imports = ["."],
)

py_library(
    name = "kserve_server",
    srcs = [
        'kserve_server.py',
    ],
    deps = [
        ":sdk",
        ":models",
    ],
    imports = ["."],
)

py_library(
    name = "plugins",
    srcs = glob([
        "plugins/*.py",
    ])
)

py_library(
    name = "tokenizer",
    srcs = glob([
        "tokenizer/*.py",
    ])
)

py_library(
    name = "embedding",
    srcs = glob([
        "embedding/*.py",
    ])
)

py_library(
    name = "lora",
    srcs = glob([
        "lora/*.py",
    ])
)

py_library(
    name = "rtp_llm_frontend_lib",
    deps = [
        ":utils",
        ":eplb",
        ":ops",
        ":pipeline",
        ":device",
        ":cli",
        ":config",
        ":structure",
        "//rtp_llm/server:server",
        ":plugins",
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
        ":openai_api",
        ":lora",
        ":sdk",
        ":frontend",
        "//rtp_llm/tools:model_assistant",
        "//rtp_llm/distribute:distribute",
        ":tokenizer",
        "//rtp_llm/aios/kmonitor:kmonitor_py",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm/access_logger:access_logger",
    ],
    data = [
        "//rtp_llm/libs:frontend_libs"
    ]
)

py_library(
    name = "rtp_llm_lib",
    deps = [
        ":rtp_llm_frontend_lib",
    ] + select({
        "@//:using_rocm": [
            ":models",
            ":async_model",
            ":embedding",
            "//rtp_llm/tools/convert:convert",
            ":aiter",
        ],
        "//conditions:default": [
            ":models",
            ":async_model",
            ":embedding",
            "//rtp_llm/tools/convert:convert",
        ],
            }) + select({
                "@//:using_cuda12": ["@deep_gemm_ext//:deep_gemm"],
                "//conditions:default": [],
            }),
    data = [
        "//rtp_llm/libs:libs"
    ]
)

py_library(
    name = "rtp_llm_package_libs",
    deps = [
        ":rtp_llm_lib",
    ],
    data = [
        "//rtp_llm/libs:whl_package_libs"
    ]
)

py_package(
    name = "rtp_llm_frontend_package",
    deps = [
        ":rtp_llm_frontend_lib",
    ],
    packages = [
        "rtp_llm"
    ],
)

py_package(
    name = "rtp_llm_package",
    deps = [
        ":rtp_llm_package_libs",
    ] + jit_deps(),
    packages = [
        "rtp_llm",
        "fla",
    ] + select({
        "@//:using_cuda12": ["deep_gemm"],
        "//conditions:default": [],
    }),
)

whl_reqs = [
    "filelock>=3.20.0",
    "jinja2",
    "sympy",
    "typing-extensions",
    "importlib_metadata",
    "transformers==4.51.2",
    "sentencepiece==0.2.0",
    "fastapi==0.115.6",
    "grpcio-tools==1.57.0",
    "uvicorn==0.30.0",
    "setuptools==60.5.0",
    "dacite",
    "pynvml",
    "thrift",
    "numpy<2.0a0,>=1.25",
    "psutil",
    "tiktoken==0.7.0",
    "lru-dict",
    "py-spy",
    "safetensors",
    "cpm_kernels",
    "pyodps",
    "Pillow",
    "pillow-heif",
    "pillow-avif-plugin",
    "protobuf==4.25",
    "einops",
    "prettytable",
    "pydantic",
    "timm==0.9.12",
    "onnx",
    "sentence-transformers==2.7.0",
    # "xfastertransformer_devel==1.8.1.1",
    # "xfastertransformer_devel_icx==1.8.1.1",
    "grpcio==1.62.0",
    #"decord==0.6.0",
    "oss2",
    "orjson",
    "aiohttp",
    "json5",
    "dashscope>=1.11.0",
    "jieba",
    "openai",
    "nest_asyncio",
    "blobfile",
    "partial_json_parser",
    "librosa",
    "matplotlib",
    "av",
    "setproctitle",
    "bitsandbytes",
    "portalocker",
    "concurrent_log_handler",
    "flashinfer-python==0.2.5",
] + whl_deps() + platform_deps() + xft_dep

py_wheel(
    name = "rtp_llm_frontend_whl",
    distribution = "rtp_llm_frontend",
    python_tag = "py3",
    tags = ["manual", "local", "no-remote"],
    version = RELEASE_VERSION,
    deps = [
        ":rtp_llm_frontend_package",
        "//deps:extension_package_frontend"
    ],
    requires = whl_reqs,
)

rename_wheel(
    name = "rtp_llm_frontend",
    package_name = "rtp_llm_frontend-%s" % RELEASE_VERSION,
    src = ":rtp_llm_frontend_whl",
)

# target for wheel
py_wheel(
    name = "rtp_llm_whl",
    distribution = "rtp_llm",
    python_tag = "py3",
    tags = ["manual", "local", "no-remote"],
    version = RELEASE_VERSION,
    entry_points = {
        "console_scripts": [
            "rtp-llm = rtp_llm.cli.main:main",
        ]
    },
    deps = [
        ":rtp_llm_package",
        "//deps:extension_package"
    ],
    requires = whl_reqs,
)

py_wheel(
    name = "rtp_llm_kserve_whl",
    distribution = "rtp_llm",
    python_tag = "py3",
    tags = ["manual", "local"],
    version = RELEASE_VERSION,
    deps = [
        ":rtp_llm_package",
        "//deps:extension_package",
    ],
    requires = whl_reqs + [
        "kserve",
    ] + xft_dep + arch_with_version_dep,
)

rename_wheel_aarch64(
    name = "rtp_llm_aarch64",
    package_name = "rtp_llm-%s" % RELEASE_VERSION,
    src = ":rtp_llm_whl",
)

rename_wheel(
    name = "rtp_llm",
    package_name = "rtp_llm-%s" % RELEASE_VERSION,
    src = ":rtp_llm_whl",
)

rename_wheel(
    name = "rtp_llm_cuda12",
    package_name = "rtp_llm-%s+cuda121" % RELEASE_VERSION,
    src = ":rtp_llm_whl",
)

py_library(
    name = "testlib",
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//:th_transformer",
        "//:rtp_compute_ops",
    ],
    deps = [
        ":rtp_llm_lib",
        "//rtp_llm/test/utils:bench_util",
        "//rtp_llm/test/utils:numeric_util",
        "//rtp_llm/test/utils:port_util",
        "//rtp_llm/test/utils:device_resource",
        "//rtp_llm/test/utils:stream_util",
        ":aiohttp",
    ]
)

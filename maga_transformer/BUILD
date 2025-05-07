package(default_visibility = ["//visibility:public"])
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
# make different torch for different device when in compiling
load("//bazel:defs.bzl", "upload_pkg", "copy_target_to", "upload_wheel", "rename_wheel", "rename_wheel_aarch64")
load("//bazel:arch_select.bzl", "requirement", "whl_deps", "internal_deps")
load("//open_source/bazel:arch_select.bzl", "platform_deps")
load("//bazel:bundle.bzl", "bundle_files", "bundle_tar")

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

xft_dep = select({
    "@//:using_arm": [],
    "//:xft_use_icx": [
        "xfastertransformer_devel_icx",
    ],
    "//conditions:default": [
        "xfastertransformer_devel",
    ],
})

arch_dep = select({
    "@//:using_arm": [],
    "//conditions:default": [":decord"],
})

arch_with_version_dep = select({
    "@//:using_arm": [],
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
    "concurrent_log_handler"
] + tensorrt)


filegroup(
    name = "cutlass_config",
    srcs = glob(["utils/gemm_utils/luts/*"]),
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
        "//maga_transformer/aios/kmonitor:kmonitor_py",
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
        "//maga_transformer/model_loader:loader",
    ] + arch_dep + select({
        "@//:using_cuda12": tensorrt,
        "//conditions:default": []
    }) + select({
        "@//:using_arm": [],
        "//:xft_use_icx": [
            "xfastertransformer_devel_icx",
        ],
        "//conditions:default": [
            "xfastertransformer_devel",
        ],
    }) + select({
        "@//:using_cuda11": [],
        "@//:using_arm": [],
        "@//:using_rocm": ["pyrsmi", "amdsmi"],
        "@//:using_cpu": [],
        "//conditions:default": ["flash_attn"]
    }),
)

filegroup(
    name = "alog_config",
    srcs = ["config/alog.conf"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "config",
    srcs = glob([
        "config/*.py",
        "config/**/*.py",
    ]),
    deps = [
        "//maga_transformer/distribute:distribute"
    ],
    data = [":alog_config"]
)

py_library(
    name = "structure",
    srcs = glob([
        "structure/*.py",
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
        "//maga_transformer/cpp:model_rpc_client",
    ],
)

py_library(
    name = "openai_api",
    srcs = glob([
        "openai/*.py",
        "openai/**/*.py",
    ]),
    deps = [
        ":async_model",
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
        "//maga_transformer/server:server",
        ":models",
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
    ],
    imports = ["."],
)

py_library(
    name = "kserve_server",
    srcs = [
        'kserve_server.py',
    ],
    deps = [
        ":sdk",
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
    name = "maga_transformer_lib",
    deps = [
        ":utils",
        ":eplb",
        ":ops",
        ":pipeline",
        ":models",
        ":device",
        ":config",
        ":structure",
        "//maga_transformer/server:server",
        ":plugins",
        ":async_model",
        ":openai_api",
        ":embedding",
        ":lora",
        ":sdk",
        "//maga_transformer/tools:model_assistant",
        "//maga_transformer/tools/convert:convert",
        "//maga_transformer/distribute:distribute",
        ":tokenizer",
        "//maga_transformer/aios/kmonitor:kmonitor_py",
        "//maga_transformer/metrics:metrics",
        "//maga_transformer/access_logger:access_logger",
    ],
    data = [
        "//maga_transformer/libs:libs"
    ]
)

py_package(
    name = "maga_transformer_package",
    deps = [
        ":maga_transformer_lib"
    ],
    packages = [
        "maga_transformer"
    ],
)

whl_reqs = [
    "filelock==3.13.1",
    "jinja2",
    "sympy",
    "typing-extensions",
    "importlib_metadata",
    "transformers==4.46.2",
    "sentencepiece==0.2.0",
    "fastapi==0.115.6",
    "grpcio-tools==1.57.0",
    "uvicorn==0.30.0",
    "setuptools==60.5.0",
    "dacite",
    "pynvml",
    "thrift",
    "numpy==1.24.1",
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
    "pydantic==2.7.0",
    "timm==0.9.12",
    "onnx",
    "sentence-transformers==2.7.0",
    "xfastertransformer_devel==1.8.1.1",
    "xfastertransformer_devel_icx==1.8.1.1",
    "grpcio==1.62.0",
    #"decord==0.6.0",
    "accelerate==0.25.0",
    "oss2",
    "orjson",
    "aiohttp",
    "json5",
    "dashscope>=1.11.0",
    "jieba",
    "openai",
    "nest_asyncio",
    "librosa",
    "matplotlib",
    "av",
    "setproctitle",
    "bitsandbytes",
    "portalocker",
    "concurrent_log_handler",
] + whl_deps() + platform_deps() + xft_dep

# target for wheel
py_wheel(
    name = "maga_transformer_whl",
    distribution = "maga_transformer",
    python_tag = "py3",
    tags = ["manual", "local", "no-remote"],
    version = "0.2.0",
    deps = [
        ":maga_transformer_package",
        "//deps:extension_package",
    ],
    requires = whl_reqs,
)

py_wheel(
    name = "maga_transformer_kserve_whl",
    distribution = "maga_transformer",
    python_tag = "py3",
    tags = ["manual", "local"],
    version = "0.2.0",
    deps = [
        ":maga_transformer_package",
        "//deps:extension_package",
    ],
    requires = whl_reqs + [
        "kserve",
    ] + xft_dep + arch_with_version_dep,
)

rename_wheel_aarch64(
    name = "maga_transformer_aarch64",
    package_name = "maga_transformer-0.2.0",
    src = ":maga_transformer_whl",
)

rename_wheel(
    name = "maga_transformer",
    package_name = "maga_transformer-0.2.0",
    src = ":maga_transformer_whl",
)

rename_wheel(
    name = "maga_transformer_cuda11",
    package_name = "maga_transformer-0.2.0+cuda118",
    src = ":maga_transformer_whl",
)

rename_wheel(
    name = "maga_transformer_cuda12",
    package_name = "maga_transformer-0.2.0+cuda121",
    src = ":maga_transformer_whl",
)

py_library(
    name = "testlib",
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        ":maga_transformer_lib",
        "//maga_transformer/test/utils:port_util",
        "//maga_transformer/test/utils:device_resource",
        ":aiohttp",
    ]
)

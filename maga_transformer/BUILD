package(default_visibility = ["//visibility:public"])
load("@rules_python//experimental/python:wheel.bzl", "py_package", "py_wheel")
# make different torch for different device when in compiling
load("//bazel:defs.bzl", "upload_pkg", "copy_target_to", "upload_wheel", "rename_wheel")
load("//bazel:arch_select.bzl", "requirement", "whl_deps")
load("//bazel:bundle.bzl", "bundle_files", "bundle_tar")

requirement([
    "sentencepiece",
    "transformers",
    "pynvml",
    "tiktoken",
    "protobuf",
    "Pillow",
    "lru-dict",
    "cpm_kernels",
    "uvicorn",
    "fastapi",
    "psutil",
    "pyodps",
    "thrift",
    "torch",
    "torchvision",
    "einops",
    "prettytable",
    "timm",
    "aiohttp",
    "tensorrt",
    "onnx",
    "sentence-transformers",
    "orjson"
])

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
        ":torchvision",
        ":lru-dict",
        ":cpm_kernels",
        ":prettytable",
    ]
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
        ":torchvision",
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
        ":torch",
        ":torchvision",
        ":einops",
        ":utils",
        ":ops",
        ":timm",
        ":tensorrt",
        ":onnx"
    ],
)

py_library(
    name = "config",
    srcs = glob([
        "config/*.py",
        "config/**/*.py",
    ]),
    deps = [
        "//maga_transformer/distribute:distribute"
    ]
)

py_library(
    name = "structure",
    srcs = glob([
        "structure/*.py",
    ])
)

filegroup(
    name = "not_experimental_async_model_files",
    srcs = glob(["async_decoder_engine/**/*.py"], exclude=["async_decoder_engine/rpc_model.py"]),
)

filegroup(
    name = "async_model_files",
    srcs = glob(["async_decoder_engine/**/*.py"]),
)

py_library(
    name = "async_model",
    srcs = select({
        "//:use_experimental": [":async_model_files"],
        "//conditions:default": [":not_experimental_async_model_files"],
    }),
    deps = [
        ":utils",
        ":ops",
        ":config",
        ":structure",
    ] + select({
        "//:use_experimental": ["//maga_transformer/cpp:model_rpc_client"],
        "//conditions:default": [],
    }),
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
)

py_library(
    name = "sdk",
    srcs = [
        '__init__.py',
        'model_factory.py',
        'start_server.py',
        '_ft_pickler.py',
        'model_factory_register.py',
    ],
    deps = [
        "//maga_transformer/server:server",
        ":models",
        ":uvicorn",
        ":fastapi",
        ":psutil",
        ":orjson"
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
    name = "maga_transformer_lib",
    deps = [
        ":utils",
        ":ops",
        ":pipeline",
        ":models",
        ":config",
        ":structure",
        "//maga_transformer/server:server",
        ":plugins",
        ":async_model",
        ":openai_api",
        ":embedding",
        ":sdk",
        "//maga_transformer/tools:save_ft_module",
        "//maga_transformer/tools:model_assistant",
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

# target for wheel
py_wheel(
    name = "maga_transformer_whl",
    distribution = "maga_transformer",
    python_tag = "py3",
    tags = ["manual", "local"],
    version = "0.1.15",
    deps = [
        ":maga_transformer_package",
        "//deps:extension_package",
    ],
    requires=[
        "filelock==3.13.1",
        "jinja2",
        "sympy",
        "typing-extensions",
        "importlib_metadata",
        "transformers==4.39.3",
        "sentencepiece==0.1.99",
        "fastapi==0.108.0",
        "uvicorn==0.21.1",
        "dacite",
        "pynvml",
        "thrift",
        "numpy==1.24.1",
        "psutil",
        "tiktoken==0.4.0",
        "lru-dict",
        "py-spy",
        "safetensors",
        "cpm_kernels",
        "pyodps",
        "Pillow",
        "protobuf==3.20.0",
        "torchvision==0.16.0",
        "einops",
        "prettytable",
        "pydantic==2.5.3",
        "timm==0.9.12",
        "sentence-transformers==2.7.0",
        "grpcio==1.62.0"
    ] + whl_deps(),
)

py_wheel(
    name = "maga_transformer_kserve_whl",
    distribution = "maga_transformer",
    python_tag = "py3",
    tags = ["manual", "local"],
    version = "0.1.15",
    deps = [
        ":maga_transformer_package",
        "//deps:extension_package",
    ],
    requires=[
        "filelock==3.13.1",
        "jinja2",
        "sympy",
        "typing-extensions",
        "importlib_metadata",
        "transformers==4.39.3",
        "sentencepiece==0.1.99",
        "fastapi==0.108.0",
        "uvicorn==0.21.1",
        "dacite",
        "pynvml",
        "thrift",
        "numpy==1.24.1",
        "psutil",
        "tiktoken==0.4.0",
        "lru-dict",
        "py-spy",
        "safetensors",
        "cpm_kernels",
        "pyodps",
        "Pillow",
        "protobuf==3.20.0",
        "torchvision==0.16.0",
        "einops",
        "prettytable",
        "pydantic==2.5.3",
        "timm==0.9.12",
        "kserve",
    ] + whl_deps(),
)

rename_wheel(
    name = "maga_transformer",
    package_name = "maga_transformer-0.1.15",
    src = ":maga_transformer_whl",
)

rename_wheel(
    name = "maga_transformer_cuda11",
    package_name = "maga_transformer-0.1.15+cuda118",
    src = ":maga_transformer_whl",
)

rename_wheel(
    name = "maga_transformer_cuda12",
    package_name = "maga_transformer-0.1.15+cuda121",
    src = ":maga_transformer_whl",
)

py_library(
    name = "testlib",
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        ":maga_transformer_lib",
        ":aiohttp",
    ]
)

load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "requirement", "whl_deps", "torch_deps")
load("//bazel:arch_select.bzl", "embedding_arpc_deps")
package(default_visibility = ["//:__subpackages__"])


requirement([
    "grpcio"
])

embedding_arpc_deps()
# load("//bazel:tf_proto.bzl", "tf_proto_library")


cc_library(
    name = "dataclass",
    hdrs = glob([
        "dataclass/*.h",
    ]),
    srcs = glob([
        "dataclass/*.cc",
    ]),
    deps = torch_deps() + [
        ":metrics",
        ":tokenizer",
        ":system_prompt",
        ":position_ids_generator",
        "//:gpt_init_params",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp/utils:py_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/models_weight:weights_define",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@havenask//aios/autil:synchronized_queue",
        "@havenask//aios/autil:json",

    ] + select({
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [],
    }),
)


cc_library(
    name = "logits_processor",
    hdrs = glob([
        "logits_processor/*.h",
    ]),
    srcs = glob([
        "logits_processor/*.cc",
    ]),
    deps = torch_deps() + [
        ":cache",
        ":dataclass",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp/utils:py_utils",
        "//rtp_llm/cpp/config:config_modules",
    ],
)

cc_library(
    name = "stream",
    hdrs = glob([
        "stream/*.h",
    ]),
    srcs = glob([
        "stream/*.cc",
    ]),
    deps = torch_deps() + [
        ":models",
        ":cache",
        ":dataclass",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
    ],
)

cc_library(
    name = "lora",
    hdrs = glob([
        "lora/*.h",
    ]),
    srcs = glob([
        "lora/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":stream",
        ":cache",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
    ],
)

cc_library(
    name = "system_prompt",
    hdrs = glob([
        "system_prompt/SystemPrompt.h",
    ]),
    srcs = glob([
    ]),
    deps = torch_deps() + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "system_prompt_constructor",
    hdrs = glob([
        "system_prompt/SystemPromptConstructor.h",
    ]),
    srcs = glob([
        "system_prompt/SystemPromptConstructor.cc",
    ]),
    deps = torch_deps() + [
        ":cache",
        ":dataclass",
        ":engine_base",
        "//:gpt_init_params",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/models_weight:weights_define",
    ],
)

cc_library(
    name = "cache",
    hdrs = [
        "cache/BatchKVCacheResource.h",
        "cache/BlockCache.h",
        "cache/BlockRefCounter.h",
        "cache/CacheConfigCreator.h",
        "cache/CacheConfig.h",
        "cache/CacheManager.h",
        "cache/KVCacheResource.h",
        "cache/DistStorage.h",
        "cache/DistStorageLocalMem.h",
        "cache/DistStorageManager.h",
        "cache/DistKvCachePlanner.h",
        "cache/DistKvCache.h",
        "cache/DistKvCacheMetrics.h",
        "cache/KVCacheAllocator.h",
    ],
    srcs = [
        "cache/BatchKVCacheResource.cc",
        "cache/BlockCache.cc",
        "cache/CacheConfigCreator.cc",
        "cache/CacheManager.cc",
        "cache/KVCacheResource.cc",
        "cache/DistStorageLocalMem.cc",
        "cache/DistStorageManager.cc",
        "cache/DistKvCachePlanner.cc",
        "cache/DistKvCache.cc",
        "cache/KVCacheAllocator.cc",
    ],
    deps = torch_deps() + [
        ":metrics",
        ":dataclass",
        ":model_rpc_pool",
        "//:gpt_init_params",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
    ] + select({
        "//:enable_3fs": [
            ":storage_3fs",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "storage_3fs",
    hdrs = [
        "cache/DistStorage.h",
        "cache/DistStorage3FS.h",
        "cache/DistStorage3FSFile.h",
        "cache/ThreeFSCudaUtil.h",
        "cache/ThreeFSMempool.h",
        "cache/DistKvCacheMetrics.h",
    ],
    srcs = [
        "cache/DistStorage3FS.cc",
        "cache/DistStorage3FSFile.cc",
        "cache/ThreeFSMempool.cc",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_base",
        "//3rdparty/3fs:hf3fs",
        "@havenask//aios/autil:lock_free",
        "@havenask//aios/autil:scope",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "models",
    hdrs = glob([
        "models/*.h",
    ]),
    srcs = glob([
        "models/*.cc",
    ]),
    deps = torch_deps() + [
        ":logits_processor",
        ":dataclass",
        ":cache",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/models_py/bindings:op_defs"
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "@//:using_rocm": [
            "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [
            "//rtp_llm/cpp/devices/cpu_impl:cpu_impl",
        ],
    }),
)

cc_library(
    name = "components",
    hdrs = glob([
        "components/*.h",
    ]),
    srcs = glob([
        "components/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
        "//rtp_llm/cpp/devices:devices_base",
    ],
)

cc_library(
    name = "schedulers",
    hdrs = glob([
        "schedulers/*.h",
    ]),
    srcs = glob([
        "schedulers/*.cc",
    ]),
    deps = torch_deps() + [
        ":stream",
        ":dataclass",
        ":metrics",
        ":cache",
        ":components",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "engine_base",
    hdrs = glob([
        "engine_base/*.h",
    ]),
    srcs = glob([
        "engine_base/*.cc",
    ]),
    deps = torch_deps() + [
        ":schedulers",
        ":stream",
        ":dataclass",
        ":models",
        ":lora",
        ":system_prompt",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "normal_engine",
    hdrs = glob([
        "normal_engine/*.h",
        "eplb/*.h",
    ]),
    srcs = glob([
        "normal_engine/*.cc",
        "eplb/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":metrics",
        ":models",
        ":schedulers",
        ":engine_base",
        ":system_prompt",
        ":system_prompt_constructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/config:eplb_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp/devices:devices_base_impl",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "speculative_engine",
    hdrs = glob([
        "speculative_engine/*.h",
        "speculative_engine/propose_executor/*.h",
        "speculative_engine/score_executor/*.h",
        "speculative_engine/speculative_sampler/*.h",
    ]),
    srcs = glob([
        "speculative_engine/*.cc",
        "speculative_engine/propose_executor/*.cc",
        "speculative_engine/score_executor/*.cc",
        "speculative_engine/speculative_sampler/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":metrics",
        ":models",
        ":schedulers",
        ":engine_base",
        ":normal_engine",
        ":system_prompt",
        ":system_prompt_constructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp/devices:devices_base_impl",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "metrics",
    hdrs = glob([
        "metrics/*.h",
    ]),
    srcs = glob([
        "metrics/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_library(
    name = "multimodal_processor",
    hdrs = glob([
        "multimodal_processor/*.h",
    ]),
    srcs = glob([
        "multimodal_processor/*.cc",
    ]),
    deps = [
        ":dataclass",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/utils:py_utils",
        "//rtp_llm/cpp/config:config_modules",
    ],
)

cc_library(
    name = "position_ids_generator",
    hdrs = glob([
        "position_ids_generator/*.h",
    ]),
    srcs = glob([
        "position_ids_generator/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/config:config_modules",
    ],
)

cc_library(
    name = "tokenizer",
    hdrs = glob([
        "tokenizer/*.h",
    ]),
    srcs = glob([
        "tokenizer/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices:devices_base", # for pybind11
        "//rtp_llm/cpp/utils:py_utils",
    ],
)

cc_library(
    name = "openai",
    hdrs = glob([
        "openai/*.h",
    ]),
    srcs = glob([
        "openai/*.cc",
    ]),
    deps = [
        "@havenask//aios/autil:json",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp:normal_engine",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        ":embedding_arpc_deps",
        # "//rtp_llm/cpp:speculative_engine",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp:multimodal_processor",
    ],
)

cc_library(
    name = "http_api_server",
    srcs = glob([
        "api_server/*.cc",
    ]),
    hdrs = glob([
        "api_server/*.h",
    ]),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/utils:data_structures",
        "//rtp_llm/cpp/utils:domain_utils",
        "//rtp_llm/cpp:openai",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp:normal_engine",
        "//rtp_llm/cpp:multimodal_processor",
        "//rtp_llm/cpp/http_server:http_server",
        "//rtp_llm/cpp/http_server:http_client",
        ":embedding_arpc_deps",
        "@havenask//aios/autil:net",
        "@havenask//aios/autil:time",
        "//rtp_llm/cpp/api_server/common:common_service",
    ],
)

cc_library(
    name = "model_rpc_pool",
    srcs = [],
    hdrs = ["model_rpc/RPCPool.h"],
    deps = [
        "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
    ]
)

cc_library(
    name = "model_rpc_server",
    srcs =  glob(["model_rpc/*.cc"]),
    hdrs =  glob(["model_rpc/*.h"], exclude = ["RPCPool.h"]),
    deps = [
        ":embedding_arpc_deps",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp:normal_engine",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp:speculative_engine",
        "//rtp_llm/cpp:model_rpc_pool",
        "//rtp_llm/cpp:multimodal_processor",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "@havenask//aios/autil:net",
    ],
)

py_library(
    name = "model_rpc_client",
    srcs = ["model_rpc/model_rpc_client.py"],
    deps = [
        ":grpcio",
        "//rtp_llm/cpp/proto:model_rpc_service_py_proto"
    ],
)

exports_files(glob(["th_op/**/*.cc"]))
exports_files(glob(["th_op/**/*.h"]))

load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm = "requirement")
load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

alias(
    name = "cuda13_torch_nvshmem",
    actual = "@torch//:torch",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/normal_engine:normal_engine",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps() + cuda13_torch_link_deps

cc_test(
    name = "model_data_test",
    srcs = [
        "ModelDataTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + ["//rtp_llm/cpp/distribute:cpu_tp_broadcaster"],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_test(
    name = "model_inputs_logger_test",
    srcs = ["ModelInputsLoggerTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/models:model_inputs_logger",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "model_inputs_replay_test",
    srcs = ["model_inputs_replay_test.py"],
    tags = [
        "H20_CU13",
        "manual",
    ],
    deps = ["//rtp_llm/models_py/standalone:auto_model"],
    exec_properties = {"gpu": "H20_CU13"},
)

cc_library(
    name = "pywrapped_model_cache_store_integration_test_lib",
    srcs = ["PyWrappedModelCacheStoreIntegrationTest.cc"],
    alwayslink = True,
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/models",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ] + select({
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
        ],
        "//conditions:default": [
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
            "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        ],
    }) + torch_deps(),
)

cc_binary(
    name = "th_pywrapped_model_cache_store_integration_test",
    deps = [":pywrapped_model_cache_store_integration_test_lib"],
    linkshared = 1,
)

pywrapped_model_test_deps = [
    "//rtp_llm:config",
    "//rtp_llm:torch",
]

py_test(
    name = "pywrapped_model_cache_store_integration_test",
    srcs = ["pywrapped_model_cache_store_integration_test.py"],
    data = [":th_pywrapped_model_cache_store_integration_test"],
    deps = pywrapped_model_test_deps,
    env = {"TEST_USING_DEVICE": "CUDA"},
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "pywrapped_model_cache_store_integration_test_rocm",
    main = "pywrapped_model_cache_store_integration_test.py",
    srcs = ["pywrapped_model_cache_store_integration_test.py"],
    data = [":th_pywrapped_model_cache_store_integration_test"],
    deps = pywrapped_model_test_deps + [requirement_gpu_rocm("pytest")],
    env = {"TEST_USING_DEVICE": "ROCM"},
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "pywrapped_model_dirty_capture_test",
    main = "pywrapped_model_cache_store_integration_test.py",
    srcs = ["pywrapped_model_cache_store_integration_test.py"],
    args = [
        "PyWrappedModelCacheStoreIntegrationTest.test_dirty_generation_prefill_capture_does_not_retain_cache_manager",
    ],
    data = [":th_pywrapped_model_cache_store_integration_test"],
    deps = pywrapped_model_test_deps + [
        "//rtp_llm/models_py:modules",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "RTP_LLM_RUN_PYWRAPPED_DIRTY_CAPTURE_TEST": "1",
        "TEST_USING_DEVICE": "CUDA",
    },
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "pywrapped_model_dirty_capture_test_rocm",
    main = "pywrapped_model_cache_store_integration_test.py",
    srcs = ["pywrapped_model_cache_store_integration_test.py"],
    args = [
        "PyWrappedModelCacheStoreIntegrationTest.test_dirty_generation_prefill_capture_does_not_retain_cache_manager",
    ],
    data = [":th_pywrapped_model_cache_store_integration_test"],
    deps = pywrapped_model_test_deps + [
        "//rtp_llm/models_py:modules",
    ] + [requirement_gpu_rocm("pytest")],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "RTP_LLM_RUN_PYWRAPPED_DIRTY_CAPTURE_TEST": "1",
        "TEST_USING_DEVICE": "ROCM",
    },
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

# Preserve exact original Native platform contracts under explicit compatibility labels.
_native_original_test_deps = ['//rtp_llm/cpp/testing:device_test_utils', '//rtp_llm/models_py/bindings/cuda/ops:cuda_impl', '//rtp_llm/cpp/normal_engine:normal_engine', '//rtp_llm/cpp/models:models', '//rtp_llm/cpp/config:config_modules', '@com_google_googletest//:gtest', '@com_google_googletest//:gtest_main', '@local_config_cuda//cuda:cuda_headers', '@local_config_cuda//cuda:cudart'] + torch_deps()

cc_test(name='model_data_test_native_a10', srcs=['ModelDataTest.cc'], data=[], copts=test_copts, deps=_native_original_test_deps + ['//rtp_llm/cpp/distribute:cpu_tp_broadcaster'], env={'TEST_USING_DEVICE': 'CUDA'}, exec_properties={'gpu': 'A10'}, tags=['A10'])

cc_test(name='model_inputs_logger_test_native_a10', srcs=['ModelInputsLoggerTest.cc'], copts=test_copts, deps=['//rtp_llm/cpp/models:model_inputs_logger', '@com_google_googletest//:gtest', '@com_google_googletest//:gtest_main'] + torch_deps(), exec_properties={'gpu': 'A10'}, tags=['A10'])

py_test(name='model_inputs_replay_test_native_h20', srcs=['model_inputs_replay_test.py'], tags=['manual', 'H20'], deps=['//rtp_llm/models_py/standalone:auto_model'], exec_properties={'gpu': 'H20'})

py_test(name='pywrapped_model_cache_store_integration_test_native_a10', tags=['manual', 'A10'], srcs=['pywrapped_model_cache_store_integration_test.py'], data=[':th_pywrapped_model_cache_store_integration_test'], deps=pywrapped_model_test_deps, env={'TEST_USING_DEVICE': 'CUDA'}, exec_properties={'gpu': 'A10'})

py_test(name='pywrapped_model_cache_store_integration_test_rocm_native_rocm', main='pywrapped_model_cache_store_integration_test.py', srcs=['pywrapped_model_cache_store_integration_test.py'], data=[':th_pywrapped_model_cache_store_integration_test'], deps=pywrapped_model_test_deps, env={'TEST_USING_DEVICE': 'ROCM'}, tags=['manual', 'rocm'], exec_properties={'gpu': 'MI308X-ROCM7'})

py_test(name='pywrapped_model_dirty_capture_test_native_h20', main='pywrapped_model_cache_store_integration_test.py', srcs=['pywrapped_model_cache_store_integration_test.py'], args=['PyWrappedModelCacheStoreIntegrationTest.test_dirty_generation_prefill_capture_does_not_retain_cache_manager'], data=[':th_pywrapped_model_cache_store_integration_test'], deps=pywrapped_model_test_deps + ['//rtp_llm/models_py:modules'], env={'ENABLE_CUDA_GRAPH_DEBUG_MODE': '1', 'NOT_USE_DEFAULT_STREAM': '1', 'RTP_LLM_RUN_PYWRAPPED_DIRTY_CAPTURE_TEST': '1', 'TEST_USING_DEVICE': 'CUDA'}, tags=['manual', 'H20', 'H20'], exec_properties={'gpu': 'H20'})

py_test(name='pywrapped_model_dirty_capture_test_rocm_native_rocm', main='pywrapped_model_cache_store_integration_test.py', srcs=['pywrapped_model_cache_store_integration_test.py'], args=['PyWrappedModelCacheStoreIntegrationTest.test_dirty_generation_prefill_capture_does_not_retain_cache_manager'], data=[':th_pywrapped_model_cache_store_integration_test'], deps=pywrapped_model_test_deps + ['//rtp_llm/models_py:modules'], env={'ENABLE_CUDA_GRAPH_DEBUG_MODE': '1', 'NOT_USE_DEFAULT_STREAM': '1', 'RTP_LLM_RUN_PYWRAPPED_DIRTY_CAPTURE_TEST': '1', 'TEST_USING_DEVICE': 'ROCM'}, tags=['manual', 'rocm'], exec_properties={'gpu': 'MI308X-ROCM7'})

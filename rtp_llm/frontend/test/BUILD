load("@arch_config//:arch_select.bzl", "telemetry_test_deps")

py_test(
    name = "d3_test",
    srcs = ["d3_test.py"],
    deps = [
        "//rtp_llm:frontend",
        # models/__init__.py imports rtp_llm.model_factory_register eagerly;
        # this target's runfiles must provide it (see //rtp_llm:model_factory_register).
        "//rtp_llm:model_factory_register",
        "//rtp_llm:models",
        "//rtp_llm:decord",
        "//rtp_llm:numpy",
        "//rtp_llm:Pillow",
        # the video accounting tests build real group frame tensors
        "//rtp_llm:torch",
        "//rtp_llm:torchvision",
        # engine.prepare() builds real video-group entries, whose lazy
        # multimodal_util import needs the model_rpc pb2 module (the rest of
        # that module's import chain already resolves through the deps above).
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
        # ...and base_model_datatypes -> generate_config imports pydantic:
        # take it from the pinned wheel set so it matches this target's own
        # typing_extensions instead of whatever the invoking user has under
        # ~/.local (pydantic 2.12 needs typing_extensions.Sentinel, which the
        # pinned typing_extensions predates).
        "//rtp_llm:pydantic",
    ],
    env = {"GPU_COUNT": "0"},
)

py_test(
    name = "d3_joint_test",
    srcs = ["d3_joint_test.py"],
    deps = [
        "//rtp_llm:frontend",
        # models/__init__.py imports rtp_llm.model_factory_register eagerly;
        # this target's runfiles must provide it (see //rtp_llm:model_factory_register).
        "//rtp_llm:model_factory_register",
        "//rtp_llm:models",
        # the joint image tests decode real data-URL images
        "//rtp_llm:Pillow",
        # ...and base_model_datatypes -> generate_config imports pydantic from
        # the pinned wheel set (see the d3_test target note).
        "//rtp_llm:pydantic",
    ],
    env = {"GPU_COUNT": "0"},
)

py_test(
    name = "d3_systemone_test",
    srcs = ["d3_systemone_test.py"],
    deps = [
        "//rtp_llm:av",
        "//rtp_llm:decord",
        "//rtp_llm:fastapi",
        "//rtp_llm:frontend",
        # frontend_app (imported at collection, line 19) pulls the full
        # production surface — rtp_llm.embedding.embedding_type, rtp_llm.metrics,
        # distribute/pipeline/structure via frontend_server/frontend_worker —
        # which only //rtp_llm:testlib's rtp_llm_lib closure covers
        # (testlib_lite lacks it: d3_final_merged_units_20261028_01 died at
        # ModuleNotFoundError 'rtp_llm.embedding'). Same pattern as
        # frontend_shutdown_test / jev_decide_test (GPU_COUNT=0 with testlib).
        "//rtp_llm:testlib",
        # models/__init__.py imports rtp_llm.model_factory_register eagerly
        # (see //rtp_llm:model_factory_register).
        "//rtp_llm:model_factory_register",
        "//rtp_llm:models",
        "//rtp_llm:numpy",
        "//rtp_llm:Pillow",
        "//rtp_llm:torchvision",
    ] + telemetry_test_deps(),
    env = {"GPU_COUNT": "0"},
)

py_test(
    name = "jev_test",
    srcs = ["jev_test.py"],
    deps = ["//rtp_llm:frontend"],
    env = {"GPU_COUNT": "0"},
)

py_test(
    name = "jev_decide_test",
    srcs = ["jev_decide_test.py"],
    deps = ["//rtp_llm:frontend", "//rtp_llm:testlib"] + telemetry_test_deps(),
    env = {"GPU_COUNT": "0"},
)

py_test(
    name = "systemone_test",
    srcs = ["systemone_test.py"],
    deps = ["//rtp_llm:frontend"],
)

py_test(
    name = "kev_decide_test",
    srcs = ["kev_decide_test.py"],
    deps = [
        "//rtp_llm:frontend",
        "//rtp_llm:testlib",
    ] + telemetry_test_deps(),
)

py_test(
    name = "custom_output_response_test",
    srcs = ["custom_output_response_test.py"],
    deps = [
        "//rtp_llm:frontend",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "incremental_decode_test",
    srcs = [
        "incremental_decode_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata",
        "//rtp_llm/utils/test:testdata",
    ],
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:config",
        "//rtp_llm:models",
        "//rtp_llm:testlib",
        "//rtp_llm:utils",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    env = {"GPU_COUNT": "0"},
    tags = [
        "L20_CU13",
        "frontend",
    ],
)

py_test(
    name = "frontend_server_test",
    srcs = [
        "frontend_server_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ] + telemetry_test_deps(),
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "frontend_shutdown_test",
    srcs = [
        "frontend_shutdown_test.py",
    ],
    deps = [
        "//rtp_llm:fastapi",
        "//rtp_llm:frontend",
        "//rtp_llm:testlib",
        "//rtp_llm:uvicorn",
    ],
)

py_test(
    name = "frontend_graceful_shutdown_business_test",
    srcs = [
        "frontend_graceful_shutdown_business_test.py",
    ],
    deps = [
        "//rtp_llm:fastapi",
        "//rtp_llm:frontend",
        "//rtp_llm:testlib",
        "//rtp_llm:uvicorn",
    ],
)

py_test(
    name = "token_processor_test",
    srcs = ["token_processor_test.py"],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata",
    ],
    deps = [
        "//rtp_llm:frontend",
        "//rtp_llm:testlib",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "request_id_generator_test",
    srcs = ["request_id_generator_test.py"],
    deps = [
        "//rtp_llm:frontend",
        "//rtp_llm:testlib",
    ],
)

# Bazel targets for the newloader porting/verification tools.
# These MUST be run via `bazel run` so the compiled rtp_llm.ops and the full
# python dep tree are present — a bare `python3` will fail with
# "No module named 'rtp_llm'".

py_binary(
    name = "load_once",
    srcs = ["_load_once.py"],
    main = "_load_once.py",
    deps = [
        # auto_model bundles py_standalone_testlib + //rtp_llm:models + ops,
        # i.e. everything needed to construct and load any registered model.
        "//rtp_llm/models_py/standalone:auto_model",
        # FakeModelLoader lives here (not in the standalone testlib graph).
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "check_imports",
    srcs = ["check_imports.py"],
    main = "check_imports.py",
    deps = [
        "//rtp_llm/models_py/standalone:auto_model",
    ],
    visibility = ["//visibility:public"],
)

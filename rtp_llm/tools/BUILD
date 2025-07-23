package(default_visibility = ["//rtp_llm:__subpackages__", "//internal_source:__subpackages__"])

py_binary(
    name = "fake_glm_v2",
    srcs = ["fake_glm_v2.py"],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:_ft_pickler",
        ":fake_util"
    ]
)

py_library(
    name = "fake_util",
    srcs = ["fake_util.py"],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:_ft_pickler",
    ]
)

py_library(
    name = "model_assistant_api",
    srcs = glob([
        "api/*.py",
    ]),
)

py_library(
    name = "model_assistant",
    srcs = [
           "model_assistant_server.py",
           ":model_assistant_api"
    ]
)


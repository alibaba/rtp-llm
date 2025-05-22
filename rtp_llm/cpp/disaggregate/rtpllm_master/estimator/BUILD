cc_library(
    name = "estimator",
    hdrs = [
        "PrefillTimeEstimator.h",
        "LookupMapImpl.h",
        "EstimatorConfig.h",
        "LookupPrefillEstimator.h",
    ],
    srcs = [
        "PrefillTimeEstimator.cpp",
        "LookupPrefillEstimator.cpp",
        "LookupMapImpl.cpp"
    ],
    deps = [
        "//rtp_llm/cpp:dataclass",
        "@havenask//aios/autil:json"
    ],
    visibility=['//visibility:public'],
)
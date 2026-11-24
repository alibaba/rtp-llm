config_setting(
    name = "prebuilt_smoke_jar",
    define_values = {"flexlb_prebuilt_jar": "true"},
)

genrule(
    name = "flexlb_jar",
    srcs = select({
        ":prebuilt_smoke_jar": ["ci_prebuilt/flexlb-api.jar"],
        "//conditions:default": glob([".mvn/**", "mvnw", "pom.xml", "flexlb-*/pom.xml", "flexlb-*/src/main/**"]) + [
            "//rtp_llm/cpp/model_rpc/proto:flexlb_schedule_service.proto",
            "//rtp_llm/cpp/model_rpc/proto:model_rpc_service.proto",
        ],
    }),
    outs = ["flexlb-api.jar"],
    cmd = select({
        ":prebuilt_smoke_jar": "cp $(location ci_prebuilt/flexlb-api.jar) $@",
        "//conditions:default": """
        set -eu -o pipefail
        out="$$PWD/$@"
        # Maven writes target/ beside each module, including under local Bazel execution.
        build_dir=$$(mktemp -d)
        trap 'rm -rf "$$build_dir"' EXIT
        tar -chf - $(SRCS) | tar -xf - -C "$$build_dir"
        cd "$$build_dir/$$(dirname "$(location mvnw)")"
        ./mvnw -q -P '!internal' -Dmaven.test.skip=true -Dflexlb.package.skip=true package
        cp flexlb-api/target/flexlb-api-*.jar "$$out"
        """,
    }),
    local = True,
    tags = ["manual"],
    visibility = ["//rtp_llm/test/smoke:__pkg__"],
)

filegroup(
    name = "smoke_java_runtime",
    srcs = select({
        ":prebuilt_smoke_jar": glob(["ci_prebuilt/java_arm/**", "ci_prebuilt/java_x86/**"]),
        "//conditions:default": [],
    }),
    visibility = ["//rtp_llm/test/smoke:__pkg__"],
)

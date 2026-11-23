genrule(
    name = "flexlb_jar",
    srcs = glob([".mvn/**", "mvnw", "pom.xml", "flexlb-*/pom.xml", "flexlb-*/src/main/**"]) + [
        "//rtp_llm/cpp/model_rpc/proto:flexlb_schedule_service.proto",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service.proto",
    ],
    outs = ["flexlb-api.jar"],
    cmd = """
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
    local = True,
    tags = ["manual"],
    visibility = ["//rtp_llm/test/smoke:__pkg__"],
)

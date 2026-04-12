genrule(
    name = "flexlb_jar",
    srcs = glob(["**"]),
    outs = ["flexlb-api.jar"],
    cmd = """
        set -e
        OUT=$$(realpath $@)
        FLEXLB_DIR=$$(dirname $(location mvnw))
        cd $$FLEXLB_DIR
        ./mvnw package -DskipTests -q >&2
        JAR=$$(ls flexlb-api/target/flexlb-api-*.jar | grep -v sources | head -1)
        cp $$JAR $$OUT
    """,
    local = True,
    visibility = ["//visibility:public"],
)

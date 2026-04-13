genrule(
    name = "flexlb_jar",
    srcs = glob(["**"]),
    outs = ["flexlb-api.jar"],
    cmd = """
        set -e
        OUT=$$(realpath $@)
        # Find Java in known locations if not in PATH
        if ! command -v java >/dev/null 2>&1; then
            for d in /opt/taobao/install/ajdk21_*/bin /opt/jdk21/bin /usr/lib/jvm/java-21/bin; do
                if [ -x "$$d/java" ]; then
                    export JAVA_HOME=$$(dirname $$d)
                    export PATH="$$d:$$PATH"
                    break
                fi
            done
        fi
        FLEXLB_DIR=$$(dirname $(location mvnw))
        cd $$FLEXLB_DIR
        ./mvnw package -DskipTests -q -P opensource >&2
        JAR=$$(ls flexlb-api/target/flexlb-api-*.jar | grep -v sources | head -1)
        cp $$JAR $$OUT
    """,
    local = True,
    visibility = ["//visibility:public"],
)

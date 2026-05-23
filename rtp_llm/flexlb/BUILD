package(default_visibility = ["//visibility:public"])

# ----------------------------------------------------------------------------
# Bazel-side wrapper around the Maven build of the flexlb-api fat jar.
#
# Rationale: flexlb is a Spring Boot project managed entirely by Maven; we do
# not attempt to model its (large) Java dep graph in Bazel. Instead we expose
# the packaged uber-jar as a Bazel-visible artifact so smoke tests can take
# it as a data dependency.
#
# This rule shells out to ``./mvnw package`` and is therefore *not* hermetic:
#   * the host must have Java 21 + a network-enabled Maven local repo, or a
#     pre-warmed ``~/.m2`` cache;
#   * it runs with ``local = 1`` and ``no-sandbox`` so the embedded Maven
#     wrapper can read user-level state (settings.xml, .m2 cache);
#   * it is tagged ``manual`` so plain ``bazel build //...`` does not pick
#     it up; suites that need it depend on it explicitly.
#
# Callers that already have the jar staged outside Bazel can bypass this
# genrule by setting ``FLEXLB_JAR_PATH`` in the smoke test env — the runner
# (see ``rtp_llm/test/smoke/flexlb_server_manager.py``) honours both.
# ----------------------------------------------------------------------------

filegroup(
    name = "flexlb_srcs",
    srcs = glob(
        [
            "mvnw",
            "mvnw.cmd",
            ".mvn/**",
            "flexlb-api/pom.xml",
            "flexlb-api/src/main/**",
            "flexlb-common/pom.xml",
            "flexlb-common/src/main/**",
            "flexlb-sync/pom.xml",
            "flexlb-sync/src/main/**",
            "flexlb-grpc/pom.xml",
            "flexlb-grpc/src/main/**",
            "flexlb-cache/pom.xml",
            "flexlb-cache/src/main/**",
        ],
        allow_empty = True,
    ),
)

# Built artifact lives at flexlb-api/target/flexlb-api-<version>.jar; we
# rename to a stable name so consumers don't depend on the maven version.
# ``local = 1`` + ``no-sandbox`` lets mvnw reach the user-level ~/.m2 cache
# and resolve dependencies over the network.
#
# The Maven ``internal`` profile is auto-activated whenever ``internal_source/``
# is present and pulls in proprietary kmonitor/vipserver SNAPSHOTs that only
# resolve inside the internal CI Maven repo. The smoke test uses
# NoOpServiceDiscovery + a no-op kmonitor stub (see flexlb-common), so those
# code paths are never exercised here; disabling the profile keeps the build
# reproducible in dev containers that only have access to public mirrors.
genrule(
    name = "flexlb_api_jar",
    srcs = [
        "pom.xml",
        ":flexlb_srcs",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service.proto",
    ],
    outs = ["flexlb-api.jar"],
    cmd = """
        set -eu
        out="$$PWD/$@"
        flexlb_src_dir="$$(dirname $(location pom.xml))"
        proto_src="$(location //rtp_llm/cpp/model_rpc/proto:model_rpc_service.proto)"
        work_root="$$(mktemp -d "$${TMPDIR:-/tmp}/flexlb-api-build.XXXXXX")"
        work_dir="$$work_root/rtp_llm/flexlb"
        trap 'rm -rf "$$work_root"' EXIT
        mkdir -p "$$work_dir" "$$work_root/rtp_llm/cpp/model_rpc/proto"
        tar -C "$$flexlb_src_dir" \
            --exclude='./target' \
            --exclude='./*/target' \
            -cf - . | tar -C "$$work_dir" -xf -
        cp "$$proto_src" "$$work_root/rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto"
        cd "$$work_dir"
        if [ -z "$${JAVA_HOME:-}" ]; then
            for java_home in /opt/taobao/java21 /opt/taobao/java /usr/lib/jvm/java-21-openjdk /usr/lib/jvm/default-java; do
                if [ -x "$$java_home/bin/javac" ]; then
                    export JAVA_HOME="$$java_home"
                    break
                fi
            done
        fi
        if [ -n "$${JAVA_HOME:-}" ]; then
            export PATH="$$JAVA_HOME/bin:$$PATH"
        fi
        command -v javac >/dev/null || {
            echo "javac not found; set JAVA_HOME or install Java 21" >&2
            exit 1
        }
        chmod +x ./mvnw
        ./mvnw -B -q -pl flexlb-api -am package \\
            -P '!internal' \\
            -DskipTests \\
            -Dmaven.test.skip=true \\
            -Dspotless.check.skip=true
        cp "$$(ls flexlb-api/target/flexlb-api-*.jar | head -1)" "$$out"
    """,
    local = 1,
    tags = [
        "manual",
        "no-cache",
        "no-remote",
        "no-sandbox",
    ],
)

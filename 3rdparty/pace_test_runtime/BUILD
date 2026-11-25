load(":repositories.bzl", "PACE_TEST_ARTIFACTS")

package(default_visibility = ["//visibility:public"])

# Service binaries only: the KVCM SDK remains the branch's existing dependency.
genrule(
    name = "meta_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": ["@pace_test_meta_rpm_aarch64//file"],
        "@//:using_arm": ["@pace_test_meta_rpm_aarch64//file"],
        "//conditions:default": ["@pace_test_meta_rpm//file"],
    }),
    outs = ["tair_mempool_meta_service"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_meta_service" "$@"
    """,
)

genrule(
    name = "server_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": ["@pace_test_server_rpm_aarch64//file"],
        "@//:using_arm": ["@pace_test_server_rpm_aarch64//file"],
        "//conditions:default": ["@pace_test_server_rpm//file"],
    }),
    outs = ["tair_mempool_server", "libtent_shared.so.5"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_server" "$(@D)/tair_mempool_server"
        cp -L "$$stage/usr/local/tair_mempool/lib/libtent_shared.so.5" "$(@D)/libtent_shared.so.5"
    """,
)

filegroup(
    name = "runtime",
    testonly = True,
    tags = ["manual"],
    srcs = [":meta_binary", ":server_binary", ":consumer_binary"] + select({
        "@//:using_cuda13_x86": [":runtime_libraries_x86"],
        "//conditions:default": [],
    }),
)

genrule(
    name = "runtime_libraries_x86",
    testonly = True,
    tags = ["manual"],
    srcs = ["@%s//file" % name for name in PACE_TEST_ARTIFACTS["runtime_rpms"]],
    outs = ["runtime-libs/" + library for library in [
        "libboost_system.so.1.66.0",
        "libboost_thread.so.1.66.0",
        "libgflags.so.2.2",
        "libglog.so.0",
        "libhiredis.so.0.13",
        "libjemalloc.so.2",
        "libmysqlclient.so.21",
        "libpcap.so.1",
        "libprometheus-cpp-core.so.1.1",
        "liburing.so.2",
        "libxtables.so.12",
    ]] + ["tools/tc"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        for rpm in $(SRCS); do
            rpm2cpio "$$rpm" | (cd "$$stage" && cpio -idm --quiet)
        done
        mkdir -p "$(@D)/runtime-libs" "$(@D)/tools"
        cp "$$stage/usr/sbin/tc" "$(@D)/tools/tc"
        patchelf --force-rpath --set-rpath '$$ORIGIN/../runtime-libs' "$(@D)/tools/tc"
        for output in $(OUTS); do
            [[ "$$output" != */tools/tc ]] || continue
            library=$$(find "$$stage/usr/lib64" -name "$${output##*/}" -print -quit)
            cp -L -- "$$library" "$$output"
        done
    """,
)

genrule(
    name = "consumer_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": ["@pace_test_consumer_rpm_aarch64//file"],
        "@//:using_arm": ["@pace_test_consumer_rpm_aarch64//file"],
        "//conditions:default": ["@pace_test_consumer_rpm//file"],
    }),
    outs = ["consumer/tair_mempool_server", "consumer/libtent_shared.so.5"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        mkdir -p "$(@D)/consumer"
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_server" "$(@D)/consumer/tair_mempool_server"
        cp -L "$$stage/usr/local/tair_mempool/lib/libtent_shared.so.5" "$(@D)/consumer/libtent_shared.so.5"
    """,
)

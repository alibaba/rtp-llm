# FlexLB Bazel Build Configuration
# This provides native Bazel targets for FlexLB modules

package(default_visibility = ["//visibility:public"])

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_java//java:defs.bzl", "java_library", "java_proto_library", "java_test")
load("@io_grpc_grpc_java//:java_grpc_library.bzl", "java_grpc_library")

# Proto definitions
proto_library(
    name = "engine_rpc_proto",
    srcs = ["flexlb-grpc/src/main/resources/proto/engine_rpc_service.proto"],
    deps = ["@com_google_protobuf//:wrappers_proto"],
)

java_proto_library(
    name = "engine_rpc_java_proto",
    deps = [":engine_rpc_proto"],
)

java_grpc_library(
    name = "engine_rpc_java_grpc",
    srcs = [":engine_rpc_proto"],
    deps = [":engine_rpc_java_proto"],
)

java_plugin(
    name = "lombok_plugin",
    processor_class = "lombok.launch.AnnotationProcessorHider$AnnotationProcessor",
    deps = ["@flexlb_maven//:org_projectlombok_lombok"],
)
java_library(
    name = "lombok",
    exports = ["@flexlb_maven//:org_projectlombok_lombok"],
    exported_plugins = [":lombok_plugin"],
    visibility = ["//visibility:public"],
)



# ==== FlexLB Common Module ====
java_library(
    name = "flexlb-common",
    srcs = glob(["flexlb-common/src/main/java/**/*.java"]),
    resources = glob(["flexlb-common/src/main/resources/**"]),
    plugins = ["//rtp_llm/flexlb:lombok_plugin"],
    javacopts = [
        "-Xep:MustBeClosedChecker:OFF",  # Disable MustBeClosedChecker
    ],
    deps = [
        "//rtp_llm/flexlb:lombok",
        "@flexlb_maven//:org_springframework_boot_spring_boot_autoconfigure",
        "@flexlb_maven//:org_springframework_boot_spring_boot_starter_webflux",
        "@flexlb_maven//:com_fasterxml_jackson_core_jackson_databind",
        "@flexlb_maven//:com_fasterxml_jackson_datatype_jackson_datatype_jsr310",
        "@flexlb_maven//:org_apache_commons_commons_lang3",
        "@flexlb_maven//:org_apache_commons_commons_collections4",
        "@flexlb_maven//:io_netty_netty_transport",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_sdk",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_exporter_otlp",
        "@flexlb_maven//:com_google_guava_guava",
        "@flexlb_maven//:io_micrometer_micrometer_core",
        # Additional dependencies from error messages
        "@flexlb_maven//:com_fasterxml_jackson_core_jackson_annotations",
        "@flexlb_maven//:com_fasterxml_jackson_core_jackson_core",
        "@flexlb_maven//:io_netty_netty_buffer",
        "@flexlb_maven//:io_netty_netty_codec_http",
        "@flexlb_maven//:io_netty_netty_common",
        "@flexlb_maven//:io_netty_netty_handler",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_api",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_context",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_sdk_common",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_sdk_trace",
        "@flexlb_maven//:io_projectreactor_reactor_core",
        "@flexlb_maven//:org_slf4j_slf4j_api",
        "@flexlb_maven//:org_springframework_boot_spring_boot",
        "@flexlb_maven//:org_springframework_spring_beans",
        "@flexlb_maven//:org_springframework_spring_context",
        "@flexlb_maven//:org_springframework_spring_core",
    ],
)

# ==== FlexLB Cache Module ====
java_library(
    name = "flexlb-cache",
    srcs = glob(["flexlb-cache/src/main/java/**/*.java"]),
    resources = glob(["flexlb-cache/src/main/resources/**"]),
    plugins = ["//rtp_llm/flexlb:lombok_plugin"],
    javacopts = [
        "-Xep:MustBeClosedChecker:OFF",  # Disable MustBeClosedChecker
    ],
    deps = [
        ":flexlb-common",
        "//rtp_llm/flexlb:lombok",
        "@flexlb_maven//:org_springframework_boot_spring_boot_starter_webflux",
        "@flexlb_maven//:com_github_ben_manes_caffeine_caffeine",
        "@flexlb_maven//:com_google_guava_guava",
        "@flexlb_maven//:org_apache_commons_commons_collections4",
        "@flexlb_maven//:org_apache_commons_commons_lang3",
        "@flexlb_maven//:jakarta_annotation_jakarta_annotation_api",
        "@flexlb_maven//:org_projectlombok_lombok",
        "@flexlb_maven//:org_springframework_spring_beans",
        "@flexlb_maven//:org_springframework_spring_context",
        "@flexlb_maven//:org_slf4j_slf4j_api",
    ],
)

# ==== FlexLB gRPC Module ====
java_library(
    name = "flexlb-grpc",
    srcs = glob(["flexlb-grpc/src/main/java/**/*.java"]),
    resources = glob(["flexlb-grpc/src/main/resources/**"]),
    plugins = ["//rtp_llm/flexlb:lombok_plugin"],
    javacopts = [
        "-Xep:MustBeClosedChecker:OFF",  # Disable MustBeClosedChecker
    ],
    deps = [
        ":flexlb-common",
        ":flexlb-cache",
        ":engine_rpc_java_grpc",
        ":engine_rpc_java_proto",
        "@flexlb_maven//:io_grpc_grpc_all",
        "@flexlb_maven//:io_grpc_grpc_api",
        "@flexlb_maven//:io_grpc_grpc_context",
        "@flexlb_maven//:io_grpc_grpc_core",
        "@flexlb_maven//:io_grpc_grpc_services",
        "@flexlb_maven//:io_grpc_grpc_stub",
        "@flexlb_maven//:io_grpc_grpc_protobuf",
        "@flexlb_maven//:io_grpc_grpc_netty",
        "@flexlb_maven//:io_grpc_grpc_census",
        "@flexlb_maven//:com_google_protobuf_protobuf_java",
        "@flexlb_maven//:com_google_protobuf_protobuf_java_util",
        "@flexlb_maven//:com_google_guava_guava",
    ],
)

# ==== FlexLB Sync Module ====
java_library(
    name = "flexlb-sync",
    srcs = glob(["flexlb-sync/src/main/java/**/*.java"]),
    resources = glob(["flexlb-sync/src/main/resources/**"]),
    javacopts = [
        "-Xep:MustBeClosedChecker:OFF",  # Disable MustBeClosedChecker
    ],
    deps = [
        ":flexlb-common",
        ":flexlb-cache",
        ":flexlb-grpc",
        "@flexlb_maven//:com_github_ben_manes_caffeine_caffeine",
        "@flexlb_maven//:org_apache_curator_curator_framework",
        "@flexlb_maven//:org_apache_curator_curator_recipes",
        "@flexlb_maven//:org_apache_zookeeper_zookeeper",
        "@flexlb_maven//:io_projectreactor_reactor_core",
        "@flexlb_maven//:org_springframework_boot_spring_boot_starter_webflux",
        "@flexlb_maven//:org_slf4j_slf4j_api",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_sdk",
        "@flexlb_maven//:io_opentelemetry_opentelemetry_exporter_otlp",
    ],
)

# ==== FlexLB API Module ====
java_library(
    name = "flexlb-api",
    srcs = glob(["flexlb-api/src/main/java/**/*.java"]),
    resources = glob(["flexlb-api/src/main/resources/**"]),
    javacopts = [
        "-Xep:MustBeClosedChecker:OFF",  # Disable MustBeClosedChecker
    ],
    deps = [
        ":flexlb-sync",
        "@flexlb_maven//:org_springframework_boot_spring_boot_starter_webflux",
        "@flexlb_maven//:org_springframework_boot_spring_boot_starter_test",
    ],
    # main_class = "org.flexlb.Application",
)

load("//:def.bzl", "copts")

package(default_visibility = ["//visibility:public"])

# Dependency-light OTel trace runtime: process-level TracerProvider /
# sampler / BSP / propagator management, gRPC metadata carriers and idempotent
# span finish guard. Business modules depend on this target only; OTel headers
# must not spread into scheduler/GPU core modules.
cc_library(
    name = "telemetry",
    srcs = ["TelemetryRuntime.cc"],
    hdrs = [
        "GrpcTraceCarrier.h",
        "PhaseSpanSynthesizer.h",
        "RequestSpanGuard.h",
        "RpcTraceHelper.h",
        "TelemetryRuntime.h",
        "TraceAttributes.h",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@grpc//:grpc++",
        "@io_opentelemetry_cpp//api",
        "@io_opentelemetry_cpp//exporters/otlp:otlp_http_exporter",
        "@io_opentelemetry_cpp//sdk/src/resource",
        "@io_opentelemetry_cpp//sdk/src/trace",
    ],
)

"""RTP-LLM OTel trace telemetry (trace-only signal, off unless enabled by env)."""


def __getattr__(name):
    # 配置解析器可独立导入；仅请求运行时 API 时加载 SDK。
    if name in __all__:
        from rtp_llm.telemetry import tracing

        return getattr(tracing, name)
    raise AttributeError(name)


__all__ = [
    "CURRENT_TRACE_STATE",
    "OTEL_AVAILABLE",
    "ClientSpanHandle",
    "RequestTraceState",
    "TelemetryState",
    "extract_context_from_headers",
    "get_tracer",
    "init_telemetry",
    "init_telemetry_for_test",
    "inject_context_to_metadata",
    "is_telemetry_active",
    "record_response_attributes",
    "resolve_pod_ip",
    "shutdown_telemetry",
    "start_client_span",
    "start_internal_span",
    "start_server_span",
    "telemetry_state",
]

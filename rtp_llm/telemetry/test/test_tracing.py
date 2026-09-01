"""Unit tests for rtp_llm.telemetry.tracing.

Run directly (conda310 interpreter) or via Bazel:
    /opt/conda310/bin/python -m unittest rtp_llm.telemetry.test.test_tracing -v
    bazelisk test //rtp_llm/telemetry/test:test_tracing

unittest style on purpose (repo convention; pytest is not part of the test
runtime). The dependency-contract test checks that the tracing SDK is available
in the configured test environment rather than relying on an ambient import.
The unskipped dependency-contract test prevents a missing runtime from turning
the functional suite into an all-skip success.
"""

import json
import os
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from rtp_llm.telemetry import attributes as attrs
from rtp_llm.telemetry import tracing

try:  # probe only; real imports live in tracing
    import opentelemetry  # noqa: F401

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on bare images
    OTEL_AVAILABLE = False

TELEMETRY_ENVS = [
    "RTP_LLM_OTEL_TRACE_ENABLE",
    "RTP_LLM_OTEL_REGION",
    "RTP_LLM_OTEL_REGION_CONFIG_FILE",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_CERTIFICATE",
    "RTP_LLM_OTEL_TRACE_SAMPLER_RATIO",
    "RTP_LLM_OTEL_TRUST_REMOTE_SAMPLING",
    "RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE",
    "RTP_LLM_OTEL_BSP_SCHEDULE_DELAY_MS",
    "RTP_LLM_OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
    "RTP_LLM_OTEL_HTTP_TIMEOUT_MS",
    "RTP_LLM_OTEL_SERVICE_NAME",
    "RTP_LLM_OTEL_SCOPE_VERSION",
    "POD_IP",
    "RequestedIP",
]


def _reset_runtime():
    assert tracing.reset_telemetry_for_test()


def _start_in_memory_runtime():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    assert tracing.init_telemetry_for_test(exporter, role="test", tp_rank=0)
    return exporter


class TestDependencyContract(unittest.TestCase):
    def test_opentelemetry_runtime_is_available(self):
        self.assertTrue(
            tracing.OTEL_AVAILABLE,
            f"opentelemetry runtime unavailable: {tracing._OTEL_IMPORT_ERROR!r}",
        )

    def test_otlp_http_trace_exporter_is_available(self):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        self.assertIsNotNone(OTLPSpanExporter)

    def test_metadata_to_headers_is_lowercase_last_value_wins(self):
        metadata = (
            ("TraceParent", "first"),
            ("traceparent", "second"),
            (b"X-Request-ID", "request"),
            ("binary-bin", b"\x00\x01"),
            ("malformed",),
            (None, "ignored"),
        )

        self.assertEqual(
            tracing.metadata_to_headers(metadata),
            {
                "traceparent": "second",
                "x-request-id": "request",
                "binary-bin": b"\x00\x01",
            },
        )


@unittest.skipUnless(OTEL_AVAILABLE, "opentelemetry not installed")
class TracingTestCase(unittest.TestCase):
    """Shared env/runtime isolation, mirroring the old autouse fixture."""

    def setUp(self):
        self._saved_env = {env: os.environ.pop(env, None) for env in TELEMETRY_ENVS}
        _reset_runtime()

    def tearDown(self):
        _reset_runtime()
        for env, value in self._saved_env.items():
            if value is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = value


class TestConfig(TracingTestCase):
    def test_disabled_by_default(self):
        assert not tracing.init_telemetry("frontend", 0)
        assert tracing.telemetry_state() == tracing.TelemetryState.DISABLED
        assert not tracing.is_telemetry_active()

    def test_disabled_region_resolution_has_no_side_effects(self):
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        os.environ["RequestedIP"] = "10.4.5.6"
        with (
            mock.patch.object(tracing, "_resolve_region_config") as resolve_config,
            mock.patch.object(socket, "gethostbyname") as gethostbyname,
        ):
            tracing.resolve_region_env()
        resolve_config.assert_not_called()
        gethostbyname.assert_not_called()
        assert "RTP_LLM_OTEL_SCOPE_VERSION" not in os.environ
        assert "POD_IP" not in os.environ

    def test_enabled_without_endpoint_disabled(self):
        os.environ["RTP_LLM_OTEL_TRACE_ENABLE"] = "1"
        assert not tracing.init_telemetry("frontend", 0)
        assert tracing.telemetry_state() == tracing.TelemetryState.DISABLED

    def test_non_rank0_disabled(self):
        os.environ["RTP_LLM_OTEL_TRACE_ENABLE"] = "1"
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
            "http://127.0.0.1:4318/v1/traces"
        )
        assert not tracing.init_telemetry("prefill", 1)
        assert tracing.telemetry_state() == tracing.TelemetryState.DISABLED

    def test_endpoint_priority_signal_specific_wins(self):
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
            "http://signal:4318/v1/traces"
        )
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://generic:4318"
        assert tracing.resolve_endpoint() == "http://signal:4318/v1/traces"

    def test_endpoint_generic_appends_path(self):
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://generic:4318/"
        assert tracing.resolve_endpoint() == "http://generic:4318/v1/traces"

    def test_endpoint_missing_empty(self):
        assert tracing.resolve_endpoint() == ""

    def test_endpoint_log_target_omits_credentials_path_and_query(self):
        endpoint = "https://user:secret@collector.example:4318/token/path?sig=abc"
        assert tracing._endpoint_log_target(endpoint) == (
            "https://collector.example:4318"
        )
        assert "secret" not in tracing._endpoint_log_target(endpoint)
        assert "sig" not in tracing._endpoint_log_target(endpoint)

    def test_invalid_env_values_fall_back(self):
        os.environ["RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE"] = "-5"
        assert (
            tracing._env_positive_int("RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE", 2048) == 2048
        )
        for value in ("3.5", "nan", "inf", "-inf", "0.5junk"):
            with self.subTest(value=value):
                os.environ["RTP_LLM_OTEL_TRACE_SAMPLER_RATIO"] = value
                assert (
                    tracing._env_ratio("RTP_LLM_OTEL_TRACE_SAMPLER_RATIO", 1.0) == 1.0
                )

    def test_otel_unavailable_disables(self):
        os.environ["RTP_LLM_OTEL_TRACE_ENABLE"] = "1"
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
            "http://127.0.0.1:4318/v1/traces"
        )
        with mock.patch.object(tracing, "OTEL_AVAILABLE", False), mock.patch.object(
            tracing, "_OTEL_IMPORT_ERROR", ImportError("simulated"), create=True
        ):
            assert not tracing.init_telemetry("frontend", 0)
            assert tracing.telemetry_state() == tracing.TelemetryState.DISABLED

    def test_invalid_region_config_shape_uses_explicit_endpoint(self):
        os.environ["RTP_LLM_OTEL_TRACE_ENABLE"] = "1"
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
            "http://127.0.0.1:4318/v1/traces"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            config_file.write("[]")
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            with self.assertLogs(tracing._LOGGER, level="WARNING") as logs:
                assert tracing.init_telemetry("frontend", 0)

        assert tracing.telemetry_state() == tracing.TelemetryState.ACTIVE
        assert any(
            "region config resolution failed" in message for message in logs.output
        )

    def test_invalid_region_entry_does_not_partially_update_environment(self):
        os.environ["RTP_LLM_OTEL_TRACE_ENABLE"] = "1"
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        config = {
            "regions": {
                "cn-test": {
                    "endpoint": "http://region-collector:4318/v1/traces",
                    "headers": ["invalid"],
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            with self.assertLogs(tracing._LOGGER, level="WARNING"):
                assert not tracing.init_telemetry("frontend", 0)

        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ
        assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ

    def test_explicit_generic_endpoint_rejects_region_carrier(self):
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://explicit:4318"
        config = {
            "regions": {
                "cn-test": {
                    "endpoint": "http://region-collector:4318/v1/traces",
                    "headers": "authorization=region",
                    "certificate": "/region/ca.pem",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            tracing._resolve_region_config()

        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ
        assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
        assert "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE" not in os.environ
        assert tracing.resolve_endpoint() == "http://explicit:4318/v1/traces"

    def test_explicit_signal_endpoint_rejects_region_credentials(self):
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
            "http://explicit:4318/v1/traces"
        )
        config = {
            "regions": {
                "cn-test": {
                    "endpoint": "http://region-collector:4318/v1/traces",
                    "headers": "authorization=region",
                    "certificate": "/region/ca.pem",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            tracing._resolve_region_config()

        assert os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"].startswith(
            "http://explicit:"
        )
        assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
        assert "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE" not in os.environ

    def test_explicit_signal_credentials_reject_region_endpoint(self):
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = "authorization=explicit"
        config = {
            "regions": {
                "cn-test": {
                    "endpoint": "http://region-collector:4318/v1/traces",
                    "headers": "authorization=region",
                    "certificate": "/region/ca.pem",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            tracing._resolve_region_config()

        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ
        assert (
            os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] == "authorization=explicit"
        )
        assert "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE" not in os.environ

    def test_explicit_generic_credentials_reject_region_endpoint(self):
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        os.environ["OTEL_EXPORTER_OTLP_CERTIFICATE"] = "/explicit/ca.pem"
        config = {
            "regions": {
                "cn-test": {
                    "endpoint": "http://region-collector:4318/v1/traces",
                    "headers": "authorization=region",
                    "certificate": "/region/ca.pem",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            tracing._resolve_region_config()

        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ
        assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
        assert os.environ["OTEL_EXPORTER_OTLP_CERTIFICATE"] == "/explicit/ca.pem"

    def test_region_config_preserves_explicit_signal_environment(self):
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        explicit_env = {
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://explicit:4318/v1/traces",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "authorization=explicit",
            "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": "/explicit/ca.pem",
        }
        os.environ.update(explicit_env)
        config = {
            "regions": {
                "cn-test": {
                    "endpoint": "http://region-collector:4318/v1/traces",
                    "headers": "authorization=region",
                    "certificate": "/region/ca.pem",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            tracing._resolve_region_config()

        for env_name, value in explicit_env.items():
            assert os.environ[env_name] == value

    def test_region_credentials_without_endpoint_are_not_injected(self):
        os.environ["RTP_LLM_OTEL_REGION"] = "cn-test"
        config = {
            "regions": {
                "cn-test": {
                    "headers": "authorization=region",
                    "certificate": "/region/ca.pem",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            os.environ["RTP_LLM_OTEL_REGION_CONFIG_FILE"] = config_file.name
            with self.assertLogs(tracing._LOGGER, level="WARNING"):
                tracing._resolve_region_config()

        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ
        assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
        assert "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE" not in os.environ


class TestInactiveNoop(TracingTestCase):
    def test_apis_safe_when_inactive(self):
        assert tracing.get_tracer() is None
        assert tracing.extract_context_from_headers({"traceparent": "x"}) is None
        assert tracing.inject_context_to_metadata(None) == []
        assert tracing.start_server_span("noop", {}) is None
        # finish on empty state is a no-op
        tracing.RequestTraceState().finish()


class TestScopeVersion(TracingTestCase):
    """otel.scope.version: env override > rtp_llm wheel metadata > empty."""

    def setUp(self):
        super().setUp()
        tracing._scope_version_cache = None

    def tearDown(self):
        tracing._scope_version_cache = None
        super().tearDown()

    def test_scope_version_from_env_on_spans(self):
        os.environ["RTP_LLM_OTEL_SCOPE_VERSION"] = "9.9.9-test"
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("scope_probe", {})
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].instrumentation_scope.name == "rtp_llm"
        assert spans[0].instrumentation_scope.version == "9.9.9-test"

    def test_resolve_region_env_exports_scope_version(self):
        os.environ["RTP_LLM_OTEL_TRACE_ENABLE"] = "1"
        os.environ.pop("RTP_LLM_OTEL_SCOPE_VERSION", None)
        with mock.patch.object(tracing, "_scope_version_cache", "7.7.7-launcher"):
            tracing.resolve_region_env()
            assert os.environ.get("RTP_LLM_OTEL_SCOPE_VERSION") == "7.7.7-launcher"


class TestResource(TracingTestCase):
    """Parity with the C++ runtime: host.ip only from POD_IP."""

    def setUp(self):
        super().setUp()
        os.environ["RTP_LLM_OTEL_TRACE_ENABLE"] = "1"

    def _finished_resource_attributes(self, exporter):
        state = tracing.start_server_span("resource_probe", {})
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        return spans[0].resource.attributes

    def test_host_ip_from_pod_ip(self):
        os.environ["POD_IP"] = "10.1.2.3"
        exporter = _start_in_memory_runtime()
        attributes = self._finished_resource_attributes(exporter)
        assert attributes.get("host.ip") == "10.1.2.3"
        assert attributes.get("rtp_llm.role") == "test"

    def test_host_ip_absent_without_pod_ip(self):
        exporter = _start_in_memory_runtime()
        attributes = self._finished_resource_attributes(exporter)
        assert "host.ip" not in attributes

    def test_resolve_region_env_preserves_existing_pod_ip(self):
        os.environ["POD_IP"] = "10.1.2.3"
        os.environ["RequestedIP"] = "10.4.5.6"
        with mock.patch.object(socket, "gethostbyname") as gethostbyname:
            tracing.resolve_region_env()
        assert os.environ["POD_IP"] == "10.1.2.3"
        gethostbyname.assert_not_called()

    def test_resolve_region_env_uses_requested_ip(self):
        os.environ["RequestedIP"] = "10.4.5.6"
        with mock.patch.object(socket, "gethostbyname") as gethostbyname:
            tracing.resolve_region_env()
        assert os.environ["POD_IP"] == "10.4.5.6"
        gethostbyname.assert_not_called()

    def test_resolve_region_env_replaces_empty_pod_ip(self):
        os.environ["POD_IP"] = ""
        os.environ["RequestedIP"] = "10.4.5.6"
        tracing.resolve_region_env()
        assert os.environ["POD_IP"] == "10.4.5.6"

    def test_resolve_region_env_rejects_invalid_requested_ip(self):
        os.environ["RequestedIP"] = "127.0.0.1"
        with mock.patch.object(socket, "gethostbyname", return_value="10.7.8.9"):
            tracing.resolve_region_env()
        assert os.environ["POD_IP"] == "10.7.8.9"

    def test_resolve_region_env_uses_hostname_ip(self):
        with mock.patch.object(
            socket, "gethostname", return_value="test-host"
        ), mock.patch.object(
            socket, "gethostbyname", return_value="10.7.8.9"
        ) as gethostbyname:
            tracing.resolve_region_env()
        assert os.environ["POD_IP"] == "10.7.8.9"
        gethostbyname.assert_called_once_with("test-host")

    def test_resolve_region_env_dns_failure_is_fail_open(self):
        with mock.patch.object(
            socket, "gethostbyname", side_effect=socket.gaierror("not found")
        ):
            tracing.resolve_region_env()
        assert "POD_IP" not in os.environ

    def test_resolve_region_env_rejects_invalid_automatic_ips(self):
        for resolved_ip in ("", "127.0.0.1", "0.0.0.0"):
            with self.subTest(resolved_ip=resolved_ip):
                os.environ.pop("POD_IP", None)
                os.environ.pop("RequestedIP", None)
                with mock.patch.object(
                    socket, "gethostbyname", return_value=resolved_ip
                ):
                    tracing.resolve_region_env()
                assert "POD_IP" not in os.environ

    def test_resolve_region_env_is_idempotent(self):
        os.environ["RequestedIP"] = "10.4.5.6"
        tracing.resolve_region_env()
        os.environ["RequestedIP"] = "10.7.8.9"
        tracing.resolve_region_env()
        assert os.environ["POD_IP"] == "10.4.5.6"

    def test_resolved_pod_ip_populates_span_resource(self):
        os.environ["RequestedIP"] = "10.4.5.6"
        tracing.resolve_region_env()
        exporter = _start_in_memory_runtime()
        attributes = self._finished_resource_attributes(exporter)
        assert attributes.get("host.ip") == "10.4.5.6"

    def test_service_name_derived_from_role(self):
        # no env override -> "rtp_llm_" + role (role-split components)
        os.environ.pop("RTP_LLM_OTEL_SERVICE_NAME", None)
        exporter = _start_in_memory_runtime()
        attributes = self._finished_resource_attributes(exporter)
        assert attributes.get("service.name") == "rtp_llm_test"

    def test_service_name_env_override(self):
        os.environ["RTP_LLM_OTEL_SERVICE_NAME"] = "custom_svc"
        exporter = _start_in_memory_runtime()
        attributes = self._finished_resource_attributes(exporter)
        assert attributes.get("service.name") == "custom_svc"


class TestActiveRuntime(TracingTestCase):
    def test_span_export_and_attributes(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("rtp_llm.http_server", {})
        assert state is not None
        state.set_attribute("rtp_llm.request_id", 42)
        state.finish()
        tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "rtp_llm.http_server"
        assert spans[0].attributes["rtp_llm.request_id"] == 42

    def test_master_route_internal_span_attributes(self):
        """PD node-selection span contract: INTERNAL kind (in-process routing
        stage, may involve zero outbound calls), child of the SERVER span,
        carries route.source + the platform request_id index key."""
        from opentelemetry.trace import SpanKind

        from rtp_llm.telemetry import attributes as trace_attrs

        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("POST /v1/chat/completions", {})
        handle = tracing.start_internal_span("rtp_llm.master_route")
        assert handle is not None
        handle.set_attribute("request_id", "42")
        handle.set_attribute(trace_attrs.RTP_LLM_ROUTE_SOURCE, "master")
        handle.finish()
        state.finish()
        tracing.shutdown_telemetry()

        spans = {s.name: s for s in exporter.get_finished_spans()}
        route = spans["rtp_llm.master_route"]
        server = spans["POST /v1/chat/completions"]
        assert route.kind == SpanKind.INTERNAL
        assert route.parent.span_id == server.context.span_id
        assert route.attributes["rtp_llm.route.source"] == "master"
        assert route.attributes["request_id"] == "42"

    def test_master_route_failure_uses_internal_description(self):
        from opentelemetry.trace import StatusCode

        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("POST /v1/chat/completions", {})
        handle = tracing.start_internal_span("rtp_llm.master_route")
        assert handle is not None
        handle.finish(error=RuntimeError("raw route detail"), error_type="TrafficLimit")
        state.finish()
        tracing.shutdown_telemetry()

        spans = {s.name: s for s in exporter.get_finished_spans()}
        route = spans["rtp_llm.master_route"]
        assert route.status.status_code == StatusCode.ERROR
        assert (
            route.status.description == "Request routing was rejected by traffic limits"
        )
        assert "raw route detail" not in route.status.description
        assert route.attributes["error.type"] == "TrafficLimit"

    def test_http_status_dual_write_on_server_span(self):
        """Root SERVER span carries both semconv generations of http attrs
        (platform views read old/new with inconsistent priority)."""
        from rtp_llm.telemetry import attributes as trace_attrs

        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("POST /v1/chat/completions", {})
        assert state is not None
        state.set_attribute(trace_attrs.HTTP_REQUEST_METHOD, "POST")
        state.set_attribute(trace_attrs.HTTP_METHOD, "POST")
        state.set_attribute(trace_attrs.HTTP_RESPONSE_STATUS_CODE, 200)
        state.set_attribute(trace_attrs.HTTP_STATUS_CODE, 200)
        state.finish()
        tracing.shutdown_telemetry()

        span_attrs = exporter.get_finished_spans()[0].attributes
        assert span_attrs["http.request.method"] == "POST"
        assert span_attrs["http.method"] == "POST"
        assert span_attrs["http.response.status_code"] == 200
        assert span_attrs["http.status_code"] == 200

    def test_remote_parent_from_traceparent_header(self):
        exporter = _start_in_memory_runtime()
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }
        state = tracing.start_server_span("child", headers)
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert (
            format(spans[0].context.trace_id, "032x")
            == "0af7651916cd43dd8448eb211c80319c"
        )
        assert format(spans[0].parent.span_id, "016x") == "b7ad6b7169203331"

    def test_server_span_accepts_explicit_start_time(self):
        exporter = _start_in_memory_runtime()
        start_time = time.time_ns() - 1_000_000
        state = tracing.start_server_span("delayed", {}, start_time=start_time)
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].start_time == start_time

    def test_untrusted_unsampled_remote_parent_uses_local_sampler(self):
        exporter = _start_in_memory_runtime()
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00"
        }
        state = tracing.start_server_span("unsampled_child", headers)
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert format(spans[0].parent.span_id, "016x") == "b7ad6b7169203331"

    def test_untrusted_sampled_remote_parent_cannot_bypass_zero_ratio(self):
        os.environ["RTP_LLM_OTEL_TRACE_SAMPLER_RATIO"] = "0"
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span(
            "untrusted_sampled",
            {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"},
        )
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()
        assert exporter.get_finished_spans() == ()

    def test_explicit_trust_preserves_remote_sampling_decision(self):
        os.environ["RTP_LLM_OTEL_TRACE_SAMPLER_RATIO"] = "0"
        os.environ["RTP_LLM_OTEL_TRUST_REMOTE_SAMPLING"] = "1"
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span(
            "trusted_sampled",
            {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"},
        )
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert format(spans[0].context.trace_id, "032x") == (
            "0af7651916cd43dd8448eb211c80319c"
        )

    def test_invalid_traceparent_falls_back_to_local_root(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("root", {"traceparent": "garbage"})
        assert state is not None
        state.finish()
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].parent is None

    def test_delayed_server_span_uses_rpc_monotonic_start_for_ttft(self):
        exporter = _start_in_memory_runtime()
        request_start_ns = time.monotonic_ns()
        state = tracing.start_server_span(
            "delayed_ttft",
            {},
            start_time=time.time_ns(),
            request_start_ns=request_start_ns,
        )
        assert state is not None
        state.record_frontend_output_tokens(1, request_start_ns + 50_000_000)
        state.finish()
        tracing.shutdown_telemetry()
        span = exporter.get_finished_spans()[0]
        assert abs(span.attributes["gen_ai.response.time_to_first_token"] - 50.0) < 1e-6

    def test_inject_extract_roundtrip(self):
        _start_in_memory_runtime()
        state = tracing.start_server_span("parent", {})
        metadata = tracing.inject_context_to_metadata(state.server_context)
        keys = {k for k, _ in metadata}
        assert "traceparent" in keys

        headers = dict(metadata)
        extracted = tracing.extract_context_from_headers(headers)
        from opentelemetry import trace as otel_trace

        remote_span_context = otel_trace.get_current_span(extracted).get_span_context()
        parent_span_context = state.server_span.get_span_context()
        assert remote_span_context.trace_id == parent_span_context.trace_id
        assert remote_span_context.span_id == parent_span_context.span_id
        state.finish()

    def test_finish_is_idempotent(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("idempotent", {})
        state.finish()
        state.finish()
        state.finish(error=RuntimeError("late error must be ignored"))
        tracing.shutdown_telemetry()
        assert len(exporter.get_finished_spans()) == 1

    def test_finish_with_error_sets_status(self):
        from opentelemetry.trace import StatusCode

        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("failed", {})
        state.finish(error=ValueError("boom"))
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        assert spans[0].status.description == "Request processing failed"
        assert "boom" not in spans[0].status.description
        assert spans[0].attributes["error.type"] == "ValueError"

    def test_finish_success_sets_ok_status(self):
        # Explicit OK on success (Unset renders as a blank status in the
        # platform UI and would diverge from the C++ GrpcStatusSpanGuard kOk
        # behavior).
        from opentelemetry.trace import StatusCode

        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("ok_root", {})
        handle, _ = tracing.start_client_span("ok_client")
        assert handle is not None
        handle.finish()
        state.finish()
        tracing.shutdown_telemetry()
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert spans["ok_root"].status.status_code == StatusCode.OK
        assert spans["ok_root"].status.description is None
        assert spans["ok_client"].status.status_code == StatusCode.OK
        assert spans["ok_client"].status.description is None

    def test_settled_ok_reports_root_span_outcome(self):
        """settled_ok lets a child span classify its own teardown.

        A CLIENT span whose generator is closed only after the response was
        fully delivered cannot tell a real interruption from plain cleanup by
        the exception type (both are GeneratorExit), so it reads the parent
        outcome instead.
        """
        _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        assert state.settled_ok is None, "unsettled while the request runs"
        state.finish()
        assert state.settled_ok is True
        state.finish(error_type="Cancelled")  # idempotent, must not flip
        assert state.settled_ok is True
        tracing.shutdown_telemetry()

    def test_settled_ok_is_false_after_error_finish(self):
        _start_in_memory_runtime()
        state = tracing.start_server_span("srv_err", {})
        state.finish(error_type="Cancelled")
        assert state.settled_ok is False
        tracing.shutdown_telemetry()

    def test_error_descriptions_are_bounded_and_predictable(self):
        assert (
            tracing._request_error_description("Cancelled")
            == "Request processing was cancelled"
        )
        assert (
            tracing._request_error_description("TrafficLimit")
            == "Request was rejected by traffic limits"
        )
        assert (
            tracing._request_error_description("raw-secret-request-error")
            == "Request processing failed"
        )
        assert (
            tracing._client_error_description("Cancelled")
            == "Client operation was cancelled"
        )
        assert (
            tracing._client_error_description("TrafficLimit")
            == "Request routing was rejected by traffic limits"
        )
        assert (
            tracing._client_error_description("raw-secret-client-error")
            == "Client operation failed"
        )
        assert (
            tracing._internal_error_description("TrafficLimit")
            == "Request routing was rejected by traffic limits"
        )
        assert (
            tracing._internal_error_description("raw-secret-route-error")
            == "Request routing failed"
        )

    def test_attribute_after_finish_dropped(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("frozen", {})
        state.finish()
        state.set_attribute("rtp_llm.late", "dropped")
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert "rtp_llm.late" not in spans[0].attributes

    def test_current_trace_state_contextvar(self):
        _start_in_memory_runtime()
        assert tracing.CURRENT_TRACE_STATE.get() is None
        state = tracing.start_server_span("ctxvar", {})
        assert tracing.CURRENT_TRACE_STATE.get() is state
        state.finish()

    def test_client_span_child_of_server_span(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("server", {})
        tracer = tracing.get_tracer()
        from opentelemetry import trace as otel_trace

        client_span = tracer.start_span(
            "client", context=state.server_context, kind=otel_trace.SpanKind.CLIENT
        )
        client_span.end()
        state.finish()
        tracing.shutdown_telemetry()

        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert spans["client"].parent.span_id == spans["server"].context.span_id
        assert spans["client"].context.trace_id == spans["server"].context.trace_id

    def test_shutdown_idempotent(self):
        _start_in_memory_runtime()
        assert tracing.shutdown_telemetry()
        assert tracing.telemetry_state() == tracing.TelemetryState.SHUTDOWN
        assert tracing.shutdown_telemetry()

    def test_reset_telemetry_for_test_restores_fixture_state(self):
        _start_in_memory_runtime()
        tracing.CURRENT_TRACE_STATE.set(mock.sentinel.trace_state)
        assert tracing.shutdown_telemetry()

        self.assertTrue(tracing.reset_telemetry_for_test())
        self.assertEqual(
            tracing.telemetry_state(), tracing.TelemetryState.UNINITIALIZED
        )
        self.assertIsNone(tracing.CURRENT_TRACE_STATE.get())
        self.assertIsNotNone(_start_in_memory_runtime())

    def test_shutdown_is_terminal_for_reinitialization(self):
        _start_in_memory_runtime()
        assert tracing.shutdown_telemetry()
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        self.assertFalse(tracing.init_telemetry_for_test(InMemorySpanExporter()))
        self.assertEqual(tracing.telemetry_state(), tracing.TelemetryState.SHUTDOWN)
        self.assertIsNone(tracing.get_tracer())


class TestClientSpan(TracingTestCase):
    def test_no_state_returns_noop(self):
        _start_in_memory_runtime()
        handle, metadata = tracing.start_client_span("rtp_llm.generate_stream_call")
        assert handle is None
        assert metadata == []

    def test_client_span_with_metadata_child_of_server(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("server", {})
        handle, metadata = tracing.start_client_span("rtp_llm.generate_stream_call")
        assert handle is not None
        carrier = dict(metadata)
        assert "traceparent" in carrier
        handle.finish()
        state.finish()
        tracing.shutdown_telemetry()

        spans = {s.name: s for s in exporter.get_finished_spans()}
        client = spans["rtp_llm.generate_stream_call"]
        server = spans["server"]
        assert client.parent.span_id == server.context.span_id
        # metadata traceparent must reference the CLIENT span (next hop parent)
        assert format(client.context.span_id, "016x") in carrier["traceparent"]
        assert format(client.context.trace_id, "032x") in carrier["traceparent"]

    def test_zero_ratio_still_propagates_non_recording_client_context(self):
        os.environ["RTP_LLM_OTEL_TRACE_SAMPLER_RATIO"] = "0"
        exporter = _start_in_memory_runtime()
        trace_id = "0af7651916cd43dd8448eb211c80319c"
        state = tracing.start_server_span(
            "server",
            {"traceparent": f"00-{trace_id}-b7ad6b7169203331-01"},
        )
        handle, metadata = tracing.start_client_span("client")
        assert handle is not None
        carrier = dict(metadata)
        assert carrier["traceparent"].split("-")[1] == trace_id
        handle.finish()
        state.finish()
        tracing.shutdown_telemetry()
        assert exporter.get_finished_spans() == ()

    def test_client_span_omits_rpc_system(self):
        # rpc.system on the frontend CLIENT span made the platform re-classify
        # it as an RPC client call, breaking the top-bar Total tokens
        # aggregation (measured regression). Only the C++ span factories carry
        # the marker; the Python spans must stay clean of it.
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("server", {})
        handle, _ = tracing.start_client_span("rtp_llm.generate_stream_call")
        assert handle is not None
        handle.finish()
        state.finish()
        tracing.shutdown_telemetry()

        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert "rpc.system" not in spans["rtp_llm.generate_stream_call"].attributes
        assert "rpc.system" not in spans["server"].attributes

    def test_client_endpoint_attributes_for_supported_address_forms(self):
        cases = (
            ("worker.example:50051", "worker.example", 50051),
            ("127.0.0.1:50052", "127.0.0.1", 50052),
            ("[2001:db8::1]:50053", "2001:db8::1", 50053),
        )
        for index, (target, expected_address, expected_port) in enumerate(cases):
            with self.subTest(target=target):
                exporter = _start_in_memory_runtime()
                state = tracing.start_server_span(f"server-{index}", {})
                handle, _ = tracing.start_client_span(f"client-{index}", target)
                assert handle is not None
                handle.finish()
                state.finish()
                tracing.shutdown_telemetry()

                client = next(
                    span
                    for span in exporter.get_finished_spans()
                    if span.name == f"client-{index}"
                )
                assert client.attributes[attrs.SERVER_ADDRESS] == expected_address
                assert client.attributes[attrs.SERVER_PORT] == expected_port
                _reset_runtime()

    def test_client_endpoint_omitted_for_invalid_or_resolver_targets(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("server", {})
        for index, target in enumerate(
            ("dns:///worker.example:50051", "2001:db8::1:50051", "host", "host:70000")
        ):
            handle, _ = tracing.start_client_span(f"client-{index}", target)
            assert handle is not None
            handle.finish()
        state.finish()
        tracing.shutdown_telemetry()

        for span in exporter.get_finished_spans():
            if not span.name.startswith("client-"):
                continue
            assert attrs.SERVER_ADDRESS not in span.attributes
            assert attrs.SERVER_PORT not in span.attributes

    def test_client_span_finish_idempotent_with_error(self):
        from opentelemetry.trace import StatusCode

        exporter = _start_in_memory_runtime()
        tracing.start_server_span("server", {})
        handle, _ = tracing.start_client_span("client")
        handle.finish(error=RuntimeError("boom"), error_type="RpcError")
        handle.finish()  # no-op
        tracing.shutdown_telemetry()
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert spans["client"].status.status_code == StatusCode.ERROR
        assert spans["client"].status.description == "Model RPC request failed"
        assert "boom" not in spans["client"].status.description
        assert spans["client"].attributes["error.type"] == "RpcError"


class TestStreamingLifecycle(TracingTestCase):
    """Simulates the frontend stream_response four-exit contract."""

    def _run(self, coro):
        import asyncio

        return asyncio.run(coro)

    def test_success_exit_finishes_once(self):
        exporter = _start_in_memory_runtime()

        async def scenario():
            state = tracing.start_server_span("stream", {})

            async def stream_response():
                trace_state = tracing.CURRENT_TRACE_STATE.get()
                try:
                    for chunk in ("a", "b"):
                        yield chunk
                    trace_state.finish()
                finally:
                    trace_state.finish()

            chunks = [c async for c in stream_response()]
            assert chunks == ["a", "b"]
            assert tracing.CURRENT_TRACE_STATE.get() is state

        self._run(scenario())
        tracing.shutdown_telemetry()
        assert len(exporter.get_finished_spans()) == 1

    def test_cancel_exit_marks_error_once(self):
        from opentelemetry.trace import StatusCode

        exporter = _start_in_memory_runtime()

        async def scenario():
            import asyncio

            tracing.start_server_span("cancelled_stream", {})

            async def stream_response():
                trace_state = tracing.CURRENT_TRACE_STATE.get()
                try:
                    yield "a"
                    await asyncio.sleep(30)
                    yield "b"
                except asyncio.CancelledError as e:
                    trace_state.finish(error=e, error_type="Cancelled")
                finally:
                    trace_state.finish()

            gen = stream_response()
            assert await gen.__anext__() == "a"

            task = asyncio.ensure_future(gen.__anext__())
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            await gen.aclose()

        self._run(scenario())
        tracing.shutdown_telemetry()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        assert spans[0].status.description == "Request processing was cancelled"
        assert spans[0].attributes["error.type"] == "Cancelled"


class TestResponseAttributes(TracingTestCase):
    """Request-level gen_ai.* business attributes on the SERVER span."""

    _SAMPLE = {
        "model": "qwen-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        "aux_info": {
            "first_token_cost_time": 8.5,  # ms (already converted on the py side)
            "cost_time": 20.0,
            "wait_time": 2.0,
            "iter_count": 5,
            "input_len": 12,
            "output_len": 5,
            "pd_sep": True,
            "reuse_len": 4,
            "local_reuse_len": 3,
            "remote_reuse_len": 1,
        },
    }

    def test_core_business_attributes_written(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span(
            "POST /v1/chat/completions",
            {},
            initial_attributes={attrs.GEN_AI_REQUEST_MODEL: "requested-model"},
        )
        tracing.record_response_attributes(self._SAMPLE)
        state.finish()
        tracing.shutdown_telemetry()

        span = exporter.get_finished_spans()[0]
        a = span.attributes
        assert a[attrs.GEN_AI_REQUEST_MODEL] == "requested-model"
        assert tuple(a[attrs.GEN_AI_RESPONSE_FINISH_REASONS]) == ("stop",)
        assert a[attrs.GEN_AI_USAGE_INPUT_TOKENS] == 12
        assert a[attrs.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
        # legacy aliases that some platform views read instead of the new keys
        assert a[attrs.GEN_AI_USAGE_PROMPT_TOKENS] == 12
        assert a[attrs.GEN_AI_USAGE_COMPLETION_TOKENS] == 5
        assert a[attrs.GEN_AI_USAGE_TOTAL_TOKENS] == 17
        assert attrs.GEN_AI_TIME_TO_FIRST_TOKEN not in a
        assert attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS not in a
        assert attrs.RTP_LLM_ENGINE_TIME_TO_FIRST_TOKEN_MS not in a
        assert attrs.RTP_LLM_ENGINE_TIME_PER_OUTPUT_TOKEN_MS not in a
        assert a[attrs.RTP_LLM_PD_SEP] is True
        assert a[attrs.RTP_LLM_CACHE_TOTAL_REUSE_LEN] == 4
        assert a[attrs.RTP_LLM_CACHE_LOCAL_REUSE_LEN] == 3
        assert a[attrs.RTP_LLM_CACHE_REMOTE_REUSE_LEN] == 1

    def test_trimmed_attributes_absent(self):
        # Raw engine timing fields and their derived TTFT/TPOT are not written
        # on the logical request SERVER span.
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        tracing.record_response_attributes(self._SAMPLE)
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert "rtp_llm.cost_time_ms" not in a
        assert "rtp_llm.wait_time_ms" not in a
        assert "rtp_llm.iter_count" not in a
        assert attrs.GEN_AI_TIME_TO_FIRST_TOKEN not in a
        assert attrs.RTP_LLM_ENGINE_TIME_PER_OUTPUT_TOKEN_MS not in a

    def test_multiple_finish_reasons_as_array(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        tracing.record_response_attributes(
            {
                "choices": [
                    {"index": 0, "finish_reason": "stop"},
                    {"index": 1, "finish_reason": "length"},
                ]
            }
        )
        state.finish()
        tracing.shutdown_telemetry()
        reasons = exporter.get_finished_spans()[0].attributes[
            attrs.GEN_AI_RESPONSE_FINISH_REASONS
        ]
        assert tuple(reasons) == ("stop", "length")

    def test_partial_response_is_robust(self):
        # Missing usage / aux_info / choices must not raise. A response model
        # must not overwrite the request-or-loaded model selected at entry.
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span(
            "srv",
            {},
            initial_attributes={attrs.GEN_AI_REQUEST_MODEL: "loaded-model"},
        )
        tracing.record_response_attributes({"model": "only-model"})
        state.finish()
        tracing.shutdown_telemetry()
        a = exporter.get_finished_spans()[0].attributes
        assert a[attrs.GEN_AI_REQUEST_MODEL] == "loaded-model"
        assert attrs.GEN_AI_USAGE_INPUT_TOKENS not in a
        assert attrs.GEN_AI_RESPONSE_FINISH_REASONS not in a

    def test_no_state_is_noop(self):
        _start_in_memory_runtime()
        tracing.CURRENT_TRACE_STATE.set(None)
        # no request span in scope -> safe no-op, never raises
        tracing.record_response_attributes(self._SAMPLE)

    def test_inactive_is_noop(self):
        # telemetry disabled -> no-op even when called with a payload
        tracing.record_response_attributes(self._SAMPLE)


class TestPlatformGapFixAttributes(TracingTestCase):
    """Platform gap fixes: LLM-view classification, phase latency, baggage."""

    _SAMPLE = TestResponseAttributes._SAMPLE

    def test_llm_view_classification_attributes(self):
        # platform classification trio + OTel GenAI semconv pair, both on root.
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        tracing.record_response_attributes(self._SAMPLE)
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert a[attrs.GEN_AI_SPAN_KIND] == "LLM"
        assert a[attrs.LINGJI_FLAG] is True
        assert a[attrs.ACS_ARMS_TENANT_SPAN_POLICY] == "mask"
        assert a[attrs.GEN_AI_OPERATION_NAME] == "chat"
        assert a[attrs.GEN_AI_SYSTEM] == "rtp_llm"

    def test_phase_latency_nanoseconds(self):
        # _SAMPLE: ttft=8.5ms wait=2.0ms cost=20.0ms
        # prefill = 6.5ms = 6_500_000 ns; decode = 11.5ms = 11_500_000 ns
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        sample = {
            **self._SAMPLE,
            "aux_info": {**self._SAMPLE["aux_info"], "pd_sep": False},
        }
        tracing.record_response_attributes(sample)
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert a[attrs.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL] == 6_500_000
        assert a[attrs.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE] == 11_500_000

    def test_phase_latency_omitted_for_pd_response(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        tracing.record_response_attributes(self._SAMPLE)
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert attrs.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL not in a
        assert attrs.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE not in a

    def test_phase_latency_skipped_on_missing_inputs(self):
        # no wait_time -> no prefill attr; no cost_time -> no decode attr
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        tracing.record_response_attributes({"aux_info": {"first_token_cost_time": 8.5}})
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert attrs.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL not in a
        assert attrs.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE not in a

    def test_llm_sdk_baggage_consumed_with_prefix_stripped(self):
        exporter = _start_in_memory_runtime()
        headers = {"baggage": "traffic.llm_sdk.scene=chat,other.vendor.key=ignored"}
        state = tracing.start_server_span("srv", headers)
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert a["scene"] == "chat"
        assert "other.vendor.key" not in a
        assert "vendor.key" not in a

    def test_llm_sdk_baggage_rejects_unknown_and_reserved_attributes(self):
        exporter = _start_in_memory_runtime()
        headers = {
            "baggage": (
                "traffic.llm_sdk.tenant_id=private,"
                "traffic.llm_sdk.error.type=Injected,"
                "traffic.llm_sdk.gen_ai.request.model=Injected,"
                "traffic.llm_sdk.rtp_llm.pd_sep=true,"
                "traffic.llm_sdk.scene=chat"
            )
        }
        state = tracing.start_server_span(
            "srv", headers, initial_attributes={attrs.GEN_AI_REQUEST_MODEL: "real"}
        )
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert a["scene"] == "chat"
        assert a[attrs.GEN_AI_REQUEST_MODEL] == "real"
        assert "tenant_id" not in a
        assert attrs.ERROR_TYPE not in a
        assert attrs.RTP_LLM_PD_SEP not in a

    def test_llm_sdk_baggage_allowed_value_is_bounded(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span(
            "srv", {"baggage": "traffic.llm_sdk.scene=" + "x" * 300}
        )
        state.finish()
        tracing.shutdown_telemetry()

        a = exporter.get_finished_spans()[0].attributes
        assert a["scene"] == "x" * 256

    def test_baggage_not_forwarded_downstream(self):
        # gRPC metadata carries traceparent only, never baggage.
        _start_in_memory_runtime()
        headers = {"baggage": "traffic.llm_sdk.scene=chat"}
        state = tracing.start_server_span("srv", headers)
        handle, metadata = tracing.start_client_span("client")
        keys = {key for key, _ in metadata}
        assert "baggage" not in keys
        assert "traceparent" in keys
        handle.finish()
        state.finish()
        tracing.shutdown_telemetry()

    def test_malformed_baggage_is_ignored(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {"baggage": ";;;===,,"})
        state.finish()
        tracing.shutdown_telemetry()
        # no crash, span still produced
        assert len(exporter.get_finished_spans()) == 1


class TestSpanEvents(TracingTestCase):
    """first_response_chunk span event contract."""

    def test_add_event_recorded_within_span_window(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        state.add_event(attrs.EVENT_FIRST_RESPONSE_CHUNK)
        state.finish()
        tracing.shutdown_telemetry()

        span = exporter.get_finished_spans()[0]
        events = list(span.events)
        assert [e.name for e in events] == [attrs.EVENT_FIRST_RESPONSE_CHUNK]
        # event stamped 'now' at call time must sit inside the span window
        assert span.start_time <= events[0].timestamp <= span.end_time

    def test_add_event_after_finish_dropped(self):
        exporter = _start_in_memory_runtime()
        state = tracing.start_server_span("srv", {})
        state.finish()
        state.add_event(attrs.EVENT_FIRST_RESPONSE_CHUNK)
        tracing.shutdown_telemetry()

        assert list(exporter.get_finished_spans()[0].events) == []

    def test_add_event_without_span_is_noop(self):
        # inactive-path shape: RequestTraceState with no span never raises
        state = tracing.RequestTraceState()
        state.add_event(attrs.EVENT_FIRST_RESPONSE_CHUNK)
        state.finish()


class TestFrontendTokenLatency(TracingTestCase):
    def test_visible_token_ttft_and_tpot_use_server_timeline(self):
        exporter = _start_in_memory_runtime()
        span = tracing.get_tracer().start_span("srv")
        state = tracing.RequestTraceState(
            server_span=span, request_start_ns=1_000_000_000
        )

        state.record_frontend_output_tokens(0, 1_005_000_000)
        assert attrs.GEN_AI_TIME_TO_FIRST_TOKEN not in span.attributes

        state.record_frontend_output_tokens(1, 1_012_500_000)
        assert span.attributes[attrs.GEN_AI_TIME_TO_FIRST_TOKEN] == 12.5
        assert attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS not in span.attributes

        state.record_frontend_output_tokens(2, 1_032_500_000)
        assert span.attributes[attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS] == 10.0
        state.finish()
        tracing.shutdown_telemetry()

        finished = exporter.get_finished_spans()[0]
        assert finished.attributes[attrs.GEN_AI_TIME_TO_FIRST_TOKEN] == 12.5
        assert (
            finished.attributes[attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS] == 10.0
        )

    def test_tpot_requires_two_distinct_delivery_instants(self):
        exporter = _start_in_memory_runtime()
        span = tracing.get_tracer().start_span("srv")
        state = tracing.RequestTraceState(
            server_span=span, request_start_ns=1_000_000_000
        )

        # One frame carrying 3 tokens exposes no inter-token send boundary, so
        # TPOT must stay absent rather than be reported as 0.0.
        state.record_frontend_output_tokens(3, 1_012_500_000)
        assert span.attributes[attrs.GEN_AI_TIME_TO_FIRST_TOKEN] == 12.5
        assert attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS not in span.attributes

        # A second frame delivered at the same instant adds no boundary either.
        state.record_frontend_output_tokens(2, 1_012_500_000)
        assert attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS not in span.attributes

        # A later frame supplies the second instant: 6 tokens across 10ms.
        state.record_frontend_output_tokens(1, 1_022_500_000)
        assert span.attributes[attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS] == 2.0
        state.finish()
        tracing.shutdown_telemetry()

        finished = exporter.get_finished_spans()[0]
        assert finished.attributes[attrs.GEN_AI_TIME_TO_FIRST_TOKEN] == 12.5
        assert (
            finished.attributes[attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS] == 2.0
        )

    def test_invalid_or_late_observations_are_ignored(self):
        exporter = _start_in_memory_runtime()
        span = tracing.get_tracer().start_span("srv")
        state = tracing.RequestTraceState(
            server_span=span, request_start_ns=1_000_000_000
        )
        state.record_frontend_output_tokens(True, 1_010_000_000)
        state.record_frontend_output_tokens(1, 999_000_000)
        state.finish()
        state.record_frontend_output_tokens(1, 1_020_000_000)
        tracing.shutdown_telemetry()

        assert (
            attrs.GEN_AI_TIME_TO_FIRST_TOKEN
            not in exporter.get_finished_spans()[0].attributes
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

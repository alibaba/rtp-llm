package org.flexlb;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.context.Context;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(SystemStubsExtension.class)
class OpenTelemetryBootstrapTest {
    @TempDir Path directory;
    @SystemStub private EnvironmentVariables environment = new EnvironmentVariables();
    private Path originalHostnameFile;
    private final ObjectMapper mapper = new ObjectMapper();

    @BeforeEach
    void setUp() throws Exception {
        OpenTelemetryBootstrap.resetForTest();
        environment.remove(TraceConfig.ENV);
        originalHostnameFile = OpenTelemetryBootstrap.hostnameFile;
        OpenTelemetryBootstrap.hostnameFile = directory.resolve("hostname");
        Files.writeString(OpenTelemetryBootstrap.hostnameFile, "probe-host\n");
    }

    @AfterEach
    void tearDown() {
        OpenTelemetryBootstrap.resetForTest();
        OpenTelemetryBootstrap.hostnameFile = originalHostnameFile;
    }

    @Test
    void sharedJsonContract() throws Exception {
        Path base = Path.of("").toAbsolutePath();
        Path fixture = null;
        for (int i = 0; i < 8 && base != null; i++, base = base.getParent()) {
            Path candidate = base.resolve("telemetry/test/trace_config_cases.json");
            if (Files.exists(candidate)) {
                fixture = candidate;
                break;
            }
        }
        assertNotNull(fixture);
        for (var item : mapper.readTree(Files.readString(fixture))) {
            String raw = item.get("raw").isNull() ? null : item.get("raw").asText();
            if (item.has("error")) {
                var error = assertThrows(TraceConfig.ConfigException.class, () -> TraceConfig.parse(raw),
                        item.toString());
                assertEquals(item.get("error").asText(), error.code, item.get("name").asText());
            } else {
                var config = TraceConfig.parse(raw);
                assertEquals(item.get("enabled").asBoolean(), config.enabled());
                assertEquals(item.path("ratio").asDouble(1.0), config.samplerRatio());
                if (item.has("queue")) {
                    assertEquals(item.get("queue").asInt(), config.maxQueueSize());
                }
            }
        }
    }

    @Test
    void disabledAndInvalidConfigurationsCannotCreateSpans() {
        environment.set("RTP_LLM_OTEL_TRACE_ENABLE", "1");
        environment.set("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost/ignored");
        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertNull(FlexlbTrace.startServer("disabled", Context.root()));
        // 初始化后修改环境不能重新开启。
        environment.set(TraceConfig.ENV, "{\"enabled\":true}");
        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        OpenTelemetryBootstrap.resetForTest();
        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertFalse(FlexlbTrace.isEnabled());
    }

    @Test
    void resourceIdentityDoesNotReadLegacyOverrides() {
        environment.set("OTEL_RESOURCE_ATTRIBUTES", "host.ip=wrong,poison=wrong");
        environment.set("HOSTNAME", "wrong");
        environment.set("POD_IP", "10.1.2.3");
        var attrs = OpenTelemetryBootstrap.resource().getAttributes();
        assertEquals("probe-host-" + ProcessHandle.current().pid(), attrs.get(AttributeKey.stringKey("host.ip")));
        assertEquals("10.1.2.3", attrs.get(AttributeKey.stringKey("rtp_llm.pod_ip")));
        assertEquals("loongsuite-genai-utils", attrs.get(AttributeKey.stringKey("gen_ai.instrumentation.sdk.name")));
        assertNull(attrs.get(AttributeKey.stringKey("poison")));
        OpenTelemetryBootstrap.hostnameFile = directory.resolve("missing");
        attrs = OpenTelemetryBootstrap.resource().getAttributes();
        assertNull(attrs.get(AttributeKey.stringKey("host.ip")));
        assertNull(attrs.get(AttributeKey.stringKey("host.name")));
    }

    @Test
    void wireExportIgnoresOldSdkAndJvmSettings() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        CountDownLatch received = new CountDownLatch(1);
        AtomicReference<String> header = new AtomicReference<>();
        AtomicReference<String> contentType = new AtomicReference<>();
        AtomicReference<byte[]> body = new AtomicReference<>();
        server.createContext("/custom", exchange -> {
            header.set(exchange.getRequestHeaders().getFirst("authorization"));
            contentType.set(exchange.getRequestHeaders().getFirst("content-type"));
            body.set(exchange.getRequestBody().readAllBytes());
            exchange.sendResponseHeaders(200, 0);
            exchange.close();
            received.countDown();
        });
        server.start();
        String previous = System.getProperty("otel.sdk.disabled");
        try {
            environment.set("OTEL_SDK_DISABLED", "true");
            environment.set("OTEL_TRACES_EXPORTER", "none");
            environment.set("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://invalid.example/ignored");
            System.setProperty("otel.sdk.disabled", "true");
            String raw = mapper.writeValueAsString(Map.of("enabled", true,
                    "endpoint", "http://127.0.0.1:" + server.getAddress().getPort() + "/custom",
                    "headers", Map.of("authorization", "fake-test-only"), "schedule_delay_ms", 1,
                    "sampler_ratio", 0));
            environment.set(TraceConfig.ENV, raw);
            assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
            var root = FlexlbTrace.startServer("root_not_sampled", Context.root());
            assertNotNull(root);
            assertFalse(root.isRecording());
            FlexlbTrace.finish(root);
            var parent = io.opentelemetry.api.trace.SpanContext.createFromRemoteParent(
                    "11111111111111111111111111111111", "2222222222222222",
                    io.opentelemetry.api.trace.TraceFlags.getSampled(),
                    io.opentelemetry.api.trace.TraceState.getDefault());
            var span = FlexlbTrace.startServer("wire",
                    Context.root().with(io.opentelemetry.api.trace.Span.wrap(parent)));
            assertNotNull(span);
            assertTrue(span.isRecording());
            FlexlbTrace.finish(span);
            assertTrue(received.await(5, TimeUnit.SECONDS));
            assertEquals("fake-test-only", header.get());
            assertEquals("application/x-protobuf", contentType.get());
            assertTrue(body.get().length > 0);
            assertFalse(TraceConfig.parse(raw).toString().contains("fake-test-only"));
        } finally {
            OpenTelemetryBootstrap.shutdown();
            server.stop(0);
            if (previous == null) {
                System.clearProperty("otel.sdk.disabled");
            } else {
                System.setProperty("otel.sdk.disabled", previous);
            }
        }
    }
}

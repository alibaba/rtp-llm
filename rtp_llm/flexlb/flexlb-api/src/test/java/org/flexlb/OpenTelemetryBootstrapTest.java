package org.flexlb;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(SystemStubsExtension.class)
class OpenTelemetryBootstrapTest {

    private static final String TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT";
    private static final String GENERIC_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT";

    @SystemStub
    private EnvironmentVariables environmentVariables = new EnvironmentVariables();

    @BeforeEach
    void setUp() {
        GlobalOpenTelemetry.resetForTest();
        clearBootstrapProperties();
        environmentVariables.remove(OpenTelemetryBootstrap.TRACE_ENABLE_ENV);
        environmentVariables.remove(TRACES_ENDPOINT_ENV);
        environmentVariables.remove(GENERIC_ENDPOINT_ENV);
        environmentVariables.remove("OTEL_SDK_DISABLED");
        environmentVariables.remove("OTEL_SERVICE_NAME");
        environmentVariables.remove("OTEL_METRICS_EXPORTER");
        environmentVariables.remove("OTEL_LOGS_EXPORTER");
        environmentVariables.remove("OTEL_TRACES_EXPORTER");
        environmentVariables.remove("OTEL_RESOURCE_ATTRIBUTES");
        environmentVariables.remove("POD_IP");
    }

    @AfterEach
    void tearDown() {
        GlobalOpenTelemetry.resetForTest();
        clearBootstrapProperties();
    }

    @Test
    void disabledTraceLeavesProviderAutoconfigurationOff() {
        environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, "0");
        environmentVariables.set(TRACES_ENDPOINT_ENV, "http://127.0.0.1:4318/v1/traces");

        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertNull(System.getProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY));
    }

    @Test
    void enabledTraceWithoutEndpointStaysFailOpen() {
        environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, "1");

        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertNull(System.getProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY));
    }

    @Test
    void enabledTraceConfiguresARealProviderWithSafeDefaults() {
        environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, "true");
        environmentVariables.set(TRACES_ENDPOINT_ENV, "http://127.0.0.1:4318/v1/traces");
        // Keep this unit test offline; it verifies provider creation, not export.
        environmentVariables.set("OTEL_TRACES_EXPORTER", "none");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals("true", System.getProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY));
        assertEquals("rtp_llm_flexlb", System.getProperty(OpenTelemetryBootstrap.SERVICE_NAME_PROPERTY));
        assertEquals("none", System.getProperty(OpenTelemetryBootstrap.METRICS_EXPORTER_PROPERTY));
        assertEquals("none", System.getProperty(OpenTelemetryBootstrap.LOGS_EXPORTER_PROPERTY));

        Span span = GlobalOpenTelemetry.getTracer("org.flexlb.test")
                .spanBuilder("provider_probe")
                .startSpan();
        try {
            assertTrue(span.getSpanContext().isValid());
        } finally {
            span.end();
        }
    }

    @Test
    void missingPodIpLeavesResourceAttributesUntouched() {
        enableTraceOffline();

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertNull(System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void podIpBecomesHostIpResourceAttribute() {
        enableTraceOffline();
        environmentVariables.set("POD_IP", "10.1.2.3");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals("host.ip=10.1.2.3",
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void podIpMergesIntoDeploymentResourceAttributes() {
        enableTraceOffline();
        environmentVariables.set("POD_IP", "10.1.2.3");
        environmentVariables.set("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=prod");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals("deployment.environment=prod,host.ip=10.1.2.3",
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void explicitHostIpWinsOverPodIp() {
        enableTraceOffline();
        environmentVariables.set("POD_IP", "10.1.2.3");
        environmentVariables.set("OTEL_RESOURCE_ATTRIBUTES", "host.ip=192.168.0.9");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertNull(System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    private void enableTraceOffline() {
        environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, "1");
        environmentVariables.set(TRACES_ENDPOINT_ENV, "http://127.0.0.1:4318/v1/traces");
        // Keep these unit tests offline; they verify configuration, not export.
        environmentVariables.set("OTEL_TRACES_EXPORTER", "none");
    }

    private static void clearBootstrapProperties() {
        System.clearProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY);
        System.clearProperty(OpenTelemetryBootstrap.SERVICE_NAME_PROPERTY);
        System.clearProperty(OpenTelemetryBootstrap.METRICS_EXPORTER_PROPERTY);
        System.clearProperty(OpenTelemetryBootstrap.LOGS_EXPORTER_PROPERTY);
        System.clearProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY);
        System.clearProperty("otel.sdk.disabled");
        System.clearProperty("otel.exporter.otlp.traces.endpoint");
        System.clearProperty("otel.exporter.otlp.endpoint");
    }
}

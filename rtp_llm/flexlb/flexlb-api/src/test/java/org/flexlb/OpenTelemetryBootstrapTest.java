package org.flexlb;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(SystemStubsExtension.class)
class OpenTelemetryBootstrapTest {

    private static final String TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT";
    private static final String GENERIC_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT";
    private static final String TEST_HOSTNAME = "probe-host";

    @TempDir
    private Path tempDir;

    private Path originalHostnameFile;

    @SystemStub
    private EnvironmentVariables environmentVariables = new EnvironmentVariables();

    @BeforeEach
    void setUp() {
        GlobalOpenTelemetry.resetForTest();
        clearBootstrapProperties();
        environmentVariables.remove(OpenTelemetryBootstrap.TRACE_ENABLE_ENV);
        environmentVariables.remove(OpenTelemetryBootstrap.AUTOCONFIGURE_ENV);
        FlexlbTrace.configureEnabled(false);
        environmentVariables.remove(TRACES_ENDPOINT_ENV);
        environmentVariables.remove(GENERIC_ENDPOINT_ENV);
        environmentVariables.remove("OTEL_SDK_DISABLED");
        environmentVariables.remove("OTEL_SERVICE_NAME");
        environmentVariables.remove("OTEL_METRICS_EXPORTER");
        environmentVariables.remove("OTEL_LOGS_EXPORTER");
        environmentVariables.remove("OTEL_TRACES_EXPORTER");
        environmentVariables.remove("OTEL_RESOURCE_ATTRIBUTES");
        environmentVariables.remove("POD_IP");
        // host.ip/host.name derive from the kernel hostname exposed by procfs, so the
        // tests point that at a temp file instead of depending on the build
        // machine. $HOSTNAME is cleared so a leaked outer value cannot make an
        // assertion pass for the wrong reason.
        environmentVariables.remove("HOSTNAME");
        originalHostnameFile = OpenTelemetryBootstrap.hostnameFile;
        OpenTelemetryBootstrap.hostnameFile = writeHostnameFile(TEST_HOSTNAME);
    }

    private Path writeHostnameFile(String content) {
        try {
            Path file = tempDir.resolve("hostname");
            // Trailing newline mirrors the procfs hostname representation.
            Files.writeString(file, content + "\n");
            return file;
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @AfterEach
    void tearDown() {
        FlexlbTrace.configureEnabled(false);
        OpenTelemetryBootstrap.hostnameFile = originalHostnameFile;
        GlobalOpenTelemetry.resetForTest();
        clearBootstrapProperties();
    }

    @Test
    void sdkDisabledSystemPropertyTakesPrecedenceInBothDirections() {
        enableTraceOffline();
        environmentVariables.set("OTEL_SDK_DISABLED", "true");
        System.setProperty("otel.sdk.disabled", "false");
        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertTrue(FlexlbTrace.isEnabled());

        environmentVariables.set("OTEL_SDK_DISABLED", "false");
        System.setProperty("otel.sdk.disabled", "true");
        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertFalse(FlexlbTrace.isEnabled());
    }

    @Test
    void sdkDisabledAbsentOrBlankPropertyFallsBackToEnvironment() {
        enableTraceOffline();
        environmentVariables.set("OTEL_SDK_DISABLED", "true");
        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        System.setProperty("otel.sdk.disabled", "  ");
        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertFalse(FlexlbTrace.isEnabled());
        environmentVariables.set("OTEL_SDK_DISABLED", "false");
        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertTrue(FlexlbTrace.isEnabled());
    }

    @Test
    void manualSwitchIsIndependentOfExternalProviderOwnership() {
        try (var sdk = io.opentelemetry.sdk.OpenTelemetrySdk.builder().setTracerProvider(
                io.opentelemetry.sdk.trace.SdkTracerProvider.builder().build()).build()) {
            GlobalOpenTelemetry.set(sdk);
            environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, "0");
            assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
            assertFalse(FlexlbTrace.isEnabled());
            assertNull(FlexlbTrace.startServer("disabled", io.opentelemetry.context.Context.root()));

            environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, "1");
            System.setProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY, "false");
            assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
            assertTrue(FlexlbTrace.isEnabled());
            Span span = FlexlbTrace.startServer("external-provider", io.opentelemetry.context.Context.root());
            try {
                assertTrue(span.isRecording());
            } finally {
                FlexlbTrace.finish(span);
            }
        }
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
    void legacyBooleanSpellingsKeepBootstrapConfigurationSemantics() {
        environmentVariables.set(TRACES_ENDPOINT_ENV, "http://127.0.0.1:4318/v1/traces");
        for (String value : new String[] {"true", "1", " YES ", "On"}) {
            clearBootstrapProperties();
            environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, value);
            environmentVariables.remove("OTEL_SDK_DISABLED");
            assertTrue(OpenTelemetryBootstrap.configureFromEnvironment(), value);
            environmentVariables.set("OTEL_SDK_DISABLED", value);
            assertFalse(OpenTelemetryBootstrap.configureFromEnvironment(), value);
        }
        environmentVariables.remove("OTEL_SDK_DISABLED");
        for (String value : new String[] {"false", "0", " NO ", "Off", "invalid", ""}) {
            clearBootstrapProperties();
            environmentVariables.set(OpenTelemetryBootstrap.TRACE_ENABLE_ENV, value);
            assertFalse(OpenTelemetryBootstrap.configureFromEnvironment(), value);
            assertNull(System.getProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY));
        }
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
    void resourceAttributesCarryPlatformIdentity() {
        enableTraceOffline();
        environmentVariables.set("POD_IP", "10.1.2.3");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals(sdkMarker() + "," + hostEntries() + ",rtp_llm.pod_ip=10.1.2.3",
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void defaultHostnameSourceIsKernelUtsFile() {
        assertEquals(Path.of("/proc/sys/kernel/hostname"), originalHostnameFile);
    }

    @Test
    void hostnameReadFailureSkipsHostKeysWithoutFallback() {
        enableTraceOffline();
        OpenTelemetryBootstrap.hostnameFile = tempDir;
        environmentVariables.set("HOSTNAME", "env-host");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals(sdkMarker(),
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    /**
     * $HOSTNAME is not a source at all, not even a lower-priority one. An outer
     * shell can leak a different machine's name into the env (observed in a
     * docker-exec session), which would make FlexLB report a different host prefix
     * than the engine processes in the same pod.
     */
    @Test
    void leakedEnvironmentHostnameIsNotUsed() {
        enableTraceOffline();
        environmentVariables.set("HOSTNAME", "leaked-outer-host");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals(sdkMarker() + "," + hostEntries(),
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    /**
     * The env var is not a fallback source. It is a snapshot of whoever launched
     * the JVM, so an image without a hostname file must report no host identity
     * rather than a value that may name a different machine.
     */
    @Test
    void missingHostnameFileSkipsHostKeysEvenWhenEnvironmentIsSet() {
        enableTraceOffline();
        OpenTelemetryBootstrap.hostnameFile = tempDir.resolve("absent-hostname");
        environmentVariables.set("HOSTNAME", "env-host");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals(sdkMarker(),
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    /** A present but blank file is as unusable as a missing one. */
    @Test
    void blankHostnameFileSkipsHostKeys() {
        enableTraceOffline();
        OpenTelemetryBootstrap.hostnameFile = writeHostnameFile("   ");
        environmentVariables.set("HOSTNAME", "env-host");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals(sdkMarker(),
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void missingPodIpStillWritesSdkMarkerAndHostIdentity() {
        enableTraceOffline();

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        // The pod address is the only optional part: host.ip no longer depends on
        // it, and the SDK marker is unconditional.
        assertEquals(sdkMarker() + "," + hostEntries(),
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void unknownHostnameSkipsHostKeysButKeepsSdkMarker() {
        enableTraceOffline();
        OpenTelemetryBootstrap.hostnameFile = tempDir.resolve("absent-hostname");
        environmentVariables.set("POD_IP", "10.1.2.3");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        // "unknown-<pid>" would pollute the per-instance aggregation these keys
        // exist to serve, so neither host key is synthesized.
        assertEquals(sdkMarker() + ",rtp_llm.pod_ip=10.1.2.3",
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void resourceAttributesMergeIntoDeploymentConfiguration() {
        enableTraceOffline();
        environmentVariables.set("POD_IP", "10.1.2.3");
        environmentVariables.set("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=prod");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        // The system property REPLACES the environment variable for the SDK, so
        // the deployment's own entry must be copied into it, not just left behind.
        assertEquals("deployment.environment=prod," + sdkMarker() + "," + hostEntries()
                        + ",rtp_llm.pod_ip=10.1.2.3",
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void explicitHostIpIsPreservedWhileOtherKeysAreAdded() {
        enableTraceOffline();
        environmentVariables.set("POD_IP", "10.1.2.3");
        environmentVariables.set("OTEL_RESOURCE_ATTRIBUTES", "host.ip=192.168.0.9");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        // A deployment that pins host.ip keeps it; only the keys it left undefined
        // are appended.
        assertEquals("host.ip=192.168.0.9," + sdkMarker() + ",host.name=" + TEST_HOSTNAME
                        + ",rtp_llm.pod_ip=10.1.2.3",
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    @Test
    void similarlyNamedKeyDoesNotSuppressHostIp() {
        enableTraceOffline();
        // A substring match on "host.ip=" would see this entry and wrongly skip
        // host.ip; keys must be compared exactly.
        environmentVariables.set("OTEL_RESOURCE_ATTRIBUTES", "rtp_llm.pod_ip=10.9.9.9");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals("rtp_llm.pod_ip=10.9.9.9," + sdkMarker() + "," + hostEntries(),
                System.getProperty(OpenTelemetryBootstrap.RESOURCE_ATTRIBUTES_PROPERTY));
    }

    private static String sdkMarker() {
        return OpenTelemetryBootstrap.INSTRUMENTATION_SDK_NAME_ATTRIBUTE + "="
                + OpenTelemetryBootstrap.INSTRUMENTATION_SDK_NAME_VALUE;
    }

    private static String hostEntries() {
        return OpenTelemetryBootstrap.HOST_NAME_ATTRIBUTE + "=" + TEST_HOSTNAME + ","
                + OpenTelemetryBootstrap.HOST_IP_ATTRIBUTE + "=" + TEST_HOSTNAME + "-"
                + ProcessHandle.current().pid();
    }

    /**
     * An operator who pinned the autoconfigure switch to false has made an
     * explicit decision. Enabling it anyway would assemble the provider against
     * that intent, so the trace switch loses this conflict.
     */
    @Test
    void explicitlyDisabledAutoconfigurePropertyIsNotOverridden() {
        enableTraceOffline();
        System.setProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY, "false");

        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertEquals("false", System.getProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY));
    }

    /** Same contract through the environment variable. */
    @Test
    void explicitlyDisabledAutoconfigureEnvIsNotOverridden() {
        enableTraceOffline();
        environmentVariables.set(OpenTelemetryBootstrap.AUTOCONFIGURE_ENV, "false");

        assertFalse(OpenTelemetryBootstrap.configureFromEnvironment());
        assertNull(System.getProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY));
    }

    /**
     * A deployment that pinned the switch to true keeps its own value rather than
     * having it rewritten, which is why the call is setDefault and not
     * setProperty. The switch still ends up enabled either way.
     */
    @Test
    void explicitlyEnabledAutoconfigureValueIsPreserved() {
        enableTraceOffline();
        environmentVariables.set(OpenTelemetryBootstrap.AUTOCONFIGURE_ENV, "true");

        assertTrue(OpenTelemetryBootstrap.configureFromEnvironment());
        // Left to the environment variable: setDefault must not shadow it.
        assertNull(System.getProperty(OpenTelemetryBootstrap.AUTOCONFIGURE_PROPERTY));
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

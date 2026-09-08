package org.flexlb;

import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Logger;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

/** Configures the OpenTelemetry SDK before Spring starts serving requests. */
final class OpenTelemetryBootstrap {

    static final String TRACE_ENABLE_ENV = "RTP_LLM_OTEL_TRACE_ENABLE";
    static final String AUTOCONFIGURE_PROPERTY = "otel.java.global-autoconfigure.enabled";
    static final String AUTOCONFIGURE_ENV = "OTEL_JAVA_GLOBAL_AUTOCONFIGURE_ENABLED";
    static final String SERVICE_NAME_PROPERTY = "otel.service.name";
    static final String METRICS_EXPORTER_PROPERTY = "otel.metrics.exporter";
    static final String LOGS_EXPORTER_PROPERTY = "otel.logs.exporter";
    static final String RESOURCE_ATTRIBUTES_PROPERTY = "otel.resource.attributes";
    static final String HOST_IP_ATTRIBUTE = "host.ip";
    static final String HOST_NAME_ATTRIBUTE = "host.name";
    static final String POD_IP_ATTRIBUTE = "rtp_llm.pod_ip";
    static final String INSTRUMENTATION_SDK_NAME_ATTRIBUTE = "gen_ai.instrumentation.sdk.name";
    static final String INSTRUMENTATION_SDK_NAME_VALUE = "loongsuite-genai-utils";
    /**
     * The current UTS namespace's kernel hostname, the only source this class
     * reads. Package-private and mutable purely as a test seam: the tests point it
     * at a temp file so the absent and blank cases can be asserted deterministically
     * instead of depending on the build machine.
     */
    static Path hostnameFile = Path.of("/proc/sys/kernel/hostname");

    private OpenTelemetryBootstrap() {
    }

    /**
     * Enables SDK autoconfiguration when the shared RTP-LLM trace switch is on.
     *
     * <p>This must run before the first GlobalOpenTelemetry access. The SDK's
     * autoconfigure extension then initializes the provider lazily and registers
     * its own JVM shutdown hook. Missing exporter configuration remains fail-open.
     */
    static boolean configureFromEnvironment() {
        FlexlbTrace.configureEnabled(readBooleanEnvironment(TRACE_ENABLE_ENV) && !sdkExplicitlyDisabled());
        if (!readBooleanEnvironment(TRACE_ENABLE_ENV)) {
            return false;
        }
        if (sdkExplicitlyDisabled()) {
            Logger.warn("{} is enabled but OTEL_SDK_DISABLED is true; FlexLB tracing remains disabled",
                    TRACE_ENABLE_ENV);
            return false;
        }
        if (autoconfigureExplicitlyDisabled()) {
            Logger.warn("{} is enabled but {} is explicitly false; SDK initialization remains external",
                    TRACE_ENABLE_ENV, AUTOCONFIGURE_PROPERTY);
            return false;
        }
        if (!hasExporterEndpoint()) {
            Logger.warn("{} is enabled but no OTLP endpoint is configured; SDK initialization remains external",
                    TRACE_ENABLE_ENV);
            return false;
        }

        // setDefault, not setProperty: an operator who pinned the autoconfigure
        // switch keeps their value. The explicitly-false case already returned
        // above, so reaching here with it set means it was set to true.
        setDefault(AUTOCONFIGURE_PROPERTY, AUTOCONFIGURE_ENV, "true");
        setDefault(SERVICE_NAME_PROPERTY, "OTEL_SERVICE_NAME", "rtp_llm_flexlb");
        // FlexLB only owns manual trace spans. Do not start unrelated metrics or
        // log exporters unless the deployment explicitly requests them.
        setDefault(METRICS_EXPORTER_PROPERTY, "OTEL_METRICS_EXPORTER", "none");
        setDefault(LOGS_EXPORTER_PROPERTY, "OTEL_LOGS_EXPORTER", "none");
        addResourceAttributes();
        Logger.info("FlexLB OpenTelemetry provider autoconfiguration enabled");
        return true;
    }

    private static boolean hasExporterEndpoint() {
        return hasText(System.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
                || hasText(System.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
                || hasText(System.getProperty("otel.exporter.otlp.traces.endpoint"))
                || hasText(System.getProperty("otel.exporter.otlp.endpoint"));
    }

    private static boolean sdkExplicitlyDisabled() {
        String property = System.getProperty("otel.sdk.disabled");
        return hasText(property) ? Boolean.parseBoolean(property.trim())
                : readBooleanEnvironment("OTEL_SDK_DISABLED");
    }

    // Bootstrap precedes Spring/config initialization. Preserve the existing
    // Trace switch spellings without depending on the removed EnvUtils class.
    private static boolean readBooleanEnvironment(String name) {
        String value = System.getenv(name);
        if (value == null) {
            return false;
        }
        return switch (value.trim().toLowerCase(java.util.Locale.ROOT)) {
            case "true", "1", "yes", "on" -> true;
            case "false", "0", "no", "off" -> false;
            default -> {
                Logger.warn("Invalid {}='{}'; using default value false", name, value);
                yield false;
            }
        };
    }

    /**
     * Whether the deployment pinned the autoconfigure switch to false. Enabling it
     * anyway would override an explicit operator decision, and the provider would
     * then be assembled against the operator's intent.
     */
    private static boolean autoconfigureExplicitlyDisabled() {
        String fromProperty = System.getProperty(AUTOCONFIGURE_PROPERTY);
        if (hasText(fromProperty)) {
            return !Boolean.parseBoolean(fromProperty.trim());
        }
        String fromEnv = System.getenv(AUTOCONFIGURE_ENV);
        return hasText(fromEnv) && !Boolean.parseBoolean(fromEnv.trim());
    }

    /**
     * Mirrors the Python and C++ runtimes' Resource identity.
     *
     * <p>Observability dashboards filter their request, error and latency panels
     * by {@code host.ip}, so spans without it are missing from per-instance
     * statistics even when the trace itself is complete. That attribute therefore
     * carries {@code "{hostname}-{pid}"} rather than an IP: a pod IP is not
     * process-unique, so every process sharing a pod would collapse into one
     * bucket. The real pod address stays reported under {@code rtp_llm.pod_ip}.
     * Nothing is synthesized when the hostname is unknown.
     *
     * <p>Resource attributes are one aggregate setting, and for the SDK a system
     * property REPLACES the environment variable instead of merging with it. The
     * existing configuration is therefore copied into the property first and only
     * keys it does not already define are appended, so a deployment that pins its
     * own value keeps it.
     */
    private static void addResourceAttributes() {
        String configured = System.getProperty(RESOURCE_ATTRIBUTES_PROPERTY);
        if (!hasText(configured)) {
            configured = System.getenv("OTEL_RESOURCE_ATTRIBUTES");
        }
        Set<String> definedKeys = parseDefinedKeys(configured);
        StringBuilder merged = new StringBuilder(hasText(configured) ? configured.trim() : "");

        appendAttribute(merged, definedKeys, INSTRUMENTATION_SDK_NAME_ATTRIBUTE, INSTRUMENTATION_SDK_NAME_VALUE);
        String hostname = hostname();
        if (hasText(hostname)) {
            appendAttribute(merged, definedKeys, HOST_NAME_ATTRIBUTE, hostname);
            appendAttribute(merged, definedKeys, HOST_IP_ATTRIBUTE,
                    hostname + "-" + ProcessHandle.current().pid());
        }
        String podIp = System.getenv("POD_IP");
        if (hasText(podIp)) {
            appendAttribute(merged, definedKeys, POD_IP_ATTRIBUTE, podIp.trim());
        }
        if (merged.length() > 0) {
            System.setProperty(RESOURCE_ATTRIBUTES_PROPERTY, merged.toString());
        }
    }

    private static void appendAttribute(StringBuilder merged, Set<String> definedKeys, String key, String value) {
        if (!definedKeys.add(key)) {
            return;
        }
        if (merged.length() > 0) {
            merged.append(',');
        }
        merged.append(key).append('=').append(value);
    }

    /**
     * Keys the deployment already defines. The SDK parses this setting as
     * comma-separated {@code k=v} pairs, so the key is matched exactly rather
     * than by substring: {@code rtp_llm.pod_ip=...} must not be mistaken for a
     * configured {@code host.ip}.
     */
    private static Set<String> parseDefinedKeys(String configured) {
        Set<String> keys = new HashSet<>();
        if (!hasText(configured)) {
            return keys;
        }
        for (String entry : configured.split(",")) {
            int separator = entry.indexOf('=');
            if (separator > 0) {
                keys.add(entry.substring(0, separator).trim());
            }
        }
        return keys;
    }

    /**
     * Reads the live Linux UTS hostname through procfs, matching the kernel
     * source used by C++ gethostname(2) and Python socket.gethostname().
     * No fallback to /etc/hostname or $HOSTNAME: either may be stale or describe
     * a different namespace. Failure to read procfs leaves the host keys absent.
     *
     * @return the hostname, or an empty string when it cannot be read or is blank.
     *         Callers then omit the host keys instead of synthesizing a value.
     */
    private static String hostname() {
        try {
            String fromFile = Files.readString(hostnameFile, StandardCharsets.UTF_8).trim();
            if (hasText(fromFile)) {
                return fromFile;
            }
        } catch (IOException | RuntimeException ignored) {
            // Reading host identity must never break startup.
        }
        return "";
    }

    private static void setDefault(String propertyName, String environmentName, String value) {
        if (!hasText(System.getProperty(propertyName)) && !hasText(System.getenv(environmentName))) {
            System.setProperty(propertyName, value);
        }
    }

    private static boolean hasText(String value) {
        return value != null && !value.trim().isEmpty();
    }
}

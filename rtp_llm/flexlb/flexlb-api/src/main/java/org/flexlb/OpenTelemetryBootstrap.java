package org.flexlb;

import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Logger;

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
        addPodIpResourceAttribute();
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
     * Mirrors the Python and C++ runtimes by deriving Resource {@code host.ip}
     * from POD_IP. Observability dashboards filter the request, error and latency
     * panels by that attribute, so spans without it are missing from per-instance
     * statistics even when the trace itself is complete. Only a real pod IP is
     * written, matching TelemetryRuntime.
     *
     * <p>Resource attributes are an aggregate setting rather than a single value,
     * so an existing configuration is merged instead of replaced, and a deployment
     * that already pins {@code host.ip} keeps its own value.
     */
    private static void addPodIpResourceAttribute() {
        String podIp = System.getenv("POD_IP");
        if (!hasText(podIp)) {
            return;
        }
        String configured = System.getProperty(RESOURCE_ATTRIBUTES_PROPERTY);
        if (!hasText(configured)) {
            configured = System.getenv("OTEL_RESOURCE_ATTRIBUTES");
        }
        if (hasText(configured) && configured.contains(HOST_IP_ATTRIBUTE + "=")) {
            return;
        }
        String hostIpEntry = HOST_IP_ATTRIBUTE + "=" + podIp.trim();
        System.setProperty(RESOURCE_ATTRIBUTES_PROPERTY,
                hasText(configured) ? configured + "," + hostIpEntry : hostIpEntry);
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

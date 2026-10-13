package org.flexlb;

import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.samplers.Sampler;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Logger;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/** Configures the OpenTelemetry SDK before Spring starts serving requests. */
final class OpenTelemetryBootstrap {

    private static boolean initialized;
    private static SdkTracerProvider provider;
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

    /** Spring 启动前读取一次 JSON；只有完整构造成功才发布 enabled。 */
    static synchronized boolean configureFromEnvironment() {
        if (initialized) {
            return FlexlbTrace.isEnabled();
        }
        initialized = true;
        FlexlbTrace.configure(null, "");
        try {
            return configure(TraceConfig.parse(System.getenv(TraceConfig.ENV)));
        } catch (TraceConfig.ConfigException error) {
            Logger.warn("Trace 已关闭 role=flexlb field={} reason={}", error.field, error.code);
        } catch (RuntimeException | LinkageError ignored) {
            Logger.warn("Trace 已关闭 role=flexlb field=sdk reason=initialization_failed");
        }
        return false;
    }

    static boolean configure(TraceConfig config) {
        if (!config.enabled()) {
            return false;
        }
        OtlpHttpSpanExporter exporter = null;
        BatchSpanProcessor processor = null;
        SdkTracerProvider candidate = null;
        try {
            var builder = OtlpHttpSpanExporter.builder().setEndpoint(config.endpoint())
                    .setTimeout(config.httpTimeoutMs(), TimeUnit.MILLISECONDS)
                    .setCompression("none");
            config.headers().forEach(builder::addHeader);
            if (!config.certificate().isEmpty()) {
                builder.setTrustedCertificates(Files.readAllBytes(Path.of(config.certificate())));
            }
            exporter = builder.build();
            processor = BatchSpanProcessor.builder(exporter)
                    .setMaxQueueSize(config.maxQueueSize())
                    .setMaxExportBatchSize(config.maxExportBatchSize())
                    .setScheduleDelay(config.scheduleDelayMs(), TimeUnit.MILLISECONDS)
                    .setExporterTimeout(config.httpTimeoutMs(), TimeUnit.MILLISECONDS).build();
            candidate = SdkTracerProvider.builder().setResource(resource())
                    .setSampler(Sampler.parentBased(Sampler.traceIdRatioBased(config.samplerRatio())))
                    .addSpanProcessor(processor).build();
            OpenTelemetrySdk sdk = OpenTelemetrySdk.builder().setTracerProvider(candidate).build();
            Runtime.getRuntime().addShutdownHook(new Thread(OpenTelemetryBootstrap::shutdown, "flexlb-trace-shutdown"));
            provider = candidate;
            FlexlbTrace.configure(sdk, "");
            Logger.info("Trace 已启用 role=flexlb source={} root_ratio={} parent_based=true",
                    config.source(), config.samplerRatio());
            return true;
        } catch (Exception | LinkageError ignored) {
            FlexlbTrace.configure(null, "");
            if (candidate != null) {
                candidate.shutdown().join(2, TimeUnit.SECONDS);
            } else if (processor != null) {
                processor.shutdown().join(2, TimeUnit.SECONDS);
            } else if (exporter != null) {
                exporter.shutdown().join(2, TimeUnit.SECONDS);
            }
            Logger.warn("Trace 已关闭 role=flexlb field=sdk reason=initialization_failed");
            return false;
        }
    }

    static synchronized void shutdown() {
        FlexlbTrace.configure(null, "");
        if (provider != null) {
            provider.shutdown().join(2, TimeUnit.SECONDS);
            provider = null;
        }
    }

    static synchronized void resetForTest() {
        shutdown();
        initialized = false;
    }

    static Resource resource() {
        long pid = ProcessHandle.current().pid();
        String hostname = hostname();
        AttributesBuilder attrs = Attributes.builder()
                .put("service.name", "rtp_llm_flexlb")
                .put("service.instance.id", (hostname.isEmpty() ? "unknown" : hostname) + "-" + pid)
                .put("process.pid", pid).put("rtp_llm.role", "flexlb")
                .put(INSTRUMENTATION_SDK_NAME_ATTRIBUTE, INSTRUMENTATION_SDK_NAME_VALUE);
        if (hasText(hostname)) {
            attrs.put(HOST_NAME_ATTRIBUTE, hostname).put(HOST_IP_ATTRIBUTE, hostname + "-" + pid);
        }
        String podIp = System.getenv("POD_IP");
        if (hasText(podIp)) {
            attrs.put(POD_IP_ATTRIBUTE, podIp.trim());
        }
        // SDK 的默认 Resource 仅含固有 SDK 身份，不启动环境自动探测。
        return Resource.getDefault().merge(Resource.create(attrs.build()));
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

    private static boolean hasText(String value) {
        return value != null && !value.trim().isEmpty();
    }
}

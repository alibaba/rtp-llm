package org.flexlb.config;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.config.MeterFilter;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;
import java.util.Set;

/**
 * When {@code flexlb.monitor.mode=critical-only} is active, only a curated set of
 * core link latency metrics are registered/reported, reducing the
 * overhead of metrics collection and exposition across 94+ registered meters.
 *
 * <p>This config performs filtering at two layers:
 * <ol>
 *   <li><b>Early-return allowlist</b> — via {@link MicrometerFlexMonitor#setAllowedMetrics(Set)},
 *       non-allowlisted metrics are skipped at register() and report() time before any
 *       Micrometer interaction, avoiding unnecessary object allocation and map lookups.</li>
 *   <li><b>MeterFilter</b> — a defensive {@code MeterRegistry} filter that denies any
 *       {@code flexlb.} prefixed meter not in the allowlist, in case a meter was registered
 *       before the allowlist took effect or via a different code path.</li>
 * </ol>
 *
 * <p>Non-{@code flexlb.} metrics (jvm.*, process.*, http.server.requests, etc.) are always allowed.
 */
@Configuration
@ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
@ConditionalOnProperty(name = "flexlb.monitor.mode", havingValue = "critical-only")
public class CriticalMetricsFilterConfig {

    private static final String METRIC_PREFIX = "flexlb.";

    /**
     * The curated allowlist of metric names (without the {@code flexlb.} prefix).
     *
     * <p>These are the core link latency metrics that must be retained
     * when operating in critical-only mode:
     * <ul>
     *   <li>Client-to-gRPC-server: network delay (network transfer)</li>
     *   <li>gRPC server processing: server entry to BalanceContext start</li>
     *   <li>Master decision: route+submit time (decision start to batcher queue placement)</li>
     *   <li>Queue wait: batcher queue wait time (enqueue to dispatch trigger)</li>
     *   <li>Dispatch: dispatch-to-ACK time (gRPC dispatch to engine ACK)</li>
     * </ul>
     */
    public static final Set<String> CRITICAL_METRICS = Set.of(
            // === Core link latency metrics (5 metrics, 4 stages) ===
            "app.request.network.delay.ms",             // client to gRPC server entry (network transfer)
            "app.grpc.server.process.ms",              // gRPC server entry to BalanceContext start
            "app.flexlb.route.submit.time.ms",          // master decision start to batcher queue
            "app.routing.queue.wait.time.ms",           // batcher queue wait to dispatch
            "app.flexlb.dispatch.ack.time.ms"           // dispatch gRPC to engine ACK
    );

    /**
     * Defensive MeterRegistry filter: allows all non-flexlb metrics, and only allows
     * flexlb metrics whose unprefixed name is in {@link #CRITICAL_METRICS}.
     */
    @Bean
    public MeterFilter criticalMetricsOnlyFilter() {
        return new MeterFilter() {
            @Override
            public Meter.Id map(Meter.Id id) {
                String name = id.getName();
                if (!name.startsWith(METRIC_PREFIX)) {
                    return id;
                }
                String unprefixed = name.substring(METRIC_PREFIX.length());
                return CRITICAL_METRICS.contains(unprefixed) ? id : null;
            }
        };
    }

    /**
     * Sets the early-return allowlist on {@link MicrometerFlexMonitor} so that
     * non-critical metrics are skipped before any Micrometer interaction.
     */
    @PostConstruct
    public void init() {
        MicrometerFlexMonitor.setAllowedMetrics(CRITICAL_METRICS);
    }
}

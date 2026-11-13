package org.flexlb.balance.metrics;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Logger;
import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.flexlb.util.metrics.SafeFlexMonitor;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.EndpointStats;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.flexlb.util.metrics.MetricConstant.DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS;
import static org.flexlb.util.metrics.MetricConstant.DECODE_INFLIGHT_KV_RESERVED_TOKENS;
import static org.flexlb.util.metrics.MetricConstant.DECODE_TOTAL_LOAD;
import static org.flexlb.util.metrics.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.util.metrics.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.util.metrics.MetricConstant.INFLIGHT_REQUEST_COUNT;
import static org.flexlb.util.metrics.MetricConstant.INFLIGHT_TTL_EXPIRED_QPS;
import static org.flexlb.util.metrics.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.util.metrics.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.util.metrics.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.flexlb.util.metrics.MetricConstant.SCHEDULER_INFLIGHT_SIZE;

/**
 * Reports route-response latency, queue depth and request resource usage.
 */
@Slf4j
@Component
public class DeliveryMetricsReporter implements org.flexlb.worker.endpoint.EndpointMetrics {

    /**
     * role tag value for scheduler-ledger series (vs PREFILL/DECODE endpoint ledgers).
     */
    public static final String SCHEDULER_ROLE = "SCHEDULER";

    /**
     * engineIp tag value for scheduler-ledger series (no real engine behind them).
     */
    public static final String SCHEDULER_ENGINE_IP = "scheduler";

    @Autowired private ObjectProvider<RequestRepository> requestSources;
    @Autowired private ObjectProvider<EndpointStats> endpointSources;
    @Autowired private ObjectProvider<RequestSchedulerReporter> requestReporters;

    private final FlexMonitor monitor;

    @Autowired
    public DeliveryMetricsReporter(FlexMonitor monitor) {
        this.monitor = SafeFlexMonitor.wrap(monitor);
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    public void report() {
        var requests = requestSources.getObject();
        var endpoints = endpointSources.getObject();
        var requestReporter = requestReporters.getObject();
        if (requests.isClosed()) {
            return;
        }
        try {
            reportSchedulerInflight(requests.liveRequestCount(), requests.oldestLiveRequestAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated("Failed to report scheduler inflight metrics", failure);
        }
        reportEndpoints(
                "Prefill",
                () ->
                        endpoints.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                org.flexlb.config.service.RoleType.PREFILL, null),
                this::reportPrefill,
                (address, endpoint) -> requestReporter.reportPrefillQueueDepth(address, 0));
        reportEndpoints(
                "Decode",
                () ->
                        endpoints.<org.flexlb.worker.endpoint.DecodeEndpoint>endpoints(
                                org.flexlb.config.service.RoleType.DECODE, null),
                this::reportDecode,
                (address, endpoint) ->
                        requestReporter.reportDecodePreemption(address, endpoint.stats()));
    }

    // ==================== Inflight metrics ====================

    /**
     * Scheduler size retains role=PREFILL; oldest age uses the distinct SCHEDULER ledger role.
     */
    public void reportSchedulerInflight(int size, long oldestAgeMs) {
        monitor.report(
                SCHEDULER_INFLIGHT_SIZE,
                FlexMetricTags.of("role", RoleType.PREFILL.name(), "engineIp", SCHEDULER_ENGINE_IP),
                size);
        monitor.report(
                INFLIGHT_MAX_AGE_MS,
                FlexMetricTags.ofEngine(SCHEDULER_ENGINE_IP, "role", SCHEDULER_ROLE),
                oldestAgeMs);
    }

    public void reportPrefill(EndpointStats.Prefill endpoint) {
        String ip = endpoint.getIp();
        var stats = endpoint.stats();
        reportPrefillInflight(ip, stats.locallyOwnedRequests(), stats.maxObservedAgeMs());
    }

    // ==================== Queue metrics ====================

    /**
     * Report one Prefill ownership snapshot using the existing per-worker series.
     */
    public void reportPrefillInflight(
            String engineIp, int locallyOwnedRequests, long maxObservedAgeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.PREFILL.name());
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, locallyOwnedRequests);
        monitor.report(INFLIGHT_MAX_AGE_MS, tags, maxObservedAgeMs);
    }

    public void reportDecode(EndpointStats.Decode endpoint) {
        var stats = endpoint.stats();
        reportDecodeInflight(
                endpoint.getIp(),
                stats.inflight(),
                stats.totalLoad(),
                stats.expectedKv(),
                stats.hardKv(),
                stats.oldestAgeMs());
    }

    /**
     * Report one Decode snapshot, keeping expected KV separate from non-reclaimable hard KV.
     */
    public void reportDecodeInflight(
            String engineIp,
            int inflight,
            int totalLoad,
            long expectedKv,
            long hardKv,
            long oldestAgeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.DECODE.name());
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, inflight);
        monitor.report(DECODE_TOTAL_LOAD, tags, totalLoad);
        monitor.report(DECODE_INFLIGHT_KV_RESERVED_TOKENS, tags, expectedKv);
        monitor.report(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, tags, hardKv);
        monitor.report(INFLIGHT_MAX_AGE_MS, tags, oldestAgeMs);
    }

    @PostConstruct
    public void init() {
        for (String metric :
                new String[] {
                    ROUTING_QUEUE_LENGTH,
                    INFLIGHT_REQUEST_COUNT,
                    SCHEDULER_INFLIGHT_SIZE,
                    DECODE_TOTAL_LOAD,
                    DECODE_INFLIGHT_KV_RESERVED_TOKENS,
                    DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS,
                    INFLIGHT_MAX_AGE_MS,
                }) {
            monitor.register(metric, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    ROUTING_QUEUE_WAIT_TIME_MS, ROUTE_SUBMIT_TIME_MS,
                }) {
            monitor.register(metric, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    ENGINE_BALANCING_MASTER_DISPATCH_REASON, INFLIGHT_TTL_EXPIRED_QPS,
                }) {
            monitor.register(metric, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        }
    }

    // ==================== Dispatch reason metrics ====================

    /**
     * Report inflight entries evicted from an endpoint ledger (prefill/decode
     * orphan sweeps via {@link EndpointRegistry}
     * and friends) via the same {@code app.flexlb.inflight.ttl.expired.qps}
     * series family.
     * <p>DiscoveryEndpoint-side evictions were previously log-only
     * (event=endpoint_inflight_ttl_eviction); this closes the gap with the
     * shared {role, engineIp, reason} tag schema. On this architecture the
     * endpoint ledgers have a single stale-unobserved exit, so the reason
     * bucket is {@code ttl}; only non-zero counts are reported, keeping the
     * series sparse.
     */
    public void reportEndpointInflightTtlExpired(
            String role, String engineIp, String reason, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", role, "reason", reason);
        monitor.report(INFLIGHT_TTL_EXPIRED_QPS, tags, count);
    }

    /**
     * Report an existing schedule-path latency series with its role and engine tags.
     */
    public void reportLatency(Latency latency, String role, String engineIp, long durationMs) {
        monitor.report(latency.metric, FlexMetricTags.ofEngine(engineIp, "role", role), durationMs);
    }

    private static <T extends org.flexlb.worker.endpoint.WorkerEndpoint> void reportEndpoints(
            String role,
            Supplier<java.util.List<T>> snapshot,
            Consumer<T> endpointMetrics,
            BiConsumer<String, T> admissionMetrics) {
        java.util.List<T> endpoints;
        try {
            endpoints = snapshot.get();
        } catch (RuntimeException failure) {
            warnIsolated("Failed to snapshot " + role + " endpoints for metrics", failure);
            return;
        }
        for (T entry : endpoints) {
            try {
                endpointMetrics.accept(entry);
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " endpoint metrics: endpoint="
                                + entry.getStatus().getMetricIpPort(),
                        failure);
            }
            try {
                admissionMetrics.accept(entry.getStatus().getMetricIpPort(), entry);
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " admission metrics: endpoint="
                                + entry.getStatus().getMetricIpPort(),
                        failure);
            }
        }
    }

    private static void warnIsolated(String message, RuntimeException failure) {
        try {
            Logger.warn(message, failure);
        } catch (Throwable ignored) {
            // Telemetry must never couple otherwise independent metric leaves.
        }
    }

    /**
     * Maps delivery timing observations to their exported metric names.
     */
    public enum Latency {
        ROUTE_SUBMIT(ROUTE_SUBMIT_TIME_MS);

        private final String metric;

        Latency(String metric) {
            this.metric = metric;
        }
    }
}

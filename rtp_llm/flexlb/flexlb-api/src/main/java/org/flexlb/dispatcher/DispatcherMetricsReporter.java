package org.flexlb.dispatcher;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.DISPATCHER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_ALL_RT;
import static org.flexlb.constant.MetricConstant.DISPATCHER_BATCH_CHUNKS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_BATCH_ITEMS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_CHUNK_DETAIL_QPS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_CHUNK_RT;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FANOUT_RT;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FEPOOL_ALIVE;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FEPOOL_SIZE;
import static org.flexlb.constant.MetricConstant.DISPATCHER_PREASSIGN_RT;

/**
 * KMonitor reporter for the dispatcher batch-fanout subsystem. Follows the same idiom as
 * {@code org.flexlb.service.monitor.EngineHealthReporter} / {@code RoutingQueueReporter}: a
 * {@code @Component} that registers every metric in {@link #init()} and exposes {@code reportXxx}
 * helpers that build {@link FlexMetricTags} and call {@link FlexMonitor#report}.
 *
 * <p>Gated on the same {@code dispatch.fe-pool-service-id} property as the rest of the dispatcher,
 * so it is only wired when the dispatcher is enabled. When monitoring is not configured the
 * injected {@link FlexMonitor} is the no-op default, so every {@code report} call is a cheap no-op.
 *
 * <p>Serving-path metrics ({@link #reportRequest}) mirror the master balancing idiom — a single
 * {@code .qps} (reported as {@code 1.0}) plus a {@code .rt} GAUGE, both tagged by {@code code} —
 * rather than a separate metric per outcome. Tag cardinality is kept bounded on purpose: only
 * {@code type} (2 values), {@code path} (the registered specs), {@code code} (HTTP status), and the
 * chunk {@code result}/{@code reason} categories — never a host/ip or raw error message.
 */
@Slf4j
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatcherMetricsReporter {

    private final FlexMonitor monitor;

    public DispatcherMetricsReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        monitor.register(DISPATCHER_ALL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_ALL_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_PREASSIGN_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_FANOUT_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_CHUNK_DETAIL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_CHUNK_RT, FlexMetricType.GAUGE);
        monitor.register(DISPATCHER_BATCH_ITEMS, FlexMetricType.GAUGE);
        monitor.register(DISPATCHER_BATCH_CHUNKS, FlexMetricType.GAUGE);
        monitor.register(DISPATCHER_FEPOOL_SIZE, FlexMetricType.GAUGE);
        monitor.register(DISPATCHER_FEPOOL_ALIVE, FlexMetricType.GAUGE);
        log.info("DispatcherMetricsReporter initialized and registered with FlexMonitor");
    }

    /**
     * One inbound dispatcher request (batch or passthrough): bumps the request QPS and records
     * end-to-end latency, both tagged by {@code type}/{@code path}/{@code code}.
     */
    public void reportRequest(String type, String path, int code, long costMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "type", type,
                "path", path,
                "code", String.valueOf(code));
        monitor.report(DISPATCHER_ALL_QPS, tags, 1.0);
        monitor.report(DISPATCHER_ALL_RT, tags, costMs);
    }

    /**
     * Batch split shape for a request that actually fanned out (tagged by {@code path}).
     */
    public void reportBatchShape(String path, int totalItems, int chunkCount) {
        FlexMetricTags tags = FlexMetricTags.of("path", path);
        monitor.report(DISPATCHER_BATCH_ITEMS, tags, totalItems);
        monitor.report(DISPATCHER_BATCH_CHUNKS, tags, chunkCount);
    }

    /**
     * In-process pre-assign (batch_schedule) latency; {@code gotTargets=false} means it fell back
     * to no pre-assignment (empty target list).
     */
    public void reportPreassignRt(long ms, boolean gotTargets) {
        monitor.report(DISPATCHER_PREASSIGN_RT, FlexMetricTags.of("result", gotTargets ? "ok" : "empty"), ms);
    }

    /**
     * Fanout latency: first chunk dispatch to all sub-batch responses collected.
     */
    public void reportFanoutRt(long ms) {
        monitor.report(DISPATCHER_FANOUT_RT, FlexMetricTags.of(), ms);
    }

    /**
     * Per-chunk fanout outcome and FE-call latency. {@code result} is {@code ok}/{@code failed};
     * {@code reason} is a bounded failure category (e.g. {@code http_4xx}, {@code http_5xx},
     * {@code transport}, {@code pick_failed}) or {@code ok} on success.
     */
    public void reportChunk(String result, String reason, long rtMs) {
        monitor.report(DISPATCHER_CHUNK_DETAIL_QPS, FlexMetricTags.of("result", result, "reason", reason), 1.0);
        monitor.report(DISPATCHER_CHUNK_RT, FlexMetricTags.of("result", result), rtMs);
    }

    /**
     * Current FE pool size and alive count (reported periodically from the health-probe loop).
     * {@code alive == 0} is the all-FE-dead fallback signal.
     */
    public void reportFePool(int size, int alive) {
        FlexMetricTags tags = FlexMetricTags.of();
        monitor.report(DISPATCHER_FEPOOL_SIZE, tags, size);
        monitor.report(DISPATCHER_FEPOOL_ALIVE, tags, alive);
    }
}

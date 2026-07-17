package org.flexlb.dispatcher;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatcherMetricsReporter {

    /** Chunk outcome reasons. Bounded set — {@link #CHUNK_OK} means success, everything else is a failure category. */
    public static final String CHUNK_OK = "ok";
    public static final String CHUNK_TRANSPORT = "transport";
    public static final String CHUNK_PICK_FAILED = "pick_failed";
    public static final String CHUNK_HTTP_4XX = "http_4xx";
    public static final String CHUNK_HTTP_5XX = "http_5xx";
    /** 200 with a JSON body whose response array is absent or the wrong length — merged as a failure. */
    public static final String CHUNK_MALFORMED = "malformed_body";

    private static final FlexMetricTags NO_TAGS = FlexMetricTags.of();
    private static final FlexMetricTags RESULT_OK = FlexMetricTags.of("result", "ok");
    private static final FlexMetricTags RESULT_FAILED = FlexMetricTags.of("result", "failed");
    private static final FlexMetricTags RESULT_EMPTY = FlexMetricTags.of("result", "empty");

    /** The reason set is closed, so every {@code (result, reason)} tag combination is pre-built. */
    private static final Map<String, FlexMetricTags> CHUNK_DETAIL_TAGS = Map.of(
            CHUNK_OK, FlexMetricTags.of("result", "ok", "reason", CHUNK_OK),
            CHUNK_TRANSPORT, FlexMetricTags.of("result", "failed", "reason", CHUNK_TRANSPORT),
            CHUNK_PICK_FAILED, FlexMetricTags.of("result", "failed", "reason", CHUNK_PICK_FAILED),
            CHUNK_HTTP_4XX, FlexMetricTags.of("result", "failed", "reason", CHUNK_HTTP_4XX),
            CHUNK_HTTP_5XX, FlexMetricTags.of("result", "failed", "reason", CHUNK_HTTP_5XX),
            CHUNK_MALFORMED, FlexMetricTags.of("result", "failed", "reason", CHUNK_MALFORMED));

    /**
     * Request/batch tag sets are memoized: {@code type × path × code} (and {@code path} alone) draw
     * from a small closed set, so building a fresh {@link FlexMetricTags} on every request is pure
     * garbage. The keyspace is bounded (a handful of paths × a handful of HTTP codes), so the maps
     * never grow beyond that.
     */
    private final Map<String, FlexMetricTags> requestTags = new ConcurrentHashMap<>();
    private final Map<String, FlexMetricTags> pathTags = new ConcurrentHashMap<>();

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
    }

    /**
     * One inbound dispatcher request (batch or passthrough): bumps the request QPS and records
     * end-to-end latency, both tagged by {@code type}/{@code path}/{@code code}.
     */
    public void reportRequest(String type, String path, int code, long costMs) {
        FlexMetricTags tags = requestTags.computeIfAbsent(type + '\u0000' + path + '\u0000' + code,
                k -> FlexMetricTags.of("type", type, "path", path, "code", String.valueOf(code)));
        monitor.report(DISPATCHER_ALL_QPS, tags, 1.0);
        monitor.report(DISPATCHER_ALL_RT, tags, costMs);
    }

    /**
     * Batch split shape for a request that actually fanned out (tagged by {@code path}).
     */
    public void reportBatchShape(String path, int totalItems, int chunkCount) {
        FlexMetricTags tags = pathTags.computeIfAbsent(path, p -> FlexMetricTags.of("path", p));
        monitor.report(DISPATCHER_BATCH_ITEMS, tags, totalItems);
        monitor.report(DISPATCHER_BATCH_CHUNKS, tags, chunkCount);
    }

    /**
     * In-process pre-assign (batch_schedule) latency; {@code gotTargets=false} means it fell back
     * to no pre-assignment (empty target list).
     */
    public void reportPreassignRt(long ms, boolean gotTargets) {
        monitor.report(DISPATCHER_PREASSIGN_RT, gotTargets ? RESULT_OK : RESULT_EMPTY, ms);
    }

    /**
     * Fanout latency: first chunk dispatch to all sub-batch responses collected.
     */
    public void reportFanoutRt(long ms) {
        monitor.report(DISPATCHER_FANOUT_RT, NO_TAGS, ms);
    }

    /**
     * Per-chunk fanout outcome and FE-call latency. {@code reason} is {@link #CHUNK_OK} on success
     * or one of the bounded failure categories ({@link #CHUNK_HTTP_4XX}, {@link #CHUNK_HTTP_5XX},
     * {@link #CHUNK_TRANSPORT}, {@link #CHUNK_PICK_FAILED}); the {@code result} tag follows from it.
     */
    public void reportChunk(String reason, long rtMs) {
        boolean ok = CHUNK_OK.equals(reason);
        monitor.report(DISPATCHER_CHUNK_DETAIL_QPS, CHUNK_DETAIL_TAGS.get(reason), 1.0);
        monitor.report(DISPATCHER_CHUNK_RT, ok ? RESULT_OK : RESULT_FAILED, rtMs);
    }

    /**
     * Current FE pool size and alive count (reported periodically from the health-probe loop).
     * {@code alive == 0} is the all-FE-dead fallback signal.
     */
    public void reportFePool(int size, int alive) {
        monitor.report(DISPATCHER_FEPOOL_SIZE, NO_TAGS, size);
        monitor.report(DISPATCHER_FEPOOL_ALIVE, NO_TAGS, alive);
    }
}

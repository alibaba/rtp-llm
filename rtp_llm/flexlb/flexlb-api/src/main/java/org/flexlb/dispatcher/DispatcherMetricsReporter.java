package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
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
import static org.flexlb.constant.MetricConstant.DISPATCHER_CHUNK_DETAIL_QPS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_CHUNK_RT;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FEPOOL_ALIVE;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FEPOOL_SIZE;

/** Dispatcher KMonitor metrics with cached, bounded tag sets. Hosts and raw errors belong in logs. */
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatcherMetricsReporter {

    public static final String CHUNK_OK = "ok";
    public static final String CHUNK_TRANSPORT = "transport";
    public static final String CHUNK_NO_FE = "no_fe_assignment";
    public static final String CHUNK_HTTP_4XX = "http_4xx";
    public static final String CHUNK_HTTP_5XX = "http_5xx";
    public static final String CHUNK_MALFORMED = "malformed_body";

    private static final FlexMetricTags NO_TAGS = FlexMetricTags.of();
    private static final FlexMetricTags RESULT_OK = FlexMetricTags.of("result", "ok");
    private static final FlexMetricTags RESULT_FAILED = FlexMetricTags.of("result", "failed");

    private static final Map<String, FlexMetricTags> CHUNK_DETAIL_TAGS = Map.of(
            CHUNK_OK, FlexMetricTags.of("result", "ok", "reason", CHUNK_OK),
            CHUNK_TRANSPORT, FlexMetricTags.of("result", "failed", "reason", CHUNK_TRANSPORT),
            CHUNK_NO_FE, FlexMetricTags.of("result", "failed", "reason", CHUNK_NO_FE),
            CHUNK_HTTP_4XX, FlexMetricTags.of("result", "failed", "reason", CHUNK_HTTP_4XX),
            CHUNK_HTTP_5XX, FlexMetricTags.of("result", "failed", "reason", CHUNK_HTTP_5XX),
            CHUNK_MALFORMED, FlexMetricTags.of("result", "failed", "reason", CHUNK_MALFORMED));

    private final Map<String, FlexMetricTags> tagCache = new ConcurrentHashMap<>();

    private final FlexMonitor monitor;

    @PostConstruct
    public void init() {
        monitor.register(DISPATCHER_ALL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_ALL_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_CHUNK_DETAIL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(DISPATCHER_CHUNK_RT, FlexMetricType.GAUGE);
        monitor.register(DISPATCHER_FEPOOL_SIZE, FlexMetricType.GAUGE);
        monitor.register(DISPATCHER_FEPOOL_ALIVE, FlexMetricType.GAUGE);
    }

    public void reportRequest(String type, String path, int code, long costMs) {
        FlexMetricTags tags = tagCache.computeIfAbsent(type + '\u0000' + path + '\u0000' + code,
                k -> FlexMetricTags.of("type", type, "path", path, "code", String.valueOf(code)));
        monitor.report(DISPATCHER_ALL_QPS, tags, 1.0);
        monitor.report(DISPATCHER_ALL_RT, tags, costMs);
    }

    public void reportChunk(String reason, long rtMs) {
        boolean ok = CHUNK_OK.equals(reason);
        monitor.report(DISPATCHER_CHUNK_DETAIL_QPS, CHUNK_DETAIL_TAGS.get(reason), 1.0);
        monitor.report(DISPATCHER_CHUNK_RT, ok ? RESULT_OK : RESULT_FAILED, rtMs);
    }

    public void reportFePool(int size, int alive) {
        monitor.report(DISPATCHER_FEPOOL_SIZE, NO_TAGS, size);
        monitor.report(DISPATCHER_FEPOOL_ALIVE, NO_TAGS, alive);
    }
}

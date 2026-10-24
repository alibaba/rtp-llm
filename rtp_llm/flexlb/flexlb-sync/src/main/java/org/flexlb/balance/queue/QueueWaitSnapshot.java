package org.flexlb.balance.queue;

import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 请求在队首等待时记录的现场：队列深度、各优先级数量、Prefill 占用和 Decode 负载。
 * 记录时只保存数字，日志用的 Map 要等到失败时才生成。
 */
public final class QueueWaitSnapshot {

    public static final Map<String, Object> INITIAL_DIAGNOSTICS =
            Map.of("cause", "waiting for Prefill decision");

    private final String endpoint;
    private final String cause;
    private final long capturedAtMs;
    private final long queueVersion;
    private final int[] requestCountsByPriority;
    private final long prefillRequests;
    private final int prefillBatchSlots;
    private final DecodeRoutingView decode;
    private volatile Map<String, Object> diagnostics;

    public QueueWaitSnapshot(String endpoint,
                             String cause,
                             long capturedAtMs,
                             long queueVersion,
                             int[] requestCountsByPriority,
                             long prefillRequests,
                             int prefillBatchSlots,
                             DecodeRoutingView decode) {
        this.endpoint = endpoint;
        this.cause = cause;
        this.capturedAtMs = capturedAtMs;
        this.queueVersion = queueVersion;
        this.requestCountsByPriority = requestCountsByPriority;
        this.prefillRequests = prefillRequests;
        this.prefillBatchSlots = prefillBatchSlots;
        this.decode = decode;
    }

    public Map<String, Object> diagnostics() {
        Map<String, Object> cached = diagnostics;
        if (cached != null) {
            return cached;
        }
        Map<Integer, Integer> priorityCounts = new HashMap<>();
        int depth = 0;
        for (int priority = 0; priority < requestCountsByPriority.length; priority++) {
            int count = requestCountsByPriority[priority];
            if (count > 0) {
                priorityCounts.put(priority, count);
            }
            depth += count;
        }
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("cause", cause);
        details.put("capturedAtMs", capturedAtMs);
        details.put("endpoint", endpoint);
        details.put("queueDepth", depth);
        details.put("queueVersion", queueVersion);
        details.put("priorityCounts", Collections.unmodifiableMap(priorityCounts));
        details.put("prefillRequests", prefillRequests);
        details.put("prefillBatchSlots", prefillBatchSlots);
        if (decode != null) {
            details.put("decode", Map.of("endpoint", decode.address(),
                    "version", decode.admissionVersion(),
                    "engineLoad", decode.engineLoad(),
                    "totalLoad", decode.totalLoad(),
                    "kvTotal", decode.totalKv(),
                    "kvAvailable", decode.placementUsage().availableKvAfterReservations()));
        }
        cached = Collections.unmodifiableMap(details);
        diagnostics = cached;
        return cached;
    }
}

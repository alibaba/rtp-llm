package org.flexlb.dao;

import com.google.protobuf.ByteString;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.pv.DecisionGroup;
import org.flexlb.dao.pv.RoutingDecision;
import org.flexlb.dao.route.RoleType;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Data
@ToString
public class BalanceContext {

    //======================== Basic =======================//

    private FlexlbConfig config;

    private Request request;

    private Response response;

    @ToString.Exclude
    private ByteString generateInputPb;

    //======================== Queue ========================//

    private CompletableFuture<Response> future;

    //======================== Meters =======================//

    private long startTime = System.currentTimeMillis();

    /** Monotonic timestamp captured when server-side request processing starts. */
    private long serviceStartNanos = System.nanoTime();

    private long totalTimeUs;

    private Long requestArrivalDelayMs;

    private Long requestBodyReadAndDeserializeTimeUs;

    /** Number of input IDs on the incoming request, before hash preparation releases them. */
    private Long inputIdsCount;

    /** Null when the request body did not declare a Content-Length. */
    private Long requestBodyBytes;

    /** Serialized protobuf bytes, excluding gRPC framing and compression; used by payload metrics. */
    private Long requestMessageBytes;

    /** Per-request accumulator written by sequential processing stages and read after their completion. */
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    @ToString.Exclude
    private final RoutingTelemetryState routingTelemetryState = new RoutingTelemetryState();

    private volatile DecisionGroup decisionGroup;

    /**
     * Timestamp (ms) when the request entered the gRPC server pipeline,
     * recorded by {@code GrpcServerTimingInterceptor}. Used to split the
     * total arrival delay into network delay and gRPC server processing time.
     * Remains 0 if the interceptor did not set it (e.g. non-gRPC code path).
     */
    private long grpcEntryTime;

    /** Monotonic counterpart of {@link #grpcEntryTime} for duration measurements. */
    private long grpcEntryNanos;

    /** Monotonic timestamp immediately before the request enters its worker batcher. */
    private long routeSubmittedNanos;

    /** Monotonic timestamp immediately before the batch is dispatched to the engine. */
    private long batchDispatchedNanos;

    private long enqueueTime;

    /**
     * Timestamp (ms) when the engine acknowledges the batch in BATCH mode.
     * Set when RequestScheduler confirms the EnqueueBatch acknowledgement.
     * Used to compute ack_to_response_time_ms in FlexlbServiceImpl.completeSchedule().
     * Remains 0 for non-BATCH paths or when ACK was not received.
     */
    private long ackAtMs;

    /** Monotonic counterpart of {@link #ackAtMs}. */
    private long ackAtNanos;

    private boolean success = true;

    private String errorMessage;

    //===================== Scheduling =================//

    /**
     * Immutable per-request scheduling metadata. The expiration is an
     * absolute caller-supplied Unix timestamp and is shared by DIRECT, QUEUE,
     * and BATCH without being reset on retry or rescue.
     */
    private SchedulingMetadata schedulingMetadata;

    /**
     * priority scheduling plan type that finally placed the request:
     * normal / prefill_evict / decode_evict. Empty when not applicable.
     */
    private String planType = "";

    /** Cost of the committed eviction plan; 0 for normal placement. */
    private long planCost;

    /** Victims preempted to place this request; 0 for normal placement. */
    private int victimCount;

    //===================== Method ===================//

    public String getRequestId() {
        return request.getRequestId();
    }

    /**
     * Normalized priority of the request (1-100, higher = more important).
     * Immutable scheduling metadata is authoritative; the request fallback
     * supports manually constructed internal contexts.
     */
    public int getPriority() {
        return schedulingMetadata != null
                ? schedulingMetadata.priority()
                : request == null ? 50 : request.getPriority();
    }

    /**
     * Absolute request expiration timestamp in Unix epoch milliseconds.
     */
    public long getRequestExpiresAtMs() {
        if (schedulingMetadata != null) {
            return schedulingMetadata.expiresAtMs();
        }
        if (config != null && config.isQueue()) {
            return config.queueScheduler().resolveExpiresAtMs(startTime);
        }
        return Long.MAX_VALUE;
    }

    public boolean requestExpired(long nowMs) {
        long expiresAtMs = getRequestExpiresAtMs();
        return expiresAtMs <= 0 || nowMs >= expiresAtMs;
    }

    /** Direct accessor for immutable scheduling metadata. */
    public SchedulingMetadata schedulingMetadata() {
        return schedulingMetadata;
    }

    public void recordRequestTiming(long requestTimeMs, Long bodyReadAndDeserializeTimeUs) {
        if (requestTimeMs > 0) {
            this.requestArrivalDelayMs = startTime - requestTimeMs;
        }
        this.requestBodyReadAndDeserializeTimeUs = bodyReadAndDeserializeTimeUs;
    }

    public void finishRequestTiming() {
        this.totalTimeUs = (System.nanoTime() - serviceStartNanos) / 1_000;
    }

    public void recordBlockHashTiming(long queueWaitTimeUs, long executionTimeUs) {
        routingTelemetryState.hashWaitUs = queueWaitTimeUs;
        routingTelemetryState.hashUs = executionTimeUs;
    }

    public void recordCacheQuery(String source, long queryTimeUs) {
        routingTelemetryState.cacheSource = source;
        routingTelemetryState.cacheQueryUs += Math.max(0L, queryTimeUs);
        routingTelemetryState.cacheQueryCount++;
    }

    public void recordCacheSelection(RoleType role, String selectedIp, long hitCacheTokens) {
        CacheMatchSelection selection = new CacheMatchSelection(role, selectedIp, hitCacheTokens);
        routingTelemetryState.cacheSelections.put(role, selection);
    }

    public void beginRoutingAttempt(RoleType role) {
        routingTelemetryState.routingAttempts.merge(role, 1, Integer::sum);
        routingTelemetryState.cacheSelections.remove(role);
        routingTelemetryState.selectionReasons.remove(role);
        routingTelemetryState.routingDecisions.remove(role);
    }

    public void recordRoutingDecision(RoutingDecision decision) {
        Objects.requireNonNull(decision.role(), "role");
        Objects.requireNonNull(decision.selectionReason(), "selectionReason");
        routingTelemetryState.selectionReasons.put(decision.role(), decision.selectionReason());
        routingTelemetryState.routingDecisions.put(decision.role(), decision);
    }

    public void recordSelectionReason(RoleType role, String selectionReason) {
        Objects.requireNonNull(selectionReason, "selectionReason");
        routingTelemetryState.selectionReasons.put(role, selectionReason);
    }

    public int routingAttempt(RoleType role) {
        return routingTelemetryState.routingAttempts.getOrDefault(role, 0);
    }

    public String selectionReason(RoleType role) {
        return routingTelemetryState.selectionReasons.get(role);
    }

    /** Copies an immutable view after processing has settled; callers must not race an active writer. */
    public RoutingTelemetry getRoutingTelemetry() {
        return new RoutingTelemetry(routingTelemetryState.hashWaitUs, routingTelemetryState.hashUs,
                routingTelemetryState.cacheSource, routingTelemetryState.cacheQueryUs,
                routingTelemetryState.cacheQueryCount, routingTelemetryState.cacheSelections,
                routingTelemetryState.selectionReasons, routingTelemetryState.routingDecisions,
                routingTelemetryState.routingAttempts);
    }

    private static final class RoutingTelemetryState {
        private long hashWaitUs;
        private long hashUs;
        private String cacheSource;
        private long cacheQueryUs;
        private int cacheQueryCount;
        private final EnumMap<RoleType, CacheMatchSelection> cacheSelections = new EnumMap<>(RoleType.class);
        private final EnumMap<RoleType, String> selectionReasons = new EnumMap<>(RoleType.class);
        private final EnumMap<RoleType, RoutingDecision> routingDecisions = new EnumMap<>(RoleType.class);
        private final EnumMap<RoleType, Integer> routingAttempts = new EnumMap<>(RoleType.class);
    }

    /** Each terminal snapshot is immutable and can be retained independently of the request context. */
    public record RoutingTelemetry(long hashWaitUs,
                                   long hashUs,
                                   String cacheSource,
                                   long cacheQueryUs,
                                   int cacheQueryCount,
                                   Map<RoleType, CacheMatchSelection> cacheSelections,
                                   Map<RoleType, String> selectionReasons,
                                   Map<RoleType, RoutingDecision> routingDecisions,
                                   Map<RoleType, Integer> routingAttempts) {
        public RoutingTelemetry {
            cacheSelections = Map.copyOf(cacheSelections);
            selectionReasons = Map.copyOf(selectionReasons);
            routingDecisions = Map.copyOf(routingDecisions);
            routingAttempts = Map.copyOf(routingAttempts);
        }
    }

    public record CacheMatchSelection(RoleType role, String selectedIp, long hitCacheTokens) {
    }

}

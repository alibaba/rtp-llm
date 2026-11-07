package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.opentelemetry.context.Context;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.DecisionGroup;
import org.flexlb.dao.pv.RoutingDecision;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.util.Failures;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

import static com.google.common.base.Preconditions.checkState;

/** Request input and telemetry. Lifecycle decisions belong to the exact RequestContext. */
@Getter
public final class RequestData {
    private static final java.util.concurrent.atomic.AtomicLong WORKER_SEQUENCE =
            new java.util.concurrent.atomic.AtomicLong();
    private final FlexlbConfig config;
    private final long startTime = System.currentTimeMillis();
    private final long serviceStartNanos = System.nanoTime();
    private volatile Request request;
    private long originalNumericRequestId;
    private org.flexlb.dao.route.RequestPhase phase = org.flexlb.dao.route.RequestPhase.GENERATION;
    private java.util.List<org.flexlb.dao.route.RoleType> requestedRoles = java.util.List.of();
    private volatile SchedulingMetadata schedulingMetadata;
    private volatile CompletableFuture<Response> future;
    private volatile AbstractRequestScheduler requestScheduler;

    @Getter(lombok.AccessLevel.NONE)
    private volatile FutureTask<EngineRpcService.GenerateInputPB> generateInput;

    private volatile int generateInputSize;

    @Setter private Context traceContext;
    @Setter private long grpcEntryTime;
    @Setter private long grpcEntryNanos;
    @Setter private long routeSubmittedNanos;
    @Setter private long batchDispatchedNanos;
    @Setter private long enqueueTime;
    private long firstWorkerEnqueueTime;
    private long workerEnqueueSequence;
    @Setter private volatile long placementSequence;
    private long ackAtMs;
    private long ackAtNanos;
    @Setter private volatile Map<String, Object> schedulingDiagnostics;
    @Setter private String planType = "";
    @Setter private long planCost;
    @Setter private int victimCount;

    @Setter private Long requestMessageBytes;
    @Setter private Long requestBodyBytes;
    @Setter private DecisionGroup decisionGroup;
    private long totalTimeUs;
    private Long requestArrivalDelayMs;
    private Long requestBodyReadAndDeserializeTimeUs;
    private String cacheSource;
    private long cacheQueryUs;
    private int cacheQueryCount;
    private final EnumMap<RoleType, CacheMatchSelection> cacheSelections =
            new EnumMap<>(RoleType.class);
    private final EnumMap<RoleType, RoutingDecision> routingDecisions =
            new EnumMap<>(RoleType.class);
    private final EnumMap<RoleType, Integer> routingAttempts = new EnumMap<>(RoleType.class);

    public synchronized void setOriginalNumericRequestId(long id) {
        checkState(requestScheduler == null, "registered request identity is immutable");
        originalNumericRequestId = id;
    }

    public synchronized void setPhase(org.flexlb.dao.route.RequestPhase value) {
        checkState(requestScheduler == null, "registered request phase is immutable");
        phase = Objects.requireNonNull(value);
    }

    public synchronized void setRequestedRoles(java.util.List<RoleType> roles) {
        checkState(requestScheduler == null, "registered requested roles are immutable");
        requestedRoles = java.util.List.copyOf(roles);
    }

    public RequestData(FlexlbConfig config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    public synchronized void setRequest(Request value) {
        checkState(
                !(future instanceof RequestFuture) || request == value,
                "registered request inputs are immutable");
        request = Objects.requireNonNull(value, "request");
    }

    public synchronized void setSchedulingMetadata(SchedulingMetadata value) {
        checkState(
                !(future instanceof RequestFuture) || schedulingMetadata == value,
                "registered scheduling metadata is immutable");
        schedulingMetadata = value;
    }

    public synchronized void setFuture(CompletableFuture<Response> value) {
        checkState(
                !(future instanceof RequestFuture) && !(value instanceof RequestFuture)
                        || future == value,
                "registered request future cannot be replaced or rebound");
        future = value;
    }

    /** Binding happens under the Repository -> Context lock; no callbacks execute here. */
    synchronized void register(AbstractRequestScheduler owner, RequestFuture response) {
        checkState(requestScheduler == null, "request data already registered");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(owner, "requestScheduler");
        Objects.requireNonNull(response, "response");
        if (schedulingMetadata == null) {
            schedulingMetadata =
                    SchedulingMetadata.explicit(
                            request.getRawPriority(), config.resolveExpiresAtMs(startTime));
        }
        requestScheduler = owner;
        future = response;
    }

    /**
     * Classifies scheduling timeouts before request resources have been fully released.
     */
    public StrategyErrorType schedulingTimeoutError() {
        return config.isQueue()
                ? StrategyErrorType.RESOURCE_EXHAUSTED
                : StrategyErrorType.BATCH_SLO_EXPIRED;
    }

    public String getRequestId() {
        return request.getRequestId();
    }

    public int getPriority() {
        return schedulingMetadata == null
                ? (request == null ? 0 : request.getRawPriority())
                : schedulingMetadata.priority();
    }

    public long getRequestExpiresAtMs() {
        return schedulingMetadata == null
                ? config.resolveExpiresAtMs(startTime)
                : schedulingMetadata.expiresAtMs();
    }

    public Response getResponse() {
        var response = future;
        if (response == null) {
            return null;
        }
        try {
            return response.getNow(null);
        } catch (CompletionException | CancellationException failed) {
            return null;
        }
    }

    /** Parsing remains deferred until delivery, and FutureTask memoizes both value and failure. */
    public synchronized void setGenerateInputPb(ByteString bytes) {
        checkState(!(future instanceof RequestFuture), "registered delivery input is immutable");
        ByteString inputBytes = config.getDispatcher().requiresGenerateInput() ? bytes : null;
        generateInputSize = inputBytes == null ? 0 : inputBytes.size();
        generateInput =
                inputBytes == null || inputBytes.isEmpty()
                        ? null
                        : new FutureTask<>(() -> Delivery.parseInput(inputBytes));
    }

    boolean hasGenerateInput() {
        return generateInput != null;
    }

    void prepareGenerateInput() {
        var input = generateInput;
        if (input != null) {
            input.run();
        }
    }

    EngineRpcService.GenerateInputPB getGenerateInput()
            throws InvalidProtocolBufferException, InterruptedException {
        var input = generateInput;
        if (input == null) {
            return null;
        }
        input.run();
        try {
            return input.get();
        } catch (ExecutionException failure) {
            if (failure.getCause() instanceof InvalidProtocolBufferException malformed) {
                throw malformed;
            }
            throw Failures.propagate(failure.getCause(), "Generate input parsing failed");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw interrupted;
        }
    }

    /** Confirmed delivery or archival releases cached input; in-progress readers keep a local reference. */
    synchronized void discardGenerateInput() {
        generateInput = null;
    }

    /** Routing is final after confirmed delivery; execution retains only scalar request metadata. */
    synchronized void releaseSchedulingInput() {
        generateInput = null;
        if (request != null
                && (!request.getBlockCacheKeys().isEmpty()
                        || !request.getCacheAffinityKeys().isEmpty())) {
            request =
                    request.toBuilder()
                            .blockCacheKeys(java.util.List.of())
                            .cacheAffinityKeys(java.util.List.of())
                            .build();
        }
    }

    public synchronized void initializeExecutionOrder(long nowMs) {
        if (workerEnqueueSequence == 0L) {
            initializeExecutionOrder(nowMs, WORKER_SEQUENCE.incrementAndGet());
        }
    }

    synchronized void initializeExecutionOrder(long nowMs, long sequence) {
        if (workerEnqueueSequence == 0L) {
            firstWorkerEnqueueTime = nowMs;
            workerEnqueueSequence = sequence;
        }
    }

    synchronized void acknowledge(long nowMs, long nowNanos) {
        if (ackAtMs == 0L) {
            ackAtMs = nowMs;
            ackAtNanos = nowNanos;
        }
    }

    public synchronized void recordRequestTiming(
            long requestTimeMs, Long bodyReadAndDeserializeTimeUs) {
        if (requestTimeMs > 0) {
            this.requestArrivalDelayMs = startTime - requestTimeMs;
        }
        this.requestBodyReadAndDeserializeTimeUs = bodyReadAndDeserializeTimeUs;
    }

    public void finishRequestTiming() {
        this.totalTimeUs = (System.nanoTime() - serviceStartNanos) / 1_000;
    }

    public synchronized void recordCacheQuery(String source, long queryTimeUs) {
        cacheSource = source;
        cacheQueryUs += Math.max(0L, queryTimeUs);
        cacheQueryCount++;
    }

    public synchronized void recordCacheSelection(
            RoleType role, String selectedIp, long hitCacheTokens) {
        cacheSelections.put(role, new CacheMatchSelection(role, selectedIp, hitCacheTokens));
    }

    public synchronized void beginRoutingAttempt(RoleType role) {
        routingAttempts.merge(role, 1, Integer::sum);
        cacheSelections.remove(role);
        routingDecisions.remove(role);
    }

    public synchronized void recordRoutingDecision(RoutingDecision decision) {
        Objects.requireNonNull(decision.role(), "role");
        Objects.requireNonNull(decision.selectionReason(), "selectionReason");
        routingDecisions.put(
                decision.role(),
                new RoutingDecision(
                        decision.role(),
                        decision.group(),
                        decision.strategy(),
                        decision.selectionReason(),
                        decision.decisionTimeMs(),
                        routingAttempt(decision.role()),
                        decision.selectedEndpoint(),
                        decision.totalWorkerCount(),
                        decision.candidateWorkerCount(),
                        decision.snapshotTruncated(),
                        decision.rejections(),
                        decision.candidates(),
                        decision.prefillPolicy()));
    }

    public synchronized int routingAttempt(RoleType role) {
        return routingAttempts.getOrDefault(role, 0);
    }

    public synchronized String selectionReason(RoleType role) {
        var decision = routingDecisions.get(role);
        return decision == null ? null : decision.selectionReason();
    }

    public synchronized RoutingTelemetry getRoutingTelemetry() {
        return new RoutingTelemetry(
                cacheSource,
                cacheQueryUs,
                cacheQueryCount,
                cacheSelections,
                routingDecisions,
                routingAttempts);
    }

    public record RoutingTelemetry(
            String cacheSource,
            long cacheQueryUs,
            int cacheQueryCount,
            Map<RoleType, CacheMatchSelection> cacheSelections,
            Map<RoleType, RoutingDecision> routingDecisions,
            Map<RoleType, Integer> routingAttempts) {
        public RoutingTelemetry {
            cacheSelections = Map.copyOf(cacheSelections);
            routingDecisions = Map.copyOf(routingDecisions);
            routingAttempts = Map.copyOf(routingAttempts);
        }
    }

    public record CacheMatchSelection(RoleType role, String selectedIp, long hitCacheTokens) {}
}

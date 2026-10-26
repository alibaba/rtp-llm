package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.opentelemetry.context.Context;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.pv.DecisionGroup;
import org.flexlb.dao.pv.RoutingDecision;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

import static com.google.common.base.Preconditions.checkState;

/**
 * Request input and telemetry. Lifecycle decisions belong to the exact RequestContext.
 */
@Getter
public final class RequestData {
    private final FlexlbConfig config;
    private final org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity
            decodeAdmissionCapacity;
    private final org.flexlb.balance.prediction.DecodeCostFormula decodeCostFormula;
    private long originalNumericRequestId;
    private Set<RoleType> requestedRoles;
    private RequestPhase requestPhase = RequestPhase.GENERATION;
    @Setter private long totalTimeUs;
    @Setter private Long requestArrivalDelayMs;
    @Setter private Long requestBodyReadAndDeserializeTimeUs;
    @Setter private Long requestBodyBytes;
    @Setter private Long requestMessageBytes;
    @Setter private volatile DecisionGroup decisionGroup;
    @Setter private boolean success = true;
    @Setter private String errorMessage;

    @Getter(lombok.AccessLevel.NONE)
    private final RoutingTelemetryState routingTelemetryState = new RoutingTelemetryState();

    private final long startTime = System.currentTimeMillis();
    private final long serviceStartNanos = System.nanoTime();
    private volatile Request request;

    @Getter(lombok.AccessLevel.NONE)
    private volatile Response reportedResponse;

    private volatile SchedulingMetadata schedulingMetadata;
    private volatile CompletableFuture<Response> future;
    private volatile AbstractRequestScheduler scheduler;

    @Getter(lombok.AccessLevel.NONE)
    private volatile FutureTask<GenerateInputPayload> generateInput;

    @Getter(lombok.AccessLevel.NONE)
    private String generateInputRequestId;

    @Getter(lombok.AccessLevel.NONE)
    private IllegalArgumentException generateInputIdentityFailure;

    private volatile int generateInputSize;

    @Getter(lombok.AccessLevel.NONE)
    private volatile boolean registered;

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

    public synchronized void setOriginalNumericRequestId(long value) {
        checkState(!registered, "registered numeric request identity is immutable");
        originalNumericRequestId = value;
    }

    public synchronized void setRequestedRoles(Set<RoleType> roles) {
        checkState(!registered, "registered role selection is immutable");
        requestedRoles = roles == null ? null : Set.copyOf(roles);
    }

    public synchronized void setRequestPhase(RequestPhase value) {
        checkState(!registered, "registered request phase is immutable");
        requestPhase = Objects.requireNonNull(value, "requestPhase");
    }

    public boolean requestExpired(long nowMs) {
        long expires = getRequestExpiresAtMs();
        return expires <= 0L || nowMs >= expires;
    }

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

    public void recordCacheQuery(String source, long queryTimeUs) {
        routingTelemetryState.cacheSource = source;
        routingTelemetryState.cacheQueryUs += Math.max(0L, queryTimeUs);
        routingTelemetryState.cacheQueryCount++;
    }

    public void recordCacheSelection(RoleType role, String selectedIp, long hitCacheTokens) {
        routingTelemetryState.cacheSelections.put(
                role, new CacheMatchSelection(role, selectedIp, hitCacheTokens));
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
        routingTelemetryState.selectionReasons.put(
                role, Objects.requireNonNull(selectionReason, "selectionReason"));
    }

    public int routingAttempt(RoleType role) {
        return routingTelemetryState.routingAttempts.getOrDefault(role, 0);
    }

    public String selectionReason(RoleType role) {
        return routingTelemetryState.selectionReasons.get(role);
    }

    public RoutingTelemetry getRoutingTelemetry() {
        return new RoutingTelemetry(
                routingTelemetryState.cacheSource,
                routingTelemetryState.cacheQueryUs,
                routingTelemetryState.cacheQueryCount,
                routingTelemetryState.cacheSelections,
                routingTelemetryState.selectionReasons,
                routingTelemetryState.routingDecisions,
                routingTelemetryState.routingAttempts);
    }

    private static final class RoutingTelemetryState {
        private String cacheSource;
        private long cacheQueryUs;
        private int cacheQueryCount;
        private final EnumMap<RoleType, CacheMatchSelection> cacheSelections =
                new EnumMap<>(RoleType.class);
        private final EnumMap<RoleType, String> selectionReasons = new EnumMap<>(RoleType.class);
        private final EnumMap<RoleType, RoutingDecision> routingDecisions =
                new EnumMap<>(RoleType.class);
        private final EnumMap<RoleType, Integer> routingAttempts = new EnumMap<>(RoleType.class);
    }

    public record RoutingTelemetry(
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

    public record CacheMatchSelection(RoleType role, String selectedIp, long hitCacheTokens) {}

    public RequestData(FlexlbConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        var decode = config.getRouter().getRoles().getDecode();
        var availability = decode.getAvailability();
        this.decodeAdmissionCapacity =
                new org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity(
                        availability.getMaxEngineRequests() == null
                                ? 0L
                                : availability.getMaxEngineRequests(),
                        availability.getMaxKvUsagePercent());
        this.decodeCostFormula = decode.getCostEstimator().compiledFormula();
    }

    public synchronized void setRequest(Request value) {
        checkState(!registered || request == value, "registered request inputs are immutable");
        request = Objects.requireNonNull(value, "request");
    }

    public synchronized void setSchedulingMetadata(SchedulingMetadata value) {
        checkState(
                !registered || schedulingMetadata == value,
                "registered scheduling metadata is immutable");
        schedulingMetadata = value;
    }

    public synchronized void setFuture(CompletableFuture<Response> value) {
        checkState(
                !registered && !(value instanceof RequestFuture) || future == value,
                "registered request future cannot be replaced or rebound");
        future = value;
    }

    /**
     * Binding happens under the Repository -> Context lock; no callbacks execute here.
     */
    synchronized void register(AbstractRequestScheduler owner, RequestFuture response) {
        checkState(!registered && scheduler == null, "request data already registered");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(owner, "scheduler");
        Objects.requireNonNull(response, "response");
        if (schedulingMetadata == null) {
            schedulingMetadata =
                    SchedulingMetadata.of(
                            request.getRawPriority(),
                            null,
                            config.resolveExpiresAtMs(startTime),
                            config.defaultPriority());
        }
        scheduler = owner;
        future = response;
        registered = true;
    }

    public String getRequestId() {
        return request.getRequestId();
    }

    public int getPriority() {
        return schedulingMetadata == null
                ? org.flexlb.util.PriorityNormalizer.normalize(
                        request == null ? 0 : request.getRawPriority(),
                        null,
                        config.defaultPriority())
                : schedulingMetadata.priority();
    }

    public long getRequestExpiresAtMs() {
        return schedulingMetadata == null
                ? config.resolveExpiresAtMs(startTime)
                : schedulingMetadata.expiresAtMs();
    }

    public void setResponse(Response response) {
        reportedResponse = response;
    }

    public Response getResponse() {
        var response = future;
        if (response == null) {
            return reportedResponse;
        }
        try {
            return response.getNow(null);
        } catch (CompletionException | CancellationException failed) {
            return null;
        }
    }

    /**
     * Parsing remains deferred until delivery, and FutureTask memoizes both value and failure.
     */
    public synchronized void setGenerateInputPb(ByteString bytes) {
        checkState(!registered, "registered delivery input is immutable");
        ByteString inputBytes = config.getDispatcher().requiresGenerateInput() ? bytes : null;
        generateInputSize = inputBytes == null ? 0 : inputBytes.size();
        generateInputRequestId = null;
        generateInputIdentityFailure = null;
        if (inputBytes != null && !inputBytes.isEmpty()) {
            try {
                generateInputRequestId =
                        org.flexlb.engine.grpc.RequestId.readGenerateInputRequestId(inputBytes);
            } catch (IllegalArgumentException malformed) {
                generateInputIdentityFailure = malformed;
            }
        }
        generateInput =
                inputBytes == null || inputBytes.isEmpty()
                        ? null
                        : new FutureTask<>(() -> GenerateInputPayload.parse(inputBytes));
    }

    String generateInputRequestId() {
        if (generateInputIdentityFailure != null) {
            throw generateInputIdentityFailure;
        }
        return generateInputRequestId;
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

    GenerateInputPayload getGenerateInput()
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

    /**
     * Confirmed delivery or archival releases cached input; in-progress readers keep a local reference.
     */
    synchronized void discardGenerateInput() {
        generateInput = null;
    }

    /**
     * Routing is final after confirmed delivery; execution retains only scalar request metadata.
     */
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

    synchronized void initializeWorkerQueue(long nowMs, long sequence) {
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
}

package org.flexlb.balance.scheduler;

import com.google.common.collect.MapMaker;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.Delivery.Member;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.constant.GrpcConstants;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkState;

/**
 * Commits a selected Prefill prefix and hands it to the bounded sender.
 * The sender builds a complete payload, asks Scheduler to start each exact member once,
 * then invokes RPC and reports transport or stop evidence. It does not arbitrate request endings.
 * Payload failures precede the request send state; temporary permits are always returned.
 */
@Component
public class DeliveryDispatcher {
    static final long CLEANUP_TIMEOUT_MS = 5_000L;
    private static final int MAX_ATTEMPTS = 8;

    private final EngineGrpcClient grpcClient;
    private final DispatcherConfig.Type mode;
    private final ConfigService configService;
    private final DeliveryMetricsReporter telemetry;
    private final TaskTimer timer;
    final DecodeResources.AdmissionCapacity decodeAdmissionCapacity;
    @Autowired private final DeliveryExecution execution;
    // MapMaker weak keys use identity: equal-valued claims must not share an operation.
    private final ConcurrentMap<Object, StopTask> sessions = new MapMaker().weakKeys().makeMap();

    public DeliveryDispatcher(
            EngineGrpcClient grpcClient,
            ConfigService configService,
            DeliveryMetricsReporter telemetry,
            TaskTimer timer,
            DecodeResources.AdmissionCapacity capacity,
            DeliveryExecution execution) {
        this.grpcClient = grpcClient;
        this.configService = Objects.requireNonNull(configService);
        this.mode = configService.loadBalanceConfig().getDispatcher().getType();
        this.telemetry = Objects.requireNonNull(telemetry);
        this.timer = Objects.requireNonNull(timer);
        this.decodeAdmissionCapacity = Objects.requireNonNull(capacity);
        this.execution = Objects.requireNonNull(execution);
    }

    public DispatcherConfig.Type mode() {
        return mode;
    }

    /**
     * Returns normal progress, including UNKNOWN on timeout; late replies still apply exact evidence.
     */
    public CompletableFuture<PreemptionRegistration.Reply> preempt(
            PreemptionRegistration registration,
            DecodeEndpoint endpoint,
            String detail,
            long timeoutMs) {
        var delivery = registration.owner().snapshot().delivery();
        var claim = delivery == null ? null : delivery.identity();
        var answer = new CompletableFuture<PreemptionRegistration.Reply>();
        long expiresAt = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        var timeout =
                timer.schedule(
                        () -> answer.complete(PreemptionRegistration.Reply.UNKNOWN),
                        Math.max(1L, timeoutMs));
        answer.whenComplete((ignored, failure) -> timeout.cancel(false));
        var task = record(claim == null ? registration : claim, claim, registration.owner());
        CompletableFuture<PreemptionRegistration.Reply> response;
        boolean rejected;
        synchronized (task) {
            rejected =
                    task.closed
                            || (task.stopRequested
                                    && (task.preemption == null
                                            || task.preemption.registration() != registration));
            if (!rejected
                    && (task.preemption == null
                            || task.preemption.registration() != registration)) {
                task.preemption =
                        new PreemptionCall(
                                registration,
                                endpoint,
                                detail,
                                expiresAt,
                                new CompletableFuture<>());
            }
            response = rejected ? null : task.preemption.response();
        }
        if (rejected) {
            answer.complete(PreemptionRegistration.Reply.UNKNOWN);
            return answer;
        }
        response.whenComplete(
                (reply, failure) ->
                        answer.complete(
                                failure == null ? reply : PreemptionRegistration.Reply.UNKNOWN));
        task.start();
        if (task.closed) {
            answer.complete(PreemptionRegistration.Reply.UNKNOWN);
        }
        return answer;
    }

    public void awaitStops() {
        Throwable problem = null;
        long deadline =
                System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS + 1_000L);
        for (var task : sessions.values()) {
            try {
                synchronized (task) {
                    while (!task.closed && (task.queued != 0 || task.pending != 0)) {
                        long remaining = deadline - System.nanoTime();
                        if (remaining <= 0) {
                            throw new java.util.concurrent.TimeoutException(
                                    "stop callbacks pending");
                        }
                        TimeUnit.NANOSECONDS.timedWait(task, remaining);
                    }
                }
            } catch (Exception failure) {
                if (failure instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                }
                task.finished.completeExceptionally(failure);
                task.close(StopResult.UNRESOLVED);
                problem = Failures.append(problem, failure);
            }
        }
        Failures.rethrow(problem, "delivery stop callbacks did not drain");
    }

    /**
     * Commit exactly the selected prefix, then send it or publish its routes.
     */
    public CapacityBoundary submit(
            Delivery delivery,
            List<RouteResult> candidates,
            OptionalLong prediction,
            RouteResult bypassedHead,
            Delivery.PrefixPolicy prefixPolicy,
            BooleanSupplier selectionStillAllowed) {
        Throwable primaryFailure = null;
        try {
            var endpoint = delivery.endpoint;
            boolean current =
                    !candidates.isEmpty()
                            && selectionStillAllowed.getAsBoolean()
                            && endpoint.isSelectionCurrent(candidates, System.currentTimeMillis());
            if (!current) return CapacityBoundary.OWNERSHIP_LOST;
            Objects.requireNonNull(prefixPolicy, "prefixPolicy");
            try {
                prepare(delivery, candidates, delivery.evaluator);
                if (delivery.members.isEmpty()
                        || prefixPolicy == Delivery.PrefixPolicy.ALL_SELECTED
                                && delivery.members.size() != candidates.size())
                    return delivery.boundary == null
                            ? CapacityBoundary.OWNERSHIP_LOST
                            : delivery.boundary;
                commit(
                        delivery,
                        delivery.evaluator,
                        prediction,
                        candidates.size(),
                        bypassedHead,
                        selectionStillAllowed);
                if (!delivery.committed())
                    return !delivery.batch
                                    && !endpoint.requestCapacityAvailable(endpoint.requestLimit, 1)
                            ? CapacityBoundary.unavailable(
                                    CapacityBoundary.Resource.PREFILL_REQUEST)
                            : CapacityBoundary.OWNERSHIP_LOST;
                if (delivery.removedFailedMember()) {
                    Throwable failure =
                            Failures.run(
                                    null, () -> endpoint.signal(PrefillEndpoint.Signal.CAPACITY));
                    try {
                        AbstractRequestScheduler.preparationFailed(
                                delivery.blocked,
                                String.valueOf(delivery.boundary.cause().getMessage()));
                    } catch (Throwable callbackFailure) {
                        delivery.blocked
                                .ctx()
                                .data()
                                .getRequestScheduler()
                                .taskPool
                                .recordFailure(callbackFailure);
                        failure =
                                Failures.run(
                                        failure,
                                        () ->
                                                Logger.error(
                                                        "Failed member notification",
                                                        callbackFailure));
                    }
                    Failures.rethrow(failure, "failed member notification failed");
                }
                dispatchCommitted(delivery);
                endpoint.signal(PrefillEndpoint.Signal.PLACEMENT);
                return null;
            } catch (Throwable failure) {
                primaryFailure = failure;
                if (!delivery.committed())
                    throw Failures.propagate(failure, "delivery preparation failed");
                if (!submitted(delivery)) {
                    String detail = String.valueOf(failure.getMessage());
                    for (Member member : delivery.members)
                        failure =
                                Failures.run(
                                        failure,
                                        () ->
                                                AbstractRequestScheduler.preparationFailed(
                                                        member.route(), detail));
                }
                primaryFailure = failure;
                Logger.error("Committed delivery failed", failure);
                return null;
            }
        } finally {
            if (!submitted(delivery)) {
                if (primaryFailure == null) delivery.close();
                else Failures.run(primaryFailure, delivery::close);
            }
        }
    }

    static ContextResult.RouteDelivered routeDelivered(
            Delivery.Member claim, WorkSnapshot precedingWork, long unstartedWorkMs) {
        long now = System.currentTimeMillis();
        long visibility =
                PrefillTimePredictor.predictedVisibilityDeadlineMs(
                        precedingWork,
                        unstartedWorkMs,
                        claim.owner()
                                .data()
                                .getConfig()
                                .getRequestLifecycle()
                                .getDecision()
                                .getLifetime(),
                        now);
        return new ContextResult.RouteDelivered(claim, visibility, now);
    }

    Response validateRequestInput(RequestData data) {
        boolean required =
                mode == DispatcherConfig.Type.BATCH && data.getPhase() == RequestPhase.GENERATION;
        if (required && !data.hasGenerateInput()) {
            return Response.buildErrorResponse(
                    StrategyErrorType.GENERATE_INPUT_REQUIRED,
                    "missing serialized generate_input for batch dispatch");
        }
        return null;
    }

    void releaseUnusedInput(RequestData data) {
        if (mode != DispatcherConfig.Type.BATCH || data.getPhase() != RequestPhase.GENERATION) {
            data.discardGenerateInput();
        }
    }

    boolean capacityAvailable(RouteResult item, CapacityBoundary boundary) {
        return switch (boundary.resource()) {
            case DISPATCH_EXECUTOR -> execution.deliveryCapacityAvailable();
            case PREFILL_REQUEST ->
                    item.prefillEndpoint()
                            .requestCapacityAvailable(item.prefillEndpoint().requestLimit, 1);
            case PREFILL_BATCH ->
                    item.prefillEndpoint().batchCapacityAvailable(execution.maxInflightBatches);
            case DECODE_ENGINE ->
                    item.decodeEndpoint()
                            .shouldRetryDispatch(
                                    item.decodeReservation(), execution.decodeDeliveryCapacity);
        };
    }

    Runnable watchCapacity(RouteResult item, CapacityBoundary boundary, Runnable wake) {
        return switch (boundary.resource()) {
            case DISPATCH_EXECUTOR -> execution.watchDeliveryCapacity(wake);
            case DECODE_ENGINE -> {
                yield item.decodeEndpoint().watchEngineDispatchCapacity(wake);
            }
            case PREFILL_BATCH, PREFILL_REQUEST -> () -> {};
        };
    }

    void deliverBatch(Delivery batch) {
        com.google.common.base.Preconditions.checkArgument(
                !batch.members().isEmpty()
                        && batch.batchId() > 0L
                        && batch.predictedMs() >= 0L
                        && batch.remainingQueueDepth() >= 0,
                "invalid batch metadata");
        Objects.requireNonNull(batch.reason(), "reason");
        batch.executionPermit()
                .submit(
                        () -> deliverCommitted(batch),
                        failure -> failBeforeSend(batch, List.of(), failure));
    }

    /**
     * Accept the entire stop operation; callers do not drive retries or sender completion.
     */
    CompletableFuture<StopResult> stop(Delivery.Member claim) {
        for (; ; ) {
            if (claim.completion().toCompletableFuture().isDone()) {
                return CompletableFuture.completedFuture(StopResult.ALREADY_SETTLED);
            }
            var delivery = claim.owner().snapshot().delivery();
            if (delivery == null || delivery.identity() != claim) {
                return CompletableFuture.completedFuture(StopResult.STALE);
            }
            var task = record(claim, claim, claim.owner());
            synchronized (task) {
                if (task.closed) {
                    sessions.remove(claim, task);
                    continue;
                }
                if (task.stopRequested) {
                    return task.finished;
                }
                task.stopRequested = true;
            }
            task.start();
            return task.finished;
        }
    }

    /**
     * Drain late RPC callbacks before Scheduler closes its fact ingress.
     */
    void shutdown(java.util.List<RequestContext> active) {
        Throwable failure = Failures.run(null, execution::awaitDeliveries);
        failure = Failures.run(failure, () -> awaitDeliveryCleanup(active, true));
        Failures.rethrow(failure, "delivery shutdown failed");
    }

    void close(java.util.List<RequestContext> active) {
        Throwable failure = Failures.run(null, () -> awaitDeliveryCleanup(active, false));
        failure = Failures.run(failure, this::awaitStops);
        Failures.rethrow(failure, "delivery cleanup failed");
    }

    void commit(
            Delivery prepared,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPrediction,
            int candidateCount,
            RouteResult bypassedHead,
            java.util.function.BooleanSupplier selectionStillAllowed) {
        if (prepared.batch) {
            prepared.members = List.copyOf(prepared.members);
            prepared.features =
                    PrefillBatchFeatures.from(
                            prepared.members,
                            member -> member.route().seqLen(),
                            member -> member.route().hitCache());
        }
        long prediction =
                !prepared.batch
                        ? 0L
                        : plannedPrediction.isPresent() && prepared.members.size() == candidateCount
                                ? plannedPrediction.getAsLong()
                                : PrefillTimePredictor.predictCommittedBatchMs(
                                        evaluator, prepared.features);
        prepared.predictedMs = prediction;
        prepared.selectionTimeMs = System.currentTimeMillis();
        prepared.bypassedHead = bypassedHead;
        if (selectionStillAllowed.getAsBoolean()) {
            prepared.endpoint.commitDelivery(prepared);
        }
    }

    private static AbstractRequestScheduler.PreemptionEvent preemptionReply(
            PreemptionCall operation,
            ContextResult.CleanupAcknowledgement cleanup,
            ContextResult.CancelEvidenceKind kind) {
        var phase =
                switch (kind) {
                    case ACKNOWLEDGED, FENCED -> PreemptionCancelPhase.CANCEL_REQUESTED;
                    case NOT_FOUND -> PreemptionCancelPhase.NOT_FOUND_STALE;
                    case UNKNOWN -> PreemptionCancelPhase.CANCEL_UNKNOWN;
                    case CLEANED -> null;
                };
        boolean cleaned = kind == ContextResult.CancelEvidenceKind.CLEANED;
        return new AbstractRequestScheduler.PreemptionEvent(
                phase,
                false,
                cleaned ? operation.endpoint() : null,
                cleaned ? operation.registration().reservation() : null,
                cleaned ? operation.detail() : null,
                cleanup);
    }

    private static ContextResult.CancelEvidenceKind cancelEvidence(
            EngineGrpcClient.CancelReply reply) {
        return switch (reply) {
            case REQUEST_CLEANED -> ContextResult.CancelEvidenceKind.CLEANED;
            case REQUEST_FENCED -> ContextResult.CancelEvidenceKind.FENCED;
            case ACCEPTED -> ContextResult.CancelEvidenceKind.ACKNOWLEDGED;
            case NOT_FOUND -> ContextResult.CancelEvidenceKind.NOT_FOUND;
            case FAILED -> ContextResult.CancelEvidenceKind.UNKNOWN;
        };
    }

    private void deliverCommitted(Delivery batch) {
        var members = new ArrayList<Delivery.Member>(batch.members().size());
        long prediction = batch.predictedMs();
        int startedMembers = 0;
        EngineRpcService.EnqueueBatchRequestPB payload;
        try {
            for (var member : batch.members()) {
                var route = member.route();
                var state = member.owner().snapshot();
                if (!state.preparedDelivery() || state.route() != route) continue;
                if (!route.selectedEndpointsAvailable()) {
                    member.owner()
                            .data()
                            .getRequestScheduler()
                            .onRouteFailure(
                                    route,
                                    ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                                    "Selected Endpoint unavailable before payload preparation");
                    continue;
                }
                members.add(member);
            }
            if (members.isEmpty()) {
                batch.executionPermit().releaseIfNoResponse();
                var cleanup = batch.releaseMembers();
                if (cleanup != null) batch.members().getFirst().recordFailure(cleanup);
                return;
            }
            long payloadBytes = 16L;
            for (var member : members)
                payloadBytes += BatchEnqueueHelper.payloadSizeUpperBound(member.route());
            checkState(
                    batch.executionPermit().tryChargeBytes(payloadBytes),
                    "in-flight payload byte budget exhausted");
            payload = BatchEnqueueHelper.prepare(batch.batchId(), members);
            int candidateCount = members.size();
            for (int index = 0; index < candidateCount; index++) {
                var member = members.get(index);
                try {
                    var started =
                            member.owner()
                                    .data()
                                    .getRequestScheduler()
                                    .commitDelivery(
                                            member,
                                            Delivery.SendMode.BATCH_ENQUEUE,
                                            batch.batchId());
                    if (started != null) members.set(startedMembers++, started);
                } catch (Throwable failure) {
                    try {
                        AbstractRequestScheduler.preparationFailed(
                                member.route(), String.valueOf(failure.getMessage()));
                    } catch (Throwable ownerFailure) {
                        member.recordFailure(ownerFailure);
                        logFailure(
                                "Delivery start failure notification",
                                batch.batchId(),
                                ownerFailure);
                    }
                }
            }
            if (startedMembers != members.size())
                members.subList(startedMembers, members.size()).clear();
            if (startedMembers != candidateCount)
                payload = BatchEnqueueHelper.retain(payload, members);
            if (members.size() != batch.members().size() && !members.isEmpty()) {
                prediction =
                        PrefillTimePredictor.predictCommittedBatchMs(
                                batch.evaluator(),
                                PrefillBatchFeatures.from(
                                        members,
                                        value -> value.route().seqLen(),
                                        value -> value.route().hitCache()));
            }
            for (Delivery.Member claim : members) {
                long now = System.currentTimeMillis();
                var data = claim.owner().data();
                data.getRequestScheduler()
                        .onDeliveryResult(
                                new ContextResult.Prediction(
                                        claim,
                                        PrefillTimePredictor.predictedVisibilityDeadlineMs(
                                                batch.precedingWork(),
                                                prediction,
                                                data.getConfig()
                                                        .getRequestLifecycle()
                                                        .getDecision()
                                                        .getLifetime(),
                                                now),
                                        now));
            }
        } catch (Throwable failure) {
            if (startedMembers != members.size())
                members.subList(startedMembers, members.size()).clear();
            failBeforeSend(batch, members, failure);
            return;
        }
        BatchCall call;
        try {
            call = dispatchBatch(batch, members, payload, prediction);
        } finally {
            Throwable cleanup = batch.releaseMembers();
            if (cleanup != null) {
                batch.members().getFirst().recordFailure(cleanup);
            }
        }
        if (call.response() == null) {
            batch.executionPermit().releaseIfNoResponse();
        }
        if (call.failure() != null) {
            report(batch.batchId(), members, Collections.nCopies(members.size(), call.failure()));
        } else if (call.response() != null) {
            batch.executionPermit()
                    .observeResponse(
                            call.response(),
                            (response, failure) ->
                                    report(
                                            batch.batchId(),
                                            members,
                                            BatchEnqueueHelper.validate(
                                                    batch.batchId(), members, response, failure)));
        }
        reportDelivery(batch, members, prediction);
    }

    private static void logFailure(String operation, long batchId, Throwable failure) {
        try {
            Logger.error("{} batch_id={}", operation, batchId, failure);
        } catch (Throwable ignored) {
        }
    }

    private void failBeforeSend(Delivery batch, List<Delivery.Member> claims, Throwable failure) {
        batch.executionPermit().releaseIfNoResponse();
        try {
            for (Member member : batch.members()) {
                if (claims.stream().anyMatch(value -> value.route() == member.route())) {
                    continue;
                }
                try {
                    AbstractRequestScheduler.preparationFailed(
                            member.route(), String.valueOf(failure.getMessage()));
                } catch (Throwable ownerFailure) {
                    member.recordFailure(ownerFailure);
                    logFailure("Preparation failure notification", batch.batchId(), ownerFailure);
                }
            }
        } finally {
            Throwable cleanup = batch.releaseMembers();
            if (cleanup != null) {
                batch.members().getFirst().recordFailure(cleanup);
            }
        }
        report(
                batch.batchId(),
                claims,
                Collections.nCopies(claims.size(), Delivery.Result.notSent(failure)));
    }

    private void report(long batchId, List<Delivery.Member> claims, List<Delivery.Result> results) {
        long responseNanos = System.nanoTime();
        for (int i = 0; i < claims.size(); i++) {
            var claim = claims.get(i);
            var data = claim.owner().data();
            var result = results.get(i);
            if (result.status() == Delivery.Result.Status.DELIVERED
                    || result.status() == Delivery.Result.Status.PREFILL_REJECTED) {
                try {
                    FlexlbTrace.setScheduleDuration(
                            data.getTraceContext(),
                            FlexlbTrace.ENQUEUE_BATCH_MS,
                            data.getBatchDispatchedNanos(),
                            responseNanos);
                } catch (Throwable failure) {
                    logFailure("Response trace", batchId, failure);
                }
            }
            try {
                data.getRequestScheduler()
                        .onDeliveryResult(
                                new ContextResult.TransportResult(
                                        claim, result, System.currentTimeMillis()));
            } catch (Throwable failure) {
                data.getRequestScheduler().taskPool.recordFailure(failure);
                logFailure(
                        "Delivery completion request_id=" + claim.route().requestId(),
                        batchId,
                        failure);
            } finally {
                var task = sessions.get(claim);
                if (task != null) {
                    task.start();
                }
            }
        }
    }

    private BatchCall dispatchBatch(
            Delivery batch,
            List<Delivery.Member> claims,
            EngineRpcService.EnqueueBatchRequestPB request,
            long prediction) {
        if (claims.isEmpty()) {
            return new BatchCall(null, null);
        }
        boolean invoked = false;
        try {
            checkState(
                    configService.loadBalanceConfig().getDispatcher().getType()
                            == DispatcherConfig.Type.BATCH,
                    "batch submission requires BATCH dispatcher configuration");
            var items = claims.stream().map(value -> value.route()).toList();
            var endpoint = items.getFirst().prefillEndpoint();
            try {
                logDispatch(batch.batchId(), items, endpoint, prediction, batch.reason());
            } catch (Throwable failure) {
                logFailure("Dispatch logging", batch.batchId(), failure);
            }
            long now = System.nanoTime();
            for (var item : items) {
                var data = item.ctx().data();
                data.setBatchDispatchedNanos(now);
                FlexlbTrace.setScheduleAttribute(
                        data.getTraceContext(), FlexlbTrace.BATCH_ID, batch.batchId());
                FlexlbTrace.setScheduleAttribute(
                        data.getTraceContext(), FlexlbTrace.BATCH_SIZE, (long) items.size());
                FlexlbTrace.setScheduleAttribute(
                        data.getTraceContext(), FlexlbTrace.DISPATCH_REASON, batch.reason());
            }
            String ip = endpoint.getIp();
            int port = endpoint.getGrpcPort();
            invoked = true;
            var response =
                    Objects.requireNonNull(
                            grpcClient.batchEnqueueAsync(ip, port, request),
                            "EnqueueBatch client returned null future after invocation");
            return new BatchCall(response, null);
        } catch (Throwable failure) {
            // Before invocation even already-begun members were not sent; afterwards delivery is
            // uncertain.
            logFailure("Dispatch batch", batch.batchId(), failure);
            return new BatchCall(
                    null,
                    invoked
                            ? Delivery.Result.uncertain(failure)
                            : Delivery.Result.notSent(failure));
        }
    }

    private void logDispatch(
            long batchId,
            List<RouteResult> items,
            PrefillEndpoint prefillEp,
            long predMs,
            String reason) {
        if (!Logger.isDebugEnabled()) {
            return;
        }
        long totalTokens = 0;
        long totalHit = 0;
        StringBuilder itemDetail = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            RouteResult item = items.get(i);
            long seqLen = item.seqLen();
            long hitCache = item.hitCache();
            totalTokens += seqLen;
            totalHit += hitCache;
            if (i > 0) {
                itemDetail.append(", ");
            }
            itemDetail
                    .append("{req_id=")
                    .append(item.requestId())
                    .append(" seq_len=")
                    .append(seqLen)
                    .append(" hit_cache=")
                    .append(hitCache)
                    .append('}');
        }

        RouteResult head = items.get(0);
        long now = System.currentTimeMillis();
        long waitMs = now - head.enqueuedAtMs();
        long remainingMs = head.expiresAtMs() - now;

        Logger.debug(
                "flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
                        + "pred_ms={} reason={} wait_ms={} request_remaining_ms={} "
                        + "prefill={}:{} items=[{}]",
                batchId,
                items.size(),
                totalTokens,
                totalHit,
                predMs,
                reason,
                waitMs,
                remainingMs,
                prefillEp.getIp(),
                prefillEp.getHttpPort(),
                itemDetail);
    }

    private void reportDelivery(Delivery batch, List<Delivery.Member> claims, long prediction) {
        if (!claims.isEmpty()) {
            try {
                long nowMs = System.currentTimeMillis();
                long hitTokens = 0L;
                long totalTokens = 0L;
                String engineIp = claims.getFirst().route().prefillEndpoint().getIp();
                for (var claim : claims) {
                    var item = claim.route();
                    telemetry.reportBatchWaitTimeMs(
                            RoleType.PREFILL.name(),
                            engineIp,
                            Math.max(0L, nowMs - item.enqueuedAtMs()),
                            item.priority());
                    hitTokens =
                            com.google.common.math.LongMath.saturatedAdd(
                                    hitTokens, Math.max(0L, item.hitCache()));
                    totalTokens =
                            com.google.common.math.LongMath.saturatedAdd(
                                    totalTokens, Math.max(0L, item.seqLen()));
                }
                telemetry.reportBatchDelivery(
                        RoleType.PREFILL.name(),
                        engineIp,
                        batch.batchId(),
                        batch.reason(),
                        batch.remainingQueueDepth(),
                        claims.size(),
                        hitTokens,
                        totalTokens,
                        prediction);
            } catch (Throwable failure) {
                logFailure("Delivery telemetry", batch.batchId(), failure);
            }
        }
    }

    private StopTask record(Object key, Delivery.Member claim, RequestContext context) {
        var existing = sessions.get(key);
        if (existing != null) {
            return existing;
        }
        var created = new StopTask(key, claim, context);
        var previous = sessions.putIfAbsent(key, created);
        if (previous != null) {
            return previous;
        }
        if (claim != null) {
            claim.completion()
                    .whenComplete((ignored, failure) -> created.close(StopResult.ALREADY_SETTLED));
        }
        return created;
    }

    private void awaitDeliveryCleanup(java.util.List<RequestContext> active, boolean batchOnly) {
        var pending =
                active.stream()
                        .map(context -> context.snapshot().delivery())
                        .filter(Objects::nonNull)
                        .map(ContextResult.DeliveryFacts::identity)
                        .filter(Objects::nonNull)
                        .filter(
                                delivery ->
                                        !batchOnly
                                                || delivery.kind()
                                                        == Delivery.SendMode.BATCH_ENQUEUE)
                        .map(delivery -> delivery.completion().toCompletableFuture())
                        .toArray(CompletableFuture[]::new);
        try {
            CompletableFuture.allOf(pending)
                    .get(DeliveryDispatcher.CLEANUP_TIMEOUT_MS + 1_000L, TimeUnit.MILLISECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(
                    "Interrupted waiting for delivery cleanup", interrupted);
        } catch (java.util.concurrent.ExecutionException
                | java.util.concurrent.TimeoutException failure) {
            throw new IllegalStateException(
                    "Unsettled deliveries at shutdown: "
                            + active.stream()
                                    .map(context -> context.data().getRequestId())
                                    .toList(),
                    failure);
        }
    }

    private static boolean submitted(Delivery delivery) {
        return delivery.executionPermit != null && delivery.executionPermit.hasTaskOwner();
    }

    private void dispatchCommitted(Delivery delivery) {
        delivery.precedingWork = delivery.precedingWork().calculate();
        if (delivery.batch) {
            if (delivery.boundary != null && delivery.boundary.unavailable())
                delivery.reason = "delivery_capacity_prefix";
            deliverBatch(delivery);
        } else {
            publishRoutes(delivery);
        }
    }

    /**
     * Route publication and its committed queue owner finish before telemetry.
     */
    private void publishRoutes(Delivery delivery) {
        var members = delivery.members;
        var predictions = delivery.predictions;
        var precedingWork = delivery.precedingWork;
        var failures = members.getFirst().route().ctx().data().getRequestScheduler().taskPool;
        Delivery.Member[] published = new Delivery.Member[members.size()];
        long unstarted = 0L;
        int delivered = 0;
        Throwable failure = null;
        for (int index = 0; index < members.size(); index++) {
            Member member = members.get(index);
            Delivery.Member started = null;
            try {
                started =
                        member.owner()
                                .data()
                                .getRequestScheduler()
                                .commitDelivery(member, Delivery.SendMode.ROUTE_DECISION, 0L);
                if (started == null) continue;
                unstarted =
                        com.google.common.math.LongMath.saturatedAdd(unstarted, predictions[index]);
                publishRoute(started, precedingWork, unstarted);
                published[index] = started;
                delivered++;
            } catch (Throwable problem) {
                if (started != null) failure = Failures.append(failure, problem);
                else {
                    try {
                        AbstractRequestScheduler.preparationFailed(
                                member.route(), String.valueOf(problem.getMessage()));
                    } catch (Throwable ownerFailure) {
                        member.recordFailure(ownerFailure);
                    }
                }
            }
        }
        failure = Failures.append(failure, delivery.releaseMembers());
        members.clear();
        if (failure != null) {
            failures.recordFailure(failure);
            try {
                Logger.error("Route publication failed", failure);
            } catch (Throwable ignored) {
            }
        }
        reportRoutes(published, delivered, delivery.remainingQueueDepth());
    }

    private void publishRoute(
            Delivery.Member claim, WorkSnapshot precedingWork, long unstartedWorkMs) {
        claim.owner()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(routeDelivered(claim, precedingWork, unstartedWorkMs));
    }

    private void reportRoutes(Delivery.Member[] claims, int delivered, int remainingQueueDepth) {
        if (delivered != 0) {
            try {
                long nowMs = System.currentTimeMillis();
                String engineIp =
                        java.util.Arrays.stream(claims)
                                .filter(Objects::nonNull)
                                .findFirst()
                                .orElseThrow()
                                .route()
                                .prefillEndpoint()
                                .getIp();
                for (var claim : claims) {
                    if (claim != null) {
                        telemetry.reportBatchWaitTimeMs(
                                RoleType.PREFILL.name(),
                                engineIp,
                                Math.max(0L, nowMs - claim.route().enqueuedAtMs()),
                                claim.route().priority());
                    }
                }
                telemetry.reportBatcherQueueSize(
                        RoleType.PREFILL.name(), engineIp, remainingQueueDepth);
            } catch (Throwable monitoringFailure) {
                Logger.error("Route delivery telemetry failed", monitoringFailure);
            }
        }
    }

    private void prepare(
            Delivery prepared,
            List<RouteResult> candidates,
            PrefillTimePredictor.Evaluator evaluator) {
        long payloadBytes = 16L;
        for (RouteResult item : candidates) {
            long itemBytes = prepared.batch ? BatchEnqueueHelper.payloadSizeUpperBound(item) : 0L;
            if (prepared.batch
                    && !prepared.members.isEmpty()
                    && itemBytes > GrpcConstants.MAX_MESSAGE_SIZE - payloadBytes) {
                break;
            }
            synchronized (item.ctx()) {
                var state = item.ctx().snapshot();
                prepared.boundary =
                        state.preparedDelivery()
                                        && state.route() == item
                                        && item.selectedEndpointsAvailable()
                                ? prepareMember(prepared, item, evaluator)
                                : CapacityBoundary.OWNERSHIP_LOST;
                if (prepared.boundary != null) {
                    prepared.blocked = item;
                    break;
                }
                payloadBytes += itemBytes;
            }
        }
    }

    private CapacityBoundary prepareMember(
            Delivery prepared, RouteResult item, PrefillTimePredictor.Evaluator evaluator) {
        if (!prepared.batch
                && !prepared.endpoint.requestCapacityAvailable(
                        prepared.endpoint.requestLimit, prepared.members.size() + 1)) {
            return CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_REQUEST);
        }
        if (prepared.batch && prepared.members.isEmpty()) {
            try {
                var capacity = execution.tryReserveDelivery();
                if (!capacity.accepted()) {
                    return capacity.boundary();
                }
                prepared.executionPermit = capacity.value();
                prepared.batchId = execution.nextBatchId();
                checkState(prepared.batchId > 0L, "batch id supplier returned a non-positive id");
                prepared.maxInflightBatches = execution.maxInflightBatches;
                if (!prepared.endpoint.batchCapacityAvailable(prepared.maxInflightBatches))
                    return CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH);
            } catch (Throwable failure) {
                return CapacityBoundary.failed(failure);
            }
        }
        long prediction;
        try {
            prediction =
                    prepared.batch
                            ? 0L
                            : PrefillTimePredictor.predictSingleRequestMs(
                                    evaluator, item.seqLen(), item.hitCache());
        } catch (InvalidPrefillPredictionException failure) {
            return CapacityBoundary.failed(failure);
        }
        var attempt = Member.prepare(item, execution.decodeDeliveryCapacity);
        if (!attempt.accepted()) {
            return attempt.boundary();
        }
        Member member = attempt.value();
        try {
            if (!prepared.batch) {
                prepared.predictions[prepared.members.size()] = prediction;
            }
            prepared.members.add(member);
            return null;
        } catch (Throwable failure) {
            return CapacityBoundary.failed(Failures.run(failure, member::close));
        }
    }

    private record BatchCall(
            CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> response,
            Delivery.Result failure) {}

    /**
     * Result of the remote stop workflow.
     *
     * <p>This is an intermediate delivery fact. {@link #REMOTE_CLEANED} confirms only that the
     * remote RPC accepted cleanup; it does not close the request, release the Decode ledger, or
     * archive the Context. Those transitions require the matching Scheduler evidence.
     */
    enum StopResult {
        NOT_SENT,
        /**
         * The remote stop endpoint confirmed cleanup; Context and ledger settlement may remain
         * pending.
         */
        REMOTE_CLEANED,
        UNRESOLVED,
        ALREADY_SETTLED,
        STALE
    }

    private record PreemptionCall(
            PreemptionRegistration registration,
            DecodeEndpoint endpoint,
            String detail,
            long expiresAt,
            CompletableFuture<PreemptionRegistration.Reply> response) {}

    /**
     * Serializes cancellation attempts and applies their replies to the owning request before completing.
     */
    private final class StopTask implements Runnable {
        private final Object key;
        private final Delivery.Member claim;
        private final RequestContext context;
        private final AbstractRequestScheduler owner;
        private final CompletableFuture<StopResult> finished = new CompletableFuture<>();

        private volatile PreemptionCall preemption;
        private volatile StopResult terminal;

        private volatile boolean closed, stopRequested;

        private volatile Thread runner;

        // The queue owns serialization; the monitor only accounts for work and callback drain.
        private int queued, pending;

        // Only the running workflow changes its retry state.
        // Completion means the Scheduler has applied the receipt, including late receipts.
        private CompletableFuture<Long> rpc;
        private PreemptionCall attemptedPreemption;

        private boolean postSend;
        private int attempts;
        private long deadline, rpcDeadline;

        StopTask(Object key, Delivery.Member claim, RequestContext context) {
            this.key = key;
            this.claim = claim;
            this.context = context;
            this.owner = context.data().getRequestScheduler();
        }

        @Override
        public void run() {
            runner = Thread.currentThread();
            var operation = preemption;
            try {
                var result = result();
                if (result != null) {
                    finish(result);
                } else if (stopRequested) {
                    stopRemotely();
                } else {
                    if (operation != null && operation != attemptedPreemption) {
                        attemptedPreemption = operation;
                        if (System.nanoTime() < operation.expiresAt()) {
                            attempts = 0;
                            requestProof(operation);
                        }
                    }
                }
            } catch (RuntimeException | Error failure) {
                finished.completeExceptionally(failure);
                owner.taskPool.recordFailure(failure);
            } finally {
                boolean unused;
                synchronized (this) {
                    runner = null;
                    queued--;
                    unused =
                            !stopRequested
                                    && attempts == 0
                                    && pending == 0
                                    && queued == 0
                                    && operation == preemption;
                    if (unused) {
                        closed = true;
                    }
                    notifyAll();
                }
                if (unused) {
                    close(StopResult.UNRESOLVED);
                }
            }
        }

        private void start() {
            synchronized (this) {
                if (closed) {
                    return;
                }
                queued++;
            }
            signal();
            boolean accepted = false;
            try {
                accepted = owner.taskPool.executeStopWork(context, this);
            } catch (RuntimeException | Error failure) {
                finished.completeExceptionally(failure);
                close(StopResult.UNRESOLVED);
                owner.taskPool.recordFailure(failure);
            } finally {
                if (!accepted) {
                    synchronized (this) {
                        queued--;
                        notifyAll();
                    }
                }
            }
        }

        private void signal() {
            LockSupport.unpark(runner);
        }

        private void close(StopResult result) {
            boolean drained;
            synchronized (this) {
                closed = true;
                drained = pending == 0;
                sessions.remove(key, this);
                notifyAll();
            }
            signal();
            finish(result);
            if (drained) {
                var operation = preemption;
                if (operation != null) {
                    operation.response().complete(PreemptionRegistration.Reply.UNKNOWN);
                }
            }
        }

        private void finish(StopResult result) {
            if (finished.complete(result) && result == StopResult.UNRESOLVED && stopRequested) {
                owner.taskPool.recordFailure(
                        new IllegalStateException(
                                "delivery stop ended without release proof: request_id="
                                        + context.data().getRequestId()));
            }
        }

        private void stopRemotely() {
            if (deadline == 0) {
                deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS);
            }
            while (result() == null) {
                var delivery = context.snapshot().delivery();
                var send = delivery == null ? ContextResult.SendState.UNKNOWN : delivery.send();
                boolean sending =
                        send == ContextResult.SendState.NOT_STARTED
                                || send == ContextResult.SendState.SENDING;
                if (!postSend && !sending) {
                    postSend = true;
                    deadline =
                            System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS);
                }
                if (System.nanoTime() >= deadline) {
                    break;
                }
                if (send == ContextResult.SendState.NOT_STARTED || sending && attempts > 0) {
                    await(deadline, () -> context.snapshot().delivery().send() != send);
                    continue;
                }
                if (rpc != null) {
                    await(Math.min(deadline, rpcDeadline), rpc::isDone);
                    long completedAt = rpc.getNow(System.nanoTime());
                    rpc = null;
                    await(
                            Math.min(
                                    deadline,
                                    completedAt
                                            + TimeUnit.MILLISECONDS.toNanos(
                                                    Math.min(400L, 25L << (attempts - 1)))),
                            () -> false);
                }
                if (result() != null || System.nanoTime() >= deadline || attempts >= MAX_ATTEMPTS) {
                    break;
                }
                requestProof(preemption);
            }
            var result = result();
            finish(result == null ? StopResult.UNRESOLVED : result);
        }

        private StopResult result() {
            if (closed) {
                return StopResult.ALREADY_SETTLED;
            }
            var delivery = context.snapshot().delivery();
            if (claim != null && (delivery == null || delivery.identity() != claim)) {
                close(StopResult.STALE);
                return StopResult.STALE;
            }
            if (delivery != null
                    && (delivery.send() == ContextResult.SendState.NOT_SENT
                            || delivery.send() == ContextResult.SendState.ROUTE_CLOSED)) {
                return StopResult.NOT_SENT;
            }
            if (delivery != null && delivery.prefillProof() && delivery.decodeProof()) {
                return StopResult.REMOTE_CLEANED;
            }
            return terminal;
        }

        private void await(long until, BooleanSupplier ready) {
            while (result() == null && !ready.getAsBoolean()) {
                long remaining = until - System.nanoTime();
                if (remaining <= 0) {
                    return;
                }
                LockSupport.parkNanos(this, remaining);
                if (Thread.currentThread().isInterrupted()) {
                    throw new IllegalStateException("delivery stop interrupted");
                }
            }
        }

        private void requestProof(PreemptionCall operation) {
            var state = context.snapshot();
            var reason =
                    state.cancellation() != null
                            ? state.cancellation().reason()
                            : state.delivery() != null && state.delivery().abandonment() != null
                                    ? state.delivery().abandonment()
                                    : CancelReason.PRIORITY_PREEMPTED;
            // An expired, unsent preemption must not replace a later ordinary cancellation.
            if (attempts == 0 && operation != null && System.nanoTime() >= operation.expiresAt()) {
                if (!stopRequested) {
                    return;
                }
                operation = null;
            }
            var target = operation == null ? null : operation.registration().cancelTarget();
            long timeout =
                    attempts++ == 0 && operation != null
                            ? Math.max(
                                    1L,
                                    TimeUnit.NANOSECONDS.toMillis(
                                            operation.expiresAt() - System.nanoTime()))
                            : 500L;
            synchronized (this) {
                pending++;
            }
            rpcDeadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeout);
            var exact = operation;
            try {
                var remote =
                        Objects.requireNonNull(
                                grpcClient.cancelAsync(
                                        target == null
                                                ? claim.route().prefillEndpoint().getIp()
                                                : target.prefillIp(),
                                        target == null
                                                ? claim.route().prefillEndpoint().getGrpcPort()
                                                : target.prefillGrpcPort(),
                                        reason.toRequest(context.data().getRequestId()),
                                        timeout),
                                "cancel future");
                rpc =
                        remote.handle(
                                (reply, failure) ->
                                        applyReply(
                                                failure == null && reply != null
                                                        ? reply
                                                        : EngineGrpcClient.CancelReply.FAILED,
                                                exact));
                rpc.whenComplete((ignored, failure) -> signal());
            } catch (RuntimeException | Error failed) {
                if (failed instanceof Error) {
                    owner.taskPool.recordFailure(failed);
                }
                rpc =
                        CompletableFuture.completedFuture(
                                applyReply(EngineGrpcClient.CancelReply.FAILED, exact));
            }
        }

        private long applyReply(EngineGrpcClient.CancelReply reply, PreemptionCall operation) {
            var progress = PreemptionRegistration.Reply.UNKNOWN;
            var evidence = EngineGrpcClient.CancelReply.FAILED;
            try {
                var scheduler = context.data().getRequestScheduler();
                var kind = cancelEvidence(reply);
                var cleanup =
                        claim == null
                                ? null
                                : new ContextResult.CleanupAcknowledgement(
                                        claim, kind, System.currentTimeMillis());
                if (operation != null) {
                    var event = preemptionReply(operation, cleanup, kind);
                    progress =
                            event.progress(scheduler.onPreemption(operation.registration(), event));
                } else if (cleanup != null) {
                    scheduler.onDeliveryResult(cleanup);
                }
                evidence = reply;
            } catch (RuntimeException | Error failed) {
                owner.taskPool.recordFailure(failed);
            } finally {
                synchronized (this) {
                    if (evidence == EngineGrpcClient.CancelReply.REQUEST_CLEANED) {
                        terminal = StopResult.REMOTE_CLEANED;
                    }
                }
                if (operation != null) {
                    operation.response().complete(progress);
                }
                synchronized (this) {
                    pending--;
                    notifyAll();
                }
                if (claim == null) {
                    close(StopResult.UNRESOLVED);
                }
            }
            return System.nanoTime();
        }
    }
}

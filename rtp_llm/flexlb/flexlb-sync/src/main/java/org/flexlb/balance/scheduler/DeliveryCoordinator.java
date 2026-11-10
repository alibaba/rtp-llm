package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.Delivery.Member;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.constant.GrpcConstants;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static com.google.common.base.Preconditions.checkState;

/**
 * Prepares and dequeues selected requests, then hands them to the dispatcher.
 * Coordinates delivery admission and shutdown; the dispatcher owns request commit and publication.
 */
@Component
public class DeliveryCoordinator {
    static final long CLEANUP_TIMEOUT_MS = 5_000L;

    private final DispatcherConfig.Type mode;
    private final DeliveryExecution execution;
    private final EngineDelivery engine;
    private final DeliveryDispatcher dispatcher;

    public DeliveryCoordinator(
            RequestRepository requests,
            EngineGrpcClient grpcClient,
            ConfigService configService,
            DeliveryMetricsReporter reporter,
            DecodeResources.AdmissionCapacity capacity,
            DeliveryExecution execution) {
        this.engine = new EngineDelivery(grpcClient);
        this.mode = configService.loadBalanceConfig().deliveryType(RequestPhase.GENERATION);
        this.execution = Objects.requireNonNull(execution);
        this.dispatcher =
                new DeliveryDispatcher(
                        Objects.requireNonNull(requests),
                        engine,
                        Objects.requireNonNull(capacity),
                        configService,
                        Objects.requireNonNull(reporter));
    }

    public DispatcherConfig.Type mode() {
        return mode;
    }

    public void awaitStops() {
        engine.awaitStops();
    }

    public boolean isStoppingOnCurrentThread() {
        return engine.isStoppingOnCurrentThread();
    }

    public CompletableFuture<EngineDelivery.CancelResult> cancel(
            RequestContext context,
            Object identity,
            String ip,
            int port,
            CancelReason reason,
            long timeoutMs) {
        return engine.cancel(context, identity, ip, port, reason, timeoutMs);
    }

    /**
     * Own preparation, queue commit and dispatch for the selected requests.
     * The caller receives a rejection reason; temporary permits and partial preparation failures stay here.
     */
    public CapacityBoundary submit(
            List<RouteResult> candidates,
            String reason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong prediction,
            Delivery.PrefixPolicy prefixPolicy) {
        if (candidates.isEmpty()) return CapacityBoundary.OWNERSHIP_LOST;
        var endpoint = candidates.getFirst().prefillEndpoint();
        var delivery =
                new Delivery(
                        endpoint,
                        mode == DispatcherConfig.Type.BATCH,
                        candidates.size(),
                        evaluator,
                        reason);
        var boundary = submitPreparedGroup(delivery, candidates, prediction, prefixPolicy);
        if (boundary == null
                && delivery.boundary != null
                && delivery.boundary.status() == CapacityBoundary.Status.FAILED) {
            failQueuedPreparation(delivery.blocked, delivery.boundary.cause());
        }
        return boundary;
    }

    /**
     * Remove a failed queued request before reporting its preparation failure or expiry.
     */
    void failQueuedPreparation(RouteResult route, Throwable failure) {
        var owner = route.ctx().data().getRequestScheduler();
        if (!route.selectedEndpointsAvailable()) {
            owner.onRouteFailure(
                    route,
                    ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                    "Selected Endpoint unavailable during delivery preparation");
            return;
        }
        var endpoint = route.prefillEndpoint();
        if (endpoint.queueStopped() || !endpoint.releaseQueuedRequest(route)) return;
        try {
            if (route.requestExpired(System.currentTimeMillis())) {
                owner.cancel(route.ctx(), 0L, CancelReason.DEADLINE_EXCEEDED);
            } else {
                AbstractRequestScheduler.preparationFailed(
                        route, String.valueOf(failure.getMessage()));
            }
        } catch (Throwable callbackFailure) {
            owner.recordRequestFailure(callbackFailure);
            Logger.error(
                    "Queued preparation failure callback failed request_id={}",
                    route.requestId(),
                    callbackFailure);
        }
    }

    CompletableFuture<Void> stop(Delivery.Member claim) {
        return engine.stop(claim);
    }

    Response validateRequestInput(RequestData data) {
        boolean required =
                data.getConfig().deliveryType(data.getPhase()) == DispatcherConfig.Type.BATCH;
        if (required && !data.hasGenerateInput()) {
            return Response.buildErrorResponse(
                    StrategyErrorType.GENERATE_INPUT_REQUIRED,
                    "missing serialized generate_input for batch dispatch");
        }
        return null;
    }

    void releaseUnusedInput(RequestData data) {
        if (data.getConfig().deliveryType(data.getPhase()) != DispatcherConfig.Type.BATCH) {
            data.discardGenerateInput();
        }
    }

    boolean capacityAvailable(RouteResult item, CapacityBoundary boundary) {
        return execution.capacityAvailable(item, boundary.resource());
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

    /**
     * Drain late RPC callbacks before Scheduler closes its fact ingress.
     */
    void shutdown(List<RequestContext> active) {
        Throwable failure = Failures.run(null, execution::awaitDeliveries);
        failure = Failures.run(failure, () -> awaitDeliveryCleanup(active, true));
        Failures.rethrow(failure, "delivery shutdown failed");
    }

    void close(List<RequestContext> active) {
        Throwable failure = Failures.run(null, () -> awaitDeliveryCleanup(active, false));
        failure = Failures.run(failure, engine::awaitStops);
        Failures.rethrow(failure, "delivery cleanup failed");
    }

    RoutingResult<ContextResult.RouteDelivered, CapacityKey> dispatchSelectedRoute(
            RouteResult route) {
        checkState(
                route.ctx().data().getConfig().deliveryType(route.ctx().data().getPhase())
                        == DispatcherConfig.Type.NON_BATCH,
                "selected route requires ROUTE delivery");
        var data = route.ctx().data();
        if (route.prefillEndpoint() == null)
            checkState(
                    data.getConfig().decisionPolicy(data.getPhase()).getType()
                            == DecisionPolicyConfig.Type.SINGLE,
                    "delivery without a worker queue requires a SINGLE decision");
        var result = dispatcher.dispatchDirect(route);
        if (result.status() == RoutingResult.Status.BLOCKED
                || result.status() == RoutingResult.Status.REJECTED
                || result.status() == RoutingResult.Status.CLOSED
                        && !data.getFuture().isDone()
                        && route.ctx().snapshot().cancellation() == null)
            return route.validateEndpoints(result);
        return result;
    }

    private CapacityBoundary submitPreparedGroup(
            Delivery delivery,
            List<RouteResult> candidates,
            OptionalLong prediction,
            Delivery.PrefixPolicy prefixPolicy) {
        Throwable primaryFailure = null;
        WorkSnapshot dequeued = null;
        boolean dispatched = false;
        try {
            var endpoint = delivery.endpoint;
            Objects.requireNonNull(prefixPolicy, "prefixPolicy");
            try {
                prepare(delivery, candidates);
                if (delivery.members.isEmpty()
                        || prefixPolicy == Delivery.PrefixPolicy.ALL_SELECTED
                                && delivery.members.size() != candidates.size())
                    return delivery.boundary == null
                            ? CapacityBoundary.OWNERSHIP_LOST
                            : delivery.boundary;
                var attempt = dequeuePrepared(delivery, prediction, candidates.size());
                if (!attempt.accepted()) return attempt.boundary();
                dequeued = attempt.value();
                delivery.precedingWork = dequeued.calculate();
                if (delivery.batch && delivery.boundary != null && delivery.boundary.unavailable())
                    delivery.reason = "delivery_capacity_prefix";
                dispatcher.dispatch(delivery);
                dispatched = true;
                endpoint.signal(PrefillEndpoint.Signal.PLACEMENT);
                return null;
            } catch (Throwable failure) {
                primaryFailure = failure;
                if (dequeued == null)
                    throw Failures.propagate(failure, "delivery preparation failed");
                if (!dispatched) {
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
            if (!dispatched) {
                if (primaryFailure == null) delivery.close();
                else Failures.run(primaryFailure, delivery::close);
            }
        }
    }

    private void prepare(Delivery prepared, List<RouteResult> candidates) {
        long payloadBytes = 16L;
        for (RouteResult item : candidates) {
            long itemBytes = prepared.batch ? EngineDelivery.payloadSizeUpperBound(item) : 0L;
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
                                ? prepareMember(prepared, item)
                                : CapacityBoundary.OWNERSHIP_LOST;
                if (prepared.boundary != null) {
                    prepared.blocked = item;
                    break;
                }
                payloadBytes += itemBytes;
            }
        }
    }

    private CapacityBoundary prepareMember(Delivery prepared, RouteResult item) {
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
                if (!prepared.endpoint.batchCapacityAvailable(execution.maxInflightBatches))
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
                                    prepared.evaluator, item.seqLen(), item.hitCache());
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

    private CapacityBoundary.Attempt<WorkSnapshot> dequeuePrepared(
            Delivery prepared, OptionalLong plannedPrediction, int candidateCount) {
        var features =
                prepared.batch
                        ? PrefillBatchFeatures.from(
                                prepared.members,
                                member -> member.route().seqLen(),
                                member -> member.route().hitCache())
                        : null;
        long prediction =
                !prepared.batch
                        ? 0L
                        : plannedPrediction.isPresent() && prepared.members.size() == candidateCount
                                ? plannedPrediction.getAsLong()
                                : PrefillTimePredictor.predictCommittedBatchMs(
                                        prepared.evaluator, features);
        prepared.predictedMs = prediction;
        if (prepared.endpoint.queueStopped())
            return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
        return prepared.endpoint.dequeueForDelivery(
                new PrefillEndpoint.DeliverySelection(
                        prepared.members.stream().map(Member::route).toList(),
                        prepared.batch,
                        prepared.batchId,
                        execution.maxInflightBatches,
                        prediction,
                        features,
                        prepared.predictions));
    }

    private void awaitDeliveryCleanup(List<RequestContext> active, boolean batchOnly) {
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
                    .get(DeliveryCoordinator.CLEANUP_TIMEOUT_MS + 1_000L, TimeUnit.MILLISECONDS);
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
}

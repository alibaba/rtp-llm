package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.Delivery.Member;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkState;

/**
 * Dispatches selected routes immediately, queued routes individually, or queued groups as EnqueueBatch.
 * Commits request state before sending and reports delivery results to the request scheduler.
 * EngineDelivery owns serialization and remote communication.
 */
final class DeliveryDispatcher {
    private final RequestRepository requests;
    private final EngineDelivery engine;
    private final org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity;
    private final ConfigService configService;
    private final DeliveryMetricsReporter reporter;

    DeliveryDispatcher(
            RequestRepository requests,
            EngineDelivery engine,
            org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity,
            ConfigService configService,
            DeliveryMetricsReporter reporter) {
        this.requests = requests;
        this.engine = engine;
        this.capacity = capacity;
        this.configService = configService;
        this.reporter = reporter;
    }

    /**
     * Returns after taking responsibility for member cleanup, including executor rejection.
     * Invalid input fails before the transfer; asynchronous failures stay with the delivery task.
     */
    void dispatch(Delivery delivery) {
        if (delivery.batch) dispatchBatchGroup(delivery);
        else dispatchNonBatch(delivery);
    }

    RoutingResult<ContextResult.RouteDelivered, CapacityKey> dispatchDirect(RouteResult route) {
        if (route.encoderEndpoint() != null) return commitEncoderRoute(route);
        checkState(
                route.decodeEndpoint() == null || route.decodeReservation() != null,
                "selected route requires Decode registration");
        Delivery.Member member = null;
        boolean deliveryCommitted = false;
        Throwable failure = null;
        try {
            var attempt = Delivery.Member.prepare(route, capacity);
            if (!attempt.accepted()) {
                if (attempt.boundary().unavailable())
                    return RoutingResult.blocked(
                            route.capacityKey(RoleType.DECODE),
                            Response.buildErrorResponse(
                                    StrategyErrorType.RESOURCE_EXHAUSTED,
                                    "Decode delivery admission unavailable at "
                                            + route.decodeEndpoint().getIp()
                                            + "; resource="
                                            + attempt.boundary().resource()),
                            Map.of(
                                    "cause",
                                    "Decode delivery admission unavailable",
                                    "resource",
                                    attempt.boundary().resource().name(),
                                    "endpoint",
                                    route.decodeEndpoint().getIp()));
                if (attempt.boundary().status() == CapacityBoundary.Status.FAILED)
                    throw new IllegalStateException(
                            "route admission failed", attempt.boundary().cause());
                return route.validateEndpoints(
                        RoutingResult.retry(
                                Response.buildErrorResponse(
                                        StrategyErrorType.SCHEDULER_PLAN_CONFLICT,
                                        "Selected route changed before delivery admission")));
            }
            member = attempt.value();
            org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot committed;
            var routingContext =
                    route.ctx().data().getRequestScheduler().currentRoutingContext(route);
            if (routingContext == null) return RoutingResult.closed();
            committed = route.prefillEndpoint().commitDirect(route);
            if (committed == null)
                return route.validateEndpoints(
                        RoutingResult.rejected(
                                Response.buildErrorResponse(
                                        StrategyErrorType.DISPATCH_FAILED,
                                        "Selected Prefill registration was not valid for commit at "
                                                + route.prefillEndpoint().getIp())));
            var preceding = committed.calculate();
            var delivered = routeDelivered(member, preceding, route.prefillWorkMs());
            if (!route.selectedEndpointsAvailable())
                return route.validateEndpoints(RoutingResult.closed());
            var claim = commit(member, Delivery.SendMode.ROUTE_DECISION, 0L, routingContext);
            if (claim == null) return RoutingResult.closed();
            deliveryCommitted = true;
            return RoutingResult.success(delivered);
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            var cleanup = member == null ? null : Failures.close(member);
            if (failure != null) Failures.append(failure, cleanup);
            else if (deliveryCommitted) {
                if (cleanup != null)
                    route.ctx().data().getRequestScheduler().recordRequestFailure(cleanup);
            } else Failures.rethrow(cleanup, "DIRECT preparation cleanup failed");
        }
    }

    Delivery.Member commit(
            Delivery.Member member,
            Delivery.SendMode mode,
            long batchId,
            RoutingContext selecting) {
        if (selecting != null)
            checkState(selecting.requestContext() == member.owner(), "foreign routing context");
        Objects.requireNonNull(mode, "delivery mode");
        com.google.common.base.Preconditions.checkArgument(
                mode != Delivery.SendMode.NONE
                        && (mode != Delivery.SendMode.BATCH_ENQUEUE || batchId > 0L),
                "invalid delivery identity");
        com.google.common.base.Preconditions.checkArgument(
                member.decode() == null
                        || member.decode()
                                .belongsTo(
                                        member.route().decodeEndpoint(),
                                        member.route().decodeReservation()),
                "foreign Decode dispatch permit");
        var scheduler = member.owner().data().getRequestScheduler();
        try {
            return tryCommitDelivery(member, mode, batchId, selecting);
        } catch (RuntimeException | Error failure) {
            var recorded = member.owner().snapshot().delivery();
            try {
                if (recorded != null && recorded.identity() == member) {
                    if (mode == Delivery.SendMode.BATCH_ENQUEUE) {
                        report(batchId, List.of(member), List.of(Delivery.Result.notSent(failure)));
                    } else {
                        scheduler.recordPreparationFailure(
                                new ContextResult.PreparationFailure(
                                        member,
                                        member.route(),
                                        failure.getMessage(),
                                        System.currentTimeMillis()));
                    }
                } else if (mode == Delivery.SendMode.BATCH_ENQUEUE) {
                    AbstractRequestScheduler.preparationFailed(
                            member.route(), failure.getMessage());
                }
            } catch (RuntimeException | Error notificationFailure) {
                scheduler.recordRequestFailure(notificationFailure);
                if (notificationFailure != failure) failure.addSuppressed(notificationFailure);
            }
            throw failure;
        }
    }

    private Delivery.Member tryCommitDelivery(
            Delivery.Member member,
            Delivery.SendMode mode,
            long batchId,
            RoutingContext selecting) {
        var route = member.route();
        var ctx = member.owner();
        var permit = member.decode();
        DecodeResources.ResourceChange<DecodeResources.EngineDispatchPermitTransferStatus>
                transfer = null;
        ContextResult.DeliveryNotStarted rejected;
        try {
            synchronized (ctx) {
                if (!requests.isCurrent(ctx) || ctx.data().getFuture().isDone()) return null;
                long nowMs = System.currentTimeMillis();
                var before = ctx.snapshot();
                boolean inactive = nowMs >= before.inactivityExpiresAtMs();
                boolean expired = nowMs >= ctx.data().getRequestExpiresAtMs();
                ExpirationTimer.Token deadline = null;
                if ((before.preparedDelivery()
                                || selecting != null && ctx.canPublishRoute(selecting))
                        && before.cancellation() == null
                        && (inactive || expired)) {
                    deadline =
                            inactive
                                    ? before.deadlines().inactivity()
                                    : before.deadlines().request();
                }
                if (deadline == null
                        && ctx.validateDelivery(route, selecting, nowMs)
                        && route.selectedEndpointsAvailable()) {
                    member.initialize(mode, batchId);
                    if (permit != null) {
                        transfer = permit.dispatch();
                        checkState(
                                transfer.status()
                                        == DecodeResources.EngineDispatchPermitTransferStatus
                                                .TRANSFERRED,
                                "Decode dispatch ownership lost");
                    }
                    if (selecting == null) ctx.recordDeliveryStarted(member, nowMs);
                    else ctx.recordDirectDeliveryStarted(selecting, member, nowMs);
                    return member;
                }
                rejected =
                        new ContextResult.DeliveryNotStarted(
                                member,
                                selecting,
                                deadline,
                                !route.selectedEndpointsAvailable(),
                                nowMs);
            }
        } finally {
            if (transfer != null && transfer.capacityReleased()) {
                Failures.recover(
                        member::recordFailure,
                        () ->
                                route.decodeEndpoint()
                                        .publishCapacity(
                                                org.flexlb.balance.endpoint.DecodeEndpoint
                                                        .CapacityEdge.DISPATCH));
            }
        }
        report(rejected);
        return null;
    }

    private void report(ContextResult.DeliveryEvidence result) {
        result.delivery().owner().data().getRequestScheduler().onDeliveryResult(result);
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
                report(
                        new ContextResult.TransportResult(
                                claim, result, System.currentTimeMillis()));
            } catch (Throwable failure) {
                data.getRequestScheduler().recordRequestFailure(failure);
                logFailure(
                        "Delivery completion request_id=" + claim.route().requestId(),
                        batchId,
                        failure);
            } finally {
                engine.deliveryChanged(claim);
            }
        }
    }

    private static void logFailure(String operation, long batchId, Throwable failure) {
        try {
            Logger.error("{} batch_id={}", operation, batchId, failure);
        } catch (Throwable ignored) {
        }
    }

    /**
     * Route publication and its committed queue owner finish before telemetry.
     */
    private void dispatchNonBatch(Delivery delivery) {
        var members = delivery.members;
        var predictions = delivery.predictions;
        var precedingWork = delivery.precedingWork;
        var failures = members.getFirst().route().ctx().data().getRequestScheduler();
        var published = new ArrayList<Delivery.Member>(members.size());
        long unstarted = 0L;
        Throwable failure = null;
        for (int index = 0; index < members.size(); index++) {
            Member member = members.get(index);
            Delivery.Member started = null;
            try {
                started = commit(member, Delivery.SendMode.ROUTE_DECISION, 0L, null);
                if (started == null) continue;
                unstarted =
                        com.google.common.math.LongMath.saturatedAdd(unstarted, predictions[index]);
                report(routeDelivered(started, precedingWork, unstarted));
                published.add(started);
            } catch (Throwable problem) {
                if (started != null) failure = Failures.append(failure, problem);
            }
        }
        failure = Failures.append(failure, delivery.releaseMembers());
        if (failure != null) {
            try {
                failures.recordRequestFailure(failure);
            } catch (Throwable reportingFailure) {
                failure = Failures.append(failure, reportingFailure);
            }
            logFailure("Route publication failed", 0L, failure);
        }
        reportRoutes(published);
    }

    private static ContextResult.RouteDelivered routeDelivered(
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

    private void reportRoutes(List<Delivery.Member> claims) {
        if (!claims.isEmpty()) {
            try {
                long nowMs = System.currentTimeMillis();
                String engineIp = claims.getFirst().route().prefillEndpoint().getIp();
                for (var claim : claims) {
                    reporter.reportBatchWaitTimeMs(
                            RoleType.PREFILL.name(),
                            engineIp,
                            Math.max(0L, nowMs - claim.route().enqueuedAtMs()),
                            claim.route().priority());
                }
                reporter.reportBatcherQueueSize(
                        RoleType.PREFILL.name(),
                        engineIp,
                        claims.getFirst().route().prefillEndpoint().queueDepth());
            } catch (Throwable monitoringFailure) {
                logFailure("Route delivery telemetry failed", 0L, monitoringFailure);
            }
        }
    }

    private RoutingResult<ContextResult.RouteDelivered, CapacityKey> commitEncoderRoute(
            RouteResult route) {
        var context = route.ctx();

        var routingContext = route.ctx().data().getRequestScheduler().currentRoutingContext(route);
        if (routingContext == null) return RoutingResult.closed();
        var request = context.data().getRequest();
        long hit =
                request.getEncoderCacheHitLen() == null
                        ? 0L
                        : Math.max(0L, request.getEncoderCacheHitLen());
        if (!route.encoderEndpoint()
                .commit(
                        route,
                        context.data().getRequestScheduler().encoderRequestLimit(),
                        Math.max(0L, request.getSeqLen() - hit)))
            return RoutingResult.blocked(
                    new CapacityKey(
                            RoleType.ENCODER,
                            route.encoderEndpoint().group(),
                            route.encoderEndpoint().ipPort()));
        if (!route.selectedEndpointsAvailable())
            return route.validateEndpoints(RoutingResult.closed());
        var claim =
                commit(
                        new Delivery.Member(route, null),
                        Delivery.SendMode.ROUTE_DECISION,
                        0L,
                        routingContext);
        if (claim == null) return RoutingResult.closed();
        return RoutingResult.success(
                new ContextResult.RouteDelivered(
                        claim,
                        context.snapshot().inactivityExpiresAtMs(),
                        System.currentTimeMillis()));
    }

    private void dispatchBatchGroup(Delivery batch) {
        com.google.common.base.Preconditions.checkArgument(
                !batch.members.isEmpty() && batch.batchId > 0L && batch.predictedMs >= 0L,
                "invalid batch metadata");
        Objects.requireNonNull(batch.reason, "reason");
        batch.executionPermit.submit(
                () -> deliverCommitted(batch),
                failure -> failBeforeSend(batch, List.of(), failure));
    }

    private void deliverCommitted(Delivery batch) {
        var members = new ArrayList<Delivery.Member>(batch.members.size());
        long prediction = batch.predictedMs;
        int startedMembers = 0;
        EngineRpcService.EnqueueBatchRequestPB payload;
        try {
            for (var member : batch.members) {
                var route = member.route();
                var state = member.owner().snapshot();
                if (!state.preparedDelivery() || state.route() != route) continue;
                if (!route.selectedEndpointsAvailable()) {
                    report(
                            new ContextResult.DeliveryNotStarted(
                                    member, null, null, true, System.currentTimeMillis()));
                    continue;
                }
                members.add(member);
            }
            if (members.isEmpty()) {
                batch.executionPermit.releaseIfNoResponse();
                var cleanup = batch.releaseMembers();
                if (cleanup != null) batch.members.getFirst().recordFailure(cleanup);
                return;
            }
            long payloadBytes = 16L;
            for (var member : members)
                payloadBytes += EngineDelivery.payloadSizeUpperBound(member.route());
            checkState(
                    batch.executionPermit.tryChargeBytes(payloadBytes),
                    "in-flight payload byte budget exhausted");
            payload = EngineDelivery.prepare(batch.batchId, members);
            int candidateCount = members.size();
            for (int index = 0; index < candidateCount; index++) {
                var member = members.get(index);
                try {
                    var started =
                            commit(member, Delivery.SendMode.BATCH_ENQUEUE, batch.batchId, null);
                    if (started != null) members.set(startedMembers++, started);
                } catch (Throwable failure) {
                    logFailure("Delivery commit failed", batch.batchId, failure);
                }
            }
            if (startedMembers != members.size())
                members.subList(startedMembers, members.size()).clear();
            if (startedMembers != candidateCount) payload = EngineDelivery.retain(payload, members);
            if (members.size() != batch.members.size() && !members.isEmpty()) {
                prediction =
                        PrefillTimePredictor.predictCommittedBatchMs(
                                batch.evaluator,
                                PrefillBatchFeatures.from(
                                        members,
                                        value -> value.route().seqLen(),
                                        value -> value.route().hitCache()));
            }
            for (Delivery.Member claim : members) {
                long now = System.currentTimeMillis();
                var data = claim.owner().data();
                data.getRequestScheduler()
                        .recordDeliveryPrediction(
                                new ContextResult.Prediction(
                                        claim,
                                        PrefillTimePredictor.predictedVisibilityDeadlineMs(
                                                batch.precedingWork,
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
                batch.members.getFirst().recordFailure(cleanup);
            }
        }
        if (call.response() == null) {
            batch.executionPermit.releaseIfNoResponse();
        }
        if (call.failure() != null) {
            report(batch.batchId, members, Collections.nCopies(members.size(), call.failure()));
        } else if (call.response() != null) {
            batch.executionPermit.observeResponse(
                    call.response(),
                    (response, failure) ->
                            report(
                                    batch.batchId,
                                    members,
                                    EngineDelivery.validate(
                                            batch.batchId, members, response, failure)));
        }
        reportDelivery(batch, members, prediction);
    }

    private void failBeforeSend(Delivery batch, List<Delivery.Member> claims, Throwable failure) {
        batch.executionPermit.releaseIfNoResponse();
        try {
            for (Member member : batch.members) {
                if (claims.stream().anyMatch(value -> value.route() == member.route())) {
                    continue;
                }
                try {
                    AbstractRequestScheduler.preparationFailed(
                            member.route(), String.valueOf(failure.getMessage()));
                } catch (Throwable ownerFailure) {
                    member.recordFailure(ownerFailure);
                    logFailure("Preparation failure notification", batch.batchId, ownerFailure);
                }
            }
        } finally {
            Throwable cleanup = batch.releaseMembers();
            if (cleanup != null) {
                batch.members.getFirst().recordFailure(cleanup);
            }
        }
        report(
                batch.batchId,
                claims,
                Collections.nCopies(claims.size(), Delivery.Result.notSent(failure)));
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
                    configService
                                    .loadBalanceConfig()
                                    .deliveryType(org.flexlb.dao.route.RequestPhase.GENERATION)
                            == DispatcherConfig.Type.BATCH,
                    "batch submission requires BATCH dispatcher configuration");
            var items = claims.stream().map(value -> value.route()).toList();
            var endpoint = items.getFirst().prefillEndpoint();
            try {
                logDispatch(batch.batchId, items, endpoint, prediction, batch.reason);
            } catch (Throwable failure) {
                logFailure("Dispatch logging", batch.batchId, failure);
            }
            long now = System.nanoTime();
            for (var item : items) {
                var data = item.ctx().data();
                data.setBatchDispatchedNanos(now);
                FlexlbTrace.setScheduleAttribute(
                        data.getTraceContext(), FlexlbTrace.BATCH_ID, batch.batchId);
                FlexlbTrace.setScheduleAttribute(
                        data.getTraceContext(), FlexlbTrace.BATCH_SIZE, (long) items.size());
                FlexlbTrace.setScheduleAttribute(
                        data.getTraceContext(), FlexlbTrace.DISPATCH_REASON, batch.reason);
            }
            String ip = endpoint.getIp();
            int port = endpoint.getGrpcPort();
            invoked = true;
            var response =
                    Objects.requireNonNull(
                            engine.batchEnqueueAsync(ip, port, request),
                            "EnqueueBatch client returned null future after invocation");
            return new BatchCall(response, null);
        } catch (Throwable failure) {
            // Before invocation even already-begun members were not sent; afterwards delivery is
            // uncertain.
            logFailure("Dispatch batch", batch.batchId, failure);
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
                    reporter.reportBatchWaitTimeMs(
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
                reporter.reportBatchDelivery(
                        RoleType.PREFILL.name(),
                        engineIp,
                        batch.batchId,
                        batch.reason,
                        claims.getFirst().route().prefillEndpoint().queueDepth(),
                        claims.size(),
                        hitTokens,
                        totalTokens,
                        prediction);
            } catch (Throwable failure) {
                logFailure("Delivery telemetry", batch.batchId, failure);
            }
        }
    }

    private record BatchCall(
            CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> response,
            Delivery.Result failure) {}
}

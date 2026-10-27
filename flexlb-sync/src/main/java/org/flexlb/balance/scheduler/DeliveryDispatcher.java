package org.flexlb.balance.scheduler;

import com.google.common.collect.MapMaker;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
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
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** Complete payload preparation precedes sending; Scheduler owns all request transitions. */
@Component
public class DeliveryDispatcher {
    static final long CLEANUP_TIMEOUT_MS = 5_000L;
    private final EngineGrpcClient grpcClient;
    private final DispatcherConfig.Type mode;
    private final ConfigService configService;
    private final DeliveryMetricsReporter telemetry;

    @Autowired
    public DeliveryDispatcher(
            EngineGrpcClient grpcClient,
            ConfigService configService,
            DeliveryMetricsReporter telemetry) {
        this.grpcClient = grpcClient;
        this.configService = Objects.requireNonNull(configService);
        this.mode = configService.loadBalanceConfig().getDispatcher().getType();
        this.telemetry = Objects.requireNonNull(telemetry);
    }

    public DispatcherConfig.Type mode() {
        return mode;
    }

    record Member(RequestRoute route, DecodeEndpoint.EngineDispatchPermit decode)
            implements AutoCloseable {
        private static final RequestTimeEstimation.AdmissionBlockSemantics DECODE_BLOCK =
                new RequestTimeEstimation.AdmissionBlockSemantics(
                        "DELIVERY_CAPACITY_DECODE_ENGINE",
                        RequestTimeEstimation.AfterProbeAdmission.UNAVAILABLE,
                        "DECODE_CAPACITY_SCOPE_UNKNOWN",
                        RoleType.DECODE);

        static CapacityBoundary.Attempt<Member> prepare(
                RequestRoute route, AdmissionCapacity capacity) {
            var endpoint = route.decodeEp();
            var reservation = route.decodeReservation();
            if (endpoint == null && reservation == null) {
                return CapacityBoundary.Attempt.accepted(new Member(route, null));
            }
            if (endpoint == null || reservation == null) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.failed(
                                new IllegalStateException(
                                        "Decode reservation is unavailable: request_id="
                                                + route.requestId())));
            }
            DecodeEndpoint.EngineDispatchPermit permit = null;
            try {
                var acquisition = endpoint.acquireDispatchPermit(reservation, capacity);
                switch (acquisition.status()) {
                    case ACQUIRED, ALREADY_ACCEPTED -> permit = acquisition.handle();
                    case CAPACITY_FULL -> {
                        return CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.unavailable(
                                        CapacityBoundary.Resource.DECODE_ENGINE, DECODE_BLOCK));
                    }
                    case NOT_OWNED, NOT_QUEUED -> {
                        return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
                    }
                    case ENDPOINT_RETIRED -> {
                        return CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.failed(
                                        new IllegalStateException(
                                                "Decode endpoint generation retired: request_id="
                                                        + route.requestId())));
                    }
                    case ALREADY_ACQUIRED -> {
                        return CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.failed(
                                        new IllegalStateException(
                                                "Decode dispatch permit already acquired:"
                                                        + " request_id="
                                                        + route.requestId())));
                    }
                }
                return CapacityBoundary.Attempt.accepted(new Member(route, permit));
            } catch (Throwable failure) {
                if (permit != null) {
                    failure = Failures.run(failure, permit::release);
                }
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(failure));
            }
        }

        @Override
        public void close() {
            if (decode != null) {
                decode.release();
            }
        }
    }

    /** Entering deliverBatch transfers all temporary resources, including on rejection. */
    record Batch(
            long batchId,
            List<Member> members,
            WorkSnapshot precedingWork,
            long predictedMs,
            PrefillTimePredictor.Evaluator evaluator,
            String reason,
            int remainingQueueDepth,
            PrefillState.CommittedHandoff handoff,
            DeliveryExecution.DeliveryPermit executionPermit) {
        Batch {
            members = List.copyOf(members);
            checkArgument(
                    !members.isEmpty()
                            && batchId > 0
                            && predictedMs >= 0
                            && remainingQueueDepth >= 0,
                    "invalid batch metadata");
            Objects.requireNonNull(precedingWork);
            Objects.requireNonNull(evaluator);
            Objects.requireNonNull(reason);
            Objects.requireNonNull(executionPermit);
        }
    }

    private record BatchCall(
            List<DeliveryClaim> pending,
            List<DeliveryClaim> skipped,
            CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> response,
            DeliveryResult failure) {}

    void deliverBatch(Batch batch) {
        batch.executionPermit()
                .submit(
                        () -> deliverCommitted(batch),
                        failure -> failBeforeSend(batch, List.of(), failure));
    }

    private void deliverCommitted(Batch batch) {
        var claims = new ArrayList<DeliveryClaim>(batch.members().size());
        long prediction = batch.predictedMs();
        try {
            claim(batch, claims);
            if (claims.size() != batch.members().size() && !claims.isEmpty()) {
                prediction =
                        PrefillPredictionBoundary.predictCommittedBatchMs(
                                batch.evaluator(),
                                PrefillBatchFeatures.from(
                                        claims,
                                        value -> value.item().seqLen(),
                                        value -> value.item().hitCache()));
            }
            for (DeliveryClaim claim : claims) {
                long now = System.currentTimeMillis();
                claim.owner()
                        .data()
                        .getScheduler()
                        .onDeliveryEvidence(
                                claim,
                                new ContextResult.Prediction(
                                        claim,
                                        PrefillPredictionBoundary.predictedVisibilityDeadlineMs(
                                                batch.precedingWork(),
                                                prediction,
                                                claim.owner()
                                                        .data()
                                                        .getConfig()
                                                        .getRequestLifecycle()
                                                        .getDecision()
                                                        .getLifetime(),
                                                now),
                                        now));
            }
        } catch (Throwable failure) {
            failBeforeSend(batch, claims, failure);
            return;
        }
        BatchCall call;
        try {
            call = dispatchBatch(batch, claims, prediction);
        } finally {
            Throwable cleanup = closeCommitted(batch.members(), batch.handoff());
            if (cleanup != null) {
                batch.members()
                        .getFirst()
                        .route()
                        .ctx()
                        .data()
                        .getScheduler()
                        .runtime
                        .recordFailure(cleanup);
            }
        }
        if (call.response() == null) {
            batch.executionPermit().releaseIfNoResponse();
        }
        report(
                batch.batchId(),
                call.skipped(),
                Collections.nCopies(
                        call.skipped().size(),
                        DeliveryResult.notSent(
                                new CancellationException("delivery abandoned before send"))));
        if (call.failure() != null) {
            report(
                    batch.batchId(),
                    call.pending(),
                    Collections.nCopies(call.pending().size(), call.failure()));
        } else if (call.response() != null) {
            batch.executionPermit()
                    .observeResponse(
                            call.response(),
                            (response, failure) ->
                                    report(
                                            batch.batchId(),
                                            call.pending(),
                                            BatchCodec.validate(
                                                    batch.batchId(),
                                                    call.pending(),
                                                    response,
                                                    failure)));
        }
        if (!claims.isEmpty()) {
            try {
                telemetry.reportDelivery(
                        batch.batchId(),
                        batch.reason(),
                        batch.remainingQueueDepth(),
                        claims.stream().map(value -> value.item()).toList(),
                        prediction);
            } catch (Throwable failure) {
                logFailure("Delivery telemetry", batch.batchId(), failure);
            }
        }
    }

    private void claim(Batch batch, List<DeliveryClaim> claims) {
        for (Member member : batch.members()) {
            try {
                var claim =
                        member.route()
                                .ctx()
                                .data()
                                .getScheduler()
                                .claimDelivery(
                                        member.route(),
                                        DeliveryClaimKind.BATCH_ENQUEUE,
                                        batch.batchId(),
                                        member.decode());
                if (claim != null) {
                    claims.add(claim);
                }
            } catch (Throwable failure) {
                try {
                    member.route()
                            .ctx()
                            .data()
                            .getScheduler()
                            .onDeliveryEvidence(
                                    null,
                                    new ContextResult.PreparationFailure(
                                            null,
                                            member.route(),
                                            String.valueOf(failure.getMessage()),
                                            System.currentTimeMillis()));
                } catch (Throwable ownerFailure) {
                    member.route().ctx().data().getScheduler().runtime.recordFailure(ownerFailure);
                    logFailure("Claim failure notification", batch.batchId(), ownerFailure);
                }
            }
        }
    }

    private BatchCall dispatchBatch(Batch batch, List<DeliveryClaim> claims, long prediction) {
        var skipped = new ArrayList<DeliveryClaim>();
        var sending = new ArrayList<DeliveryClaim>(claims.size());
        if (claims.isEmpty()) {
            return new BatchCall(List.of(), List.of(), null, null);
        }
        boolean invoked = false;
        try {
            long payloadBytes = 16L;
            for (var claim : claims) {
                payloadBytes += claim.item().batchPayloadSizeUpperBound();
            }
            checkState(
                    batch.executionPermit().tryChargeBytes(payloadBytes),
                    "in-flight payload byte budget exhausted");
            var request = BatchCodec.prepare(batch.batchId(), claims);
            checkState(
                    configService.loadBalanceConfig().getDispatcher().getType()
                            == DispatcherConfig.Type.BATCH,
                    "batch submission requires BATCH dispatcher configuration");
            for (DeliveryClaim claim : claims) {
                if (claim.owner().data().getScheduler().tryStartSend(claim)) {
                    sending.add(claim);
                } else {
                    skipped.add(claim);
                }
            }
            if (sending.isEmpty()) {
                return new BatchCall(List.of(), List.copyOf(skipped), null, null);
            }
            if (sending.size() != claims.size()) {
                request = BatchCodec.retain(request, sending);
            }
            var items = sending.stream().map(value -> value.item()).toList();
            var endpoint = items.getFirst().prefillEp();
            try {
                logDispatch(batch.batchId(), items, endpoint, prediction, batch.reason());
            } catch (Throwable failure) {
                logFailure("Dispatch logging", batch.batchId(), failure);
            }
            long now = System.nanoTime();
            for (var item : items) {
                item.ctx().data().setBatchDispatchedNanos(now);
                FlexlbTrace.setScheduleAttribute(
                        item.ctx().data().getTraceContext(), FlexlbTrace.BATCH_ID, batch.batchId());
                FlexlbTrace.setScheduleAttribute(
                        item.ctx().data().getTraceContext(),
                        FlexlbTrace.BATCH_SIZE,
                        (long) items.size());
                FlexlbTrace.setScheduleAttribute(
                        item.ctx().data().getTraceContext(),
                        FlexlbTrace.DISPATCH_REASON,
                        batch.reason());
            }
            String ip = endpoint.getIp();
            int port = endpoint.getGrpcPort();
            var pending = List.copyOf(sending);
            invoked = true;
            var response =
                    Objects.requireNonNull(
                            grpcClient.batchEnqueueAsync(ip, port, request),
                            "EnqueueBatch client returned null future after invocation");
            return new BatchCall(pending, List.copyOf(skipped), response, null);
        } catch (Throwable failure) {
            // Before invocation even already-begun members were not sent; afterwards delivery is
            // uncertain.
            var unresolved = new ArrayList<DeliveryClaim>(claims);
            unresolved.removeAll(skipped);
            logFailure("Dispatch batch", batch.batchId(), failure);
            return new BatchCall(
                    List.copyOf(unresolved),
                    List.copyOf(skipped),
                    null,
                    invoked ? DeliveryResult.uncertain(failure) : DeliveryResult.notSent(failure));
        }
    }

    private void report(long batchId, List<DeliveryClaim> claims, List<DeliveryResult> results) {
        long responseNanos = System.nanoTime();
        for (int i = 0; i < claims.size(); i++) {
            var claim = claims.get(i);
            var result = results.get(i);
            if (result.status() == DeliveryResult.Status.DELIVERED
                    || result.status() == DeliveryResult.Status.PREFILL_REJECTED) {
                try {
                    FlexlbTrace.setScheduleDuration(
                            claim.owner().data().getTraceContext(),
                            FlexlbTrace.ENQUEUE_BATCH_MS,
                            claim.owner().data().getBatchDispatchedNanos(),
                            responseNanos);
                } catch (Throwable failure) {
                    logFailure("Response trace", batchId, failure);
                }
            }
            try {
                claim.owner()
                        .data()
                        .getScheduler()
                        .onDeliveryEvidence(
                                claim,
                                new ContextResult.TransportResult(
                                        claim, result, System.currentTimeMillis()));
            } catch (Throwable failure) {
                claim.owner().data().getScheduler().runtime.recordFailure(failure);
                logFailure(
                        "Delivery completion request_id=" + claim.item().requestId(),
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

    private void failBeforeSend(Batch batch, List<DeliveryClaim> claims, Throwable failure) {
        batch.executionPermit().releaseIfNoResponse();
        try {
            for (Member member : batch.members()) {
                if (claims.stream().anyMatch(value -> value.item() == member.route())) {
                    continue;
                }
                try {
                    member.route()
                            .ctx()
                            .data()
                            .getScheduler()
                            .onDeliveryEvidence(
                                    null,
                                    new ContextResult.PreparationFailure(
                                            null,
                                            member.route(),
                                            String.valueOf(failure.getMessage()),
                                            System.currentTimeMillis()));
                } catch (Throwable ownerFailure) {
                    member.route().ctx().data().getScheduler().runtime.recordFailure(ownerFailure);
                    logFailure("Preparation failure notification", batch.batchId(), ownerFailure);
                }
            }
        } finally {
            Throwable cleanup = closeCommitted(batch.members(), batch.handoff());
            if (cleanup != null) {
                batch.members()
                        .getFirst()
                        .route()
                        .ctx()
                        .data()
                        .getScheduler()
                        .runtime
                        .recordFailure(cleanup);
            }
        }
        report(
                batch.batchId(),
                claims,
                Collections.nCopies(claims.size(), DeliveryResult.notSent(failure)));
    }

    static Throwable closeCommitted(List<Member> members, PrefillState.CommittedHandoff handoff) {
        Throwable failure = null;
        for (Member member : members) {
            failure = Failures.append(failure, Failures.close(member));
        }
        return Failures.append(failure, handoff == null ? null : Failures.close(handoff));
    }

    private static void logFailure(String operation, long batchId, Throwable failure) {
        try {
            Logger.error("{} batch_id={}", operation, batchId, failure);
        } catch (Throwable ignored) {
        }
    }

    private void logDispatch(
            long batchId,
            List<RequestRoute> items,
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
            RequestRoute item = items.get(i);
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

        RequestRoute head = items.get(0);
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

    private static final int MAX_ATTEMPTS = 8;

    enum StopResult {
        NOT_SENT,
        REMOTE_CLEANED,
        UNRESOLVED,
        UNSUPPORTED,
        ALREADY_SETTLED,
        STALE
    }

    // MapMaker weak keys use identity: equal-valued claims must not share an operation.
    private final ConcurrentMap<Object, StopTask> sessions = new MapMaker().weakKeys().makeMap();

    /** Accept the entire stop operation; callers do not drive retries or sender completion. */
    CompletableFuture<StopResult> stop(DeliveryClaim claim) {
        var runtime = claim.owner().data().getScheduler().runtime;
        for (; ; ) {
            if (runtime.signals().delivery(claim).toCompletableFuture().isDone()) {
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

    public CompletableFuture<PreemptionRegistration.Reply> preempt(
            PreemptionRegistration registration,
            DecodeEndpoint endpoint,
            String detail,
            long timeoutMs) {
        var runtime = registration.owner().data().getScheduler().runtime;
        var delivery = registration.owner().snapshot().delivery();
        var claim = delivery == null ? null : delivery.identity();
        var answer = new CompletableFuture<PreemptionRegistration.Reply>();
        long expiresAt = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        var timeout =
                runtime.scheduleTimeout(
                        () -> answer.complete(PreemptionRegistration.Reply.UNKNOWN),
                        Math.max(1L, timeoutMs));
        answer.whenComplete((ignored, failure) -> timeout.cancel());
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

    private StopTask record(Object key, DeliveryClaim claim, RequestContext context) {
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
            created.runtime
                    .signals()
                    .delivery(claim)
                    .whenComplete((ignored, failure) -> created.close(StopResult.ALREADY_SETTLED));
        }
        return created;
    }

    /** Drain late RPC callbacks before Scheduler closes its fact ingress. */
    void awaitStops(SchedulerRuntime runtime) {
        Throwable problem = null;
        long deadline =
                System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS + 1_000L);
        for (var task : sessions.values()) {
            if (task.runtime != runtime) {
                continue;
            }
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

    private record PreemptionCall(
            PreemptionRegistration registration,
            DecodeEndpoint endpoint,
            String detail,
            long expiresAt,
            CompletableFuture<PreemptionRegistration.Reply> response) {}

    private final class StopTask implements Runnable {
        private final Object key;
        private final DeliveryClaim claim;
        private final RequestContext context;
        private final SchedulerRuntime runtime;
        private final CompletableFuture<StopResult> finished = new CompletableFuture<>();
        private final CompletableFuture<PreemptionRegistration.Reply> firstReply =
                new CompletableFuture<>();
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

        StopTask(Object key, DeliveryClaim claim, RequestContext context) {
            this.key = key;
            this.claim = claim;
            this.context = context;
            this.runtime = context.data().getScheduler().runtime;
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
                accepted = runtime.executeStopWork(context, this);
            } catch (RuntimeException | Error failure) {
                finished.completeExceptionally(failure);
                close(StopResult.UNRESOLVED);
                runtime.recordFailure(failure);
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
                firstReply.complete(PreemptionRegistration.Reply.UNKNOWN);
                runtime.recordFailure(failure);
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
                                runtime.engineClient.cancelAsync(
                                        target == null
                                                ? claim.item().prefillEp().getIp()
                                                : target.prefillIp(),
                                        target == null
                                                ? claim.item().prefillEp().getGrpcPort()
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
                    runtime.recordFailure(failed);
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
                var scheduler = context.data().getScheduler();
                if (operation != null) {
                    var event =
                            AbstractRequestScheduler.PreemptionEvent.receipt(
                                    operation.registration(),
                                    operation.endpoint(),
                                    reply,
                                    operation.detail(),
                                    claim);
                    progress =
                            event.progress(scheduler.onPreemption(operation.registration(), event));
                } else if (claim != null) {
                    scheduler.onDeliveryEvidence(
                            claim,
                            AbstractRequestScheduler.PreemptionEvent.acknowledgement(claim, reply));
                }
                evidence = reply;
            } catch (RuntimeException | Error failed) {
                runtime.recordFailure(failed);
            } finally {
                boolean unsupported;
                synchronized (this) {
                    unsupported =
                            evidence == EngineGrpcClient.CancelReply.UNSUPPORTED
                                    && terminal == null;
                    if (evidence == EngineGrpcClient.CancelReply.REQUEST_CLEANED) {
                        terminal = StopResult.REMOTE_CLEANED;
                    } else if (unsupported) {
                        terminal = StopResult.UNSUPPORTED;
                    }
                }
                if (unsupported) {
                    try {
                        Logger.warn(
                                "[delivery-cleanup] remote cancellation unsupported: request_id={};"
                                        + " waiting for Worker completion or retirement",
                                context.data().getRequestId());
                    } catch (RuntimeException | Error ignored) {
                    }
                }
                if (operation != null) {
                    operation.response().complete(progress);
                }
                firstReply.complete(progress);
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

        private void finish(StopResult result) {
            if (finished.complete(result) && result == StopResult.UNRESOLVED && stopRequested) {
                runtime.recordFailure(
                        new IllegalStateException(
                                "delivery stop ended without release proof: request_id="
                                        + context.data().getRequestId()));
            }
            if (result == StopResult.NOT_SENT || result == StopResult.UNSUPPORTED) {
                firstReply.complete(PreemptionRegistration.Reply.UNKNOWN);
            }
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
                firstReply.complete(PreemptionRegistration.Reply.UNKNOWN);
                var operation = preemption;
                if (operation != null) {
                    operation.response().complete(PreemptionRegistration.Reply.UNKNOWN);
                }
            }
        }
    }
}

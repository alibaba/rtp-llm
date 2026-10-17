package org.flexlb.balance.scheduler;

import com.google.common.collect.MapMaker;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;

/** One Dispatcher-owned stop record per exact delivery; events advance it without restarting a runner. */
final class DeliveryCleanup {
    static final long TIMEOUT_MS = 5_000L;
    private static final int MAX_ATTEMPTS = 8;
    enum Result { NOT_SENT, REMOTE_CLEANED, UNRESOLVED, UNSUPPORTED, ALREADY_SETTLED, STALE }
    private final ConcurrentMap<Object, StopRecord> sessions = new MapMaker().weakKeys().makeMap();

    CompletableFuture<Result> start(DeliveryClaim claim, EngineGrpcClient client, SchedulerRuntime runtime) {
        if (runtime.signals().delivery(claim).toCompletableFuture().isDone()) {
            return CompletableFuture.completedFuture(Result.ALREADY_SETTLED);
        }
        var delivery = claim.owner().snapshot().delivery();
        if (delivery == null || delivery.identity() != claim) { return CompletableFuture.completedFuture(Result.STALE); }
        var record = record(claim, claim, claim.owner(), client, runtime);
        synchronized (record) {
            if (!record.stopSubmitted) {
                record.stopSubmitted = true;
                record.post(() -> {
                    record.stopRequested = true;
                    record.deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MS);
                });
            }
        }
        return record.finished;
    }

    CompletableFuture<EngineGrpcClient.CancelReply> preempt(PreemptionRegistration registration,
            EngineGrpcClient client, SchedulerRuntime runtime, long timeoutMs) {
        var delivery = registration.owner().snapshot().delivery();
        var claim = delivery == null ? null : delivery.identity();
        var record = record(claim == null ? registration : claim, claim, registration.owner(), client, runtime);
        var answer = new CompletableFuture<EngineGrpcClient.CancelReply>();
        record.post(() -> {
            if (!record.stopRequested && record.preemption != registration) {
                record.preemption = registration;
                record.firstReply = new CompletableFuture<>();
                record.attempts = 0;
                record.timeoutMs = timeoutMs;
            }
            record.firstReply.whenComplete((reply, failure) -> answer.complete(
                    failure == null ? reply : EngineGrpcClient.CancelReply.FAILED));
        }).whenComplete((ignored, failure) -> {
            if (failure != null || record.closed) { answer.complete(EngineGrpcClient.CancelReply.FAILED); }
        });
        return answer;
    }

    private StopRecord record(Object key, DeliveryClaim claim, RequestContext context,
                              EngineGrpcClient client, SchedulerRuntime runtime) {
        var existing = sessions.get(key);
        if (existing != null) { return existing; }
        var created = new StopRecord(key, claim, context, client, runtime);
        var previous = sessions.putIfAbsent(key, created);
        if (previous != null) { return previous; }
        created.settled.whenComplete((ignored, failure) -> created.post(() -> created.close(Result.ALREADY_SETTLED)));
        return created;
    }

    void sent(DeliveryClaim claim) {
        var record = sessions.get(claim);
        if (record != null) { record.post(() -> { }); }
    }

    /** Drain accepted events and late RPC replies while Scheduler still accepts evidence. */
    void await(SchedulerRuntime runtime) {
        Throwable problem = null;
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MS + 1_000L);
        for (var record : sessions.values()) {
            if (record.runtime != runtime) { continue; }
            var drained = new CompletableFuture<Void>();
            record.post(() -> record.draining = drained);
            if (record.closed) { continue; }
            try { drained.get(Math.max(0L, deadline - System.nanoTime()), TimeUnit.NANOSECONDS); }
            catch (Exception failure) {
                if (failure instanceof InterruptedException) { Thread.currentThread().interrupt(); }
                record.closed = true; // Explicit shutdown cutoff; never fabricate resource-release proof.
                record.finished.completeExceptionally(failure);
                record.firstReply.complete(EngineGrpcClient.CancelReply.FAILED);
                sessions.remove(record.key, record);
                problem = Failures.append(problem, failure);
            }
        }
        Failures.rethrow(problem, "delivery stop callbacks did not drain");
    }

    static void publish(DeliveryClaim claim, ContextResult.DeliveryEvidence evidence) {
        Throwable failure = null;
        for (int attempt = 0; attempt < 3; attempt++) {
            try { claim.owner().data().getScheduler().onDeliveryEvidence(claim, evidence); return; }
            catch (RuntimeException failed) { failure = Failures.append(failure, failed); }
        }
        throw Failures.propagate(failure, "delivery fact could not be applied");
    }

    private final class StopRecord {
        private final Object key;
        private final DeliveryClaim claim;
        private final RequestContext context;
        private final EngineGrpcClient client;
        private final SchedulerRuntime runtime;
        private final CompletableFuture<Void> settled;
        private final CompletableFuture<Result> finished = new CompletableFuture<>();
        private volatile CompletableFuture<EngineGrpcClient.CancelReply> firstReply = new CompletableFuture<>();
        private CompletableFuture<Void> tail = CompletableFuture.completedFuture(null), draining;
        private CompletableFuture<EngineGrpcClient.CancelReply> rpc;
        private RequestTaskScheduler.Cancellation alarm;
        private PreemptionRegistration preemption;
        private ContextResult.SendState lastSend = ContextResult.SendState.NOT_STARTED;
        private Result terminal;
        private volatile boolean closed;
        private boolean stopSubmitted, stopRequested;
        private int attempts, pending;
        private long deadline, readyAt, timeoutMs = 500L;

        StopRecord(Object key, DeliveryClaim claim, RequestContext context, EngineGrpcClient client, SchedulerRuntime runtime) {
            this.key = key; this.claim = claim; this.context = context; this.client = client; this.runtime = runtime;
            settled = claim == null ? new CompletableFuture<>() : runtime.signals().delivery(claim).toCompletableFuture();
        }

        /** Only this ordered event stream changes execution state; no effects run under the record monitor. */
        private synchronized CompletableFuture<Void> post(Runnable event) {
            if (closed) { return CompletableFuture.completedFuture(null); }
            tail = tail.handleAsync((ignored, previousFailure) -> {
                if (!closed) {
                    if (alarm != null) { alarm.cancel(); alarm = null; }
                    event.run();
                    advance();
                }
                if (draining != null && (closed || pending == 0 && alarm == null)) { draining.complete(null); }
                return (Void) null;
            }, task -> runtime.executeStopWork(context, task)).whenComplete((ignored, failure) -> {
                if (failure == null) { return; }
                terminal = Result.UNRESOLVED;
                finished.completeExceptionally(failure);
                firstReply.complete(EngineGrpcClient.CancelReply.FAILED);
                runtime.recordFailure(failure);
                if (draining != null && pending == 0) { draining.complete(null); }
            });
            return tail;
        }

        private void advance() {
            if (closed) { return; }
            if (settled.isDone()) { close(Result.ALREADY_SETTLED); return; }
            var delivery = context.snapshot().delivery();
            if (claim != null && (delivery == null || delivery.identity() != claim)) { close(Result.STALE); return; }
            if (delivery != null && (delivery.send() == ContextResult.SendState.NOT_SENT
                    || delivery.send() == ContextResult.SendState.ROUTE_CLOSED)) {
                firstReply.complete(EngineGrpcClient.CancelReply.FAILED);
                finish(Result.NOT_SENT); return;
            }
            if (delivery != null && delivery.prefillProof() && delivery.decodeProof()) { terminal = Result.REMOTE_CLEANED; }
            if (terminal != null) {
                firstReply.complete(terminal == Result.REMOTE_CLEANED ? EngineGrpcClient.CancelReply.REQUEST_CLEANED
                        : terminal == Result.UNSUPPORTED ? EngineGrpcClient.CancelReply.UNSUPPORTED : EngineGrpcClient.CancelReply.FAILED);
                finish(terminal); return;
            }
            if (!stopRequested) {
                if (attempts == 0) { requestProof(); }
                return;
            }
            long now = System.nanoTime();
            var send = delivery == null ? ContextResult.SendState.UNKNOWN : delivery.send();
            boolean sending = send == ContextResult.SendState.NOT_STARTED || send == ContextResult.SendState.SENDING;
            if (!sending && (lastSend == ContextResult.SendState.NOT_STARTED || lastSend == ContextResult.SendState.SENDING)) {
                // One post-send budget, sharing the same attempt count with the pre-ACK cancellation.
                deadline = now + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MS);
                lastSend = send;
                if (rpc == null) { readyAt = 0L; }
            }
            if (attempts >= MAX_ATTEMPTS && rpc == null || now >= deadline) { finish(Result.UNRESOLVED); return; }
            if (send == ContextResult.SendState.NOT_STARTED || sending && attempts > 0) { wakeAt(deadline); return; }
            if (rpc != null) {
                if (rpc.isDone()) { return; } // Its queued reply owns fact publication, even when the timer fires first.
                if (now < readyAt) { wakeAt(Math.min(readyAt, deadline)); return; }
                rpc = null;
                readyAt = now + TimeUnit.MILLISECONDS.toNanos(Math.min(400L, 25L << (attempts - 1)));
            }
            if (now < readyAt) { wakeAt(Math.min(readyAt, deadline)); return; }
            if (attempts >= MAX_ATTEMPTS) { finish(Result.UNRESOLVED); return; }
            requestProof();
        }

        private void requestProof() {
            var state = context.snapshot();
            var reason = state.cancellation() != null ? state.cancellation().reason()
                    : state.delivery() != null && state.delivery().abandonment() != null ? state.delivery().abandonment()
                    : CancelReason.PRIORITY_PREEMPTED;
            var target = preemption == null ? null : preemption.cancelTarget();
            var response = firstReply;
            long timeout = attempts++ == 0 ? timeoutMs : 500L;
            lastSend = state.delivery() == null ? ContextResult.SendState.UNKNOWN : state.delivery().send();
            pending++;
            try {
                var remote = Objects.requireNonNull(client.cancelAsync(
                        target == null ? claim.item().prefillEp().getIp() : target.prefillIp(),
                        target == null ? claim.item().prefillEp().getGrpcPort() : target.prefillGrpcPort(),
                        reason.toRequest(context.data().getRequestId()), timeout), "cancel future");
                rpc = remote;
                readyAt = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeout);
                remote.whenComplete((reply, failure) -> post(() -> receive(remote, response,
                        failure == null && reply != null ? reply : EngineGrpcClient.CancelReply.FAILED)));
            } catch (RuntimeException failed) { receive(null, response, EngineGrpcClient.CancelReply.FAILED); }
            if (stopRequested) { wakeAt(Math.min(readyAt, deadline)); }
        }

        private void receive(CompletableFuture<EngineGrpcClient.CancelReply> remote,
                             CompletableFuture<EngineGrpcClient.CancelReply> response, EngineGrpcClient.CancelReply reply) {
            try {
                if (claim != null) { publish(claim, new ContextResult.CleanupAcknowledgement(claim,
                        switch (reply) {
                            case REQUEST_CLEANED -> ContextResult.CancelEvidenceKind.CLEANED;
                            case REQUEST_FENCED -> ContextResult.CancelEvidenceKind.FENCED;
                            case ACCEPTED -> ContextResult.CancelEvidenceKind.ACKNOWLEDGED;
                            case NOT_FOUND -> ContextResult.CancelEvidenceKind.NOT_FOUND;
                            case UNSUPPORTED -> ContextResult.CancelEvidenceKind.UNSUPPORTED;
                            case FAILED -> ContextResult.CancelEvidenceKind.UNKNOWN;
                        }, System.currentTimeMillis())); }
                if (reply == EngineGrpcClient.CancelReply.REQUEST_CLEANED) { terminal = Result.REMOTE_CLEANED; }
                if (reply == EngineGrpcClient.CancelReply.UNSUPPORTED && terminal == null) {
                    terminal = Result.UNSUPPORTED;
                    try { Logger.warn("[delivery-cleanup] remote cancellation unsupported: request_id={}; waiting for Worker completion or retirement",
                            context.data().getRequestId()); } catch (RuntimeException ignored) { }
                }
                if (rpc == remote) {
                    rpc = null;
                    readyAt = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.min(400L, 25L << (attempts - 1)));
                }
                response.complete(reply);
            } finally { pending--; }
            if (claim == null && pending == 0) { close(Result.UNRESOLVED); }
        }

        private void wakeAt(long at) {
            for (int attempt = 0; ; attempt++) {
                try {
                    alarm = runtime.scheduleTimeout(() -> post(() -> { }),
                            Math.max(1L, TimeUnit.NANOSECONDS.toMillis(at - System.nanoTime())));
                    return;
                } catch (RuntimeException rejected) { if (attempt == 2) { throw rejected; } }
            }
        }

        private void finish(Result result) {
            if (finished.complete(result) && result == Result.UNRESOLVED && stopRequested) {
                runtime.recordFailure(new IllegalStateException("delivery stop ended without release proof: request_id=" + context.data().getRequestId()));
            }
        }

        private void close(Result result) {
            closed = true;
            finish(result);
            firstReply.complete(EngineGrpcClient.CancelReply.FAILED);
            sessions.remove(key, this);
        }
    }
}

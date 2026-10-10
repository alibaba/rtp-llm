package org.flexlb.balance.scheduler;

import com.google.common.collect.MapMaker;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Bounded transport sessions. Only Scheduler settles resources or publishes frontend responses. */
final class DeliveryCleanup {
    static final long TIMEOUT_MS = 5_000L;
    private static final long ACK_TIMEOUT_MS = 500L;
    private static final int MAX_ATTEMPTS = 8;
    private final ConcurrentMap<DeliveryClaim, CompletableFuture<Void>> sessions = new MapMaker().weakKeys().makeMap();

    CompletableFuture<Void> start(DeliveryClaim claim, EngineCancelChannel channel, SchedulerRuntime runtime) {
        var found = sessions.get(claim);
        if (found != null) { return found; }
        var finished = new CompletableFuture<Void>();
        var previous = sessions.putIfAbsent(claim, finished);
        if (previous != null) { return previous; }
        var settled = runtime.signals().delivery(claim).toCompletableFuture();
        settled.whenComplete((ignored, failure) -> sessions.remove(claim, finished));
        if (settled.isDone()) { finished.complete(null); return finished; }
        var session = new Session(claim, channel, runtime, settled, finished);
        try { runtime.deliveryCleanupExecutor.execute(session); }
        catch (RuntimeException | Error rejected) {
            if (settled.isDone()) { finished.complete(null); }
            else { runtime.recordFailure(rejected); finished.completeExceptionally(rejected); }
        }
        return finished;
    }

    private record Facts(CancelReason reason, boolean needsRemote, boolean remoteSettled) { }
    private static final class Session implements Runnable {
        private final DeliveryClaim claim;
        private final EngineCancelChannel channel;
        private final SchedulerRuntime runtime;
        private final CompletableFuture<Void> settled;
        private final CompletableFuture<Void> finished;
        private final CompletableFuture<Void> unsupported = new CompletableFuture<>();
        private long deadline;

        Session(DeliveryClaim claim, EngineCancelChannel channel, SchedulerRuntime runtime,
                CompletableFuture<Void> settled, CompletableFuture<Void> finished) {
            this.claim = claim;
            this.channel = channel;
            this.runtime = runtime;
            this.settled = settled;
            this.finished = finished;
        }

        @Override public void run() {
            deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MS);
            try {
                for (int attempt = 1; !settled.isDone(); attempt++) {
                    remaining();
                    Facts evidence = facts();
                    if (unsupported.isDone() || evidence.remoteSettled() || !evidence.needsRemote()) {
                        settled.get(remaining(), TimeUnit.NANOSECONDS);
                        break;
                    }
                    EngineCancelChannel.CancelAck reply = null;
                    Throwable lastFailure = null;
                    try { reply = requestProof(evidence.reason()); }
                    catch (ExecutionException | TimeoutException | RuntimeException failed) { lastFailure = failed; }
                    if (settled.isDone()) { break; }
                    if (unsupported.isDone() || facts().remoteSettled()) { continue; }
                    if (attempt >= MAX_ATTEMPTS) {
                        throw new IllegalStateException("cancel lacks cleanup proof: request_id=" + claim.item().requestId() + " ack=" + reply, lastFailure);
                    }
                    try {
                        CompletableFuture.anyOf(settled, unsupported).get(
                                Math.min(TimeUnit.MILLISECONDS.toNanos(Math.min(400L, 25L << (attempt - 1))), remaining()),
                                TimeUnit.NANOSECONDS);
                    } catch (TimeoutException retryDelay) { }
                }
                finished.complete(null);
            } catch (Throwable failure) {
                if (failure instanceof InterruptedException) { Thread.currentThread().interrupt(); }
                if (!settled.isDone()) { runtime.recordFailure(failure); }
                finished.completeExceptionally(failure);
            }
        }

        private EngineCancelChannel.CancelAck requestProof(CancelReason reason) throws Exception {
            var remote = Objects.requireNonNull(channel.cancel(
                    new CancelTarget(claim.item().prefillEp().getIp(), claim.item().prefillEp().getGrpcPort()),
                    claim.item().requestId(), reason, ACK_TIMEOUT_MS), "cancel future");
            var applied = new CompletableFuture<EngineCancelChannel.CancelAck>();
            remote.whenComplete((reply, failure) -> {
                if (failure != null) { applied.completeExceptionally(failure); }
                else if (reply == null) { applied.complete(null); }
                else { apply(reply, applied); }
            });
            CompletableFuture.anyOf(remote, settled, unsupported).get(Math.min(TimeUnit.MILLISECONDS.toNanos(ACK_TIMEOUT_MS), remaining()),
                    TimeUnit.NANOSECONDS);
            CompletableFuture.anyOf(applied, settled, unsupported).get(remaining(), TimeUnit.NANOSECONDS);
            return applied.isDone() && !applied.isCompletedExceptionally() ? applied.getNow(null) : null;
        }

        /** Late replies retain their own accepted request-work owner after this session stops retrying. */
        private void apply(EngineCancelChannel.CancelAck reply, CompletableFuture<EngineCancelChannel.CancelAck> applied) {
            var accepted = runtime.tryRegisterRequestWork(claim.owner());
            if (accepted == null) { applied.complete(reply); return; }
            try (accepted) {
                accepted.submit(() -> {
                    try {
                        claim.owner().data().getScheduler().onDeliveryEvidence(claim,
                                new ContextResult.CleanupAcknowledgement(claim, switch (reply) {
                                    case REQUEST_CLEANED -> ContextResult.CancelEvidenceKind.CLEANED;
                                    case REQUEST_FENCED -> ContextResult.CancelEvidenceKind.FENCED;
                                    case ACCEPTED -> ContextResult.CancelEvidenceKind.ACKNOWLEDGED;
                                    case FAILED -> ContextResult.CancelEvidenceKind.UNKNOWN;
                                    case NOT_FOUND -> ContextResult.CancelEvidenceKind.NOT_FOUND;
                                    case UNSUPPORTED -> ContextResult.CancelEvidenceKind.UNSUPPORTED;
                                }, System.currentTimeMillis()));
                        if (reply == EngineCancelChannel.CancelAck.UNSUPPORTED && unsupported.complete(null)) {
                            try { Logger.warn("[delivery-cleanup] remote cancellation unsupported: request_id={}, worker={}:{}; "
                                            + "active cancellation retries stopped; waiting for Worker completion or retirement",
                                    claim.item().requestId(), claim.item().prefillEp().getIp(), claim.item().prefillEp().getGrpcPort()); }
                            catch (Throwable ignored) { }
                        }
                        applied.complete(reply);
                    } catch (Throwable failure) { applied.completeExceptionally(failure); throw failure; }
                });
            } catch (Throwable failure) { applied.completeExceptionally(failure); runtime.recordFailure(failure); }
        }

        private Facts facts() {
            var snapshot = claim.owner().snapshot();
            var delivery = snapshot.delivery();
            if (delivery == null || delivery.identity() != claim) { return new Facts(null, false, true); }
            var send = delivery.send();
            var reason = delivery.abandonment() != null ? delivery.abandonment()
                    : snapshot.cancellation() == null ? null : snapshot.cancellation().reason();
            return new Facts(reason, send != ContextResult.SendState.NOT_STARTED
                    && send != ContextResult.SendState.NOT_SENT && send != ContextResult.SendState.ROUTE_CLOSED,
                    delivery.prefillProof() && delivery.decodeProof());
        }

        private long remaining() throws TimeoutException {
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0L) { throw new TimeoutException("delivery cleanup did not settle: request_id=" + claim.item().requestId()); }
            return remaining;
        }
    }
}

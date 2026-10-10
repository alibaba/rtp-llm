package org.flexlb.balance.scheduler;

import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Consumer;

/** Executes bounded remote cleanup; the claim remains the authority for release evidence. */
final class DeliveryCleanupTask {
    static final long CLEANUP_TIMEOUT_MS = 5_000L;
    private static final int MAX_CANCEL_ATTEMPTS = 8;
    private static final long ACK_TIMEOUT_MS = 500L;

    private final DeliveryClaim claim;
    private final EngineCancelChannel channel;
    private final ScheduledExecutorService timer;
    private final Consumer<Runnable> facts;
    private final CompletableFuture<?> settlement;
    private ScheduledFuture<?> deadline;
    private ScheduledFuture<?> retry;
    private ScheduledFuture<?> ackDeadline;
    private int attempt;
    private boolean stopped;

    DeliveryCleanupTask(DeliveryClaim claim, EngineCancelChannel channel,
                        ScheduledExecutorService timer, Consumer<Runnable> facts) {
        this.claim = Objects.requireNonNull(claim);
        this.channel = Objects.requireNonNull(channel);
        this.timer = Objects.requireNonNull(timer);
        this.facts = Objects.requireNonNull(facts);
        this.settlement = claim.settlement().toCompletableFuture();
    }

    synchronized void start() {
        settlement.thenRun(() -> stop(null));
        try {
            deadline = timer.schedule(() -> stop(new TimeoutException(
                    "delivery cleanup did not settle: request_id=" + claim.item.requestId())),
                    CLEANUP_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            if (settlement.isDone()) { cancelTimers(); return; }
            attemptCancel();
        } catch (RuntimeException failure) { stop(failure); }
    }

    private synchronized void attemptCancel() {
        var evidence = claim.cleanupEvidence();
        if (stopped || settlement.isDone() || evidence.remoteSettled() || !evidence.needsRemoteCancel()) { return; }
        int current = ++attempt;
        CompletableFuture<EngineCancelChannel.CancelAck> reply = new CompletableFuture<>();
        try {
            ackDeadline = timer.schedule(() -> {
                reply.completeExceptionally(new TimeoutException("cancel acknowledgement timed out"));
            }, ACK_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            reply.whenComplete((ack, failure) -> {
                try { facts.accept(() -> finishCancel(current, ack, failure)); }
                catch (RuntimeException rejected) { stop(rejected); }
            });
            var remote = Objects.requireNonNull(channel.cancel(
                    new CancelTarget(claim.item.prefillEp().getIp(), claim.item.prefillEp().getGrpcPort()),
                    claim.item.requestId(), evidence.reason(), ACK_TIMEOUT_MS), "cancel future");
            remote.whenComplete((ack, failure) -> {
                if (failure == null) {
                    if (!reply.complete(ack) && ack != null) {
                        // A timed-out attempt still carries exact cleanup proof when its RPC finishes later.
                        try { facts.accept(() -> claim.item.ctx().scheduler().acceptDeliveryCleanup(claim, ack)); }
                        catch (RuntimeException rejected) { stop(rejected); }
                    }
                }
                else { reply.completeExceptionally(failure); }
            });
        } catch (Throwable failure) { reply.completeExceptionally(failure); }
    }

    private synchronized void finishCancel(int current, EngineCancelChannel.CancelAck ack, Throwable failure) {
        // Exact proof remains valid after this attempt timed out, a retry started, or the task stopped.
        if (ack != null) { claim.item.ctx().scheduler().acceptDeliveryCleanup(claim, ack); }
        if (stopped || settlement.isDone() || current != attempt) { return; }
        if (ackDeadline != null) { ackDeadline.cancel(false); }
        if (settlement.isDone() || claim.cleanupEvidence().remoteSettled()) { return; }
        if (current >= MAX_CANCEL_ATTEMPTS) {
            stop(failure == null ? new IllegalStateException(
                    "cancel lacks cleanup proof: request_id=" + claim.item.requestId() + " ack=" + ack) : failure);
            return;
        }
        try {
            retry = timer.schedule(this::attemptCancel, Math.min(400L, 25L << (current - 1)), TimeUnit.MILLISECONDS);
        } catch (RuntimeException rejected) { stop(rejected); }
    }

    private void stop(Throwable failure) {
        synchronized (this) {
            if (stopped) { return; }
            stopped = true;
            cancelTimers();
        }
        if (failure != null && !settlement.isDone()) { claim.item.ctx().scheduler().recordFailure(failure); }
    }

    private synchronized void cancelTimers() {
        if (deadline != null) { deadline.cancel(false); }
        if (retry != null) { retry.cancel(false); }
        if (ackDeadline != null) { ackDeadline.cancel(false); }
    }
}

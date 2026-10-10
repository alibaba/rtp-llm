package org.flexlb.balance.endpoint;

import org.flexlb.util.Failures;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** One endpoint-generation admission gate and its retirement drain. */
final class EndpointGenerationLifecycle {

    private static final long DEFAULT_RETIREMENT_TIMEOUT_MS =
            Long.getLong("flexlb.endpoint.retirement.timeout.ms", 30_000L);

    private enum RetirementPhase {
        ACCEPTING_HANDOFFS,
        RETIRING,
        WAITING_HANDOFFS,
        CLEANUP_SCHEDULED,
        CLEANING,
        RETIRED
    }

    private volatile RetirementPhase phase =
            RetirementPhase.ACCEPTING_HANDOFFS;
    private final Runnable handoffsDrained;
    private int activeHandoffs;
    private Thread cleanupThread;
    private Throwable retirementFailure;

    EndpointGenerationLifecycle(Runnable handoffsDrained) {
        this.handoffsDrained = Objects.requireNonNull(
                handoffsDrained, "handoffsDrained");
    }

    synchronized HandoffPermit tryAcquireHandoff() {
        if (phase != RetirementPhase.ACCEPTING_HANDOFFS) {
            return null;
        }
        activeHandoffs++;
        return new HandoffPermit(this);
    }

    boolean isRetiringOrRetired() {
        return phase != RetirementPhase.ACCEPTING_HANDOFFS;
    }

    /** Close the gate without waiting or running endpoint cleanup. */
    synchronized void beginRetirement() {
        if (phase == RetirementPhase.ACCEPTING_HANDOFFS) {
            phase = RetirementPhase.RETIRING;
        }
    }

    /** Claim cleanup atomically; return true only when this caller should run it. */
    synchronized boolean tryStartCleanup() {
        checkState(phase != RetirementPhase.ACCEPTING_HANDOFFS, "endpoint retirement gate is still open");
        if (phase != RetirementPhase.RETIRING) {
            return false;
        }
        phase = activeHandoffs == 0
                ? RetirementPhase.CLEANUP_SCHEDULED : RetirementPhase.WAITING_HANDOFFS;
        return phase == RetirementPhase.CLEANUP_SCHEDULED;
    }

    /** Bind the claimed cleanup to its execution thread for reentrant close. */
    synchronized void beginCleanup() {
        checkState(phase == RetirementPhase.CLEANUP_SCHEDULED && activeHandoffs == 0,
                "endpoint retirement cleanup owner is invalid");
        phase = RetirementPhase.CLEANING;
        cleanupThread = Thread.currentThread();
    }

    synchronized void completeRetirement(Throwable failure) {
        checkState(phase == RetirementPhase.CLEANING && activeHandoffs == 0,
                "endpoint generation cleanup is not ready to complete");
        retirementFailure = failure;
        cleanupThread = null;
        phase = RetirementPhase.RETIRED;
        notifyAll();
    }

    void awaitRetirement() {
        awaitRetirement(DEFAULT_RETIREMENT_TIMEOUT_MS);
    }

    void awaitRetirement(long timeoutMs) {
        checkArgument(timeoutMs > 0L, "endpoint retirement timeout must be positive");
        boolean interrupted = false;
        Throwable failure;
        long deadlineNanos = System.nanoTime()
                + java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        try {
            synchronized (this) {
                checkState(phase != RetirementPhase.ACCEPTING_HANDOFFS, "endpoint retirement has not begun");
                checkState(phase != RetirementPhase.RETIRING, "endpoint retirement cleanup has not been initiated");
                checkState(phase != RetirementPhase.CLEANING || cleanupThread != Thread.currentThread(),
                        "endpoint retirement cleanup cannot await itself");
                while (phase != RetirementPhase.RETIRED) {
                    long remainingNanos = deadlineNanos - System.nanoTime();
                    if (remainingNanos <= 0L) {
                        throw new IllegalStateException(
                                "endpoint retirement timed out after "
                                        + timeoutMs + "ms: phase=" + phase
                                        + ", activeHandoffs=" + activeHandoffs);
                    }
                    try {
                        long waitMillis = Math.max(
                                1L,
                                java.util.concurrent.TimeUnit.NANOSECONDS
                                        .toMillis(remainingNanos));
                        wait(waitMillis);
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
                failure = retirementFailure;
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        Failures.rethrow(failure, "endpoint generation retirement failed");
    }

    private void releaseHandoff() {
        boolean runContinuation = false;
        synchronized (this) {
            checkState(activeHandoffs > 0, "endpoint handoff permit released more than once");
            activeHandoffs--;
            if (activeHandoffs == 0 && phase == RetirementPhase.WAITING_HANDOFFS) {
                phase = RetirementPhase.CLEANUP_SCHEDULED;
                runContinuation = true;
            }
        }
        if (runContinuation) {
            handoffsDrained.run();
        }
    }

    static final class HandoffPermit implements AutoCloseable {
        private final EndpointGenerationLifecycle lifecycle;
        private final AtomicBoolean open = new AtomicBoolean(true);

        private HandoffPermit(EndpointGenerationLifecycle lifecycle) {
            this.lifecycle = lifecycle;
        }

        @Override
        public void close() {
            if (open.compareAndSet(true, false)) {
                lifecycle.releaseHandoff();
            }
        }

        boolean isOpen() {
            return open.get();
        }
    }
}

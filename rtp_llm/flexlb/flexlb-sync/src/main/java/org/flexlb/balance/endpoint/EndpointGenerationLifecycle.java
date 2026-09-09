package org.flexlb.balance.endpoint;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/** One endpoint-generation admission gate and its retirement drain. */
final class EndpointGenerationLifecycle {

    private static final long DEFAULT_RETIREMENT_TIMEOUT_MS =
            Long.getLong("flexlb.endpoint.retirement.timeout.ms", 30_000L);

    private enum RetirementPhase {
        ACCEPTING_HANDOFFS,
        RETIRING,
        RETIRED
    }

    private volatile RetirementPhase phase =
            RetirementPhase.ACCEPTING_HANDOFFS;
    private final Runnable handoffsDrained;
    private int activeHandoffs;
    private boolean cleanupClaimed;
    private boolean drainContinuationArmed;
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

    /** Exactly one caller may own the endpoint-local drain and cleanup. */
    synchronized boolean tryClaimCleanup() {
        if (phase == RetirementPhase.ACCEPTING_HANDOFFS) {
            throw new IllegalStateException(
                    "endpoint retirement gate is still open");
        }
        if (phase == RetirementPhase.RETIRED || cleanupClaimed) {
            return false;
        }
        cleanupClaimed = true;
        return true;
    }

    /** Arm one continuation only when accepted handoffs still need to drain. */
    synchronized boolean armDrainContinuation() {
        if (phase != RetirementPhase.RETIRING || !cleanupClaimed
                || cleanupThread != null || drainContinuationArmed) {
            throw new IllegalStateException(
                    "endpoint retirement continuation is invalid");
        }
        if (activeHandoffs == 0) {
            return false;
        }
        drainContinuationArmed = true;
        return true;
    }

    /** Bind the claimed cleanup to its execution thread for reentrant close. */
    synchronized void beginCleanup() {
        if (phase != RetirementPhase.RETIRING || !cleanupClaimed
                || cleanupThread != null || drainContinuationArmed
                || activeHandoffs != 0) {
            throw new IllegalStateException(
                    "endpoint retirement cleanup owner is invalid");
        }
        cleanupThread = Thread.currentThread();
    }

    synchronized void completeRetirement(Throwable failure) {
        if (phase != RetirementPhase.RETIRING || activeHandoffs != 0) {
            throw new IllegalStateException(
                    "endpoint generation cannot retire with active handoffs");
        }
        retirementFailure = failure;
        cleanupThread = null;
        phase = RetirementPhase.RETIRED;
        notifyAll();
    }

    void awaitRetirement() {
        awaitRetirement(DEFAULT_RETIREMENT_TIMEOUT_MS);
    }

    void awaitRetirement(long timeoutMs) {
        if (timeoutMs <= 0L) {
            throw new IllegalArgumentException(
                    "endpoint retirement timeout must be positive");
        }
        boolean interrupted = false;
        Throwable failure = null;
        IllegalStateException timeoutFailure = null;
        long deadlineNanos = System.nanoTime()
                + java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        synchronized (this) {
            if (phase == RetirementPhase.ACCEPTING_HANDOFFS) {
                throw new IllegalStateException(
                        "endpoint retirement has not begun");
            }
            if (phase == RetirementPhase.RETIRING && !cleanupClaimed) {
                throw new IllegalStateException(
                        "endpoint retirement cleanup has not been initiated");
            }
            if (phase == RetirementPhase.RETIRING
                    && cleanupThread == Thread.currentThread()) {
                throw new IllegalStateException(
                        "endpoint retirement cleanup cannot await itself");
            }
            while (phase != RetirementPhase.RETIRED) {
                long remainingNanos = deadlineNanos - System.nanoTime();
                if (remainingNanos <= 0L) {
                    timeoutFailure = new IllegalStateException(
                            "endpoint retirement timed out after "
                                    + timeoutMs + "ms: phase=" + phase
                                    + ", activeHandoffs=" + activeHandoffs
                                    + ", cleanupClaimed=" + cleanupClaimed
                                    + ", drainContinuationArmed="
                                    + drainContinuationArmed);
                    break;
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
            if (timeoutFailure == null) {
                failure = retirementFailure;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        if (timeoutFailure != null) {
            throw timeoutFailure;
        }
        rethrow(failure);
    }

    private void releaseHandoff() {
        boolean runContinuation = false;
        synchronized (this) {
            if (activeHandoffs <= 0) {
                throw new IllegalStateException(
                        "endpoint handoff permit released more than once");
            }
            activeHandoffs--;
            if (activeHandoffs == 0) {
                notifyAll();
                if (drainContinuationArmed) {
                    drainContinuationArmed = false;
                    runContinuation = true;
                }
            }
        }
        if (runContinuation) {
            handoffsDrained.run();
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "endpoint generation retirement failed", failure);
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

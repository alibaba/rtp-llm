package org.flexlb.balance.endpoint;

import java.util.concurrent.atomic.AtomicBoolean;

/** One endpoint-generation admission gate and its retirement drain. */
final class EndpointGenerationLifecycle {

    private enum RetirementPhase {
        ACCEPTING_HANDOFFS,
        RETIRING,
        RETIRED
    }

    private volatile RetirementPhase phase =
            RetirementPhase.ACCEPTING_HANDOFFS;
    private int activeHandoffs;
    private boolean cleanupClaimed;
    private Thread cleanupThread;
    private Throwable retirementFailure;

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

    synchronized boolean hasActiveHandoffs() {
        return activeHandoffs > 0;
    }

    /** Bind the claimed cleanup to its execution thread for reentrant close. */
    synchronized void beginCleanup() {
        if (phase != RetirementPhase.RETIRING || !cleanupClaimed
                || cleanupThread != null) {
            throw new IllegalStateException(
                    "endpoint retirement cleanup owner is invalid");
        }
        cleanupThread = Thread.currentThread();
    }

    void awaitHandoffs() {
        boolean interrupted = false;
        synchronized (this) {
            if (phase == RetirementPhase.ACCEPTING_HANDOFFS) {
                throw new IllegalStateException(
                        "endpoint retirement has not begun");
            }
            while (activeHandoffs > 0) {
                try {
                    wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
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
        boolean interrupted = false;
        Throwable failure;
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
                try {
                    wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
            failure = retirementFailure;
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        rethrow(failure);
    }

    private synchronized void releaseHandoff() {
        if (activeHandoffs <= 0) {
            throw new IllegalStateException(
                    "endpoint handoff permit released more than once");
        }
        activeHandoffs--;
        if (activeHandoffs == 0) {
            notifyAll();
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

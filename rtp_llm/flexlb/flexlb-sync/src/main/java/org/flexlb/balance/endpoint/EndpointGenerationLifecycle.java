package org.flexlb.balance.endpoint;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Coordinates generation-local ownership handoffs and retirement for one endpoint.
 *
 * <p>The lifecycle is strictly monotonic:
 * {@code ACCEPTING_HANDOFFS -> RETIRING -> RETIRED}. A handoff may cover local
 * DIRECT ledger publication, QUEUE route-delivery ownership transfer, or QUEUE
 * batch transport submission. Retirement first rejects new handoffs, then waits
 * for every accepted handoff to finish. It does not wait for delivery ACKs or
 * Engine terminals. One retirement action is installed after retirement begins.
 * The thread which observes the final handoff release runs that action and
 * publishes {@code RETIRED} only after generation-local cleanup is complete.
 * Concurrent close callers wait for that final state rather than mistaking an
 * empty handoff set for completed retirement.
 */
final class EndpointGenerationLifecycle {

    private enum State {
        ACCEPTING_HANDOFFS,
        RETIRING,
        RETIRED
    }

    private final ReentrantLock lock = new ReentrantLock();
    private final Condition lifecycleChanged = lock.newCondition();
    private volatile State state = State.ACCEPTING_HANDOFFS;
    private int activeHandoffs;
    private final Map<Thread, Integer> handoffsByAcquiringThread =
            new IdentityHashMap<>();
    private Thread retirementOwner;
    private Runnable drainedRetirementAction;
    private boolean drainedRetirementActionClaimed;
    private Throwable retirementFailure;

    HandoffPermit tryAcquireHandoff() {
        lock.lock();
        try {
            if (state != State.ACCEPTING_HANDOFFS) {
                return null;
            }
            activeHandoffs++;
            Thread acquiringThread = Thread.currentThread();
            handoffsByAcquiringThread.merge(acquiringThread, 1, Integer::sum);
            return new HandoffPermit(this, acquiringThread);
        } finally {
            lock.unlock();
        }
    }

    boolean isRetiringOrRetired() {
        return state != State.ACCEPTING_HANDOFFS;
    }

    /** Claim exclusive responsibility for retiring this generation. */
    boolean tryBeginRetirement() {
        lock.lock();
        try {
            if (state != State.ACCEPTING_HANDOFFS) {
                return false;
            }
            state = State.RETIRING;
            retirementOwner = Thread.currentThread();
            return true;
        } finally {
            lock.unlock();
        }
    }

    /** True only for a reentrant close on the thread which began retirement. */
    boolean currentThreadOwnsRetirement() {
        lock.lock();
        try {
            return state == State.RETIRING
                    && retirementOwner == Thread.currentThread();
        } finally {
            lock.unlock();
        }
    }

    /** Whether returning from close is required to let this thread release a handoff. */
    boolean currentThreadOwnsAcceptedHandoff() {
        lock.lock();
        try {
            return handoffsByAcquiringThread.containsKey(Thread.currentThread());
        } finally {
            lock.unlock();
        }
    }

    /**
     * Install the single retirement action. It runs immediately when already
     * drained, otherwise the final handoff release runs it outside the lock.
     */
    void runWhenAcceptedHandoffsDrain(Runnable retirementAction) {
        Objects.requireNonNull(retirementAction, "retirementAction");
        Runnable actionToRun = null;
        lock.lock();
        try {
            if (state != State.RETIRING) {
                throw new IllegalStateException("endpoint retirement is not active");
            }
            if (drainedRetirementAction != null || drainedRetirementActionClaimed) {
                throw new IllegalStateException("endpoint retirement action already installed");
            }
            if (activeHandoffs == 0) {
                drainedRetirementActionClaimed = true;
                actionToRun = retirementAction;
            } else {
                drainedRetirementAction = retirementAction;
            }
        } finally {
            lock.unlock();
        }
        if (actionToRun != null) {
            actionToRun.run();
        }
    }

    /** Wait uninterruptibly until every handoff accepted by this generation ends. */
    void awaitAcceptedHandoffs() {
        boolean interrupted = false;
        lock.lock();
        try {
            if (state == State.ACCEPTING_HANDOFFS) {
                throw new IllegalStateException(
                        "endpoint retirement has not begun");
            }
            while (activeHandoffs > 0) {
                try {
                    lifecycleChanged.await();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        } finally {
            lock.unlock();
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    /** Publish that all generation-local retirement work is complete. */
    void completeRetirement() {
        completeRetirement(null);
    }

    void completeRetirement(Throwable failure) {
        lock.lock();
        try {
            if (state != State.RETIRING || activeHandoffs != 0) {
                throw new IllegalStateException(
                        "endpoint generation cannot retire with active handoffs");
            }
            retirementFailure = failure;
            retirementOwner = null;
            drainedRetirementAction = null;
            state = State.RETIRED;
            lifecycleChanged.signalAll();
        } finally {
            lock.unlock();
        }
    }

    /** Wait uninterruptibly for the retiring owner to finish all cleanup. */
    void awaitRetirement() {
        boolean interrupted = false;
        Throwable failure;
        lock.lock();
        try {
            while (state != State.RETIRED) {
                try {
                    lifecycleChanged.await();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
            failure = retirementFailure;
        } finally {
            lock.unlock();
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        rethrowRetirementFailure(failure);
    }

    private static void rethrowRetirementFailure(Throwable failure) {
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

    private void releaseHandoff(Thread acquiringThread) {
        Runnable actionToRun = null;
        lock.lock();
        try {
            Integer ownerCount = handoffsByAcquiringThread.get(acquiringThread);
            if (ownerCount == null || ownerCount <= 0) {
                throw new IllegalStateException(
                        "endpoint handoff acquiring thread was not registered");
            }
            activeHandoffs--;
            if (activeHandoffs < 0) {
                activeHandoffs++;
                throw new IllegalStateException(
                        "endpoint handoff permit released more than once");
            }
            if (ownerCount == 1) {
                handoffsByAcquiringThread.remove(acquiringThread);
            } else {
                handoffsByAcquiringThread.put(acquiringThread, ownerCount - 1);
            }
            if (state == State.RETIRING && activeHandoffs == 0) {
                if (drainedRetirementAction != null) {
                    actionToRun = drainedRetirementAction;
                    drainedRetirementAction = null;
                    drainedRetirementActionClaimed = true;
                }
                lifecycleChanged.signalAll();
            }
        } finally {
            lock.unlock();
        }
        if (actionToRun != null) {
            actionToRun.run();
        }
    }

    static final class HandoffPermit implements AutoCloseable {

        private final EndpointGenerationLifecycle lifecycle;
        private final Thread acquiringThread;
        private final AtomicBoolean released = new AtomicBoolean();

        private HandoffPermit(
                EndpointGenerationLifecycle lifecycle,
                Thread acquiringThread) {
            this.lifecycle = lifecycle;
            this.acquiringThread = acquiringThread;
        }

        @Override
        public void close() {
            if (released.compareAndSet(false, true)) {
                lifecycle.releaseHandoff(acquiringThread);
            }
        }
    }
}

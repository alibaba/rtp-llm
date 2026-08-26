package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Publishes frontend completions without running user continuations on a
 * scheduler, endpoint, or transport critical path.
 *
 * <p>The publisher never decides request ownership. External callers first
 * cross the synchronous {@link Terminalizer} boundary; only an exact
 * lifecycle-owned publication permit may mutate its public future. Internal
 * Delivery and terminal reducers call the kind-specific submit method only
 * after releasing the exact slot lock.
 */
final class RequestCompletionPublisher
        implements ExternalCompletionSink, AutoCloseable {

    private final Terminalizer terminalizer;
    private final ThreadPoolExecutor executor;
    private final Object lifecycleMonitor = new Object();
    private final ThreadLocal<ArrayDeque<Runnable>> localDrain =
            new ThreadLocal<>();
    private final ThreadLocal<Integer> publicationDepth =
            new ThreadLocal<>();
    private LifecycleState lifecycle = LifecycleState.OPEN;
    private int inFlightPublications;
    private Throwable closeFailure;

    RequestCompletionPublisher(Terminalizer terminalizer) {
        this(terminalizer, Policy.productionDefaults());
    }

    RequestCompletionPublisher(
            Terminalizer terminalizer,
            Policy policy) {
        this.terminalizer = Objects.requireNonNull(
                terminalizer, "terminalizer");
        Objects.requireNonNull(policy, "policy");

        AtomicInteger workerSequence = new AtomicInteger();
        executor = new ThreadPoolExecutor(
                policy.workerCount(),
                policy.workerCount(),
                0L,
                TimeUnit.MILLISECONDS,
                // One exact current RequestSlot can own at most one lease, so
                // the global outstanding-request bound is the real queue
                // bound. A non-blocking ingress avoids scheduler/transport
                // deadlock under user-continuation backpressure.
                new LinkedBlockingQueue<>(),
                runnable -> {
                    Thread thread = new Thread(
                            runnable,
                            "request-completion-publisher-"
                                    + workerSequence.getAndIncrement());
                    thread.setDaemon(true);
                    return thread;
                },
                new ThreadPoolExecutor.AbortPolicy());
        executor.prestartAllCoreThreads();
    }

    @Override
    public PublicationLease tryReservePublication(RequestSlot exactSlot) {
        synchronized (lifecycleMonitor) {
            if (lifecycle != LifecycleState.OPEN) {
                return null;
            }
            inFlightPublications++;
            return new PublisherLease(exactSlot);
        }
    }

    /** Queue an exact delivery-owned response publication. */
    void submitDeliveryResponse(
            RequestSlot.PublicationPermit exactPermit,
            Response response) {
        RequestSlot exactSlot = exactPermit.slot();
        requireOutsideSlotLock(exactSlot, "delivery response submission");
        RequestSlot.Publication publication;
        try {
            publication = exactPermit.claimDeliveryResponse(response);
        } catch (RuntimeException | Error claimFailure) {
            exactPermit.abandonIfUnclaimed();
            throw claimFailure;
        }
        try {
            enqueue(() -> executePublication(exactPermit, publication));
        } catch (RuntimeException | Error enqueueFailure) {
            exactPermit.abortClaimedPublication();
            throw enqueueFailure;
        }
    }

    /** Queue an exact terminal-owned response publication. */
    void submitTerminalResponse(
            RequestSlot.PublicationPermit exactPermit,
            Response response) {
        RequestSlot exactSlot = exactPermit.slot();
        requireOutsideSlotLock(exactSlot, "terminal response submission");
        RequestSlot.Publication publication;
        try {
            publication = exactPermit.claimTerminalResponse(response);
        } catch (RuntimeException | Error claimFailure) {
            exactPermit.abandonIfUnclaimed();
            throw claimFailure;
        }
        try {
            enqueue(() -> executePublication(exactPermit, publication));
        } catch (RuntimeException | Error enqueueFailure) {
            exactPermit.abortClaimedPublication();
            throw enqueueFailure;
        }
    }

    @Override
    public boolean publishResponse(
            RequestSlot exactSlot,
            Response response) {
        requireOutsideSlotLock(exactSlot, "external response publication");
        RequestSlot.PublicationPermit permit =
                terminalizer.terminalizeResponse(exactSlot, response);
        if (permit == null) {
            return false;
        }
        RequestSlot.PublicationPermit exact =
                requireExactSlot(permit, exactSlot);
        RequestSlot.Publication publication;
        try {
            publication = exact.claimTerminalResponse(response);
        } catch (RuntimeException | Error claimFailure) {
            exact.abandonIfUnclaimed();
            throw claimFailure;
        }
        try {
            return executePublication(exact, publication);
        } catch (RuntimeException | Error executionFailure) {
            exact.abortClaimedPublication();
            throw executionFailure;
        }
    }

    @Override
    public boolean publishFailure(
            RequestSlot exactSlot,
            Throwable error) {
        requireOutsideSlotLock(exactSlot, "external failure publication");
        RequestSlot.PublicationPermit permit =
                terminalizer.terminalizeFailure(exactSlot, error);
        if (permit == null) {
            return false;
        }
        RequestSlot.PublicationPermit exact =
                requireExactSlot(permit, exactSlot);
        RequestSlot.Publication publication;
        try {
            publication = exact.claimFailure(error);
        } catch (RuntimeException | Error claimFailure) {
            exact.abandonIfUnclaimed();
            throw claimFailure;
        }
        try {
            return executePublication(exact, publication);
        } catch (RuntimeException | Error executionFailure) {
            exact.abortClaimedPublication();
            throw executionFailure;
        }
    }

    @Override
    public boolean publishCancellation(
            RequestSlot exactSlot,
            boolean mayInterruptIfRunning) {
        requireOutsideSlotLock(
                exactSlot, "external cancellation publication");
        RequestSlot.PublicationPermit permit =
                terminalizer.terminalizeCancellation(exactSlot);
        if (permit == null) {
            return false;
        }
        RequestSlot.PublicationPermit exact =
                requireExactSlot(permit, exactSlot);
        RequestSlot.Publication publication;
        try {
            publication = exact.claimCancellation(mayInterruptIfRunning);
        } catch (RuntimeException | Error claimFailure) {
            exact.abandonIfUnclaimed();
            throw claimFailure;
        }
        try {
            return executePublication(exact, publication);
        } catch (RuntimeException | Error executionFailure) {
            exact.abortClaimedPublication();
            throw executionFailure;
        }
    }

    Snapshot snapshot() {
        int queueSize = executor.getQueue().size();
        return new Snapshot(
                executor.getMaximumPoolSize(),
                queueSize,
                executor.getLargestPoolSize(),
                executor.getCompletedTaskCount(),
                lifecycleSnapshot() != LifecycleState.OPEN);
    }

    boolean awaitTermination(long timeout, TimeUnit unit)
            throws InterruptedException {
        return executor.awaitTermination(timeout, unit);
    }

    @Override
    public void close() {
        boolean reentrant = publicationDepth.get() != null;
        boolean interrupted = false;
        boolean closeOwner = false;
        synchronized (lifecycleMonitor) {
            if (lifecycle == LifecycleState.CLOSED) {
                rethrow(closeFailure);
                return;
            }
            if (lifecycle == LifecycleState.CLOSING) {
                if (reentrant) {
                    return;
                }
                while (lifecycle != LifecycleState.CLOSED) {
                    try {
                        lifecycleMonitor.wait();
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                rethrow(closeFailure);
                return;
            }
            lifecycle = LifecycleState.CLOSING;
            closeOwner = true;
        }

        if (closeOwner && reentrant) {
            try {
                Thread closer = new Thread(
                        this::finishClose,
                        "request-completion-publisher-close");
                closer.setDaemon(false);
                closer.start();
            } catch (RuntimeException | Error startFailure) {
                try {
                    executor.shutdown();
                } catch (Throwable shutdownFailure) {
                    startFailure.addSuppressed(shutdownFailure);
                }
                completeClose(startFailure);
                throw startFailure;
            }
            return;
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        finishClose();
        rethrow(closeFailure);
    }

    private void finishClose() {
        boolean interrupted = false;
        synchronized (lifecycleMonitor) {
            while (inFlightPublications != 0) {
                try {
                    lifecycleMonitor.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }

        Throwable failure = null;
        try {
            executor.shutdown();
            while (!executor.isTerminated()) {
                try {
                    executor.awaitTermination(1, TimeUnit.DAYS);
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        } catch (Throwable shutdownFailure) {
            failure = shutdownFailure;
        } finally {
            completeClose(failure);
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void completeClose(Throwable failure) {
        synchronized (lifecycleMonitor) {
            closeFailure = failure;
            lifecycle = LifecycleState.CLOSED;
            lifecycleMonitor.notifyAll();
        }
    }

    private void enqueue(Runnable publication) {
        ArrayDeque<Runnable> activeDrain = localDrain.get();
        if (activeDrain != null) {
            // A user continuation re-entered the scheduler from a dedicated
            // publisher thread. Append locally so bounded-queue backpressure
            // cannot make every publisher worker wait on its own queue.
            activeDrain.addLast(publication);
            return;
        }

        Runnable drainTask = () -> drainPublications(publication);
        try {
            executor.execute(drainTask);
        } catch (java.util.concurrent.RejectedExecutionException closed) {
            throw new IllegalStateException(
                    "accepted completion publication was rejected", closed);
        }
    }

    private void drainPublications(Runnable first) {
        if (localDrain.get() != null) {
            throw new IllegalStateException(
                    "completion publication drain is already active");
        }
        ArrayDeque<Runnable> drain = new ArrayDeque<>();
        localDrain.set(drain);
        drain.addLast(first);
        Throwable failure = null;
        try {
            while (!drain.isEmpty()) {
                try {
                    drain.removeFirst().run();
                } catch (Throwable publicationFailure) {
                    failure = appendFailure(failure, publicationFailure);
                }
            }
        } finally {
            localDrain.remove();
        }
        rethrowPublicationFailure(failure);
    }

    private void exitPublication() {
        synchronized (lifecycleMonitor) {
            if (inFlightPublications <= 0) {
                throw new IllegalStateException(
                        "completion publication counter underflow");
            }
            inFlightPublications--;
            if (inFlightPublications == 0) {
                lifecycleMonitor.notifyAll();
            }
        }
    }

    private LifecycleState lifecycleSnapshot() {
        synchronized (lifecycleMonitor) {
            return lifecycle;
        }
    }

    private static RequestSlot.PublicationPermit requireExactSlot(
            RequestSlot.PublicationPermit permit,
            RequestSlot exactSlot) {
        if (permit.slot() != exactSlot) {
            permit.abandonIfUnclaimed();
            throw new IllegalStateException(
                    "terminalizer returned a publication for another slot");
        }
        return permit;
    }

    private boolean executePublication(
            RequestSlot.PublicationPermit permit,
            RequestSlot.Publication publication) {
        PublisherLease lease = requirePublisherLease(permit);
        Integer currentDepth = publicationDepth.get();
        publicationDepth.set(currentDepth == null ? 1 : currentDepth + 1);
        try {
            return publication.publish();
        } finally {
            if (currentDepth == null) {
                publicationDepth.remove();
            } else {
                publicationDepth.set(currentDepth);
            }
            lease.close();
        }
    }

    private PublisherLease requirePublisherLease(
            RequestSlot.PublicationPermit permit) {
        PublicationLease lease = permit.lease();
        if (!(lease instanceof PublisherLease owned)
                || owned.publisher != this
                || owned.exactSlot() != permit.slot()) {
            lease.close();
            throw new IllegalStateException(
                    "publication permit belongs to another publisher");
        }
        return owned;
    }

    /** Exact close-barrier owner acquired before a slot lifecycle edge. */
    private final class PublisherLease implements PublicationLease {
        private final RequestCompletionPublisher publisher =
                RequestCompletionPublisher.this;
        private final RequestSlot exactSlot;
        private final AtomicBoolean closed = new AtomicBoolean();

        private PublisherLease(RequestSlot exactSlot) {
            this.exactSlot = exactSlot;
        }

        @Override
        public RequestSlot exactSlot() {
            return exactSlot;
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                exitPublication();
            }
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "completion publisher close failed", failure);
        }
    }

    private static Throwable appendFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static void rethrowPublicationFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "completion publication failed", failure);
        }
    }

    private static void requireOutsideSlotLock(
            RequestSlot exactSlot,
            String operation) {
        if (Thread.holdsLock(exactSlot)) {
            throw new IllegalStateException(
                    operation + " must run outside the RequestSlot lock");
        }
    }

    /** Synchronous exact-slot reducer used by the public future adapter. */
    interface Terminalizer {

        RequestSlot.PublicationPermit terminalizeResponse(
                RequestSlot exactSlot, Response response);

        RequestSlot.PublicationPermit terminalizeFailure(
                RequestSlot exactSlot, Throwable error);

        RequestSlot.PublicationPermit terminalizeCancellation(
                RequestSlot exactSlot);
    }

    private enum LifecycleState {
        OPEN,
        CLOSING,
        CLOSED
    }

    record Policy(int workerCount) {

        Policy {
            if (workerCount < 1) {
                throw new IllegalArgumentException(
                        "completion publisher worker count must be positive");
            }
        }

        static Policy productionDefaults() {
            return new Policy(Math.max(2, Math.min(
                    8, Runtime.getRuntime().availableProcessors())));
        }
    }

    record Snapshot(
            int workerLimit,
            int queueSize,
            int largestPoolSize,
            long completedTaskCount,
            boolean shutdown) {
    }
}

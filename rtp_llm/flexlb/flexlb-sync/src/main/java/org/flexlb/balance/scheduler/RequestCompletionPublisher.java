package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.ArrayDeque;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.flexlb.balance.scheduler.RequestTerminalCleanup.runTerminalLeaf;

/**
 * Publishes frontend completions without running user continuations on a
 * scheduler, endpoint, or transport critical path.
 *
 * <p>The publisher never decides request ownership. External callers first
 * cross the synchronous lifecycle-owner boundary; only an exact
 * lifecycle-owned publication permit may mutate its public future. Internal
 * Delivery and terminal reducers call the kind-specific submit method only
 * after releasing the exact slot lock. Claiming the permit briefly re-enters
 * that lock to select the single frontend result; future completion and all
 * user continuations run only after the lock is released.
 */
final class RequestCompletionPublisher implements AutoCloseable {

    private static final int DEFAULT_PUBLISHER_WORKERS = 8;

    private final ExpirationTimer expirationTimer;
    private final BatchSchedulerReporter reporter;
    private final ThreadPoolExecutor executor;
    private final Object lifecycleMonitor = new Object();
    private final ThreadLocal<ArrayDeque<Runnable>> localDrain =
            new ThreadLocal<>();
    private final ThreadLocal<Integer> publicationDepth =
            new ThreadLocal<>();
    private PublisherPhase phase = PublisherPhase.OPEN;
    private int inFlightPublications;
    private Throwable closeFailure;

    RequestCompletionPublisher(ExpirationTimer timer, BatchSchedulerReporter reporter, int configuredWorkers) {
        this.expirationTimer = timer;
        this.reporter = reporter;
        int workers = configuredWorkers > 0
                ? configuredWorkers : DEFAULT_PUBLISHER_WORKERS;
        AtomicInteger workerSequence = new AtomicInteger();
        executor = new ThreadPoolExecutor(
                workers,
                workers,
                0L,
                TimeUnit.MILLISECONDS,
                // Queue completions so a busy publisher never runs client callbacks
                // inline on a decision thread. Slot owns request lifetime; the
                // publisher owns only these in-flight frontend completions.
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

    RequestSlot.PublicationPermit tryReservePublication(
            RequestSlot exactSlot,
            RequestSlot.PublicationKind kind) {
        synchronized (lifecycleMonitor) {
            if (phase != PublisherPhase.OPEN) {
                return null;
            }
            inFlightPublications++;
            return new RequestSlot.PublicationPermit(
                    this, exactSlot, kind);
        }
    }

    /** Queue an exact delivery-owned response publication. */
    void submitDeliveryResponse(
            RequestSlot.PublicationPermit exactPermit,
            Response response) {
        submit(exactPermit, "delivery response submission",
                permit -> permit.claimDeliveryResponse(response));
    }

    /** Queue an exact terminal-owned response publication. */
    void submitTerminalResponse(
            RequestSlot.PublicationPermit exactPermit,
            Response response) {
        submit(exactPermit, "terminal response submission",
                permit -> permit.claimTerminalResponse(response));
    }

    boolean publishResponse(
            RequestSlot exactSlot,
            Response response) {
        return publish(exactSlot, "external response publication",
                () -> exactSlot.publishExternalResponse(response),
                permit -> permit.claimTerminalResponse(response));
    }

    boolean publishFailure(
            RequestSlot exactSlot,
            Throwable error) {
        return publish(exactSlot, "external failure publication",
                () -> exactSlot.publishExternalFailure(error),
                permit -> permit.claimFailure(error));
    }

    boolean publishCancellation(
            RequestSlot exactSlot,
            boolean mayInterruptIfRunning) {
        return publish(exactSlot, "external cancellation publication",
                () -> exactSlot.publishExternalCancellation(),
                permit -> permit.claimCancellation(mayInterruptIfRunning));
    }

    private void submit(
            RequestSlot.PublicationPermit permit,
            String operation,
            Function<RequestSlot.PublicationPermit, BooleanSupplier> claim) {
        requireOutsideSlotLock(permit.slot(), operation);
        BooleanSupplier publication = claim(permit, claim);
        try {
            enqueue(() -> executePublication(permit, publication));
        } catch (RuntimeException | Error enqueueFailure) {
            permit.abortClaimedPublication();
            throw enqueueFailure;
        }
    }

    private boolean publish(
            RequestSlot exactSlot,
            String operation,
            Supplier<RequestSlot.PublicationPermit> acquire,
            Function<RequestSlot.PublicationPermit, BooleanSupplier> claim) {
        requireOutsideSlotLock(exactSlot, operation);
        RequestSlot.PublicationPermit permit = acquire.get();
        if (permit == null) {
            return false;
        }
        RequestSlot.PublicationPermit exact =
                requireExactSlot(permit, exactSlot);
        BooleanSupplier publication = claim(exact, claim);
        try {
            return executePublication(exact, publication);
        } catch (RuntimeException | Error executionFailure) {
            exact.abortClaimedPublication();
            throw executionFailure;
        }
    }

    private static BooleanSupplier claim(
            RequestSlot.PublicationPermit permit,
            Function<RequestSlot.PublicationPermit, BooleanSupplier> claim) {
        try {
            return claim.apply(permit);
        } catch (RuntimeException | Error claimFailure) {
            permit.abandonIfUnclaimed();
            throw claimFailure;
        }
    }

    void publishDelivery(
            RequestSlot slot,
            ScheduledRequest item,
            Response response,
            RequestSlot.DeliveryConfirmation confirmation,
            DeliveryClaimKind deliveryKind) {
        Throwable preparationFailure = null;
        preparationFailure = runTerminalLeaf(
                preparationFailure,
                confirmation.requestDeadline() == null
                        ? null : () -> expirationTimer.cancel(
                                confirmation.requestDeadline()));
        if (deliveryKind == DeliveryClaimKind.BATCH_ENQUEUE
                && item.ctx().getAckAtMs() > 0L && confirmation.batchEnqueueStartedAtMs() > 0L) {
            long latencyMs = Math.max(
                    0L,
                    item.ctx().getAckAtMs() - confirmation.batchEnqueueStartedAtMs());
            preparationFailure = runTerminalLeaf(
                    preparationFailure,
                    () -> reporter.reportDispatchAckTimeMs(
                            RoleType.PREFILL.name(),
                            item.prefillEp() == null
                                    ? ""
                                    : item.prefillEp().getIp(),
                            latencyMs));
        }
        if (preparationFailure != null) {
            Logger.error(
                    "Delivery publication preparation isolated: request_id={}",
                    item.requestId(),
                    preparationFailure);
        }
        submitDeliveryResponse(
                confirmation.publication(), response);
    }

    @Override
    public void close() {
        boolean reentrant = publicationDepth.get() != null;
        boolean interrupted = false;
        boolean closeOwner = false;
        synchronized (lifecycleMonitor) {
            if (phase == PublisherPhase.CLOSED) {
                rethrow(closeFailure);
                return;
            }
            if (phase == PublisherPhase.CLOSING) {
                if (reentrant) {
                    return;
                }
                while (phase != PublisherPhase.CLOSED) {
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
            phase = PublisherPhase.CLOSING;
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
            phase = PublisherPhase.CLOSED;
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
        } catch (RejectedExecutionException closed) {
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

    void exitPublication() {
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
            BooleanSupplier publication) {
        requireOwnedPermit(permit);
        Integer currentDepth = publicationDepth.get();
        publicationDepth.set(currentDepth == null ? 1 : currentDepth + 1);
        try {
            return publication.getAsBoolean();
        } finally {
            if (currentDepth == null) {
                publicationDepth.remove();
            } else {
                publicationDepth.set(currentDepth);
            }
            permit.closePublication();
        }
    }

    private void requireOwnedPermit(
            RequestSlot.PublicationPermit permit) {
        if (!permit.ownedBy(this)) {
            permit.closePublication();
            throw new IllegalStateException(
                    "publication permit belongs to another publisher");
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

    private enum PublisherPhase {
        OPEN,
        CLOSING,
        CLOSED
    }
}

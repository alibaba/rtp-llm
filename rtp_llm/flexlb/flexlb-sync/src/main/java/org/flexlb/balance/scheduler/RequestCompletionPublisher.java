package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.RequestSlot.PublicationKind;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.util.Logger;

import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Publishes frontend completions without running user continuations on a
 * scheduler, endpoint, or transport critical path.
 *
 * <p>RequestSlot selects a concrete publication before calling this executor.
 * The publisher manages execution, in-flight accounting and shutdown only;
 * it never invokes a request transition or arbitrates between responses.
 * External Future operations execute synchronously; internal responses are
 * queued so user continuations run outside scheduler and endpoint locks.
 */
final class RequestCompletionPublisher implements AutoCloseable {

    /**
     * Invocation-local proof that one exact lifecycle edge owns one frontend
     * publication. The capability is never stored in a slot or registry.
     */
    static final class PublicationPermit {
        private final RequestCompletionPublisher publisher;
        final RequestSlot slot;
        final PublicationKind kind;
        private final AtomicBoolean claimed = new AtomicBoolean();
        private final AtomicBoolean closed = new AtomicBoolean();

        PublicationPermit(
                RequestCompletionPublisher publisher,
                RequestSlot slot,
                PublicationKind kind) {
            this.publisher = Objects.requireNonNull(publisher, "publisher");
            this.slot = slot;
            this.kind = kind;
        }

        RequestSlot slot() {
            return slot;
        }

        boolean ownedBy(RequestCompletionPublisher expected) {
            return publisher == expected;
        }

        void closePublication() {
            if (closed.compareAndSet(false, true)) {
                publisher.exitPublication();
            }
        }

        /** Abandon a permit only when no other submitter consumed it. */
        void abandonIfUnclaimed() {
            if (claimed.compareAndSet(false, true)) {
                closePublication();
            }
        }

        /** Settle a claim whose publication could not enter its executor. */
        void abortClaimedPublication() {
            closePublication();
        }

        void claim() {
            if (!claimed.compareAndSet(false, true)) {
                throw new IllegalStateException(
                        "publication permit already consumed for request "
                                + slot.requestId());
            }
        }
    }

    enum ResponseCompletion { RESPONSE, FAILURE, CANCELLATION }

    /** Immutable result of Slot arbitration. Execution never re-enters request decisions. */
    record SelectedPublication(PublicationPermit permit, RequestFuture future, boolean selected,
            ResponseCompletion completion, Response response, Throwable failure, boolean mayInterruptIfRunning) {
        boolean complete() {
            if (!selected) { return false; }
            return switch (completion) {
                case RESPONSE -> future.completeOwned(response);
                case FAILURE -> future.completeExceptionallyOwned(failure);
                case CANCELLATION -> future.cancelOwned(mayInterruptIfRunning);
            };
        }
    }

    private static final int DEFAULT_PUBLISHER_WORKERS = 8;

    private final org.flexlb.service.monitor.BatchSchedulerReporter reporter;
    private final ThreadPoolExecutor executor;
    private final Object lifecycleMonitor = new Object();
    private final ThreadLocal<ArrayDeque<Runnable>> localDrain =
            new ThreadLocal<>();
    private final ThreadLocal<Integer> publicationDepth =
            new ThreadLocal<>();
    private PublisherPhase phase = PublisherPhase.OPEN;
    private int inFlightPublications;
    private Throwable closeFailure;

    RequestCompletionPublisher(int configuredWorkers, org.flexlb.service.monitor.BatchSchedulerReporter reporter) {
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

    // ── 发布许可：认领与归还执行计数 ──

    RequestCompletionPublisher.PublicationPermit tryReservePublication(
            RequestSlot exactSlot,
            RequestSlot.PublicationKind kind) {
        synchronized (lifecycleMonitor) {
            if (phase != PublisherPhase.OPEN) {
                return null;
            }
            inFlightPublications++;
            return new RequestCompletionPublisher.PublicationPermit(
                    this, exactSlot, kind);
        }
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

    private void requireOwnedPermit(
            RequestCompletionPublisher.PublicationPermit permit) {
        if (!permit.ownedBy(this)) {
            permit.closePublication();
            throw new IllegalStateException(
                    "publication permit belongs to another publisher");
        }
    }

    // ── 响应入口：ACK、异步提交与同步 Future 完成 ──

    void submitDelivery(RequestSlot.DeliveryPublication delivery, ExpirationTimer expirationTimer) {
        try {
            if (delivery.requestDeadline() != null) { expirationTimer.cancel(delivery.requestDeadline()); }
        } catch (Throwable failure) {
            Logger.error("Delivery deadline cancellation failed request_id={}", delivery.item().requestId(), failure);
        }
        try {
            if (delivery.batchEnqueueStartedAtMs() > 0L && delivery.item().ctx().getAckAtMs() > 0L) {
                reporter.reportDispatchAckTimeMs(org.flexlb.dao.route.RoleType.PREFILL.name(),
                        delivery.item().prefillEp() == null
                                ? ""
                                : delivery.item().prefillEp().getStatus().getMetricIpPort(),
                        Math.max(0L, delivery.item().ctx().getAckAtMs() - delivery.batchEnqueueStartedAtMs()));
            }
        } catch (Throwable failure) {
            Logger.error("Delivery ACK reporting failed request_id={}", delivery.item().requestId(), failure);
        }
        submit(delivery.publication().slot().selectPublication(delivery.publication(),
                RequestCompletionPublisher.ResponseCompletion.RESPONSE, delivery.response(), null, false));
    }

    /** Queue an already-selected result; all request arbitration has finished. */
    void submit(RequestCompletionPublisher.SelectedPublication publication) {
        RequestCompletionPublisher.PublicationPermit permit = publication.permit();
        requireOutsideSlotLock(permit.slot(), "response submission");
        try {
            enqueue(() -> executePublication(publication));
        } catch (RuntimeException | Error enqueueFailure) {
            permit.abortClaimedPublication();
            throw enqueueFailure;
        }
    }

    /** External Future operations preserve their synchronous completion semantics. */
    boolean publishNow(RequestCompletionPublisher.SelectedPublication publication) {
        try {
            return executePublication(publication);
        } catch (RuntimeException | Error executionFailure) {
            publication.permit().abortClaimedPublication();
            throw executionFailure;
        }
    }

    // ── 执行：排队、重入排空与完成 Future ──

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

    private boolean executePublication(RequestCompletionPublisher.SelectedPublication publication) {
        RequestCompletionPublisher.PublicationPermit permit = publication.permit();
        requireOutsideSlotLock(permit.slot(), "response completion");
        requireOwnedPermit(permit);
        Integer currentDepth = publicationDepth.get();
        publicationDepth.set(currentDepth == null ? 1 : currentDepth + 1);
        try {
            return publication.complete();
        } finally {
            if (currentDepth == null) {
                publicationDepth.remove();
            } else {
                publicationDepth.set(currentDepth);
            }
            permit.closePublication();
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

    // ── 关闭：停止接收、等待在途发布、关闭线程池 ──

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

    // ── 异常汇总 ──

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

    // ── 本类使用的数据类型 ──

    private enum PublisherPhase {
        OPEN,
        CLOSING,
        CLOSED
    }
}

/** Stateless public-future adapter bound to one exact canonical slot. */
final class RequestFuture extends CompletableFuture<Response> {
    private final RequestSlot slot;

    RequestFuture(RequestSlot slot) {
        this.slot = slot;
    }

    @Override
    public boolean complete(Response response) {
        return slot.completeExternal(RequestCompletionPublisher.ResponseCompletion.RESPONSE, response, null, false);
    }

    @Override
    public boolean completeExceptionally(Throwable error) {
        return slot.completeExternal(RequestCompletionPublisher.ResponseCompletion.FAILURE, null, error, false);
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
        return slot.completeExternal(RequestCompletionPublisher.ResponseCompletion.CANCELLATION, null, null, mayInterruptIfRunning);
    }

    boolean completeOwned(Response response) {
        return super.complete(response);
    }

    boolean completeExceptionallyOwned(Throwable error) {
        return super.completeExceptionally(error);
    }

    boolean cancelOwned(boolean mayInterruptIfRunning) {
        return super.cancel(mayInterruptIfRunning);
    }
}

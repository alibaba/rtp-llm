package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.BiPredicate;
import java.util.function.LongSupplier;

import static com.google.common.base.Preconditions.checkState;

/** Schedules exact deadline capabilities and drains accepted registrations on close. */
final class ExpirationTimer implements AutoCloseable {

    private enum DeadlineState {

        PREPARED, FIRED_BEFORE_INSTALL, ARMED, CONSUMED, CANCELED
    }

    /** Exact deadlines detached together for request finalization or timer shutdown. */
    record DetachedDeadlines(RequestDeadline requestDeadline, DecisionDeadline detachedDecisionDeadline,
                             InactivityDeadline inactivityDeadline) {
        void release() {
            Throwable failure = Failures.run(null,
                    inactivityDeadline == null ? null : inactivityDeadline::cancel);
            failure = Failures.run(failure,
                    requestDeadline == null ? null : requestDeadline::cancel);
            failure = Failures.run(failure,
                    detachedDecisionDeadline == null ? null : detachedDecisionDeadline::cancel);
            Failures.rethrow(failure, "request cleanup failed");
        }
    }

    abstract static class DeadlineRegistration {

        private DeadlineState state = DeadlineState.PREPARED;

        private ScheduledFuture<?> scheduled;

        private DeadlineRegistration() { }

        final synchronized void installScheduled(
                ScheduledFuture<?> exactScheduled) {
            checkState(scheduled == null, "deadline already owns a scheduled task");
            scheduled = exactScheduled;
            if (state == DeadlineState.CANCELED) {
                exactScheduled.cancel(false);
            }
        }

        /**
         * Publish only after the exact context has stored this capability.
         */
        final synchronized boolean publishAfterInstall() {
            return switch (state) {
                case PREPARED -> {
                    state = DeadlineState.ARMED;
                    yield false;
                }
                case FIRED_BEFORE_INSTALL -> {
                    state = DeadlineState.CONSUMED;
                    yield true;
                }
                case CANCELED, CONSUMED -> false;
                case ARMED -> throw new IllegalStateException(
                        "deadline capability was published twice");
            };
        }

        final synchronized boolean consume() {
            return switch (state) {
                case PREPARED -> {
                    state = DeadlineState.FIRED_BEFORE_INSTALL;
                    yield false;
                }
                case ARMED -> {
                    state = DeadlineState.CONSUMED;
                    yield true;
                }
                case FIRED_BEFORE_INSTALL, CONSUMED, CANCELED -> false;
            };
        }

        final boolean cancel() {
            ScheduledFuture<?> exactScheduled;
            synchronized (this) {
                if (state == DeadlineState.CONSUMED
                        || state == DeadlineState.CANCELED) {
                    return false;
                }
                state = DeadlineState.CANCELED;
                exactScheduled = scheduled;
            }
            if (exactScheduled != null) {
                exactScheduled.cancel(false);
            }
            return true;
        }
    }

    /**
     * Exact one-shot capability for one request's absolute scheduling deadline.
     */
    static final class RequestDeadline extends DeadlineRegistration {
        private RequestDeadline() { }
    }

    /**
     * Wake-up to recheck the latest request activity, not proof of expiration.
     */
    static final class InactivityDeadline extends DeadlineRegistration {
        private InactivityDeadline() { }
    }

    /**
     * Exact one-shot capability for request visibility and PD handoff detection.
     */
    static final class DecisionDeadline extends DeadlineRegistration {

        private final long deadlineAtMs;

        private DecisionDeadline(long deadlineAtMs) {
            this.deadlineAtMs = deadlineAtMs;
        }

        long deadlineAtMs() { return deadlineAtMs; }
    }

    private final RequestRepository requests;

    private final LongSupplier clock;

    private final ScheduledThreadPoolExecutor executor;

    private final Object registrationMonitor = new Object();

    /** Null while open; all closing callers observe the same completed result. */
    private CompletableFuture<Throwable> closeCompletion;

    private int inflightRegistrations;

    ExpirationTimer(RequestRepository requests) {
        this(requests, System::currentTimeMillis);
    }

    ExpirationTimer(RequestRepository requests, LongSupplier clock) {
        this.requests = Objects.requireNonNull(requests, "lifecycle");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.executor = new ScheduledThreadPoolExecutor(1,
                Thread.ofPlatform().daemon().name("request-scheduler-expiration").factory(),
                new ThreadPoolExecutor.AbortPolicy());
        executor.setRemoveOnCancelPolicy(true);
        executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
    }

    // ── 调度期限：注册、安装与触发 ──
    /**
     * Register one absolute request deadline.
     *
     * @return its exact context-owned capability, or null when the context rejected
     *         installation because another lifecycle transition already won
     */
    RequestDeadline scheduleRequestDeadline(RequestContext context, long deadlineAtMs) {
        return register(context, new RequestDeadline(), delayUntil(deadlineAtMs),
                RequestContext::installRequestDeadline, (requestContext, exact) -> requestContext.scheduler().onSchedulingDeadline(requestContext, exact));
    }

    // ── 可见性期限：计划、注册、触发与取消 ──
    void scheduleDecisionDeadline(RequestContext requestContext) {
        OptionalLong deadline = requestContext.decisionDeadlineAtMs();
        if (deadline.isPresent()) {
            registerDecisionDeadline(requestContext, deadline.getAsLong());
        }
    }

    /**
     * Register one exact stage deadline from the current delivery evidence.
     *
     * @return its exact context-owned capability, or null when the context rejected
     *         installation because another lifecycle transition already won
     */
    DecisionDeadline registerDecisionDeadline(RequestContext context, long deadlineAtMs) {
        return register(context, new DecisionDeadline(deadlineAtMs), delayUntil(deadlineAtMs),
                RequestContext::installDecisionDeadline, RequestContext::onDecisionVisibilityDeadline);
    }

    static void releaseDecisionDeadline(DecisionDeadline exact) {
        if (exact == null) { return; }
        try {
            exact.cancel();
        } catch (Throwable failure) {
            Logger.error("Decision deadline cancellation failed", failure);
        }
    }

    // ── 沉默期限：计划、注册与续期检查 ──
    InactivityDeadline scheduleInactivityDeadline(RequestContext context) {
        if (requests.isClosed()) { return null; }
        OptionalLong deadline;
        synchronized (context) {
            if (!requests.isCurrent(context)) { return null; }
            deadline = context.inactivityDeadlineAtMs();
        }
        return deadline.isEmpty() ? null : register(context, new InactivityDeadline(),
                delayUntil(deadline.getAsLong()), RequestContext::installInactivityDeadline,
                this::inactivityDeadlineExpired);
    }

    private void inactivityDeadlineExpired(RequestContext requestContext, InactivityDeadline exact) {
        requestContext.scheduler().enqueueInactivityDeadline(requestContext, exact, clock.getAsLong(), () -> scheduleInactivityDeadline(requestContext));
    }

    // ── 精确句柄：注册协议、调度与取消 ──
    private <D extends DeadlineRegistration> D register(RequestContext context, D exact,
            long delayMs, BiPredicate<RequestContext, D> install, BiConsumer<RequestContext, D> expire) {
        if (requests.isClosed()) { return null; }
        try {
            beginRegistration();
            try {
                boolean installed = false;
                try {
                    exact.installScheduled(executor.schedule(() -> {
                        if (exact.consume()) { expire.accept(context, exact); }
                    }, delayMs, TimeUnit.MILLISECONDS));
                    synchronized (context) {
                        installed = requests.isCurrent(context) && install.test(context, exact);
                    }
                    if (!installed) { return null; }
                    if (exact.publishAfterInstall()) { expire.accept(context, exact); }
                    return exact;
                } finally {
                    if (!installed) { exact.cancel(); }
                }
            } finally {
                endRegistration();
            }
        } catch (RuntimeException stopped) {
            if (requests.isClosed()) { return null; }
            throw stopped;
        }
    }

    private void beginRegistration() {
        synchronized (registrationMonitor) {
            if (closeCompletion != null) {
                throw new RejectedExecutionException(
                        "ExpirationTimer is closing");
            }
            inflightRegistrations++;
        }
    }

    private void endRegistration() {
        synchronized (registrationMonitor) {
            checkState(inflightRegistrations > 0, "ExpirationTimer registration count underflow");
            inflightRegistrations--;
            if (inflightRegistrations == 0) {
                registrationMonitor.notifyAll();
            }
        }
    }

    private long delayUntil(long deadlineAtMs) {
        long nowMs = clock.getAsLong();
        if (deadlineAtMs <= nowMs) {
            return 0L;
        }
        long delayMs = deadlineAtMs - nowMs;
        return delayMs < 0L ? Long.MAX_VALUE : delayMs;
    }

    // ── 关闭：等待注册完成，摘除并取消全部句柄 ──
    @Override
    public void close() {
        boolean alreadyClosing;
        CompletableFuture<Throwable> completion;
        synchronized (registrationMonitor) {
            alreadyClosing = closeCompletion != null;
            if (!alreadyClosing) {
                closeCompletion = new CompletableFuture<>();
            }
            completion = closeCompletion;
        }
        if (alreadyClosing) {
            // join waits for the owner and preserves the caller's interrupt flag.
            Failures.rethrow(completion.join(), "expiration timer failed");
            return;
        }

        boolean interrupted = false;
        synchronized (registrationMonitor) {
            while (inflightRegistrations != 0) {
                try {
                    registrationMonitor.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }

        Throwable failure = detachAllDeadlines();
        try {
            executor.shutdownNow();
            while (!executor.isTerminated()) {
                try {
                    executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        } catch (RuntimeException | Error shutdownFailure) {
            failure = Failures.append(failure, shutdownFailure);
        } finally {
            completion.complete(failure);
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        Failures.rethrow(failure, "expiration timer failed");
    }

    private Throwable detachAllDeadlines() {
        List<RequestContext> exactContexts;
        try {
            exactContexts = requests.snapshotActive();
        } catch (RuntimeException | Error snapshotFailure) {
            return snapshotFailure;
        }
        Throwable failure = null;
        for (RequestContext exactContext : exactContexts) {
            failure = Failures.run(failure,
                    () -> exactContext.detachDeadlines().release());
        }
        return failure;
    }

}

package org.flexlb.balance.schedule;

import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.LongSupplier;

/**
 * Installs, replaces and cancels deadline tasks for exact request contexts.
 * Each slot retains the current plan revision and timer handles; callbacks record facts through the request scheduler.
 * Closing prevents new registrations, waits for ongoing installations and callbacks, then cancels remaining handles.
 */
public final class ExpirationTimer implements AutoCloseable {
    private static final Kind[] KINDS = Kind.values();

    private final RequestRepository requests;
    private final LongSupplier clock;
    private final TaskTimer timer;
    private final ConcurrentMap<RequestContext, Slot> slots = new ConcurrentHashMap<>();
    private final Object registrationMonitor = new Object();

    private volatile CompletableFuture<Throwable> closing;

    private int registrations;

    ExpirationTimer(RequestRepository requests, TaskTimer timer) {
        this(requests, timer, System::currentTimeMillis);
    }

    ExpirationTimer(RequestRepository requests, TaskTimer timer, LongSupplier clock) {
        this.requests = Objects.requireNonNull(requests);
        this.timer = Objects.requireNonNull(timer);
        this.clock = Objects.requireNonNull(clock);
    }

    @Override
    public void close() {
        CompletableFuture<Throwable> completion;
        boolean owner;
        synchronized (registrationMonitor) {
            owner = closing == null;
            if (owner) {
                closing = new CompletableFuture<>();
            }
            completion = closing;
        }
        if (!owner) {
            Failures.rethrow(completion.join(), "expiration timer failed");
            return;
        }
        boolean interrupted = false;
        Throwable failure = null;
        try {
            interrupted = awaitRegistrations();
            for (RequestContext context : List.copyOf(slots.keySet())) {
                failure = Failures.run(failure, () -> remove(context));
            }
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        } finally {
            completion.complete(failure);
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        Failures.rethrow(failure, "expiration timer failed");
    }

    /**
     * Reconcile a deadline plan, ignoring unchanged or superseded revisions. Successful return
     * means this revision needs no further installation; failures retain retryable handles.
     */
    void update(RequestContext context, Plan plan) {
        Objects.requireNonNull(plan);
        if (context.snapshot().requestId() == null || !needsUpdate(context, plan)) {
            return;
        }
        List<Handle> start = new ArrayList<>(3);
        List<Handle> stop = new ArrayList<>(3);
        if (!beginRegistration()) {
            remove(context);
            return;
        }
        try {
            slots.compute(
                    context, (key, previous) -> replaceHandles(key, plan, previous, start, stop));
            Throwable failure = null;
            for (Handle handle : stop) {
                failure = Failures.run(failure, handle::cancel);
            }
            for (Handle handle : start) {
                failure = Failures.run(failure, () -> install(handle));
            }
            Failures.rethrow(failure, "deadline update failed");
        } finally {
            try {
                if (!requests.isCurrent(context)) {
                    remove(context);
                }
            } finally {
                endRegistration();
            }
        }
    }

    boolean needsUpdate(RequestContext context, Plan plan) {
        Slot observed = slots.get(context);
        if (closing != null && requests.isClosed()) {
            return observed != null;
        }
        return requests.isCurrent(context)
                && (observed == null
                        || plan.revision() > observed.revision()
                        || plan.revision() == observed.revision()
                                && observed.hasFailedInstallation());
    }

    /**
     * Archive drops infrastructure ownership after the terminal record's atomic directory commit.
     */
    void remove(RequestContext context) {
        Slot removed = slots.get(context);
        Throwable failure = null;
        if (removed != null) {
            for (Kind kind : List.of(Kind.INACTIVITY, Kind.REQUEST, Kind.VISIBILITY)) {
                Handle handle = removed.handles()[kind.ordinal()];
                if (handle != null) {
                    failure = Failures.run(failure, handle::cancel);
                }
            }
        }
        if (failure == null && removed != null) {
            slots.remove(context, removed);
        }
        Failures.rethrow(failure, "deadline removal failed");
    }

    /**
     * Replace only the exact deadline handles; scheduling and cancellation stay outside compute.
     */
    private Slot replaceHandles(
            RequestContext key, Plan plan, Slot previous, List<Handle> start, List<Handle> stop) {
        if (!requests.isCurrent(key)
                || previous != null
                        && (plan.revision() < previous.revision()
                                || plan.revision() == previous.revision()
                                        && !previous.hasFailedInstallation())) {
            return previous;
        }
        var next = new Handle[KINDS.length];
        for (Kind kind : KINDS) {
            Token desired = plan.token(kind);
            Handle old = previous == null ? null : previous.handles()[kind.ordinal()];
            if (desired != null && (desired.owner() != key || desired.kind() != kind)) {
                throw new IllegalArgumentException("deadline belongs to another request or kind");
            }
            if (old != null && old.token == desired && !old.installationFailed) {
                next[kind.ordinal()] = old;
                continue;
            }
            if (old != null) {
                stop.add(old);
            }
            if (desired != null) {
                var fresh = new Handle(desired);
                next[kind.ordinal()] = fresh;
                start.add(fresh);
            }
        }
        return new Slot(plan.revision(), next);
    }

    private void install(Handle handle) {
        if (!requests.isCurrent(handle.token.owner())) {
            handle.cancel();
            return;
        }
        try {
            handle.install(timer.schedule(handle::fire, delayUntil(handle.token.atMs())));
        } catch (RuntimeException | Error failure) {
            Failures.run(failure, handle::cancel);
            synchronized (handle) {
                handle.installationFailed = !handle.claimed;
            }
            throw failure;
        }
    }

    private long delayUntil(long atMs) {
        long nowMs = clock.getAsLong();
        if (atMs <= nowMs) {
            return 0L;
        }
        long delay = atMs - nowMs;
        return delay < 0L ? Long.MAX_VALUE : delay;
    }

    private boolean beginRegistration() {
        synchronized (registrationMonitor) {
            if (closing != null) {
                if (requests.isClosed()) {
                    return false;
                }
                throw new RejectedExecutionException("ExpirationTimer is closing");
            }
            registrations++;
            return true;
        }
    }

    private void endRegistration() {
        synchronized (registrationMonitor) {
            if (--registrations == 0) {
                registrationMonitor.notifyAll();
            }
        }
    }

    private boolean awaitRegistrations() {
        boolean interrupted = false;
        synchronized (registrationMonitor) {
            while (registrations != 0) {
                try {
                    registrationMonitor.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        return interrupted;
    }

    public enum Kind {
        REQUEST,
        VISIBILITY,
        INACTIVITY
    }

    /**
     * Identifies one request deadline by kind, version and absolute firing time.
     */
    public record Token(RequestContext owner, Kind kind, long version, long atMs) {
        public Token {
            Objects.requireNonNull(owner);
            Objects.requireNonNull(kind);
            if (version <= 0L || atMs <= 0L) {
                throw new IllegalArgumentException("invalid deadline identity");
            }
        }
    }

    /**
     * Retains exact unchanged deadline tokens while revising deadlines that need rescheduling.
     */
    public record Plan(long revision, Token request, Token visibility, Token inactivity) {
        /**
         * Keep unchanged tokens exact, and keep the earlier activity recheck already scheduled.
         */
        Plan schedule(RequestContext owner, long requestAt, long visibilityAt, long inactivityAt) {
            if (inactivity != null && inactivityAt > 0L && inactivity.atMs() <= inactivityAt) {
                inactivityAt = inactivity.atMs();
            }
            long next = revision + 1L;
            var requestToken = token(owner, request, Kind.REQUEST, next, requestAt);
            var visibilityToken = token(owner, visibility, Kind.VISIBILITY, next, visibilityAt);
            var inactivityToken = token(owner, inactivity, Kind.INACTIVITY, next, inactivityAt);
            return requestToken == request
                            && visibilityToken == visibility
                            && inactivityToken == inactivity
                    ? this
                    : new Plan(next, requestToken, visibilityToken, inactivityToken);
        }

        Token token(Kind kind) {
            return switch (kind) {
                case REQUEST -> request;
                case VISIBILITY -> visibility;
                case INACTIVITY -> inactivity;
            };
        }

        Plan without(Kind kind) {
            return new Plan(
                    revision + 1L,
                    kind == Kind.REQUEST ? null : request,
                    kind == Kind.VISIBILITY ? null : visibility,
                    kind == Kind.INACTIVITY ? null : inactivity);
        }

        private static Token token(
                RequestContext owner, Token previous, Kind kind, long revision, long atMs) {
            if (atMs <= 0L) return null;
            return previous != null && previous.atMs() == atMs
                    ? previous
                    : new Token(owner, kind, revision, atMs);
        }
    }

    /**
     * Groups the timer handles installed for one deadline plan revision.
     */
    private record Slot(long revision, Handle[] handles) {
        boolean hasFailedInstallation() {
            for (Handle handle : handles) {
                if (handle != null && handle.installationFailed) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * Coordinates timer installation, firing and cancellation so one deadline is delivered at most once.
     */
    private final class Handle {
        private final Token token;

        private volatile boolean installationFailed;

        private Future<?> scheduled;

        private boolean fired;
        private boolean claimed;
        private boolean canceled;

        private Handle(Token token) {
            this.token = token;
        }

        private void install(Future<?> task) {
            boolean discard, deliver;
            synchronized (this) {
                scheduled = Objects.requireNonNull(task, "scheduled deadline task");
                discard = canceled;
                deliver = fired && !canceled && !claimed;
                if (deliver) claimed = true;
            }
            if (discard) task.cancel(false);
            if (deliver) deliver();
        }

        private void deliver() {
            synchronized (registrationMonitor) {
                if (closing != null) return;
                registrations++;
            }
            try {
                if (requests.isCurrent(token.owner())) {
                    token.owner().getRequestScheduler().onDeadline(token, clock.getAsLong());
                }
            } finally {
                endRegistration();
            }
        }

        private void fire() {
            boolean deliver;
            synchronized (this) {
                fired = true;
                deliver = scheduled != null && !canceled && !claimed;
                if (deliver) {
                    claimed = true;
                }
            }
            if (deliver) {
                deliver();
            }
        }

        private void cancel() {
            Future<?> task;
            synchronized (this) {
                if (canceled || claimed) {
                    return;
                }
                canceled = true;
                task = scheduled;
            }
            if (task != null) {
                task.cancel(false);
            }
        }
    }
}

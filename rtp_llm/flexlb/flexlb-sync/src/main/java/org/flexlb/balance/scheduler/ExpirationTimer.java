package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/** Owns scheduled handles. Context owns only immutable desired deadline tokens. */
public final class ExpirationTimer implements AutoCloseable {
    public enum Kind { REQUEST, VISIBILITY, INACTIVITY }
    private static final Kind[] KINDS = Kind.values();
    public record Token(RequestContext owner, Kind kind, long version, long atMs) {
        public Token {
            Objects.requireNonNull(owner);
            Objects.requireNonNull(kind);
            if (version <= 0L || atMs <= 0L) { throw new IllegalArgumentException("invalid deadline identity"); }
        }
    }
    public record Plan(long revision, Token request, Token visibility, Token inactivity) {
        Token token(Kind kind) {
            return switch (kind) {
                case REQUEST -> request;
                case VISIBILITY -> visibility;
                case INACTIVITY -> inactivity;
            };
        }
    }
    private record Slot(long revision, EnumMap<Kind, Handle> handles) {
        boolean hasFailedInstallation() {
            for (Kind kind : KINDS) {
                Handle handle = handles.get(kind);
                if (handle != null && handle.installationFailed) { return true; }
            }
            return false;
        }
    }

    private final RequestRepository requests;
    private final LongSupplier clock;
    private final ScheduledThreadPoolExecutor executor;
    private final ConcurrentMap<RequestContext, Slot> slots = new ConcurrentHashMap<>();
    private final Object registrationMonitor = new Object();
    private CompletableFuture<Throwable> closing;
    private int registrations;

    ExpirationTimer(RequestRepository requests) { this(requests, System::currentTimeMillis); }
    ExpirationTimer(RequestRepository requests, LongSupplier clock) {
        this.requests = Objects.requireNonNull(requests);
        this.clock = Objects.requireNonNull(clock);
        executor = new ScheduledThreadPoolExecutor(1,
                Thread.ofPlatform().daemon().name("request-scheduler-expiration").factory());
        executor.setRemoveOnCancelPolicy(true);
        executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
    }

    /** Effect order may differ from fact order; only the greatest accepted revision installs handles. */
    void update(RequestContext context, Plan plan) {
        Objects.requireNonNull(plan);
        if (context.snapshot().key() == null) { return; }
        Slot observed = slots.get(context);
        if (!requests.isCurrent(context) || observed != null && (plan.revision() < observed.revision()
                || plan.revision() == observed.revision() && !observed.hasFailedInstallation())) { return; }
        List<Handle> start = new ArrayList<>(3);
        List<Handle> stop = new ArrayList<>(3);
        if (!beginRegistration()) { remove(context); return; }
        try {
            slots.compute(context, (key, previous) -> {
                if (!requests.isCurrent(key) || previous != null && (plan.revision() < previous.revision()
                        || plan.revision() == previous.revision() && !previous.hasFailedInstallation())) { return previous; }
                var next = new EnumMap<Kind, Handle>(Kind.class);
                for (Kind kind : KINDS) {
                    Token desired = plan.token(kind);
                    Handle old = previous == null ? null : previous.handles().get(kind);
                    if (desired != null && (desired.owner() != key || desired.kind() != kind)) {
                        throw new IllegalArgumentException("deadline belongs to another request or kind");
                    }
                    if (old != null && old.token == desired && !old.installationFailed) { next.put(kind, old); continue; }
                    if (old != null) { stop.add(old); }
                    if (desired != null) {
                        var fresh = new Handle(desired);
                        next.put(kind, fresh);
                        start.add(fresh);
                    }
                }
                return new Slot(plan.revision(), next);
            });
            Throwable failure = null;
            for (Handle handle : stop) { failure = Failures.run(failure, handle::cancel); }
            for (Handle handle : start) { failure = Failures.run(failure, () -> install(handle)); }
            Failures.rethrow(failure, "deadline update failed");
        } finally {
            try { if (!requests.isCurrent(context)) { remove(context); } }
            finally { endRegistration(); }
        }
    }

    private void install(Handle handle) {
        if (!requests.isCurrent(handle.token.owner())) { handle.cancel(); return; }
        try {
            handle.install(executor.schedule(handle::fire, delayUntil(handle.token.atMs()), TimeUnit.MILLISECONDS));
            handle.publish();
        } catch (RuntimeException | Error failure) {
            Failures.run(failure, handle::cancel);
            synchronized (handle) { handle.installationFailed = !handle.claimed; }
            throw failure;
        }
    }

    /** Archive drops infrastructure ownership after the terminal record's atomic directory commit. */
    void remove(RequestContext context) {
        Slot removed = slots.get(context);
        Throwable failure = null;
        if (removed != null) {
            for (Kind kind : List.of(Kind.INACTIVITY, Kind.REQUEST, Kind.VISIBILITY)) {
                Handle handle = removed.handles().get(kind);
                if (handle != null) { failure = Failures.run(failure, handle::cancel); }
            }
        }
        if (failure == null && removed != null) { slots.remove(context, removed); }
        Failures.rethrow(failure, "deadline removal failed");
    }

    private long delayUntil(long atMs) {
        long nowMs = clock.getAsLong();
        if (atMs <= nowMs) { return 0L; }
        long delay = atMs - nowMs;
        return delay < 0L ? Long.MAX_VALUE : delay;
    }

    private boolean beginRegistration() {
        synchronized (registrationMonitor) {
            if (closing != null) {
                if (requests.isClosed()) { return false; }
                throw new RejectedExecutionException("ExpirationTimer is closing");
            }
            registrations++;
            return true;
        }
    }
    private void endRegistration() {
        synchronized (registrationMonitor) {
            if (--registrations == 0) { registrationMonitor.notifyAll(); }
        }
    }

    @Override public void close() {
        CompletableFuture<Throwable> completion;
        boolean owner;
        synchronized (registrationMonitor) {
            owner = closing == null;
            if (owner) { closing = new CompletableFuture<>(); }
            completion = closing;
        }
        if (!owner) { Failures.rethrow(completion.join(), "expiration timer failed"); return; }
        boolean interrupted = false;
        Throwable failure = null;
        try {
            synchronized (registrationMonitor) {
                while (registrations != 0) {
                    try { registrationMonitor.wait(); }
                    catch (InterruptedException ignored) { interrupted = true; }
                }
            }
            for (RequestContext context : List.copyOf(slots.keySet())) {
                failure = Failures.run(failure, () -> remove(context));
            }
            failure = Failures.run(failure, executor::shutdownNow);
            while (!executor.isTerminated()) {
                try { executor.awaitTermination(1, TimeUnit.DAYS); }
                catch (InterruptedException ignored) { interrupted = true; }
            }
        } catch (Throwable problem) { failure = Failures.append(failure, problem); }
        finally {
            completion.complete(failure);
            if (interrupted) { Thread.currentThread().interrupt(); }
        }
        Failures.rethrow(failure, "expiration timer failed");
    }

    private final class Handle {
        private final Token token;
        private volatile boolean installationFailed;
        private ScheduledFuture<?> scheduled;
        private boolean published;
        private boolean fired;
        private boolean claimed;
        private boolean canceled;
        private Handle(Token token) { this.token = token; }

        private void install(ScheduledFuture<?> task) {
            boolean discard;
            synchronized (this) { scheduled = task; discard = canceled; }
            if (discard) { task.cancel(false); }
        }
        private void publish() {
            boolean deliver;
            synchronized (this) {
                published = true;
                deliver = fired && !canceled && !claimed;
                if (deliver) { claimed = true; }
            }
            if (deliver) { deliver(); }
        }
        private void fire() {
            boolean deliver;
            synchronized (this) {
                fired = true;
                deliver = published && !canceled && !claimed;
                if (deliver) { claimed = true; }
            }
            if (deliver) { deliver(); }
        }
        private void deliver() {
            if (requests.isCurrent(token.owner())) {
                token.owner().data().getScheduler().onDeadline(token, clock.getAsLong());
            }
        }
        private void cancel() {
            ScheduledFuture<?> task;
            synchronized (this) {
                if (canceled || claimed) { return; }
                canceled = true;
                task = scheduled;
            }
            if (task != null) { task.cancel(false); }
        }
    }
}

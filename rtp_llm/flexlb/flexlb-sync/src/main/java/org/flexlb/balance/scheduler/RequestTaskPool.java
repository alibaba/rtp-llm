package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static com.google.common.base.Preconditions.checkState;

/**
 * Executes request work and serializes each request's close effects outside Context locks.
 * Repeated cleanup notifications coalesce; a reserved response keeps its own queued task.
 * Rejection continuations run before the accepted task releases its count, so shutdown also
 * waits for their cleanup and any accepted continuation for the same request.
 */
public final class RequestTaskPool implements RequestTaskExecutor {
    private Throwable failure;

    private final ExecutorService requestWorkers =
            Executors.newFixedThreadPool(
                    Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8),
                    Thread.ofPlatform().daemon().name("request-work-", 1).factory());
    private final java.util.concurrent.Executor stopWorkers =
            work -> Thread.ofVirtual().name("delivery-stop-work").start(work);
    // Never run request code while holding this monitor. Context -> workLifecycle is the only edge.
    private final RequestWorkQueue requestWorkQueue = new RequestWorkQueue();
    private final ThreadLocal<RequestContext> executingRequest = new ThreadLocal<>();
    private final Object workLifecycle = new Object();

    private boolean acceptingWork = true;
    private long pendingWork, pendingStops;

    /** Offloads effects; Context and endpoint owners arbitrate concurrent work, not executor order. */
    @Override
    public void execute(RequestContext context, Runnable work) {
        Objects.requireNonNull(work, "work");
        try (var registration = registerRequestWork(context)) {
            registration.submit(work);
        }
    }

    public boolean isExecutingOnCurrentThread() {
        return executingRequest.get() != null;
    }

    public void close() {
        synchronized (workLifecycle) {
            acceptingWork = false;
        }
        // A running owner can still hand off its exact request through executingRequest.
        awaitIdle();
        requestWorkers.shutdown();
        boolean interrupted = false;
        while (!requestWorkers.isTerminated()) {
            try {
                requestWorkers.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    public void awaitIdle() {
        boolean interrupted = false;
        synchronized (workLifecycle) {
            // Local drains do not wait for remote cleanup; shutdown also drains stop owners.
            while (pendingWork != 0 || !acceptingWork && pendingStops != 0) {
                try {
                    workLifecycle.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    public void throwIfFailed() {
        synchronized (workLifecycle) {
            Failures.rethrow(failure, "request work failed");
        }
    }

    /** Coalesce the exact stop workflow; local close work has a separate Context key. */
    boolean executeStopWork(RequestContext context, Runnable work) {
        Objects.requireNonNull(work, "work");
        try (var registration = registerRequestWork(context)) {
            synchronized (workLifecycle) {
                pendingWork--;
                pendingStops++;
                registration.stop = true;
                if (pendingWork == 0) {
                    workLifecycle.notifyAll();
                }
            }
            return registration.submit(work, work, false, true, null);
        }
    }

    /** Register before publishing state that obliges a later effect or local cleanup. */
    WorkRegistration registerRequestWork(RequestContext context) {
        WorkRegistration registration = tryRegisterRequestWork(context);
        checkState(registration != null, "request work is closed");
        return registration;
    }

    WorkRegistration tryRegisterRequestWork(RequestContext context) {
        Objects.requireNonNull(context, "context");
        synchronized (workLifecycle) {
            if (!acceptingWork && executingRequest.get() != context) {
                return null;
            }
            pendingWork++;
        }
        return new WorkRegistration(context);
    }

    void recordFailure(Throwable cause) {
        synchronized (workLifecycle) {
            failure = Failures.append(failure, cause);
        }
    }

    /**
     * Counts both the submitting source and its optional task, including the handoff between them.
     */
    final class WorkRegistration implements AutoCloseable, Runnable {
        private final RequestContext context;

        private Runnable work;
        private Consumer<Throwable> rejected;

        private boolean propagate;
        private boolean stop;
        private int owners = 1;
        private boolean sourceClosed;
        private boolean started;
        private boolean executed;

        private WorkRegistration(RequestContext context) {
            this.context = context;
        }

        @Override
        public void run() {
            synchronized (this) {
                if (started) {
                    return;
                }
                started = true;
                executed = true;
            }
            var previous = executingRequest.get();
            executingRequest.set(context);
            try {
                work.run();
            } catch (Throwable failure) {
                recordFailure(failure);
                try {
                    Logger.error(
                            "Request work failed: request_id={}",
                            context.data().getRequestId(),
                            failure);
                } catch (Throwable ignored) {
                }
                if (propagate) {
                    throw Failures.propagate(failure, "request control failed");
                }
            } finally {
                executingRequest.set(previous);
                release();
            }
        }

        @Override
        public void close() {
            synchronized (this) {
                if (sourceClosed) {
                    return;
                }
                sourceClosed = true;
            }
            release();
        }

        void runInline(Runnable work) {
            submit(work, null, true, true, null);
        }

        void submit(Runnable work) {
            submit(work, null, false, true, null);
        }

        void submitCloseResponse(Runnable work, boolean inline, Consumer<Throwable> rejected) {
            submit(work, context, inline, false, rejected);
        }

        void submitClose(Runnable work, boolean inline, Consumer<Throwable> rejected) {
            submit(work, context, inline, true, rejected);
        }

        synchronized boolean canRetrySubmission() {
            return !executed && owners == (sourceClosed ? 0 : 1);
        }

        private boolean submit(
                Runnable work,
                Object owner,
                boolean inline,
                boolean coalesce,
                Consumer<Throwable> rejected) {
            Objects.requireNonNull(work, "work");
            checkState(!Thread.holdsLock(context), "request work submission under Context lock");
            synchronized (this) {
                checkState(!sourceClosed && this.work == null, "request work already handed off");
                this.work = work;
                this.rejected = rejected;
                propagate = inline;
                owners++;
            }
            try {
                if (owner == null) {
                    if (inline) {
                        run();
                    } else {
                        dispatch(this);
                    }
                    return true;
                }
                if (requestWorkQueue.execute(
                        owner, this, inline ? Runnable::run : this::dispatch, coalesce)) {
                    return true;
                }
                release();
                return false;
            } catch (RuntimeException | Error failure) {
                reject(failure);
                throw failure;
            }
        }

        private void dispatch(Runnable task) {
            (stop ? stopWorkers : requestWorkers).execute(task);
        }

        private void release() {
            synchronized (this) {
                if (--owners != 0) {
                    return;
                }
            }
            synchronized (workLifecycle) {
                if (stop) {
                    pendingStops--;
                } else {
                    pendingWork--;
                }
                if (pendingWork == 0) {
                    workLifecycle.notifyAll();
                }
            }
        }

        private void reject(Throwable cause) {
            synchronized (this) {
                if (started) {
                    return;
                }
                started = true;
            }
            recordFailure(cause);
            var previous = executingRequest.get();
            executingRequest.set(context);
            try {
                if (rejected != null) rejected.accept(cause);
            } finally {
                executingRequest.set(previous);
                release();
            }
        }
    }

    /**
     * Serializes each owner's effects outside Context's monitor. Cleanup notifications coalesce;
     * a task carrying a reserved frontend response must run even when cleanup is already pending.
     */
    static final class RequestWorkQueue {
        private final ConcurrentMap<Object, Lane> lanes = new ConcurrentHashMap<>();

        boolean execute(
                Object owner, WorkRegistration work, Consumer<Runnable> start, boolean coalesce) {
            Objects.requireNonNull(work, "work");
            for (; ; ) {
                Lane lane = lanes.get(owner);
                if (lane == null || lane.retired) {
                    lane =
                            lanes.compute(
                                    owner,
                                    (key, old) -> old == null || old.retired ? new Lane(key) : old);
                }
                synchronized (lane) {
                    if (lane.retired) {
                        continue;
                    }
                    if (coalesce && !lane.pending.isEmpty()) {
                        return false;
                    } // One pending notification covers the latest facts.
                    lane.pending.addLast(work);
                    if (lane.started) {
                        return true;
                    }
                }
                try {
                    start.accept(lane);
                } catch (RuntimeException | Error failure) {
                    lane.reject(failure);
                    throw failure;
                }
                return true;
            }
        }

        /**
         * Drains one owner's queued work serially and retires when no accepted work remains.
         */
        private final class Lane implements Runnable {
            private final Object owner;
            private final ArrayDeque<WorkRegistration> pending = new ArrayDeque<>();
            private boolean started;
            private volatile boolean retired;

            private Lane(Object owner) {
                this.owner = owner;
            }

            @Override
            public void run() {
                synchronized (this) {
                    if (started || retired) {
                        return;
                    }
                    started = true;
                }
                Throwable failure = null;
                for (; ; ) {
                    WorkRegistration next;
                    synchronized (this) {
                        next = pending.pollFirst();
                        if (next == null) {
                            retired = true;
                        }
                    }
                    if (next == null) {
                        lanes.remove(owner, this);
                        Failures.rethrow(failure, "request closing failed");
                        return;
                    }
                    failure = Failures.run(failure, next);
                }
            }

            /** A failed start resolves every counted task, including a notification queued during handoff. */
            private void reject(Throwable cause) {
                java.util.List<WorkRegistration> abandoned;
                synchronized (this) {
                    if (started || retired) {
                        return;
                    }
                    retired = true;
                    abandoned = new ArrayList<>(pending);
                    pending.clear();
                }
                lanes.remove(owner, this);
                for (var work : abandoned) {
                    Failures.run(cause, () -> work.reject(cause));
                }
            }
        }
    }
}

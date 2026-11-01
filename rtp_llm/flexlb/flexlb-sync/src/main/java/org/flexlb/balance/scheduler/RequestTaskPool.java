package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static com.google.common.base.Preconditions.checkState;

/** Executes and accounts for request tasks, including cleanup continuations during shutdown. */
public final class RequestTaskPool implements RequestTaskExecutor {
    private final Consumer<Throwable> onFailure;
    private Throwable failure;

    public RequestTaskPool() {
        this.onFailure = this::recordFailure;
    }

    void recordFailure(Throwable cause) {
        synchronized (workLifecycle) {
            failure = Failures.append(failure, cause);
        }
    }

    public void throwIfFailed() {
        synchronized (workLifecycle) {
            Failures.rethrow(failure, "request work failed");
        }
    }

    RequestTaskPool(Consumer<Throwable> onFailure) {
        this.onFailure = onFailure;
    }

    public boolean isExecutingOnCurrentThread() {
        return executingRequest.get() != null;
    }

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
            return registration.submit(work, work, false);
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

    /** Counts both the submitting source and its optional task, including the handoff between them. */
    final class WorkRegistration implements AutoCloseable {
        private final RequestContext context;
        private boolean stop;
        private int owners = 1;
        private boolean submitted;
        private boolean sourceClosed;
        private boolean started;
        private boolean executed;

        private WorkRegistration(RequestContext context) {
            this.context = context;
        }

        void submit(Runnable work) {
            submit(work, null, false);
        }

        void runInline(Runnable work) {
            submit(work, null, true);
        }

        void submitClose(Runnable work, boolean inline) {
            submit(work, context, inline);
        }

        private boolean submit(Runnable work, Object owner, boolean inline) {
            Objects.requireNonNull(work, "work");
            checkState(!Thread.holdsLock(context), "request work submission under Context lock");
            synchronized (this) {
                checkState(!sourceClosed && !submitted, "request work already handed off");
                submitted = true;
                owners++;
            }
            try {
                if (owner == null) {
                    if (inline) {
                        run(work, true);
                    } else {
                        dispatch(() -> run(work, false));
                    }
                    return true;
                }
                if (requestWorkQueue.execute(
                        owner,
                        () -> run(work, inline),
                        this::reject,
                        inline ? Runnable::run : this::dispatch)) {
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

        private void reject(Throwable cause) {
            synchronized (this) {
                if (started) {
                    return;
                }
                started = true;
            }
            onFailure.accept(cause);
            release();
        }

        private void run(Runnable work, boolean propagate) {
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
                onFailure.accept(failure);
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

        synchronized boolean hasTaskOwner() {
            return owners > (sourceClosed ? 0 : 1);
        }

        synchronized boolean canRetrySubmission() {
            return !executed && owners == (sourceClosed ? 0 : 1);
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
}

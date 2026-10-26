package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Deadline;
import io.grpc.ForwardingServerCall;
import io.grpc.ForwardingServerCallListener;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallExecutorSupplier;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerStreamTracer;
import io.grpc.Status;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Headers-phase receiving admission, independent of accepted business queue length. Requires a
 * direct bootstrap executor and gRPC 1.65.1 per-call SerializingExecutor ordering. Initial
 * allowances plus body wire increments form a logical budget, not a physical Netty Direct-memory
 * measurement. Normal receiving completion is decoded half-close; terminal cleanup waits for the
 * application listener, including message producers already queued when the transport closed.
 */
public final class GrpcReceivingGuard extends ServerStreamTracer.Factory
        implements ServerCallExecutorSupplier, ServerInterceptor, AutoCloseable {
    private static final Context.Key<Lease> LEASE = Context.key("receive-guard");
    private final int limit;
    private final long timeoutNanos;
    private final Executor business;
    private final Executor control;
    private final int controlLimit;
    private final AtomicInteger controlReceiving = new AtomicInteger();
    private final ScheduledThreadPoolExecutor timer;
    private final java.util.Set<Lease> active = ConcurrentHashMap.newKeySet();
    private volatile boolean stopping;
    private final AtomicInteger receiving = new AtomicInteger();
    private final AtomicLong rejected = new AtomicLong();
    private final AtomicLong timedOut = new AtomicLong();
    private final AtomicLong receivingBytes = new AtomicLong();
    private final AtomicLong controlReceivingBytes = new AtomicLong();
    private final long scheduleByteLimit;
    private final long controlByteLimit;
    private final long initialReservation;

    public long receivingBytes() {
        return receivingBytes.get();
    }

    public long controlReceivingBytes() {
        return controlReceivingBytes.get();
    }

    // Package-private convenience for deterministic lifecycle fixtures. Production always uses
    // the explicit lane byte budgets resolved at startup by GrpcReceivingBudget.
    GrpcReceivingGuard(int limit, long timeoutMillis, Executor business) {
        this(limit, 1, timeoutMillis, business, business);
    }

    GrpcReceivingGuard(
            int limit, int controlLimit, long timeoutMillis, Executor business, Executor control) {
        this(
                limit,
                controlLimit,
                timeoutMillis,
                business,
                control,
                Long.MAX_VALUE,
                Long.MAX_VALUE,
                GrpcReceivingBudget.INITIAL_RESERVATION_BYTES);
    }

    public GrpcReceivingGuard(
            int limit,
            int controlLimit,
            long timeoutMillis,
            Executor business,
            Executor control,
            long scheduleByteLimit,
            long controlByteLimit,
            long initialReservation) {
        if (limit <= 0
                || controlLimit <= 0
                || timeoutMillis <= 0
                || initialReservation <= 0
                || scheduleByteLimit < initialReservation
                || controlByteLimit < initialReservation) throw new IllegalArgumentException();
        this.scheduleByteLimit = scheduleByteLimit;
        this.controlByteLimit = controlByteLimit;
        this.initialReservation = initialReservation;
        this.limit = limit;
        this.controlLimit = controlLimit;
        this.control = control;
        this.timeoutNanos = TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
        this.business = business;
        this.timer =
                new ScheduledThreadPoolExecutor(
                        1,
                        r -> {
                            Thread t = new Thread(r, "receive-guard-timer");
                            t.setDaemon(true);
                            return t;
                        });
        timer.setRemoveOnCancelPolicy(true);
        timer.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
    }

    public int receiving() {
        return receiving.get();
    }

    public int controlReceiving() {
        return controlReceiving.get();
    }

    public int pendingTimers() {
        return timer.getQueue().size();
    }

    public long rejected() {
        return rejected.get();
    }

    public long timedOut() {
        return timedOut.get();
    }

    private boolean acquire(AtomicInteger counter, int maximum) {
        if (stopping) return false;
        for (; ; ) {
            int v = counter.get();
            if (v >= maximum) return false;
            if (counter.compareAndSet(v, v + 1)) return true;
        }
    }

    private boolean acquireBytes(AtomicLong counter, long maximum, long bytes) {
        for (; ; ) {
            long current = counter.get();
            if (bytes < 0 || bytes > maximum || current > maximum - bytes) return false;
            if (counter.compareAndSet(current, current + bytes)) return true;
        }
    }

    @Override
    public ServerStreamTracer newServerStreamTracer(String method, Metadata headers) {
        boolean isControl = method.endsWith("/Cancel");
        AtomicInteger counter = isControl ? controlReceiving : receiving;
        AtomicLong byteCounter = isControl ? controlReceivingBytes : receivingBytes;
        long byteLimit = isControl ? controlByteLimit : scheduleByteLimit;
        boolean admitted = acquire(counter, isControl ? controlLimit : limit);
        if (admitted && !acquireBytes(byteCounter, byteLimit, initialReservation)) {
            counter.decrementAndGet();
            admitted = false;
        }
        Lease lease = new Lease(admitted, counter, isControl, byteCounter, byteLimit);
        active.add(lease);
        return new ServerStreamTracer() {
            @Override
            public Context filterContext(Context c) {
                return c.withValue(LEASE, lease);
            }

            @Override
            public void inboundWireSize(long bytes) {
                lease.charge(bytes);
            }

            @Override
            public void streamClosed(Status status) {
                lease.terminal();
            }
        };
    }

    @Override
    public <Q, S> Executor getExecutor(ServerCall<Q, S> call, Metadata headers) {
        Lease lease = LEASE.get();
        if (lease == null) {
            call.close(Status.INTERNAL.withDescription("missing receiving lease"), new Metadata());
            return Runnable::run;
        }
        synchronized (lease) {
            lease.call = call;
            if (lease.closed || lease.finished) return Runnable::run;
        }
        if (!lease.admitted) {
            rejected.incrementAndGet();
            lease.close(Status.RESOURCE_EXHAUSTED.withDescription("receiving budget full"));
            return Runnable::run;
        }
        long delay = Math.max(0, timeoutNanos - (System.nanoTime() - lease.created));
        Deadline clientDeadline = Context.current().getDeadline();
        if (clientDeadline != null)
            delay =
                    Math.min(
                            delay, Math.max(0, clientDeadline.timeRemaining(TimeUnit.NANOSECONDS)));
        try {
            synchronized (lease) {
                if (lease.closed || lease.finished) return Runnable::run;
                lease.timeout = timer.schedule(lease::expire, delay, TimeUnit.NANOSECONDS);
            }
        } catch (RuntimeException | Error e) {
            lease.close(Status.UNAVAILABLE.withDescription("receiving watchdog unavailable"));
            return Runnable::run;
        }
        Executor delegate = lease.control ? control : business;
        return command -> {
            Runnable cleanupAware =
                    () -> {
                        try {
                            command.run();
                        } catch (Error failure) {
                            lease.kickAfterError(command);
                            throw failure;
                        } finally {
                            lease.afterTask(command);
                        }
                    };
            try {
                delegate.execute(cleanupAware);
            } catch (RejectedExecutionException rejected) {
                cleanupAware.run();
            }
        };
    }

    @Override
    public <Q, S> ServerCall.Listener<Q> interceptCall(
            ServerCall<Q, S> call, Metadata headers, ServerCallHandler<Q, S> next) {
        Lease lease = LEASE.get();
        if (lease == null) return new ServerCall.Listener<>() {};
        synchronized (lease) {
            if (!lease.admitted || lease.closed) return new ServerCall.Listener<>() {};
            lease.handlerStarted = true;
        }
        ServerCall<Q, S> wrapped =
                new ForwardingServerCall.SimpleForwardingServerCall<>(call) {
                    @Override
                    public void close(Status s, Metadata m) {
                        if (lease.claimClose()) super.close(s, m);
                    }

                    // The watchdog only runs before decoded half-close; ordinary unary responses
                    // are emitted after that point, when it has already been stopped.
                    @Override
                    public void sendHeaders(Metadata m) {
                        if (!lease.closed) super.sendHeaders(m);
                    }

                    @Override
                    public void sendMessage(S m) {
                        if (!lease.closed) super.sendMessage(m);
                    }
                };
        ServerCall.Listener<Q> listener;
        try {
            listener = next.startCall(wrapped, headers);
        } catch (RuntimeException | Error failure) {
            synchronized (lease) {
                lease.startCallFailed = true;
            }
            throw failure;
        }
        return new ForwardingServerCallListener.SimpleForwardingServerCallListener<>(listener) {
            @Override
            public void onHalfClose() {
                synchronized (lease) {
                    if (lease.closed) return;
                    lease.finish();
                }
                super.onHalfClose();
            }

            @Override
            public void onCancel() {
                lease.finish();
                super.onCancel();
            }

            @Override
            public void onComplete() {
                lease.finish();
                super.onComplete();
            }
        };
    }

    private final class Lease {
        final boolean admitted;
        final AtomicInteger counter;
        final boolean control;
        final AtomicLong byteCounter;
        final long byteLimit;
        long chargedBytes;
        final long created = System.nanoTime();
        volatile boolean closed;
        boolean finished;
        boolean handlerStarted;
        boolean transportClosed;
        boolean startCallFailed;
        boolean cleanupMarkerQueued;
        ServerCall<?, ?> call;
        ScheduledFuture<?> timeout;

        Lease(
                boolean admitted,
                AtomicInteger counter,
                boolean control,
                AtomicLong byteCounter,
                long byteLimit) {
            this.admitted = admitted;
            this.counter = counter;
            this.control = control;
            this.byteCounter = byteCounter;
            this.byteLimit = byteLimit;
            this.chargedBytes = admitted ? initialReservation : 0;
        }

        void charge(long bytes) {
            if (bytes <= 0) return;
            synchronized (this) {
                if (!admitted || closed || finished) return;
                if (acquireBytes(byteCounter, byteLimit, bytes)) {
                    chargedBytes += bytes;
                    return;
                }
                // Wire accounting follows one deframer read. Keep the initial window/slack
                // reservation
                // and all prior charges until the same cleanup fence as the receiving permit.
                closed = true;
                rejected.incrementAndGet();
            }
            call.close(
                    Status.RESOURCE_EXHAUSTED.withDescription("receiving byte budget full"),
                    new Metadata());
        }

        synchronized void finish() {
            if (finished) return;
            finished = true;
            if (admitted) {
                counter.decrementAndGet();
                byteCounter.addAndGet(-chargedBytes);
            }
            if (timeout != null) timeout.cancel(false);
            active.remove(this);
        }

        synchronized void terminal() {
            closed = true;
            transportClosed = true;
            if (timeout != null) timeout.cancel(false);
            if (!handlerStarted) finish();
        }

        synchronized boolean claimClose() {
            if (closed) return false;
            closed = true;
            return true;
        }

        void close(Status status) {
            if (claimClose()) call.close(status, new Metadata());
        }

        void expire() {
            synchronized (this) {
                if (finished || closed) return;
                closed = true;
                timedOut.incrementAndGet();
            }
            call.close(
                    Status.DEADLINE_EXCEEDED.withDescription("request body incomplete"),
                    new Metadata());
        }

        // grpc-core 1.65.1 SerializingExecutor implements Runnable and Executor. Its run()
        // restores STOPPED in finally but an Error skips the normal nonempty-queue reschedule.
        // Enqueueing a no-op on that SAME serial executor restarts pending producer cleanup.
        // This path never returns a reservation; the terminal listener/marker must still run.
        void kickAfterError(Runnable task) {
            if (task instanceof Executor) {
                try {
                    ((Executor) task).execute(() -> {});
                } catch (RuntimeException | Error ignored) {
                    /* retain permit if cleanup cannot run */
                }
            }
        }

        // startCall may throw before gRPC installs our terminal listener. streamClosed runs
        // after deframerClosed, so no future transport producers can be appended. Insert the
        // finish marker in the same serial queue, behind all producers already enqueued.
        // This is deliberately NOT a marker in the shared business executor queue.
        // For an unsupported runnable shape, fail closed rather than reuse retained bytes.
        void afterTask(Runnable task) {
            synchronized (this) {
                if (!transportClosed
                        || !startCallFailed
                        || finished
                        || cleanupMarkerQueued
                        || !(task instanceof Executor)) return;
                cleanupMarkerQueued = true;
            }
            try {
                ((Executor) task).execute(this::finish);
            } catch (RuntimeException | Error ignored) {
                /* fail closed, never reuse a possibly retained reservation */
            }
        }
    }

    @Override
    public void close() {
        stopping = true;
        for (Lease lease : active) {
            ServerCall<?, ?> call;
            synchronized (lease) {
                call = lease.call;
            }
            if (call != null)
                lease.close(Status.UNAVAILABLE.withDescription("receiving guard stopped"));
        }
        timer.shutdownNow();
        try {
            timer.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }
}

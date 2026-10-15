package org.flexlb.mockengine;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.ForwardingServerCall;
import io.grpc.ForwardingServerCallListener;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallExecutorSupplier;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;

import java.util.Set;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/** Logical engine quotas; protocol cancellation never waits for an engine worker. */
final class MockRpcExecutors implements AutoCloseable {
    private final ExecutorService control = Executors.newFixedThreadPool(2,
            Thread.ofPlatform().daemon().name("mock-rpc-control-", 0).factory());
    private final ExecutorService callbacks = Executors.newThreadPerTaskExecutor(
            Thread.ofVirtual().name("mock-rpc-", 0).factory());
    private final Set<Engine> engines = ConcurrentHashMap.newKeySet();
    private final AtomicBoolean closing = new AtomicBoolean();
    private final int concurrency;
    private final int maxCalls;
    private boolean terminated;

    MockRpcExecutors(int concurrency, int maxCalls) {
        if (concurrency < 1 || maxCalls < 1) throw new IllegalArgumentException("RPC limits must be positive");
        this.concurrency = concurrency;
        this.maxCalls = maxCalls;
    }

    Executor controlExecutor() { return control; }

    Engine engine() {
        if (closing.get()) throw new IllegalStateException("RPC runtime closed");
        Engine engine = new Engine();
        engines.add(engine);
        return engine;
    }

    final class Engine implements ServerCallExecutorSupplier, ServerInterceptor {
        private final Semaphore workers = new Semaphore(concurrency, true);
        private final Semaphore calls = new Semaphore(maxCalls);
        private final ConcurrentHashMap<ServerCall<?, ?>, Admission> active = new ConcurrentHashMap<>();
        private final AtomicBoolean stopped = new AtomicBoolean();
        private final Set<Thread> running = ConcurrentHashMap.newKeySet();
        private final AtomicInteger waiting = new AtomicInteger();
        private final AtomicInteger callbacksInProgress = new AtomicInteger();
        private final LongAdder rejected = new LongAdder();

        int inFlight() { return maxCalls - calls.availablePermits(); }
        Map<String, Object> snapshot() {
            return Map.of("in_flight", inFlight(), "active_callbacks", concurrency - workers.availablePermits(),
                    "waiting_callbacks", waiting.get(), "rejected_calls_total", rejected.sum(),
                    "concurrency_limit", concurrency, "in_flight_limit", maxCalls);
        }

        @Override
        public <ReqT, RespT> Executor getExecutor(ServerCall<ReqT, RespT> call, Metadata headers) {
            Context context = Context.current();
            Status failure = null;
            boolean admitted = false;
            if (closing.get() || stopped.get()) failure = Status.UNAVAILABLE.withDescription("mock RPC runtime stopping");
            else if (context.isCancelled()) failure = Contexts.statusFromCancelled(context);
            else if (!(admitted = calls.tryAcquire())) failure = Status.RESOURCE_EXHAUSTED
                    .withDescription("mock RPC harness per-engine in-flight limit");
            if (failure != null) rejected.increment();
            Admission admission = new Admission(call, context, admitted, failure);
            active.put(call, admission);
            context.addListener(ignored -> admission.cancelled(), Runnable::run);
            if (closing.get() || stopped.get()) admission.close(Status.UNAVAILABLE, new Metadata());
            // Reject through the normal RPC status path, never executor rejection.
            if (!admitted) return control;
            return command -> {
                callbacksInProgress.incrementAndGet();
                try { callbacks.execute(() -> {
                boolean acquired = false;
                Thread thread = Thread.currentThread();
                running.add(thread);
                admission.runners.add(thread);
                boolean waitingForWorker = true;
                waiting.incrementAndGet();
                try {
                    // Virtual threads wait without occupying another engine's workers.
                    // Cancelled calls still drain gRPC callbacks, bypassing this quota.
                    while (!admission.closed.get() && !context.isCancelled()) {
                        if (workers.tryAcquire(10, TimeUnit.MILLISECONDS)) {
                            acquired = true;
                            break;
                        }
                    }
                    waiting.decrementAndGet();
                    waitingForWorker = false;
                    command.run();
                } catch (InterruptedException interrupted) {
                    // Still deliver terminal callbacks after a cancelled quota wait.
                    if (waitingForWorker) { waiting.decrementAndGet(); waitingForWorker = false; }
                    command.run();
                } finally {
                    if (waitingForWorker) waiting.decrementAndGet();
                    admission.runners.remove(thread);
                    running.remove(thread);
                    if (acquired) workers.release();
                    synchronized (Engine.this) {
                        callbacksInProgress.decrementAndGet();
                        Engine.this.notifyAll();
                    }
                }
            }); } catch (RuntimeException error) {
                synchronized (this) { callbacksInProgress.decrementAndGet(); notifyAll(); }
                throw error;
            }
            };
        }

        @Override
        public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
            Admission admission = active.get(call);
            if (admission == null) {
                if (!Context.current().isCancelled()) call.close(Status.INTERNAL.withDescription("missing mock RPC admission"), new Metadata());
                return new ServerCall.Listener<>() {};
            }
            if (admission.closed.get() || Context.current().isCancelled()) return new ServerCall.Listener<>() {};

            if (admission.failure != null) {
                admission.close(admission.failure, new Metadata());
                return new ServerCall.Listener<>() {};
            }
            ServerCall<ReqT, RespT> owned = new ForwardingServerCall.SimpleForwardingServerCall<>(call) {
                @Override public void close(Status status, Metadata trailers) { admission.close(status, trailers); }
            };
            ServerCall.Listener<ReqT> listener = next.startCall(owned, headers);
            return new ForwardingServerCallListener.SimpleForwardingServerCallListener<>(listener) {
                @Override public void onHalfClose() {
                    if (!admission.closed.get() && !admission.context.isCancelled()) super.onHalfClose();
                }
            };
        }

        void stop() {
            stopped.set(true);
            active.values().forEach(admission -> admission.close(Status.UNAVAILABLE
                    .withDescription("mock RPC engine stopping"), new Metadata()));
            running.forEach(Thread::interrupt);
        }

        synchronized void awaitStopped(long deadlineNanos) {
            if (!stopped.get()) throw new IllegalStateException("engine admission is still open");
            while (callbacksInProgress.get() != 0) {
                long remaining = deadlineNanos - System.nanoTime();
                if (remaining <= 0) throw new IllegalStateException("engine RPC callbacks did not stop; ownership retained");
                try { TimeUnit.NANOSECONDS.timedWait(this, remaining); }
                catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("interrupted awaiting engine callbacks", error);
                }
            }
            engines.remove(this);
        }

        private final class Admission {
            final ServerCall<?, ?> call;
            final Context context;
            final boolean permit;
            final Status failure;
            final AtomicBoolean closed = new AtomicBoolean();
            final AtomicBoolean released = new AtomicBoolean();
            final Set<Thread> runners = ConcurrentHashMap.newKeySet();
            Admission(ServerCall<?, ?> call, Context context, boolean permit, Status failure) {
                this.call = call; this.context = context; this.permit = permit; this.failure = failure;
            }
            void release() { if (permit && released.compareAndSet(false, true)) calls.release(); }
            void cancelled() {
                closed.set(true);
                release();
                active.remove(call, this);
                if (context.cancellationCause() != null)
                    runners.forEach(thread -> { if (thread != Thread.currentThread()) thread.interrupt(); });
            }
            void close(Status status, Metadata trailers) {
                if (closed.compareAndSet(false, true)) {
                    try { call.close(status, trailers); } finally { release(); }
                }
            }
        }
    }

    @Override
    public synchronized void close() {
        if (terminated) return;
        closing.set(true);
        engines.forEach(Engine::stop);
        // The owner calls close after server transports terminate. Drain their
        // already-delivered callbacks before retiring the protocol executor.
        control.shutdown();
        boolean drained = false;
        boolean protocolDrained = false;
        try {
            protocolDrained = control.awaitTermination(2, TimeUnit.SECONDS);
            if (!protocolDrained) {
                control.shutdownNow();
                protocolDrained = control.awaitTermination(1, TimeUnit.SECONDS);
            }
            callbacks.shutdown();
            drained = callbacks.awaitTermination(1, TimeUnit.SECONDS);
            if (!drained) {
                callbacks.shutdownNow();
                drained = callbacks.awaitTermination(3, TimeUnit.SECONDS);
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        } finally {
            if (!drained) callbacks.shutdownNow();
            if (!protocolDrained) control.shutdownNow();
        }
        if (!drained || !protocolDrained) throw new IllegalStateException("mock RPC callbacks did not drain within shutdown budget");
        terminated = true;
    }
}

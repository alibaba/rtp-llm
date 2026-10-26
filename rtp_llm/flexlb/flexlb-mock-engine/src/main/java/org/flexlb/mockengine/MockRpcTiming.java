package org.flexlb.mockengine;

import io.grpc.*;
import java.util.ArrayDeque;
import java.util.Set;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/** Delays messages, never engine state. A call uses one fixed offset per leg, preserving stream order. */
final class MockRpcTiming implements ServerInterceptor, AutoCloseable {
    private final MockTiming timing;
    private final ScheduledExecutorService timers;
    private final Executor dispatch;
    private final Consumer<java.util.Map<String, Number>> metrics;
    private final Set<Call> calls = ConcurrentHashMap.newKeySet();
    private volatile boolean closed;
    MockRpcTiming(MockTiming timing, ScheduledExecutorService timers, Executor dispatch,
                  Consumer<java.util.Map<String, Number>> metrics) {
        this.timing = timing; this.timers = timers; this.dispatch = dispatch; this.metrics = metrics;
    }
    private static String method(String full) {
        return full.substring(full.lastIndexOf('/') + 1).replaceAll("([a-z0-9])([A-Z])", "$1_$2").toLowerCase(java.util.Locale.ROOT);
    }
    private final class Call {
        final Context context = Context.current();
        final AtomicBoolean done = new AtomicBoolean();
        final Context.CancellationListener cancellation = ignored -> finish();
        final Lane input, output;
        final Consumer<Status> abort;
        Call(String method, Consumer<Status> abort) {
            this.abort = abort;
            input = new Lane(this, "rpc." + method + ".request");
            output = new Lane(this, "rpc." + method + ".response");
            calls.add(this); context.addListener(cancellation, Runnable::run);
            if (closed) finish();
        }
        void finish() {
            if (!done.compareAndSet(false, true)) return;
            input.clear(); output.clear(); calls.remove(this); context.removeListener(cancellation);
        }
    }
    private record Event(long created, long due, Runnable action) { }
    private final class Lane {
        final Call call;
        final String path;
        final MockTiming.Sample sample;
        final ArrayDeque<Event> queue = new ArrayDeque<>();
        ScheduledFuture<?> wake;
        boolean draining;
        Lane(Call call, String path) { this.call = call; this.path = path; sample = timing.sample(path, 0); }
        synchronized void add(Runnable action) {
            if (call.done.get() || call.context.isCancelled()) return;
            long now = System.nanoTime();
            queue.add(new Event(now, now + sample.nanos(), action));
            arm();
        }
        private void arm() {
            if (wake != null || draining || queue.isEmpty() || call.done.get()) return;
            try { wake = timers.schedule(() -> {
                synchronized (this) { wake = null; draining = true; }
                try { dispatch.execute(() -> call.context.run(this::drain)); }
                catch (RejectedExecutionException stopping) { call.finish(); }
            }, Math.max(0, queue.peek().due() - System.nanoTime()), TimeUnit.NANOSECONDS);
            } catch (RejectedExecutionException stopped) { call.finish(); }
        }
        private void drain() {
            try {
                while (true) {
                    Event event;
                    synchronized (this) {
                        if (call.done.get() || call.context.isCancelled() || queue.isEmpty()
                                || queue.peek().due() > System.nanoTime()) return;
                        event = queue.remove();
                    }
                    try { metrics.accept(timing.metrics(path, sample, System.nanoTime() - event.created())); }
                    catch (RuntimeException ignored) { /* telemetry owns no work */ }
                    try { if (!call.done.get() && !call.context.isCancelled()) event.action().run(); }
                    catch (RuntimeException error) {
                        try { call.abort.accept(Status.UNKNOWN.withDescription("mock delayed RPC callback failed").withCause(error)); }
                        finally { call.finish(); }
                        return;
                    }
                }
            } finally { synchronized (this) { draining = false; arm(); } }
        }
        synchronized void clear() { queue.clear(); if (wake != null) wake.cancel(false); wake = null; }
    }
    @Override public <Req, Resp> ServerCall.Listener<Req> interceptCall(ServerCall<Req, Resp> server,
            Metadata headers, ServerCallHandler<Req, Resp> next) {
        var method = method(server.getMethodDescriptor().getFullMethodName());
        if (!timing.rpcEnabled(method)) return next.startCall(server, headers);
        var call = new Call(method, status -> server.close(status, new Metadata()));
        var outbound = new ForwardingServerCall.SimpleForwardingServerCall<Req, Resp>(server) {
            @Override public void sendHeaders(Metadata value) {
                var copy = new Metadata(); copy.merge(value); call.output.add(() -> super.sendHeaders(copy));
            }
            @Override public void sendMessage(Resp value) { call.output.add(() -> super.sendMessage(value)); }
            @Override public void close(Status status, Metadata trailers) {
                var copy = new Metadata(); copy.merge(trailers);
                call.output.add(() -> { try { super.close(status, copy); } finally { call.finish(); } });
            }
        };
        ServerCall.Listener<Req> listener;
        try { listener = next.startCall(outbound, headers); }
        catch (RuntimeException error) { call.finish(); throw error; }
        return new ForwardingServerCallListener.SimpleForwardingServerCallListener<>(listener) {
            @Override public void onMessage(Req value) { call.input.add(() -> super.onMessage(value)); }
            @Override public void onHalfClose() { call.input.add(super::onHalfClose); }
            @Override public void onReady() { call.input.add(super::onReady); }
            @Override public void onCancel() { call.finish(); super.onCancel(); }
            @Override public void onComplete() { call.finish(); super.onComplete(); }
        };
    }
    int pendingCalls() { return calls.size(); }
    @Override public void close() { closed = true; calls.forEach(Call::finish); }
}

package org.flexlb.mock.engine.rtp;

import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
import org.flexlb.mock.metrics.EngineMetrics;
import static org.flexlb.mock.engine.RequestFailure.Reason.*;
import static org.flexlb.mock.engine.RequestState.Phase.*;

/** RTP scheduling domains. Native frames, RPC observers and transfer clients stay in adapters. */
final class RtpScheduler implements Scheduler<RtpScheduler.Request>, AutoCloseable {
    static final int LIMIT = 1024;
    record Input(String identity, long requestId, long groupId, int tokens, int maxNewTokens, int priority,
                 long started, long deadline, List<Long> keys, boolean reuse, boolean retained) {
        Input {
            if (identity == null || identity.isBlank() || tokens < 1 || maxNewTokens < 1)
                throw new IllegalArgumentException("Invalid scheduling input");
            keys = List.copyOf(keys);
        }
    }
    interface Listener {
        default boolean writable() { return true; }
        default boolean cancelled() { return false; }
        default void allocated(Request request) {}
        void output(Request request, int committed, boolean finished);
        void terminal(Request request, RequestFailure failure);
    }
    final class Request {
        final Input input;
        final Listener listener;
        final RequestState state;
        DeviceCache.Lease lease;
        RequestState.Use use;
        ScheduledFuture<?> timeout;
        Future<?> capacity;
        int reused, stopAfter;
        long stepStarted;
        DecodeStepModel.Step step;
        boolean generating;
        Request(Input input, Listener listener) {
            this.input = input; this.listener = listener;
            state = new RequestState(input.deadline(), failure -> finished(this, failure));
        }
        RequestState.Snapshot snapshot() { return state.snapshot(); }
    }
    private interface Policy {
        void submit(Request request);
        Work<Request> next();
        void completed(Request request, SimulatedCompute.Outcome outcome);
        void remove(Request request);
    }
    private final Object lock;
    private final RtpConfig config;
    private final DeviceCache cache;
    private final Map<String, Request> active = new LinkedHashMap<>();
    private final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("mock-request-deadline").factory());
    private final ExecutorService capacity = Executors.newVirtualThreadPerTaskExecutor();
    private final EngineLoop<Request> loop;
    private final EngineMetrics metrics;
    private final Policy policy;
    private boolean closed;

    RtpScheduler(Object lock, RtpConfig config, DeviceCache cache) {
        this.lock = lock; this.config = config; this.cache = cache;
        policy = config.role() == Role.prefill ? new Prefill() : new Decode();
        loop = new EngineLoop<>(lock, this, config.compute(), config.role(), config.id());
        metrics = new EngineMetrics(config.role().name(), this::metricState);
        deadlines.setRemoveOnCancelPolicy(true);
    }
    Request submit(Input input, Listener listener) {
        synchronized (lock) {
            if (closed) throw new RequestFailure(CLOSED, "Engine is closing");
            if (active.containsKey(input.identity())) throw new RequestFailure(DUPLICATE, "Active request identity already exists");
            if (active.size() >= LIMIT) throw new RequestFailure(OVERLOADED, "Mock request limit reached");
            var request = new Request(input, Objects.requireNonNull(listener));
            active.put(input.identity(), request); metrics.started();
            if (input.deadline() != Long.MAX_VALUE)
                request.timeout = deadlines.schedule(() -> finish(request, new RequestFailure(TIMEOUT, "Request budget expired")),
                    Math.max(0, input.deadline() - System.nanoTime()), TimeUnit.NANOSECONDS);
            policy.submit(request); loop.wake();
            return request;
        }
    }
    void generate(Request request, int stopAfter) {
        synchronized (lock) {
            if (!(policy instanceof Decode decode) || request.state.terminating() || request.lease == null || request.generating
                    || stopAfter < 1 || stopAfter > request.input.maxNewTokens())
                throw new RequestFailure(INVALID, "Decode cannot start before resource and transfer readiness");
            if (request.listener.cancelled()) { finish(request, new RequestFailure(CANCELLED, "Output consumer closed")); return; }
            request.generating = true; request.stopAfter = stopAfter; request.state.transferred(1);
            if (stopAfter == 1) {
                try { request.listener.output(request, 1, true); finish(request, null); }
                catch (RuntimeException error) { finish(request, new RequestFailure(CANCELLED, "Output consumer closed", error)); }
            } else { request.state.phase(QUEUED); decode.ready.addLast(request); loop.wake(); }
        }
    }
    void resume(Request request) {
        synchronized (lock) {
            if (policy instanceof Decode decode && !request.state.terminating() && request.snapshot().phase() == OUTPUT) {
                request.state.phase(QUEUED); decode.ready.addLast(request); loop.wake();
            }
        }
    }
    void consume(Request request, Runnable deliver) {
        synchronized (lock) {
            requireReady(request); deliver.run(); finish(request, null);
        }
    }
    void requireReady(Request request) {
        synchronized (lock) {
            if (request.state.terminating() || request.snapshot().phase() != READY || request.state.expired())
                throw new RequestFailure(INVALID, "Prefill transfer data is no longer owned");
        }
    }
    void finish(Request request, RequestFailure failure) {
        synchronized (lock) {
            if (request.state.terminating()) return;
            request.state.finish(failure);
            policy.remove(request);
            loop.cancel(request); // shortcut: RTP Work owns one request; a shared-batch policy must cancel the batch only when all members stop.
            loop.wake();
        }
    }
    private void finished(Request request, RequestFailure failure) {
        if (request.timeout != null) request.timeout.cancel(false);
        if (request.capacity != null) request.capacity.cancel(true);
        active.remove(request.input.identity(), request);
        policy.remove(request);
        metrics.finished(failure == null ? EngineMetrics.Outcome.success : switch (failure.reason()) {
            case CANCELLED -> EngineMetrics.Outcome.cancelled;
            case TIMEOUT -> EngineMetrics.Outcome.timeout;
            default -> EngineMetrics.Outcome.error;
        });
        request.listener.terminal(request, failure);
    }
    List<Request> requests() { synchronized (lock) { return List.copyOf(active.values()); } }
    QueueSnapshot queues() { synchronized (lock) { return QueueSnapshot.of(active.values().stream().map(Request::snapshot).toList()); } }
    String scrapeMetrics() { return metrics.scrape(); }
    private EngineMetrics.State metricState() {
        synchronized (lock) {
            var counts = queues(); var kv = cache.snapshot(0, false); long size = config.blockSize();
            return new EngineMetrics.State(counts.waiting(), counts.running(), kv.totalBlocks() * size,
                kv.availableBlocks() * size, kv.freeBlocks() * size, kv.heldBlocks() * size);
        }
    }
    private int blocks(int tokens) { return (int) (((long) tokens + config.blockSize() - 1) / config.blockSize()); }
    private DeviceCache.Failure acquire(Request request, boolean reuse) {
        int matched = Math.min(request.input.keys().size(), (request.input.tokens() - 1) / config.blockSize());
        var admission = cache.acquire(blocks(request.input.tokens()), reuse ? request.input.keys().subList(0, matched) : List.of());
        if (admission.lease() == null) return admission.failure();
        request.lease = admission.lease(); request.reused = request.lease.reusedBlocks() * config.blockSize();
        request.state.attach(request.lease, () -> request.lease.complete(request.input.reuse() && config.reuseCache() && config.deviceCache()
            ? request.input.keys() : List.of()));
        metrics.cacheLookup(request.input.tokens(), request.reused);
        return null;
    }
    private Work<Request> work(Request request, int target, int generated, int tokens, String batchId) {
        var input = request.input;
        var item = new ForwardBatch.Request(input.identity(), input.tokens(), request.reused, target,
            generated, tokens, input.priority(), List.of());
        var batch = new ForwardBatch(batchId, List.of(new ForwardBatch.Partition(0, List.of(item))));
        request.use = request.state.forward(); request.stepStarted = System.nanoTime();
        return new Work<>(request, batch);
    }
    @Override public Work<Request> schedule() { synchronized (lock) { return closed ? null : policy.next(); } }
    @Override public void completed(Request request, SimulatedCompute.Outcome outcome) {
        synchronized (lock) {
            try {
                if (request.state.terminating()) return;
                if (outcome.state() != SimulatedCompute.State.COMPLETED || request.listener.cancelled()) finish(request, new RequestFailure(CANCELLED, "Forward cancelled"));
                else if (request.state.expired()) finish(request, new RequestFailure(TIMEOUT, "Request expired during Forward"));
                else policy.completed(request, outcome);
            } finally { releaseUse(request); }
        }
    }
    @Override public void failed(Request request, RuntimeException error) {
        synchronized (lock) {
            try { finish(request, new RequestFailure(INTERNAL, "Forward failed: " + error.getMessage(), error)); }
            finally { releaseUse(request); }
        }
    }
    private void releaseUse(Request request) {
        var use = request.use; request.use = null;
        if (use != null) use.close();
    }
    private final class Prefill implements Policy {
        private final ArrayDeque<Request> waiting = new ArrayDeque<>();
        public void submit(Request request) { waiting.addLast(request); }
        public Work<Request> next() {
            while (!waiting.isEmpty()) {
                var request = waiting.removeFirst();
                if (request.state.terminating()) continue;
                if (request.state.expired()) { finish(request, new RequestFailure(TIMEOUT, "Prefill expired in queue")); continue; }
                try {
                    var failure = acquire(request, config.reuseCache() && config.deviceCache() && request.input.reuse());
                    if (failure != null) { finish(request, new RequestFailure(RequestFailure.Reason.CAPACITY, "KV allocation failed: " + failure)); continue; }
                    return work(request, 1, 0, request.input.tokens() - request.reused, "local/" + request.input.requestId());
                } catch (RuntimeException error) { failed(request, error); }
            }
            return null;
        }
        public void completed(Request request, SimulatedCompute.Outcome outcome) {
            request.state.commit(1, outcome.elapsedNanos());
            metrics.forwardCompleted(outcome.elapsedNanos(), outcome.sample().planned().toNanos(), request.input.tokens(), request.input.tokens() - request.reused, 1);
            request.state.phase(READY);
            request.listener.output(request, 1, true);
            if (!request.input.retained()) finish(request, null);
        }
        public void remove(Request request) { waiting.remove(request); }
    }
    private final class Decode implements Policy {
        private final ArrayDeque<Request> ready = new ArrayDeque<>();
        public void submit(Request request) { request.state.phase(RequestState.Phase.CAPACITY); request.capacity = capacity.submit(() -> awaitCapacity(request, 0)); }
        public Work<Request> next() {
            while (!ready.isEmpty()) {
                var request = ready.removeFirst();
                if (request.state.terminating()) continue;
                if (request.state.expired()) { finish(request, new RequestFailure(TIMEOUT, "Decode expired in queue")); continue; }
                if (request.listener.cancelled()) { finish(request, new RequestFailure(CANCELLED, "Output consumer closed")); continue; }
                if (!request.listener.writable()) { request.state.phase(OUTPUT); continue; }
                try {
                    var progress = request.snapshot();
                    if (request.step == null) request.step = config.decodeModel().sample(Long.toString(request.input.requestId()), progress.iterations(), request.input.maxNewTokens() - progress.generated());
                    int demand = blocks(Math.addExact(Math.addExact(request.input.tokens(), progress.generated()), request.step.proposedTokens()));
                    var failure = request.lease.growTo(demand);
                    if (failure == DeviceCache.Failure.PERMANENT) { finish(request, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode growth exceeds capacity")); continue; }
                    if (failure != null) {
                        request.state.phase(RequestState.Phase.CAPACITY); request.capacity = capacity.submit(() -> awaitCapacity(request, demand)); continue;
                    }
                    return work(request, request.input.maxNewTokens(), progress.generated(), request.step.verifyTokens(), request.input.identity() + "/" + progress.generated());
                } catch (RuntimeException error) { failed(request, error); }
            }
            return null;
        }
        public void completed(Request request, SimulatedCompute.Outcome outcome) {
            int committed = Math.min(request.step.committedTokens(), request.stopAfter - request.snapshot().generated());
            request.state.commit(committed, outcome.elapsedNanos());
            metrics.forwardCompleted(outcome.elapsedNanos(), outcome.sample().planned().toNanos(), request.step.verifyTokens(), request.step.verifyTokens(), committed);
            request.step = null;
            boolean done = request.snapshot().generated() >= request.stopAfter;
            try { request.listener.output(request, committed, done); }
            catch (RuntimeException error) { finish(request, new RequestFailure(CANCELLED, "Output consumer closed", error)); return; }
            if (done) finish(request, null);
            else { request.state.phase(QUEUED); ready.addLast(request); }
        }
        public void remove(Request request) { ready.remove(request); }
        private void awaitCapacity(Request request, int growth) {
            long observed = Long.MIN_VALUE;
            try {
                while (true) {
                    synchronized (lock) {
                        if (request.state.terminating()) return;
                        if (request.state.expired()) { finish(request, new RequestFailure(TIMEOUT, "Decode expired waiting for capacity")); return; }
                        long generation = cache.allocationGeneration();
                        if (observed != generation) {
                            var failure = growth > 0 ? request.lease.growTo(growth) : acquire(request,
                                config.reuseCache() && config.deviceCache() && !config.disableDecodeFirstReuse() && request.input.reuse());
                            observed = cache.allocationGeneration();
                            if (failure == null) {
                                if (growth > 0) { request.state.phase(QUEUED); ready.addLast(request); loop.wake(); }
                                else { request.state.phase(TRANSFER); request.listener.allocated(request); }
                                return;
                            }
                            if (failure == DeviceCache.Failure.PERMANENT) { finish(request, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode demand exceeds capacity")); return; }
                        }
                    }
                    cache.awaitCapacityChange(observed, 50);
                }
            } catch (InterruptedException error) { Thread.currentThread().interrupt(); finish(request, new RequestFailure(CANCELLED, "Capacity wait interrupted", error)); }
            catch (RuntimeException error) { finish(request, new RequestFailure(INTERNAL, "Capacity wait failed", error)); }
        }
    }
    @Override public void close() {
        synchronized (lock) {
            if (closed) return;
            closed = true;
            for (var request : List.copyOf(active.values())) finish(request, new RequestFailure(CLOSED, "Engine closed"));
            loop.close(); deadlines.shutdownNow(); capacity.shutdownNow();
        }
    }
}

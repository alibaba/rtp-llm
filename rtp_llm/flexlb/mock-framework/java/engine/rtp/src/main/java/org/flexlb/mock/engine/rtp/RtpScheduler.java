package org.flexlb.mock.engine.rtp;

import java.util.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.DeviceCache;
import org.flexlb.mock.metrics.EngineMetrics;
import static org.flexlb.mock.engine.RequestFailure.Reason.*;
import static org.flexlb.mock.engine.RequestState.Phase.*;

/** RTP queue policy. No RPC objects, executors or compute loop live here. */
final class RtpScheduler implements Scheduler {
    private final Object lock;
    private final RtpConfig config;
    private final DeviceCache cache;
    private final EngineMetrics metrics;
    private final Map<String, Entry> active = new LinkedHashMap<>();
    private final Set<Entry> owned = new LinkedHashSet<>();
    private final ArrayDeque<Entry> waiting = new ArrayDeque<>();
    private final Set<Entry> running = new LinkedHashSet<>();
    private final Set<Entry> notifications = new LinkedHashSet<>();
    private Work executing;
    private long batchSequence;
    private boolean closed;

    private final class Entry implements Request {
        final Input input;
        final Listener listener;
        final RequestState state;
        final ArrayDeque<Notice> notices = new ArrayDeque<>();
        DeviceCache.Lease lease;
        int reused, stopAfter;
        long stepStarted, attemptedGeneration = Long.MIN_VALUE;
        boolean loaded, generating, outputPending, delivering, readStarted;
        DecodeStepModel.Step nextStep;

        Entry(Input input, Listener listener) {
            this.input = input;
            this.listener = listener;
            this.state = new RequestState(input.deadline(), failure -> terminal(this, failure));
        }
        RtpScheduler owner() { return RtpScheduler.this; }
        @Override public Input input() { return input; }
        @Override public RequestState.Snapshot snapshot() { synchronized (lock) { return state.snapshot(); } }
        @Override public int reused() { synchronized (lock) { return reused; } }
        @Override public long stepStarted() { synchronized (lock) { return stepStarted; } }
        @Override public boolean running() { synchronized (lock) { return !state.terminating() && running.contains(this); } }
        @Override public boolean allocated() { synchronized (lock) { return lease != null; } }
    }
    private record Member(Entry request, RequestState.Use use, DecodeStepModel.Step step) {}
    private record Batch(List<Member> members) {}
    private record Notice(Runnable action, boolean terminal) {}

    static Scheduler.Factory factory(RtpConfig config) {
        Objects.requireNonNull(config);
        return (lock, cache) -> new RtpScheduler(lock, config, cache);
    }

    RtpScheduler(Object lock, RtpConfig config, DeviceCache cache) {
        this.lock = Objects.requireNonNull(lock);
        this.config = Objects.requireNonNull(config);
        this.cache = Objects.requireNonNull(cache);
        this.metrics = new EngineMetrics(config.role().name(), this::metricState);
    }

    @Override public Request submit(Input input, Listener listener) {
        if (closed) throw new RequestFailure(CLOSED, "Engine is closing");
        if ((long) input.tokens() + (prefill() ? 1 : input.maxNewTokens()) > config.maxSeqLen()
                || prefill() && config.maxBatchTokens() > 0 && input.tokens() > config.maxBatchTokens())
            throw new RequestFailure(INVALID, "Request exceeds scheduling limits");
        if (active.containsKey(input.identity())) throw new RequestFailure(DUPLICATE, "Active request identity already exists");
        if (owned.size() >= RtpConfig.REQUEST_LIMIT) throw new RequestFailure(OVERLOADED, "Request/delivery limit reached");
        var entry = new Entry(input, Objects.requireNonNull(listener));
        active.put(input.identity(), entry);
        owned.add(entry);
        metrics.started();
        if (prefill()) waiting.addLast(entry);
        else entry.state.phase(RequestState.Phase.CAPACITY);
        return entry;
    }

    @Override public void dependencyCompleted(Request handle, Dependency dependency) {
        var entry = entry(handle);
        if (entry.state.terminating()) return;
        if (entry.state.expired()) { finish(entry, new RequestFailure(TIMEOUT, "Dependency arrived after deadline")); return; }
        if (dependency instanceof Loaded loaded) {
            if (prefill() || entry.lease == null || entry.loaded || entry.generating || loaded.stopAfter() > entry.input.maxNewTokens())
                throw new RequestFailure(INVALID, "Unexpected KV load completion");
            entry.loaded = true;
            entry.stopAfter = loaded.stopAfter();
            entry.state.phase(TRANSFER);
            return;
        }
        switch ((Signal) dependency) {
            case GENERATE -> {
                if (prefill() || !entry.loaded || entry.generating)
                    throw new RequestFailure(INVALID, "Generation requires loaded KV and a new generation permit");
                entry.generating = true;
                entry.state.transferred(1);
                if (entry.stopAfter == 1) emit(entry, 1, true);
                else { entry.state.phase(QUEUED); waiting.addLast(entry); }
            }
            case PREFILL_COMPLETE -> {
                if (prefill() || !entry.loaded || entry.generating || entry.stopAfter != 1)
                    throw new RequestFailure(INVALID, "Unexpected Prefill-only completion");
                entry.state.transferred(1);
                finish(entry, null);
            }
            case TRANSFER_ACKNOWLEDGED -> {
                if (!prefill() || !entry.input.retained() || !entry.readStarted || entry.state.snapshot().phase() != READY)
                    throw new RequestFailure(INVALID, "Transfer acknowledgement without a readable Prefill source");
                finish(entry, null);
            }
            case OUTPUT_WRITABLE -> { /* The next scheduling pass reads the consumer's readiness. */ }
        }
    }

    @Override public void maintain() {
        for (var entry : List.copyOf(active.values())) {
            if (!live(entry)) continue;
            if (!prefill() && entry.lease == null) {
                var failure = acquire(entry);
                if (failure == null) {
                    entry.state.phase(LOADING);
                    enqueue(entry, () -> { if (liveForDelivery(entry)) entry.listener.allocated(entry); }, false);
                } else if (failure == DeviceCache.Failure.PERMANENT) {
                    finish(entry, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode allocation exceeds capacity"));
                }
            }
        }
    }

    @Override public Work schedule() {
        if (closed) return null;
        if (executing != null) throw new IllegalStateException("Forward already in flight");
        int limit = prefill() ? config.maxContextBatchSize() : config.maxGenerateBatchSize();
        long budget = config.maxBatchTokens() > 0 ? config.maxBatchTokens() : Long.MAX_VALUE;
        var selected = new ArrayList<Entry>();
        if (!prefill()) {
            for (var entry : List.copyOf(running)) {
                if (selected.size() >= limit) break;
                int tokens = workTokens(entry);
                if (tokens > 0 && tokens <= budget && reserve(entry)) { selected.add(entry); budget -= tokens; }
            }
        }
        int queued = waiting.size();
        while (queued-- > 0 && selected.size() < limit && (prefill() || running.size() < limit)) {
            var entry = waiting.removeFirst();
            if (!live(entry)) continue;
            int tokens = workTokens(entry);
            if (tokens <= 0 || tokens > budget || !reserve(entry)) {
                if (!entry.state.terminating()) waiting.addLast(entry);
                continue;
            }
            running.add(entry);
            selected.add(entry);
            budget -= tokens;
        }
        if (selected.isEmpty()) return null;
        var members = new ArrayList<Member>();
        var items = new ArrayList<ForwardBatch.Request>();
        for (var entry : selected) {
            var progress = entry.state.snapshot();
            int tokens = prefill() ? entry.input.tokens() - entry.reused : entry.nextStep.verifyTokens();
            items.add(new ForwardBatch.Request(entry.input.identity(), entry.input.tokens(), entry.reused,
                prefill() ? 1 : entry.input.maxNewTokens(), progress.generated(), tokens, entry.input.priority(), List.of()));
            members.add(new Member(entry, entry.state.forward(), entry.nextStep));
            entry.stepStarted = System.nanoTime();
        }
        var batch = new ForwardBatch(config.id() + "/" + ++batchSequence, List.of(new ForwardBatch.Partition(0, items)));
        executing = new Work(new Batch(List.copyOf(members)), batch);
        return executing;
    }

    private int workTokens(Entry entry) {
        if (!live(entry) || entry.outputPending) return 0;
        if (!entry.listener.writable()) { entry.state.phase(OUTPUT); return 0; }
        if (prefill()) return entry.input.tokens();
        var progress = entry.state.snapshot();
        if (entry.nextStep == null)
            entry.nextStep = config.decodeModel().sample(Long.toString(entry.input.requestId()), progress.iterations(), entry.stopAfter - progress.generated());
        if (config.maxBatchTokens() > 0 && entry.nextStep.verifyTokens() > config.maxBatchTokens()) {
            finish(entry, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode step exceeds batch token budget"));
            return 0;
        }
        return entry.nextStep.verifyTokens();
    }

    private boolean reserve(Entry entry) {
        if (prefill()) {
            var failure = acquire(entry);
            if (failure == DeviceCache.Failure.PERMANENT)
                finish(entry, new RequestFailure(RequestFailure.Reason.CAPACITY, "Prefill allocation exceeds capacity"));
            return failure == null;
        }
        int demand = blocks(Math.addExact(Math.addExact(entry.input.tokens(), entry.state.snapshot().generated()), entry.nextStep.proposedTokens()));
        var failure = entry.lease.growTo(Math.max(demand, entry.lease.blocks()));
        if (failure != null) {
            // RTP running streams fail KV growth; only admission waits for capacity.
            finish(entry, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode KV growth failed: " + failure));
            return false;
        }
        entry.state.phase(RUNNING);
        return true;
    }

    @Override public void completed(Work work, SimulatedCompute.Outcome outcome) {
        var batch = completedBatch(work);
        long input = 0, computed = 0, output = 0;
        try {
            for (var member : batch.members()) {
                var entry = member.request();
                input += prefill() ? entry.input.tokens() : member.step().verifyTokens();
                computed += prefill() ? entry.input.tokens() - entry.reused : member.step().verifyTokens();
                if (!live(entry)) continue;
                if (outcome.state() != SimulatedCompute.State.COMPLETED) {
                    finish(entry, new RequestFailure(CANCELLED, "Forward cancelled"));
                    continue;
                }
                int committed = prefill() ? 1 : Math.min(member.step().committedTokens(), entry.stopAfter - entry.state.snapshot().generated());
                entry.state.commit(committed, outcome.elapsedNanos());
                entry.nextStep = null;
                output += committed;
                boolean done = prefill() || entry.state.snapshot().generated() >= entry.stopAfter;
                if (prefill()) { running.remove(entry); entry.state.phase(READY); }
                emit(entry, committed, done);
            }
            if (outcome.state() == SimulatedCompute.State.COMPLETED)
                metrics.forwardCompleted(outcome.elapsedNanos(), outcome.sample().planned().toNanos(), input, computed, output);
        } finally { batch.members().forEach(member -> member.use().close()); }
    }

    @Override public void failed(Work work, RuntimeException error) {
        var batch = completedBatch(work);
        var failure = error instanceof RequestFailure known ? known : new RequestFailure(INTERNAL, "Forward failed", error);
        try { batch.members().forEach(member -> finish(member.request(), failure)); }
        finally { batch.members().forEach(member -> member.use().close()); }
    }
    @Override public boolean cancelled(Work work) {
        return work == executing && ((Batch) work.owner()).members().stream().allMatch(member -> member.request().state.terminating());
    }
    private Batch completedBatch(Work work) {
        if (work != executing) throw new IllegalStateException("Stale/foreign Forward completion");
        executing = null;
        return (Batch) work.owner();
    }

    private void emit(Entry entry, int committed, boolean done) {
        var snapshot = entry.state.snapshot();
        var output = new Output(snapshot.generated(), snapshot.iterations(), committed, done);
        entry.outputPending = true;
        if (!prefill()) entry.state.phase(OUTPUT);
        enqueue(entry, () -> {
            if (!liveForDelivery(entry)) return;
            entry.listener.output(entry, output);
            synchronized (lock) {
                entry.outputPending = false;
                if (!live(entry)) return;
                if (done && !entry.input.retained()) finish(entry, null);
                else if (!prefill()) entry.state.phase(RUNNING);
            }
        }, false);
    }

    @Override public RequestState.Use readReady(Request request) {
        var entry = entry(request);
        if (!prefill() || entry.state.terminating() || entry.state.expired() || entry.state.snapshot().phase() != READY)
            throw new RequestFailure(INVALID, "Prefill transfer data is not ready or no longer owned");
        entry.readStarted = true;
        return entry.state.pin();
    }
    @Override public void finish(Request request, RequestFailure failure) {
        var entry = entry(request);
        if (entry.state.terminating()) return;
        waiting.remove(entry);
        running.remove(entry);
        entry.state.finish(failure);
    }
    private void terminal(Entry entry, RequestFailure failure) {
        active.remove(entry.input.identity(), entry);
        metrics.finished(failure == null ? EngineMetrics.Outcome.success : switch (failure.reason()) {
            case CANCELLED -> EngineMetrics.Outcome.cancelled;
            case TIMEOUT -> EngineMetrics.Outcome.timeout;
            default -> EngineMetrics.Outcome.error;
        });
        enqueue(entry, () -> {
            try { entry.listener.terminal(entry, failure); }
            finally { synchronized (lock) { owned.remove(entry); } }
        }, true);
    }
    private void enqueue(Entry entry, Runnable notification, boolean terminal) {
        // At most admission, one output, and terminal can be pending for a request.
        if (entry.notices.size() >= 3) throw new IllegalStateException("Request notification overflow");
        entry.notices.addLast(new Notice(notification, terminal));
        notifications.add(entry);
    }
    @Override public List<Runnable> takeNotifications() {
        var result = new ArrayList<Runnable>();
        for (var entry : List.copyOf(notifications)) {
            if (entry.delivering) continue;
            notifications.remove(entry);
            var notification = entry.notices.removeFirst();
            entry.delivering = true;
            result.add(() -> {
                try { notification.action().run(); }
                catch (RuntimeException error) {
                    synchronized (lock) {
                        if (notification.terminal()) throw error;
                        finish(entry, new RequestFailure(CANCELLED, "Request consumer failed", error));
                    }
                } finally {
                    synchronized (lock) {
                        entry.delivering = false;
                        if (!entry.notices.isEmpty()) notifications.add(entry);
                    }
                }
            });
        }
        return result;
    }

    private boolean liveForDelivery(Entry entry) { synchronized (lock) { return live(entry); } }
    private boolean live(Entry entry) {
        if (entry.state.terminating()) return false;
        if (entry.state.expired()) finish(entry, new RequestFailure(TIMEOUT, "Request budget expired"));
        else if (entry.listener.cancelled()) finish(entry, new RequestFailure(CANCELLED, "Output consumer closed"));
        return !entry.state.terminating();
    }
    private Entry entry(Request request) {
        if (!(request instanceof RtpScheduler.Entry entry) || entry.owner() != this)
            throw new IllegalArgumentException("Foreign request handle");
        return entry;
    }
    private boolean prefill() { return config.role() == Role.prefill; }
    private int blocks(int tokens) { return (int) (((long) tokens + config.blockSize() - 1) / config.blockSize()); }
    private DeviceCache.Failure acquire(Entry entry) {
        if (entry.lease != null) return null;
        long generation = cache.allocationGeneration();
        if (entry.attemptedGeneration == generation) return DeviceCache.Failure.RETRYABLE;
        entry.attemptedGeneration = generation;
        boolean reuse = config.reuseCache() && config.deviceCache() && entry.input.reuse()
            && (prefill() || !config.disableDecodeFirstReuse());
        int matched = Math.min(entry.input.keys().size(), (entry.input.tokens() - 1) / config.blockSize());
        var admission = cache.acquire(blocks(entry.input.tokens()), reuse ? entry.input.keys().subList(0, matched) : List.of());
        if (admission.lease() == null) return admission.failure();
        entry.lease = admission.lease();
        entry.reused = admission.lease().reusedBlocks() * config.blockSize();
        entry.state.attach(entry.lease, () -> entry.lease.complete(
            config.reuseCache() && config.deviceCache() && entry.input.reuse() ? entry.input.keys() : List.of()));
        metrics.cacheLookup(entry.input.tokens(), entry.reused);
        return null;
    }
    @Override public List<Request> requests() { return new ArrayList<>(active.values()); }
    @Override public QueueSnapshot queues() {
        int queued = 0, executing = 0, loading = 0, ready = 0;
        var waitingMembers = Set.copyOf(waiting);
        for (var entry : active.values()) {
            if (entry.state.terminating()) continue;
            if (running.contains(entry)) executing++;
            else if (waitingMembers.contains(entry) || entry.state.snapshot().phase() == RequestState.Phase.CAPACITY) queued++;
            else if (entry.state.snapshot().phase() == READY) ready++;
            else loading++;
        }
        return new QueueSnapshot(queued, executing, loading, ready);
    }
    private EngineMetrics.State metricState() {
        synchronized (lock) {
            var counts = queues();
            var kv = cache.snapshot(0, false);
            long block = config.blockSize();
            return new EngineMetrics.State(counts.waiting(), counts.running(), counts.loading(), counts.ready(), kv.totalBlocks() * block,
                kv.availableBlocks() * block, kv.freeBlocks() * block, kv.heldBlocks() * block);
        }
    }
    @Override public String scrapeMetrics() { return metrics.scrape(); }
    @Override public void close() {
        if (closed) return;
        closed = true;
        for (var entry : List.copyOf(active.values())) finish(entry, new RequestFailure(CLOSED, "Engine closed"));
    }
}

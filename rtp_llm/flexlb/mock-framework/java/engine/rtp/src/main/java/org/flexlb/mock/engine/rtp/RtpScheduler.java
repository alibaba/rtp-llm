package org.flexlb.mock.engine.rtp;

import java.util.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.metrics.EngineMetrics;
import static org.flexlb.mock.engine.RequestFailure.Reason.*;
import static org.flexlb.mock.engine.rtp.RtpScheduler.Phase.*;

/** RTP queue policy. No RPC objects, executors or compute loop live here. */
final class RtpScheduler implements Scheduler {
    enum Phase { QUEUED, CAPACITY, LOADING, TRANSFERRING, TRANSFERRED, RUNNING, OUTPUT, FORWARD, READY }
    private final TaskHistory history = new TaskHistory();
    private final Object lock;
    private final RtpConfig config;
    private final KvCache cache;
    private final EngineMetrics metrics;
    private final Map<String, Entry> active = new LinkedHashMap<>();
    private final Set<Entry> owned = new LinkedHashSet<>();
    private final ArrayDeque<Entry> waiting = new ArrayDeque<>();
    private final ArrayDeque<ArrayDeque<Entry>> groups = new ArrayDeque<>();
    private boolean preferGroup;
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
        ArrayDeque<Entry> group;
        KvCache.Lease lease;
        int reused;
        boolean prefillFinished;
        long admittedAt = Long.MIN_VALUE, attemptedGeneration = Long.MIN_VALUE;
        Phase phase = QUEUED;
        ReportedPhase finalPhase;
        boolean loaded, generating, outputPending, delivering, readStarted;
        ComputeModel.Plan nextStep;

        Entry(Input input, Listener listener) {
            this.input = input;
            this.listener = listener;
            this.state = new RequestState(input.deadline(), failure -> terminal(this, failure));
        }
        RtpScheduler owner() { return RtpScheduler.this; }
        @Override public Input input() { return input; }
        @Override public RequestState.Snapshot snapshot() { synchronized (lock) { return state.snapshot(); } }
        @Override public int reused() { synchronized (lock) { return reused; } }

    }
    private record Member(Entry request, RequestState.Use use, ComputeModel.Plan step) {}
    private record Batch(List<Member> members) {}
    private record Notice(Runnable action, boolean terminal) {}

    static Scheduler.Factory factory(RtpConfig config) {
        Objects.requireNonNull(config);
        return (lock, cache) -> new RtpScheduler(lock, config, cache);
    }

    RtpScheduler(Object lock, RtpConfig config, KvCache cache) {
        this.lock = Objects.requireNonNull(lock);
        this.config = Objects.requireNonNull(config);
        this.cache = Objects.requireNonNull(cache);
        this.metrics = new EngineMetrics(config.role().name(), this::metricState);
    }

    @Override public Request input(Input input, Listener listener) {
        if (closed) throw new RequestFailure(CLOSED, "Engine is closing");
        long length = (long) input.tokens() + (prefill() ? 1 : input.maxNewTokens());
        if (prefill() ? length > config.maxSeqLen() : length >= config.maxSeqLen())
            throw new RequestFailure(INVALID, "Request exceeds scheduling limits");
        if (active.containsKey(input.identity())) throw new RequestFailure(DUPLICATE, "Active request identity already exists");
        if (owned.size() >= RtpConfig.REQUEST_LIMIT) throw new RequestFailure(OVERLOADED, "Request/delivery limit reached");
        var entry = new Entry(input, Objects.requireNonNull(listener));
        active.put(input.identity(), entry);
        owned.add(entry);
        metrics.started();
        if (prefill()) waiting.addLast(entry);
        else entry.phase = Phase.CAPACITY;
        return entry;
    }

    @Override public List<InputResult> input(List<InputItem> batch) {
        if (closed) throw new RequestFailure(CLOSED, "Engine is closing");
        if (!prefill()) throw new RequestFailure(INVALID, "Explicit batches belong to Prefill");
        batch = List.copyOf(batch);
        var results = new ArrayList<InputResult>();
        var group = new ArrayDeque<Entry>();
        for (var item : batch) {
            try {
                var entry = (Entry) input(item.input(), item.listener());
                waiting.remove(entry); group.addLast(entry);
                results.add(new InputResult(entry, null));
            } catch (RequestFailure failure) { results.add(new InputResult(null, failure)); }
        }
        if (group.size() > config.maxGenerateBatchSize()
                || config.maxInitedKvStreams() > 0 && group.size() > config.maxInitedKvStreams()) {
            waiting.addAll(group); // RTP falls oversized groups back to ordinary FIFO admission.
        } else if (!group.isEmpty()) {
            group.forEach(entry -> entry.group = group); groups.addLast(group);
        }
        return List.copyOf(results);
    }

    @Override public void control(Request handle, Control control) {
        var entry = entry(handle);
        if (entry.state.terminating()) return;
        if (entry.state.expired()) { finish(entry, new RequestFailure(TIMEOUT, "Control arrived after deadline")); return; }
        if (!(control instanceof RtpControl event)) throw new RequestFailure(INVALID, "Foreign RTP control");
        switch (event) {
            case GENERATE -> {
                if (prefill() || !entry.loaded || entry.generating || entry.prefillFinished)
                    throw new RequestFailure(INVALID, "Generation requires loaded KV and a new generation permit");
                entry.generating = true;
                entry.state.transferred(1);
                entry.phase = QUEUED; waiting.addLast(entry);
            }
            case PREFILL_COMPLETE -> {
                if (prefill() || !entry.loaded || entry.generating || !entry.prefillFinished)
                    throw new RequestFailure(INVALID, "Unexpected Prefill-only completion");
                entry.state.transferred(1);
                finish(entry, null);
            }
            case TRANSFER_ACKNOWLEDGED -> {
                if (!prefill() || !entry.input.retained() || !entry.readStarted || entry.phase != TRANSFERRED)
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
                    entry.phase = LOADING;
                    enqueue(entry, () -> { if (liveForDelivery(entry)) entry.listener.allocated(entry); }, false);
                } else if (failure == KvCache.Failure.PERMANENT) {
                    finish(entry, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode allocation exceeds capacity"));
                }
            }
        }
    }

    @Override public Work schedule() {
        if (closed) return null;
        if (executing != null) throw new IllegalStateException("Forward already in flight");
        int limit = config.maxGenerateBatchSize();
        var selected = new ArrayList<Entry>();
        if (prefill()) {
            boolean groupFirst = !groups.isEmpty() && (preferGroup || waiting.isEmpty());
            if (groupFirst) {
                selectGroup(selected);
                if (selected.isEmpty()) selectPrefill(waiting, selected, false);
            } else {
                selectPrefill(waiting, selected, false);
                if (selected.isEmpty()) selectGroup(selected);
            }
            if (!selected.isEmpty()) preferGroup = selected.getFirst().group == null && !groups.isEmpty();
        } else {
            for (var entry : List.copyOf(running)) {
                if (selected.size() >= limit) break;
                if (workTokens(entry) > 0 && reserve(entry)) selected.add(entry);
            }
            int queued = waiting.size();
            while (queued-- > 0 && selected.size() < limit && running.size() < limit) {
                var entry = waiting.removeFirst();
                if (!live(entry)) continue;
                if (workTokens(entry) <= 0 || !reserve(entry)) {
                    if (!entry.state.terminating()) waiting.addLast(entry);
                    continue;
                }
                running.add(entry); selected.add(entry);
            }
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
            entry.phase = FORWARD;
        }
        var batch = new ForwardBatch(config.id() + "/" + ++batchSequence, List.of(new ForwardBatch.Partition(0, items)));
        executing = new Work(new Batch(List.copyOf(members)), batch);
        return executing;
    }

    private void selectGroup(List<Entry> selected) {
        if (!groups.isEmpty()) selectPrefill(groups.getFirst(), selected, true);
    }
    private void selectPrefill(ArrayDeque<Entry> queue, List<Entry> selected, boolean explicit) {
        long uncached = 0;
        for (var entry : List.copyOf(queue)) {
            if (!live(entry)) continue;
            if (selected.size() >= config.maxGenerateBatchSize()) break;
            if (!explicit && config.maxUncachedBatchTokens() > 0 && uncached >= config.maxUncachedBatchTokens()) continue;
            if (entry.lease == null && config.maxInitedKvStreams() > 0
                    && active.values().stream().filter(e -> e.lease != null).count() >= config.maxInitedKvStreams()) continue;
            if (workTokens(entry) <= 0 || !fitsPrefill(selected, entry) || !reserve(entry)) continue;
            queue.remove(entry); running.add(entry); selected.add(entry);
            uncached += entry.input.tokens() - entry.reused;
        }
        if (explicit && queue.isEmpty()) groups.remove(queue);
    }
    private boolean fitsPrefill(List<Entry> selected, Entry candidate) {
        if (selected.isEmpty()) return candidate.input.tokens() < config.maxSeqLen();
        long sum = candidate.input.tokens(), longest = candidate.input.tokens();
        for (var entry : selected) {
            sum += entry.input.tokens(); longest = Math.max(longest, entry.input.tokens());
        }
        // Native prefill-only means max_new_tokens == 0; this Mock currently accepts positive output lengths only.
        long budget = (long) config.maxBatchTokens() - 1;
        return sum <= budget && longest * (selected.size() + 1L) <= budget;
    }

    private int workTokens(Entry entry) {
        if (!live(entry) || entry.outputPending) return 0;
        if (!entry.listener.writable()) { entry.phase = OUTPUT; return 0; }
        var progress = entry.state.snapshot();
        if (entry.nextStep == null)
            entry.nextStep = config.compute().prepare(config.role(), new ComputeModel.Request(
                Long.toString(entry.input.requestId()), progress.iterations(), entry.input.tokens(), progress.generated(),
                entry.input.minNewTokens(), entry.input.maxNewTokens(), config.maxSeqLen(), entry.input.ignoreEos()));
        if (prefill()) return entry.input.tokens();
        return entry.nextStep.verifyTokens();
    }

    private boolean reserve(Entry entry) {
        if (prefill()) {
            var failure = acquire(entry);
            if (failure == KvCache.Failure.PERMANENT)
                finish(entry, new RequestFailure(RequestFailure.Reason.CAPACITY, "Prefill allocation exceeds capacity"));
            return failure == null;
        }
        long tokens = (long) entry.input.tokens() + entry.state.snapshot().generated() + entry.nextStep.additionalKvTokens();
        var failure = entry.lease.ensureCapacity(tokens);
        if (failure != null) {
            // RTP running streams fail KV growth; only admission waits for capacity.
            finish(entry, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode KV growth failed: " + failure));
            return false;
        }
        entry.phase = RUNNING;
        return true;
    }

    @Override public void computeCompleted(Work work, SimulatedCompute.Outcome outcome) {
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
                var result = config.compute().completed(member.step());
                int committed = result.committedTokens();
                entry.state.commit(committed, outcome.elapsedNanos());
                entry.nextStep = null;
                output += committed;
                if (prefill()) { running.remove(entry); entry.phase = READY; entry.prefillFinished = result.finished(); }
                emit(entry, committed, result.finished());
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
        if (!prefill()) entry.phase = OUTPUT;
        enqueue(entry, () -> {
            if (!liveForDelivery(entry)) return;
            entry.listener.output(entry, output);
            synchronized (lock) {
                entry.outputPending = false;
                if (!live(entry)) return;
                if ((prefill() || done) && !entry.input.retained()) finish(entry, null);
                else if (!prefill()) entry.phase = RUNNING;
            }
        }, false);
    }

    @Override public RequestState.Use beginTransfer(Request request) {
        var entry = entry(request);
        if (!live(entry) || entry.readStarted || (prefill() ? entry.phase != READY : entry.phase != LOADING))
            throw new RequestFailure(INVALID, "Request is not ready for a new KV transfer");
        entry.readStarted = true;
        entry.phase = TRANSFERRING;
        return entry.state.pin();
    }
    @Override public void transferCompleted(Request request, TransferResult result) {
        var entry = entry(request);
        if (!live(entry)) return;
        if (!(result instanceof RtpTransferResult transferred)) throw new RequestFailure(INVALID, "Foreign RTP transfer result");
        if (transferred.failure() != null) { finish(entry, transferred.failure()); return; }
        if (!entry.readStarted || entry.phase != TRANSFERRING)
            throw new RequestFailure(INVALID, "Stale or duplicate transfer completion");
        if (!prefill()) {
            if (entry.lease == null || entry.loaded || entry.generating)
                throw new RequestFailure(INVALID, "Unexpected KV load completion");
            entry.loaded = true;
            entry.prefillFinished = transferred.prefillFinished();
        }
        entry.phase = TRANSFERRED;
    }
    @Override public void finish(Request request, RequestFailure failure) {
        var entry = entry(request);
        if (entry.state.terminating()) return;
        entry.finalPhase = reportedPhase(entry);
        waiting.remove(entry);
        if (entry.group != null) {
            entry.group.remove(entry);
            if (entry.group.isEmpty()) groups.remove(entry.group);
        }
        running.remove(entry);
        entry.state.finish(failure);
    }
    private void terminal(Entry entry, RequestFailure failure) {
        history.add(task(entry, System.nanoTime(), System.currentTimeMillis(), failure));
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
    private KvCache.Failure acquire(Entry entry) {
        if (entry.lease != null) return null;
        long generation = cache.allocationGeneration();
        if (entry.attemptedGeneration == generation) return KvCache.Failure.RETRYABLE;
        boolean reuse = config.reuseCache() && (config.deviceCache() || config.hostBlocks() > 0) && entry.input.reuse()
            && (prefill() || !config.disableDecodeFirstReuse());
        int matched = Math.min(entry.input.keys().size(), (entry.input.tokens() - 1) / config.blockSize());
        long capacityTokens = (long) entry.input.tokens() + (prefill() ? 0 : config.compute().kvReserveTokens());
        var admission = cache.acquire(capacityTokens, reuse ? entry.input.keys().subList(0, matched) : List.of());
        entry.attemptedGeneration = admission.generation();
        if (admission.lease() == null) return admission.failure();
        if (entry.admittedAt == Long.MIN_VALUE) entry.admittedAt = System.nanoTime();
        entry.lease = admission.lease();
        entry.reused = Math.toIntExact(admission.lease().reusedTokens());
        entry.state.attach(entry.lease, () -> entry.lease.publish(
            config.reuseCache() && (config.deviceCache() || config.hostBlocks() > 0) && entry.input.reuse() ? entry.input.keys() : List.of(), entry.input.tokens()));
        metrics.cacheLookup(entry.input.tokens(), entry.reused);
        return null;
    }
    @Override public List<Request> requests() { return new ArrayList<>(active.values()); }
    @Override public QueueSnapshot queues() {
        int queued = 0, executing = 0, loading = 0, ready = 0;
        var waitingMembers = new HashSet<>(waiting);
        groups.forEach(waitingMembers::addAll);
        for (var entry : active.values()) {
            if (entry.state.terminating()) continue;
            if (running.contains(entry)) executing++;
            else if (waitingMembers.contains(entry) || entry.phase == Phase.CAPACITY) queued++;
            else if (entry.phase == READY || entry.phase == TRANSFERRING || entry.phase == TRANSFERRED) ready++;
            else loading++;
        }
        return new QueueSnapshot(queued, executing, loading, ready);
    }
    @Override public long nextDeadline() {
        return active.values().stream().filter(entry -> !entry.state.terminating())
            .mapToLong(entry -> entry.input.deadline()).min().orElse(Long.MAX_VALUE);
    }
    @Override public Status status(long latestFinishedVersion) {
        long now = System.nanoTime();
        var tasks = active.values().stream().map(entry -> task(entry, now, -1, null)).toList();
        int running = (int) tasks.stream().filter(task -> task.phase() == ReportedPhase.RUNNING).count();
        return history.snapshot(latestFinishedVersion, tasks, tasks.size() - running, running);
    }
    private Task task(Entry entry, long now, long endTimeMillis, RequestFailure failure) {
        var progress = entry.state.snapshot();
        var phase = endTimeMillis >= 0 && entry.finalPhase != null ? entry.finalPhase : reportedPhase(entry);
        long wait = Math.max(0, (entry.admittedAt == Long.MIN_VALUE ? now : entry.admittedAt) - entry.input.started());
        long execution = entry.admittedAt == Long.MIN_VALUE ? 0 : Math.max(0, now - entry.admittedAt);
        long code = failure == null ? 0 : RtpErrors.code(failure);
        if (failure != null && code == 0) code = RtpErrors.status(failure).getCode().value();
        return new Task(entry.input.requestId(), entry.input.groupId(), entry.input.tokens(), entry.reused,
            entry.input.priority(), phase, progress.iterations(), wait / 1_000_000, execution / 1_000_000,
            endTimeMillis, code, failure == null ? "" : Objects.toString(failure.getMessage(), ""));
    }
    private ReportedPhase reportedPhase(Entry entry) {
        return entry.phase == FORWARD || entry.phase == RUNNING || !prefill() && entry.phase == OUTPUT
            ? ReportedPhase.RUNNING : entry.lease != null
                ? ReportedPhase.KV_ALLOCATED : ReportedPhase.RECEIVED;
    }
    private EngineMetrics.State metricState() {
        synchronized (lock) {
            var counts = queues();
            var kv = cache.snapshot(0, false);
            var host = cache.hostSnapshot();
            long block = config.blockSize();
            return new EngineMetrics.State(counts.waiting(), counts.running(), counts.loading(), counts.ready(), kv.totalBlocks() * block,
                kv.availableBlocks() * block, kv.freeBlocks() * block, kv.heldBlocks() * block, host.total() * block, host.free() * block);
        }
    }
    @Override public String scrapeMetrics() { return metrics.scrape(); }
    @Override public void close() {
        if (closed) return;
        closed = true;
        for (var entry : List.copyOf(active.values())) finish(entry, new RequestFailure(CLOSED, "Engine closed"));
    }
}

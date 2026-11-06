package org.flexlb.mock.engine.scheduler;

import io.prometheus.client.Collector;
import java.util.*;
import java.util.function.Supplier;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.forward.*;
import static org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure.Reason.*;
import static org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.Phase.*;

/** Shared RTP request, KV, PD and Forward lifecycle; admission policies own their queues. */
public abstract class AbstractRtpScheduler implements EngineScheduler {
    public static Supplier<List<Collector.MetricFamilySamples>> metricsFor(EngineScheduler scheduler) {
        return scheduler instanceof AbstractRtpScheduler rtp ? rtp.metrics()::collect : null;
    }

    public static EngineScheduler.Factory factory(RtpConfig config) {
        Objects.requireNonNull(config);
        return (lock, cache, finished) -> config.groupScheduling()
            ? new RtpPrefillGroupScheduler(lock, config, cache, finished) : new RtpFifoScheduler(lock, config, cache, finished);
    }

    /** Native P/D handoff events carried across the engine loop. */
    public enum RtpControl implements Control {
        TRANSFER_ACKNOWLEDGED, PREFILL_COMPLETE;

        /** The first sampled token is also the repeated synthetic Decode output. */
        public record Generate(int tokenId) implements Control {}
    }
    public record RtpTransferResult(boolean prefillFinished, RequestFailure failure) implements TransferResult {}


    enum Phase { QUEUED, CAPACITY, KV_PREPARING, WAITING_LOAD, TRANSFERRING, TRANSFERRED, RUNNING, FORWARD, READY }
    private final TaskHistory history;
    private final Object lock;
    protected final RtpConfig config;
    private final KvCacheManager cache;
    private final RtpMetrics metrics;
    private final Map<String, GenerateStream> active = new LinkedHashMap<>();
    protected final ArrayDeque<GenerateStream> waiting = new ArrayDeque<>();
    private final Set<GenerateStream> running = new LinkedHashSet<>();
    private final Set<GenerateStream> notifications = new LinkedHashSet<>();
    private Work executing;
    private long batchSequence;
    protected boolean closed;

    /** One generation request: input, KV ownership, progress and ordered protocol callbacks. */
    protected final class GenerateStream implements Request {
        final Input input;
        final Listener listener;
        final RequestState state;
        final ArrayDeque<Notice> notices = new ArrayDeque<>();
        KvCacheManager.Lease lease;
        KvCacheManager.Preparation preparation;
        long preparationStarted;
        boolean cacheMatchRequested;

        int reused, remoteReused, outputToken;
        boolean prefillFinished;
        final long begun = System.nanoTime();
        long enqueuedAt = begun, waitNanos, firstToken;
        long timingStarted, canRun, firstRunning;
        Phase phase = QUEUED;
        ReportedPhase finalPhase;
        boolean loaded, generating, notifying, transferBegun;
        Progress progress = new Progress(0, 0, false);
        volatile RequestFailure terminalFailure;
        ComputeModel.Plan nextStep;

        GenerateStream(Input input, Listener listener) {
            this.input = input;
            timingStarted = input.started();
            this.listener = listener;
            this.state = new RequestState(input.deadline(), failure -> terminal(this, failure));
        }
        AbstractRtpScheduler owner() { return AbstractRtpScheduler.this; }
        @Override public Input input() { return input; }
        @Override public RequestState.Snapshot snapshot() { synchronized (lock) { return state.snapshot(); } }
        @Override public Progress progress() { synchronized (lock) { return progress; } }
        @Override public int reused() { synchronized (lock) { return reused; } }
        @Override public int remoteReused() { synchronized (lock) { return remoteReused; } }

    }
    private record Member(GenerateStream request, RequestState.Use use, ComputeModel.Plan step) {}
    private record Batch(List<Member> members, long scheduleStarted, long stepStarted, long inputStarted) {}
    private record Notice(Runnable action, boolean terminal, boolean progress) {}

    AbstractRtpScheduler(Object lock, RtpConfig config, KvCacheManager cache, TaskHistory history) {
        this.history = Objects.requireNonNull(history);
        this.lock = Objects.requireNonNull(lock);
        this.config = Objects.requireNonNull(config);
        this.cache = Objects.requireNonNull(cache);
        this.metrics = new RtpMetrics(config.role().name(), this::metricState, System::nanoTime, cache::itemCount);
        cache.onEvictionLifetime(metrics::blockEvicted);
    }

    @Override public Request enqueue(Input input, Listener listener) {
        if (closed) throw new RequestFailure(CLOSED, "Engine is closing");
        if (input.tokens() >= config.maxSeqLen())
            throw new RequestFailure(INVALID, "Prompt exceeds the context limit");
        if (input.tokens() > cache.maxRequestTokens())
            throw new RequestFailure(RequestFailure.Reason.CAPACITY, "Prompt exceeds the maximum KV capacity");
        if (active.containsKey(input.identity())) throw new RequestFailure(DUPLICATE, "Active request identity already exists");
        var stream = new GenerateStream(input, Objects.requireNonNull(listener));
        active.put(input.identity(), stream);
        metrics.started();
        if (prefill()) waiting.addLast(stream);
        else stream.phase = Phase.CAPACITY;
        return stream;
    }

    @Override public void control(Request handle, Control control) {
        var stream = requireGenerateStream(handle);
        if (stream.state.terminating()) return;
        if (stream.state.expired()) { finish(stream, new RequestFailure(TIMEOUT, "Control arrived after deadline")); return; }
        if (control instanceof RtpControl.Generate generate) {
            if (prefill() || !stream.loaded || stream.generating || stream.prefillFinished || generate.tokenId() < 0)
                throw new RequestFailure(INVALID, "Generation requires loaded KV and a new generation permit");
            stream.timingStarted = System.nanoTime();
            stream.waitNanos = 0; stream.canRun = stream.firstRunning = 0;
            stream.outputToken = generate.tokenId();
            stream.generating = true;
            stream.state.transferred(1);
            // The first P token may already reach D's context/output limit. No Decode forward is needed.
            if (stream.state.snapshot().generated() >= outputLimit(stream)) { emit(stream, true); return; }
            stream.phase = QUEUED; stream.enqueuedAt = stream.timingStarted; waiting.addLast(stream);
            return;
        }
        if (!(control instanceof RtpControl event)) throw new RequestFailure(INVALID, "Foreign RTP control");
        switch (event) {
            case PREFILL_COMPLETE -> {
                if (prefill() || !stream.loaded || stream.generating)
                    throw new RequestFailure(INVALID, "Unexpected Prefill-only completion");
                stream.prefillFinished = true;
                stream.state.transferred(1);
                finish(stream, null);
            }
            case TRANSFER_ACKNOWLEDGED -> {
                if (!prefill() || !stream.input.retained() || !stream.transferBegun || stream.phase != TRANSFERRING)
                    throw new RequestFailure(INVALID, "Transfer acknowledgement without a readable Prefill source");
                finish(stream, null);
            }
        }
    }

    @Override public void prepareKvAndCheckTimeouts() {
        for (var stream : List.copyOf(active.values())) {
            if (!live(stream)) continue;
            if (stream.preparation instanceof KvCacheManager.Loading loading) {
                var result = loading.poll();
                if (result instanceof KvCacheManager.Loading) continue;
                stream.preparation = null;
                stream.phase = prefill() ? QUEUED : Phase.CAPACITY;
                completePreparation(stream, result);
            } else if (!prefill() && stream.lease == null) {
                prepare(stream);
            }
        }
    }

    @Override public Work schedule() {
        long scheduleStarted = System.nanoTime();
        if (closed) return null;
        if (executing != null) throw new IllegalStateException("Forward already in flight");
        int limit = config.maxGenerateBatchSize();
        var selected = new ArrayList<GenerateStream>();
        if (prefill()) {
            selectPrefill(selected);
        } else {
            for (var stream : List.copyOf(running)) {
                if (selected.size() >= limit) break;
                if (workTokens(stream) > 0 && reserve(stream)) selected.add(stream);
            }
            int queued = waiting.size();
            while (queued-- > 0 && selected.size() < limit && running.size() < limit) {
                var stream = waiting.removeFirst();
                if (!live(stream)) continue;
                if (workTokens(stream) <= 0 || !reserve(stream)) {
                    if (!stream.state.terminating()) waiting.addLast(stream);
                    continue;
                }
                running.add(stream); selected.add(stream);
            }
        }
        if (selected.isEmpty()) return null;
        var members = new ArrayList<Member>();
        for (var stream : selected) {
            members.add(new Member(stream, stream.state.forward(), stream.nextStep));
            long now = System.nanoTime();
            if (stream.firstRunning == 0) {
                stream.waitNanos = Math.max(0, now - stream.timingStarted);
                stream.firstRunning = now;
            }
            stream.phase = FORWARD;
        }
        // Real prepares streams before starting its step timer, then gathers model input.
        long prepareStarted = System.nanoTime();
        var items = new ArrayList<ForwardBatch.Request>();
        for (var member : members) {
            var stream = member.request();
            int tokens = prefill() ? stream.input.tokens() - stream.reused : 1;
            items.add(new ForwardBatch.Request(stream.input.identity(), stream.input.tokens(), stream.reused,
                prefill() ? 1 : stream.input.maxNewTokens(), member.step().request().generated(), tokens, stream.input.priority(), List.of()));
        }
        var batch = new ForwardBatch(config.id() + "/" + ++batchSequence, List.of(new ForwardBatch.Partition(0, items)));
        executing = new Work(new Batch(List.copyOf(members), scheduleStarted, System.nanoTime(), prepareStarted), batch);
        metrics.computeStarted();
        metrics.batchPrepared(System.nanoTime() - prepareStarted);
        return executing;
    }

    protected abstract void selectPrefill(List<GenerateStream> selected);

    protected final void admitPrefill(ArrayDeque<GenerateStream> queue, List<GenerateStream> selected, int uncachedLimit) {
        long uncached = 0;
        var admitted = new ArrayList<GenerateStream>();
        boolean capped = config.maxInitedKvStreams() > 0;
        long initialized = capped ? active.values().stream().filter(this::ownsKv).count() : 0;
        for (var stream : List.copyOf(queue)) {
            boolean ownedBefore = capped && ownsKv(stream);
            try {
                if (!live(stream) || stream.preparation instanceof KvCacheManager.Loading) continue;
                if (admitted.size() >= config.maxGenerateBatchSize() || config.forceSinglePrefill() && !admitted.isEmpty()) break;
                if (uncachedLimit > 0 && uncached >= uncachedLimit) continue;
                if (stream.lease == null && capped && initialized >= config.maxInitedKvStreams()) continue;
                if (workTokens(stream) <= 0 || !fitsPrefill(admitted, stream)) continue;
                if (!reserve(stream)) continue;
                admitted.add(stream);
                queue.remove(stream); running.add(stream); selected.add(stream);
                uncached += uncachedTokens(stream);
            } finally {
                if (capped) initialized += (ownsKv(stream) ? 1 : 0) - (ownedBefore ? 1 : 0);
            }
        }
    }
    private boolean ownsKv(GenerateStream stream) {
        return active.get(stream.input.identity()) == stream && (stream.lease != null || stream.preparation instanceof KvCacheManager.Loading);
    }
    protected void removeWaiting(GenerateStream stream) { waiting.remove(stream); }

    protected Set<GenerateStream> waitingStreams() { return new HashSet<>(waiting); }

    private long uncachedTokens(GenerateStream stream) {
        long tokens = stream.input.tokens() - stream.reused;
        long alignment = 2L * config.cpSize();
        return config.cpSize() > 1 ? (tokens + alignment - 1) / alignment * alignment : tokens;
    }

    private boolean fitsPrefill(List<GenerateStream> selected, GenerateStream candidate) {
        if (selected.isEmpty()) return candidate.input.tokens() < config.maxSeqLen();
        long sum = candidate.input.tokens(), longest = candidate.input.tokens();
        for (var stream : selected) {
            sum += stream.input.tokens(); longest = Math.max(longest, stream.input.tokens());
        }
        // Native prefill-only means max_new_tokens == 0; this Mock currently accepts positive output lengths only.
        long budget = (long) config.maxBatchTokens() - 1;
        return sum <= budget && longest * (selected.size() + 1L) <= budget;
    }

    private int outputLimit(GenerateStream stream) {
        return Math.min(stream.input.maxNewTokens(),
            config.maxSeqLen() - stream.input.tokens());
    }

    private int workTokens(GenerateStream stream) {
        if (!live(stream)) return 0;
        var progress = stream.state.snapshot();
        if (stream.nextStep == null)
            stream.nextStep = config.compute().prepare(config.role(), new ComputeModel.Request(
                Long.toString(stream.input.requestId()), progress.iterations(), stream.input.tokens(), progress.generated(),
                stream.input.minNewTokens(), stream.input.maxNewTokens(),
                config.maxSeqLen(), stream.input.ignoreEos()));
        if (prefill()) return stream.input.tokens();
        return 1;
    }

    private boolean reserve(GenerateStream stream) {
        if (stream.canRun == 0) stream.canRun = System.nanoTime();
        if (prefill()) {
            return prepare(stream);
        }
        long tokens = (long) stream.input.tokens() + stream.state.snapshot().generated() + stream.nextStep.additionalKvTokens();
        var failure = stream.lease.ensureCapacity(tokens);
        if (failure != null) {
            // RTP running streams fail KV growth; only admission waits for capacity.
            finish(stream, new RequestFailure(RequestFailure.Reason.CAPACITY, "Decode KV growth failed: " + failure));
            return false;
        }
        stream.phase = RUNNING;
        return true;
    }

    @Override public void processOutputs(Work work, SimulatedCompute.Outcome outcome) {
        var batch = completedBatch(work);
        long input = 0, computed = 0, output = 0, maxSequence = 0;
        // Like Real's staged accept_len, this describes the completed computation,
        // before cancellation filtering and EOS/max-length output truncation.
        long tpotStepNanos = System.nanoTime() - batch.inputStarted();
        long acceptedTokens = 0;
        var byPriority = new TreeMap<Integer, RtpMetrics.TokenCounts>();
        try {
            for (var member : batch.members()) {
                var stream = member.request();
                maxSequence = Math.max(maxSequence, (long) stream.input.tokens() + member.step().request().generated());
                input += prefill() ? stream.input.tokens() : 1;
                computed += prefill() ? stream.input.tokens() - stream.reused : 1;
                acceptedTokens += member.step().candidates();
                long context = prefill() ? stream.input.tokens() - stream.reused : 0;
                long generated = prefill() ? 0 : member.step().candidates();
                byPriority.merge(stream.input.priority(), new RtpMetrics.TokenCounts(context,
                    prefill() ? stream.input.tokens() : 0, generated, context + generated), RtpMetrics.TokenCounts::plus);
                if (!live(stream)) continue;
                if (outcome.state() != SimulatedCompute.State.COMPLETED) {
                    finish(stream, new RequestFailure(CANCELLED, "Forward cancelled"));
                    continue;
                }
                var result = config.compute().completed(member.step());
                int committed = result.committedTokens();
                stream.state.commit(committed, outcome.elapsedNanos());
                if (committed > 0 && stream.firstToken == 0) stream.firstToken = System.nanoTime();
                stream.nextStep = null;
                output += committed;
                if (prefill()) { running.remove(stream); stream.phase = READY; stream.prefillFinished = result.finished(); }
                emit(stream, result.finished());
            }
            if (outcome.state() == SimulatedCompute.State.COMPLETED) {
                long completed = System.nanoTime();
                int size = batch.members().size();
                metrics.stepCompleted(new RtpMetrics.StepStats(prefill() ? size : 0, prefill() ? 0 : size,
                    maxSequence, completed - batch.scheduleStarted(), completed - batch.stepStarted(),
                    outcome.elapsedNanos(), outcome.sample().planned().toNanos(), input, computed, output, acceptedTokens, tpotStepNanos), byPriority);
            }
        } finally { metrics.computeFinished(); batch.members().forEach(member -> member.use().close()); }
    }

    @Override public void failForward(Work work, RuntimeException error) {
        var batch = completedBatch(work);
        var failure = error instanceof RequestFailure known ? known : new RequestFailure(INTERNAL, "Forward failed", error);
        try { batch.members().forEach(member -> finish(member.request(), failure)); }
        finally { metrics.computeFinished(); batch.members().forEach(member -> member.use().close()); }
    }
    @Override public boolean isForwardCancelled(Work work) {
        return work == executing && ((Batch) work.owner()).members().stream().allMatch(member -> member.request().state.terminating());
    }
    private Batch completedBatch(Work work) {
        if (work != executing) throw new IllegalStateException("Stale/foreign Forward completion");
        executing = null;
        return (Batch) work.owner();
    }

    private void emit(GenerateStream stream, boolean done) {
        var snapshot = stream.state.snapshot();
        stream.progress = new Progress(snapshot.generated(), snapshot.iterations(), done);
        // A notification carries no token data. A slow consumer sees the latest cumulative progress.
        stream.notices.removeIf(Notice::progress);
        stream.notices.addLast(new Notice(() -> {
            if (stream.terminalFailure == null) stream.listener.progressed(stream);
        }, false, true));
        notifications.add(stream);
        if ((prefill() || done) && !stream.input.retained()) finish(stream, null);
    }

    @Override public RequestState.Use beginTransfer(Request request) {
        var stream = requireGenerateStream(request);
        if (!live(stream) || stream.transferBegun || (prefill() ? stream.phase != READY : stream.phase != WAITING_LOAD))
            throw new RequestFailure(INVALID, "Request is not ready for a new KV transfer");
        stream.transferBegun = true;
        stream.phase = TRANSFERRING;
        return stream.state.pin();
    }
    @Override public void transferCompleted(Request request, TransferResult result) {
        var stream = requireGenerateStream(request);
        if (!live(stream)) return;
        if (!(result instanceof RtpTransferResult transferred)) throw new RequestFailure(INVALID, "Foreign RTP transfer result");
        if (transferred.failure() != null) { finish(stream, transferred.failure()); return; }
        if (!stream.transferBegun || stream.phase != TRANSFERRING)
            throw new RequestFailure(INVALID, "Stale or duplicate transfer completion");
        if (prefill() || stream.lease == null || stream.loaded || stream.generating)
            throw new RequestFailure(INVALID, "Unexpected Decode KV load completion");
        stream.loaded = true;
        stream.prefillFinished = transferred.prefillFinished();
        stream.phase = TRANSFERRED;
    }
    @Override public void finish(Request request, RequestFailure failure) {
        var stream = requireGenerateStream(request);
        if (stream.state.terminating()) return;
        stream.finalPhase = reportedPhase(stream);
        long now = System.nanoTime();
        if (waitingStreams().contains(stream) || stream.phase == Phase.CAPACITY)
            stream.waitNanos = Math.max(0, now - stream.timingStarted);
        removeWaiting(stream);
        running.remove(stream);
        if (stream.preparation instanceof KvCacheManager.Loading loading) loading.close();
        stream.preparation = null;
        stream.state.finish(failure);
    }
    private void terminal(GenerateStream stream, RequestFailure failure) {
        stream.terminalFailure = failure;
        long now = System.nanoTime();
        history.add(task(stream, now, System.currentTimeMillis(), failure));
        var outcome = failure == null ? RtpMetrics.Outcome.success : switch (failure.reason()) {
            case CANCELLED -> RtpMetrics.Outcome.cancelled;
            case TIMEOUT -> RtpMetrics.Outcome.timeout;
            default -> RtpMetrics.Outcome.error;
        };
        var progress = stream.state.snapshot();
        metrics.requestFinished(outcome, new RtpMetrics.RequestStats(
            Math.max(0, now - stream.timingStarted), stream.waitNanos,
            stream.canRun == 0 ? 0 : Math.max(0, stream.canRun - stream.enqueuedAt),
            stream.canRun == 0 || stream.firstRunning == 0 ? 0 : Math.max(0, stream.firstRunning - stream.canRun),
            stream.firstToken == 0 ? -1 : stream.firstToken - stream.begun,
            stream.input.tokens(), progress.generated(), progress.iterations(), stream.reused),
            prefill() || canReuseLocalKv(stream));
        metrics.streamingFinished(stream.input.streaming());
        active.remove(stream.input.identity(), stream);
        enqueue(stream, () -> stream.listener.terminal(stream, failure), true);
    }
    private void enqueue(GenerateStream stream, Runnable notification, boolean terminal) {
        // At most admission, one coalesced progress signal, and terminal per request.
        if (stream.notices.size() >= 3) throw new IllegalStateException("Request notification overflow");
        stream.notices.addLast(new Notice(notification, terminal, false));
        notifications.add(stream);
    }
    @Override public List<Runnable> takeNotifications() {
        var result = new ArrayList<Runnable>();
        for (var stream : List.copyOf(notifications)) {
            if (stream.notifying) continue;
            notifications.remove(stream);
            var notification = stream.notices.removeFirst();
            stream.notifying = true;
            result.add(() -> {
                try { notification.action().run(); }
                catch (RuntimeException error) {
                    synchronized (lock) {
                        if (notification.terminal()) throw error;
                        finish(stream, new RequestFailure(CANCELLED, "Request consumer failed", error));
                    }
                } finally {
                    synchronized (lock) {
                        stream.notifying = false;
                        if (!stream.notices.isEmpty()) notifications.add(stream);
                    }
                }
            });
        }
        return result;
    }

    private boolean liveForDelivery(GenerateStream stream) { synchronized (lock) { return live(stream); } }
    protected final boolean live(GenerateStream stream) {
        if (stream.state.terminating()) return false;
        if (stream.state.expired()) finish(stream, new RequestFailure(TIMEOUT, "Request budget expired"));
        return !stream.state.terminating();
    }
    private GenerateStream requireGenerateStream(Request request) {
        if (!(request instanceof AbstractRtpScheduler.GenerateStream stream) || stream.owner() != this)
            throw new IllegalArgumentException("Foreign request handle");
        return stream;
    }
    protected final boolean prefill() { return config.role() == Role.prefill; }
    private boolean canReuseLocalKv(GenerateStream stream) {
        return config.reuseCache() && config.deviceCache() && stream.input.reuse()
            && (prefill() || !config.disableDecodeFirstReuse());
    }

    private boolean prepare(GenerateStream stream) {
        if (stream.lease != null) return true;
        stream.preparationStarted = System.nanoTime();
        if (stream.preparation instanceof KvCacheManager.Failed failed)
            return completePreparation(stream, failed.retry());
        boolean reuse = canReuseLocalKv(stream);
        var keys = stream.input.keys();
        if (prefill())
            metrics.recordPrefillCacheKeys(keys, stream.input.tokens(), config.blockSize());
        int matched = Math.min(keys.size(), (stream.input.tokens() - 1) / config.blockSize());
        stream.cacheMatchRequested = reuse && matched > 0;
        return completePreparation(stream,
            cache.prepare(stream.input.tokens(), reuse ? keys.subList(0, matched) : List.of()));
    }

    private boolean completePreparation(GenerateStream stream, KvCacheManager.Preparation result) {
        if (result instanceof KvCacheManager.Loading loading) {
            stream.preparation = loading;
            stream.phase = KV_PREPARING;
            stream.canRun = 0;
            return false;
        }
        if (result instanceof KvCacheManager.Failed failed) {
            stream.preparation = failed;
            if (failed.failure() == KvCacheManager.Failure.PERMANENT)
                finish(stream, new RequestFailure(RequestFailure.Reason.CAPACITY, "KV allocation exceeds capacity"));
            return false;
        }
        stream.preparation = null;
        var ready = (KvCacheManager.Ready) result;
        stream.waitNanos = Math.max(0, System.nanoTime() - stream.timingStarted);
        stream.lease = ready.lease();
        stream.reused = Math.toIntExact(stream.lease.reusedTokens());
        stream.remoteReused = ready.remoteReusedTokens();
        stream.state.attach(stream.lease, () -> {
            if (!config.reuseCache() || !config.deviceCache() || !stream.input.reuse()) return;
            // The final sampled token has not entered a Forward and owns no computed KV.
            int computedOutput = !prefill() && stream.generating ? Math.max(0, stream.state.snapshot().generated() - 1) : 0;
            stream.lease.publish(stream.input.cacheSequence().keys(computedOutput, stream.outputToken),
                (long) stream.input.tokens() + computedOutput);
        });
        metrics.cacheLookup(stream.input.tokens(), stream.reused);
        if (stream.cacheMatchRequested) metrics.cacheMatched(ready.matchNanos());
        if (config.reuseCache() && stream.input.reuse())
            metrics.prepared((long) (stream.input.tokens() / config.blockSize()) * config.blockSize(),
                stream.reused - stream.remoteReused, System.nanoTime() - stream.preparationStarted);
        if (!prefill()) {
            stream.phase = WAITING_LOAD;
            enqueue(stream, () -> { if (liveForDelivery(stream)) stream.listener.allocated(stream); }, false);
        }
        return true;
    }
    @Override public List<Request> requests() { return new ArrayList<>(active.values()); }
    @Override public QueueSnapshot queues() {
        int queued = 0, executing = 0, loading = 0, ready = 0;
        var waitingMembers = waitingStreams();
        for (var stream : active.values()) {
            if (stream.state.terminating()) continue;
            if (running.contains(stream)) executing++;
            else if (stream.phase == KV_PREPARING) loading++;
            else if (waitingMembers.contains(stream) || stream.phase == Phase.CAPACITY) queued++;
            else if (stream.phase == READY || stream.phase == TRANSFERRING || stream.phase == TRANSFERRED) ready++;
            else loading++;
        }
        return new QueueSnapshot(queued, executing, loading, ready);
    }
    @Override public long nextDeadline() {
        return active.values().stream().filter(stream -> !stream.state.terminating())
            .mapToLong(stream -> stream.input.deadline()).min().orElse(Long.MAX_VALUE);
    }
    @Override public List<Task> tasks() {
        long now = System.nanoTime();
        return active.values().stream().map(stream -> task(stream, now, -1, null)).toList();
    }
    private Task task(GenerateStream stream, long now, long endTimeMillis, RequestFailure failure) {
        var progress = stream.state.snapshot();
        var phase = endTimeMillis >= 0 && stream.finalPhase != null ? stream.finalPhase : reportedPhase(stream);
        boolean finished = endTimeMillis >= 0;
        // Native runtime metadata publishes final timings only on dequeue. Live execution is -1.
        long waitMillis = finished ? stream.waitNanos / 1_000_000 : 0;
        long executionMillis = finished ? Math.max(0, (now - stream.timingStarted) / 1_000_000 - waitMillis) : -1;
        return new Task(stream.input.requestId(), stream.input.groupId(), stream.input.tokens(), stream.reused,
            stream.input.priority(), phase, progress.iterations(), waitMillis, executionMillis,
            endTimeMillis, failure);
    }
    private ReportedPhase reportedPhase(GenerateStream stream) {
        return stream.phase == FORWARD || stream.phase == RUNNING
            ? ReportedPhase.RUNNING : stream.lease != null
                ? ReportedPhase.KV_ALLOCATED : ReportedPhase.RECEIVED;
    }
    private RtpMetrics.State metricState() {
        synchronized (lock) {
            var counts = queues();
            var kv = cache.snapshot(0, false);
            long block = config.blockSize();
            return new RtpMetrics.State(counts.waiting(), counts.running(), counts.loading(), counts.ready(), kv.totalBlocks() * block,
                kv.availableBlocks() * block, kv.freeBlocks() * block, kv.heldBlocks() * block, config.blockSize());
        }
    }
    RtpMetrics metrics() { return metrics; }
    @Override public void close() {
        if (closed) return;
        closed = true;
        for (var stream : List.copyOf(active.values())) finish(stream, new RequestFailure(CLOSED, "Engine closed"));
    }
}

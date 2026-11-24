package org.flexlb.mock.frontend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;
import org.flexlb.mock.frontend.FrontendProtocol.*;
import org.flexlb.mock.protocol.JsonFiles;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;

/**
 * Owns one round of local playback, from pacing through drain.
 */
final class ReplayWorker implements AutoCloseable {
    private static final ObjectWriter JSON = new ObjectMapper().writerWithDefaultPrettyPrinter();

    private final FrontendReplayPlan plan;
    private final BackendRpcTransport frontend;
    private final RequestJournal journal;
    private final Path resultFile;
    private final MasterRoutes masterRoutes;
    private final FrontendMetrics metrics;
    private final ArrayBlockingQueue<Item> queue;
    private final Map<Long, BackendRpcTransport.Call> active = new HashMap<>();
    // Duplicate probes share the primary's request_id, so they live outside the id-keyed map.
    private final Set<BackendRpcTransport.Call> duplicateActive =
        java.util.Collections.synchronizedSet(new java.util.HashSet<>());
    // Status readers never need the monitor used for dispatch and RPC registration.
    private final Object stateLock = new Object();
    private final Map<String, Long> outcomes = new TreeMap<>();
    private final List<String> errors = new ArrayList<>();
    private final CompletableFuture<State> completion = new CompletableFuture<>();

    private final int imageToken;
    private final int physicalBlockSize;

    private Thread player, dispatcher;

    private boolean begun, producerDone;
    private volatile boolean stopped;

    private final AtomicLong produced = new AtomicLong();

    private long issued, finished, duplicateIssued, drainDeadline = Long.MAX_VALUE;

    ReplayWorker(FrontendReplayPlan plan, FrontendMetrics metrics, Path topology, Path evidence, String id,
                 int imageToken, int physicalBlockSize, long seed, int prefillCpSize) throws Exception {
        this.imageToken = imageToken; this.physicalBlockSize = physicalBlockSize;
        this.plan = plan; this.metrics = metrics;
        resultFile = evidence.resolveSibling("result.json");
        queue = new ArrayBlockingQueue<>(plan.queue_capacity());
        for (var outcome : BackendRpcTransport.Outcome.values()) outcomes.put(outcome.name(), 0L);
        masterRoutes = new MasterRoutes(topology, plan.masters());
        try { journal = new RequestJournal(evidence, id, plan); }
        catch (Exception error) { masterRoutes.close(); throw error; }
        frontend = new BackendRpcTransport(journal::record, imageToken, seed, prefillCpSize, masterRoutes::routes);
        // A lost begin command must not leave a prepared worker alive forever.
        dispatcher = Thread.ofVirtual().name("frontend-dispatch").start(this::dispatch);
    }

    @Override public void close() {
        halt();
        List<BackendRpcTransport.Call> calls;
        synchronized (this) { calls = List.copyOf(active.values()); }
        calls.forEach(BackendRpcTransport.Call::cancel);
        synchronized (duplicateActive) {
            duplicateActive.forEach(BackendRpcTransport.Call::cancel);
        }
        try { completion.get(8, TimeUnit.SECONDS); }
        catch (Exception error) { dispatcher.interrupt(); }
    }

    long sequence() { return plan.sequence(); }

    synchronized void begin(long epochMillis) {
        if (begun || stopped || completion.isDone()) throw new IllegalStateException("Frontend already begun or stopped");
        long delay = epochMillis - System.currentTimeMillis();
        if (delay < 0 || delay > 30000) throw new IllegalArgumentException("Start time must be within the next 30 seconds");
        begun = true;
        long start = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(delay);
        player = Thread.ofVirtual().name("frontend-player").start(() -> {
            try { play(plan, start, imageToken, this::offer); }
            catch (InterruptedException interrupted) { if (!stopped) fail("Player interrupted unexpectedly"); }
            catch (java.io.InterruptedIOException interrupted) { if (!stopped) fail(interrupted.toString()); }
            catch (java.io.IOException | RuntimeException error) { fail(error.toString()); }
            finally { synchronized (ReplayWorker.this) { finishInput(); } }
        });
        LockSupport.unpark(dispatcher);
    }

    static void play(FrontendReplayPlan plan, long startNanos, int imageToken, Consumer<Item> submit) throws IOException, InterruptedException {
        try (var indexed = new IndexedTrafficReader(Path.of(plan.requests_file()))) {
            long expected = FrontendReplayPlan.requestCount(indexed.count, plan.first_index(), plan.stride());
            if (!indexed.identity.equals(plan.sha256()) || expected != plan.count())
                throw new IOException("Indexed traffic changed after planning");
            checkPlaybackEnd(indexed.count - 1, indexed.lastArrival, plan.qps(), plan.replay_speed(), plan.warmup());
            for (long position = 0; position < plan.count(); position++) {
                if (Thread.interrupted()) throw new InterruptedException();
                long index = plan.first_index() + position * plan.stride();
                var entry = indexed.entry(index, imageToken);
                int outputLimit = playbackLimit(entry, plan.fallback_max_output_tokens());
                long due = startNanos + delayNanos(index, entry.arrivalMs(), plan.qps(), plan.replay_speed(), plan.warmup());
                long left;
                while ((left = due - System.nanoTime()) > 0) TimeUnit.NANOSECONDS.sleep(left);
                if (Thread.interrupted()) throw new InterruptedException();
                Integer duplicateEvery = plan.duplicate_every();
                submit.accept(new Item(plan.first_request_id() + index, entry, outputLimit,
                    duplicateEvery != null && position % duplicateEvery == 0));
            }
        }
    }

    static void checkPlaybackEnd(long index, long arrivalMs, Double qps, Double speed, Warmup warmup) throws IOException {
        long end = delayNanos(index, arrivalMs, qps, speed, warmup);
        if (warmup != null && end < TimeUnit.MILLISECONDS.toNanos(warmup.duration_ms()))
            throw new IOException("Dataset ends during warmup; no requests remain at the formal rate");
    }

    static long delayNanos(long index, long arrivalMs, Double qps, Double speed, Warmup warmup) throws IOException {
        double seconds = speed == null ? index / qps : arrivalMs / (1000.0 * speed);
        if (warmup != null) {
            double wallWarmup = warmup.duration_ms() / 1000.0;
            double replayWarmup = wallWarmup * warmup.rate_scale();
            seconds = seconds < replayWarmup ? seconds / warmup.rate_scale() : wallWarmup + (seconds - replayWarmup);
        }
        if (!Double.isFinite(seconds) || seconds < 0 || seconds > 86400)
            throw new IOException("Playback exceeds 24 hours");
        return (long) (seconds * 1_000_000_000.0);
    }

    static int playbackLimit(IndexedTrafficReader.Entry entry, Integer fallback) throws IOException {
        if (entry.observedOutputTokens() == 0)
            throw new IOException("Indexed traffic: zero observed output cannot be replayed as generation");
        if (entry.observedOutputTokens() > 0) return entry.observedOutputTokens();
        if (entry.maxOutputTokens() > 0) return entry.maxOutputTokens();
        if (fallback == null) throw new IOException("Indexed traffic: fallback_max_output_tokens is required");
        return fallback;
    }

    synchronized void fail(String error) {
        synchronized (stateLock) { if (errors.isEmpty()) errors.add(error); }
        halt();
    }

    synchronized void halt() {
        if (completion.isDone() || stopped) return;
        stopped = true; queue.clear();
        if (player != null) player.interrupt();
        finishInput();
    }

    State state() {
        var done = completion.getNow(null);
        return done == null ? snapshot(false) : done;
    }

    State await(long timeoutMillis) throws Exception { return completion.get(timeoutMillis, TimeUnit.MILLISECONDS); }

    private void offer(Item item) {
        if (stopped) return;
        if (!queue.offer(item)) {
            synchronized (this) { fail("Playback queue full: capacity=" + plan.queue_capacity() + ", inflight=" + active.size()); }
            throw new IllegalStateException("Playback queue full");
        }
        produced.incrementAndGet();
        // Stop may race an offer; it can never reopen this queue or issue that request.
        if (stopped) queue.clear();
        LockSupport.unpark(dispatcher);
    }

    private void finishInput() {
        if (!producerDone) {
            producerDone = true;
            drainDeadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(plan.drain_timeout_ms());
        }
        LockSupport.unpark(dispatcher);
    }

    private void dispatch() {
        long prepareDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        try {
            while (true) {
                masterRoutes.check();
                if (journal.failure() != null) fail(journal.failure());
                List<BackendRpcTransport.Call> cancel = List.of();
                Item item = null;
                synchronized (this) {
                    if (!begun && !stopped && System.nanoTime() >= prepareDeadline) fail("Timed out waiting for begin");
                    if (producerDone && queue.isEmpty() && active.isEmpty() && duplicateActive.isEmpty()) break;
                    if (producerDone && System.nanoTime() >= drainDeadline) {
                        fail("Frontend drain timed out");
                        cancel = List.copyOf(active.values());
                    } else if (begun && !stopped && !queue.isEmpty() && active.size() < plan.concurrency()) {
                        item = queue.remove();
                    }
                }
                if (item != null) {
                    issue(item);
                    continue;
                }
                // Native completion callbacks acquire this object's monitor; never cancel while holding it.
                cancel.forEach(BackendRpcTransport.Call::cancel);
                LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(100));
                if (Thread.interrupted()) throw new InterruptedException();
            }
        } catch (InterruptedException error) {
            fail("Frontend dispatcher interrupted"); Thread.currentThread().interrupt();
        } catch (RuntimeException error) { fail("Frontend dispatcher failed: " + error); }
        finally {
            if (player != null) try { player.join(1000); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            frontend.close(); masterRoutes.close();
            try { journal.finish(issued, finished); }
            catch (java.io.IOException error) { fail(error.getMessage()); }
            var result = snapshot(true);
            try { JsonFiles.writeAtomically(resultFile, result, JSON, true); }
            catch (IOException error) {
                fail("Cannot save Frontend result: " + error);
                result = snapshot(true);
            }
            completion.complete(result);
        }
    }

    private void issue(Item item) {
        var entry = item.entry();
        var input = entry.input().toBuilder().setRequestId(item.requestId());
        input.getGenerateConfigBuilder().setReuseCache(Boolean.TRUE.equals(plan.reuse_cache()))
            .setMaxNewTokens(item.outputLimit()).setIgnoreEos(entry.observedOutputTokens() > 0)
            .setCanUsePdSeparation(item.outputLimit() > 1).setNumBeams(1).setReturnIncremental(true).setIsStreaming(true);
        var generateInput = input.build();
        var request = FlexlbScheduleRequestPB.newBuilder().setRequestId(item.requestId())
            .setCacheKeyBlockSize(physicalBlockSize).setSeqLen(input.getTokenIdsCount())
            .setMaxNewTokens(input.getGenerateConfig().getMaxNewTokens()).setPriority(entry.priority())
            .build();
        var prepared = frontend.prepare(request, generateInput, Duration.ofMillis(plan.request_timeout_ms()), entry.expectedImageTokens());
        BackendRpcTransport.Call call;
        synchronized (this) {
            if (stopped) return;
            call = frontend.send(prepared);
            active.put(item.requestId(), call);
            synchronized (stateLock) { issued++; }
            metrics.issued();
            if (item.duplicate()) issueDuplicate(prepared);
        }
        call.completion().whenComplete((result, error) -> {
            synchronized (ReplayWorker.this) {
                active.remove(item.requestId());
                synchronized (stateLock) {
                    finished++;
                    if (error == null) outcomes.merge(result.outcome().name(), 1L, Long::sum);
                }
                if (error != null) fail("Request execution failed: " + error);
                else metrics.finished(result);
                LockSupport.unpark(dispatcher);
            }
        });
    }

    /**
     * Issues one extra concurrent call reusing the item's request_id; the Master must reject it
     * as a duplicate. Counted like any other call in issued/finished/outcomes so the playback
     * result stays consistent; tracked outside the primary active map which keys request_id.
     */
    private void issueDuplicate(BackendRpcTransport.PreparedCall prepared) {
        var duplicate = frontend.sendDuplicate(prepared);
        duplicateActive.add(duplicate);
        synchronized (stateLock) { issued++; duplicateIssued++; }
        metrics.issued();
        duplicate.completion().whenComplete((result, error) -> {
            synchronized (ReplayWorker.this) {
                duplicateActive.remove(duplicate);
                synchronized (stateLock) {
                    finished++;
                    if (error == null) outcomes.merge(result.outcome().name(), 1L, Long::sum);
                }
                if (error != null) fail("Duplicate request execution failed: " + error);
                else metrics.finished(result);
                LockSupport.unpark(dispatcher);
            }
        });
    }

    private State snapshot(boolean done) {
        synchronized (stateLock) {
            // issued/finished count duplicate probes like any call; unissued excludes them since
            // they are not planned input entries.
            return new State(done, stopped, produced.get(), issued, finished,
                plan.count() - (issued - duplicateIssued),
                queue.size(), Math.toIntExact(issued - finished), Map.copyOf(outcomes), List.copyOf(errors));
        }
    }

    /**
     * One playback entry; duplicate marks the item for an extra concurrent call reusing its
     * request_id, which the Master must reject as a duplicate while the original proceeds.
     */
    record Item(long requestId, IndexedTrafficReader.Entry entry, int outputLimit, boolean duplicate) {}
}
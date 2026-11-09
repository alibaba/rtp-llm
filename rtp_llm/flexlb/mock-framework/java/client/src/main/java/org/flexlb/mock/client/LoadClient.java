package org.flexlb.mock.client;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.mock.client.ClientProtocol.*;

/** Owns playback pacing, admission, concurrency and request lifetimes. */
final class LoadClient implements AutoCloseable {
    record Item(long requestId, IndexedTraffic.Entry entry, int outputLimit) {}
    private final WorkerPlan plan;
    private final NativeClient client;
    private final RequestJournal journal;
    private final MasterDiscovery discovery;
    private final ClientMetrics metrics;
    private final ArrayBlockingQueue<Item> queue;
    private final Map<Long, NativeClient.Call> active = new HashMap<>();
    private final Map<String, Long> outcomes = new TreeMap<>();
    private final List<String> errors = new ArrayList<>();
    private final CompletableFuture<State> completion = new CompletableFuture<>();
    private final int imageToken;
    private final int blockSize;
    private Thread player, dispatcher;
    private boolean begun, producerDone;
    private volatile boolean stopped;
    private final AtomicLong produced = new AtomicLong();
    private long issued, finished, drainDeadline = Long.MAX_VALUE;

    LoadClient(WorkerPlan plan, ClientMetrics metrics, java.nio.file.Path topology, java.nio.file.Path evidence, String id) throws Exception {
        this(plan, metrics, topology, evidence, id, org.flexlb.mock.protocol.MediaInput.IMAGE_TOKEN);
    }
    LoadClient(WorkerPlan plan, ClientMetrics metrics, java.nio.file.Path topology, java.nio.file.Path evidence, String id, int imageToken) throws Exception {
        this(plan, metrics, topology, evidence, id, imageToken, 0, 0L, 1);
    }
    LoadClient(WorkerPlan plan, ClientMetrics metrics, java.nio.file.Path topology, java.nio.file.Path evidence, String id, int imageToken, int blockSize, long seed, int prefillCpSize) throws Exception {
        this.imageToken = imageToken; this.blockSize = blockSize;
        this.plan = plan; this.metrics = metrics;
        queue = new ArrayBlockingQueue<>(plan.queue_capacity());
        for (var outcome : NativeClient.Outcome.values()) outcomes.put(outcome.name(), 0L);
        discovery = new MasterDiscovery(topology, plan.masters());
        try { journal = new RequestJournal(evidence, id, plan); }
        catch (Exception error) { discovery.close(); throw error; }
        client = new NativeClient(discovery::routes, journal::record, imageToken, seed, prefillCpSize);
        // A lost begin command must not leave a prepared worker alive forever.
        dispatcher = Thread.ofVirtual().name("client-dispatch").start(this::dispatch);
    }
    synchronized void begin(long epochMillis) {
        if (begun || stopped || completion.isDone()) throw new IllegalStateException("Client already begun or stopped");
        long delay = epochMillis - System.currentTimeMillis();
        if (delay < 0 || delay > 30000) throw new IllegalArgumentException("Start time must be within the next 30 seconds");
        begun = true;
        long start = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(delay);
        player = Thread.ofVirtual().name("client-player").start(() -> {
            try { play(plan, start, imageToken, this::offer); }
            catch (InterruptedException interrupted) { if (!isStopped()) fail("Player interrupted unexpectedly"); }
            catch (java.io.InterruptedIOException interrupted) { if (!isStopped()) fail(interrupted.toString()); }
            catch (java.io.IOException | RuntimeException error) { fail(error.toString()); }
            finally { synchronized (LoadClient.this) { finishInput(); } }
        });
        LockSupport.unpark(dispatcher);
    }
    static void play(WorkerPlan plan, long startNanos, Consumer<Item> submit) throws IOException, InterruptedException {
        play(plan, startNanos, org.flexlb.mock.protocol.MediaInput.IMAGE_TOKEN, submit);
    }

    static void play(WorkerPlan plan, long startNanos, int imageToken, Consumer<Item> submit) throws IOException, InterruptedException {
        try (var indexed = new IndexedTraffic(Path.of(plan.requests_file()))) {
            long expected = ClientProtocol.requestCount(indexed.count, plan.first_index(), plan.stride());
            if (!indexed.identity.equals(plan.sha256()) || expected != plan.count())
                throw new IOException("Indexed traffic changed after planning");
            delayNanos(indexed.count - 1, indexed.lastArrival, plan.qps(), plan.replay_speed());
            for (long position = 0; position < plan.count(); position++) {
                if (Thread.interrupted()) throw new InterruptedException();
                long index = plan.first_index() + position * plan.stride();
                var entry = indexed.entry(index, imageToken);
                int outputLimit = entry.playbackLimit(plan.fallback_max_output_tokens());
                long due = startNanos + delayNanos(index, entry.arrivalMs(), plan.qps(), plan.replay_speed());
                long left;
                while ((left = due - System.nanoTime()) > 0) TimeUnit.NANOSECONDS.sleep(left);
                if (Thread.interrupted()) throw new InterruptedException();
                submit.accept(new Item(plan.first_request_id() + index, entry, outputLimit));
            }
        }
    }

    static long delayNanos(long index, long arrivalMs, Double qps, Double speed) throws IOException {
        double seconds = speed == null ? index / qps : arrivalMs / (1000.0 * speed);
        if (!Double.isFinite(seconds) || seconds < 0 || seconds > 86400)
            throw new IOException("Playback exceeds 24 hours");
        return (long) (seconds * 1_000_000_000.0);
    }

    private synchronized boolean isStopped() { return stopped; }
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
    synchronized void halt() {
        if (completion.isDone() || stopped) return;
        stopped = true; queue.clear();
        if (player != null) player.interrupt();
        finishInput();
    }
    synchronized void fail(String error) {
        if (errors.isEmpty()) errors.add(error);
        halt();
    }
    private void dispatch() {
        long prepareDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        try {
            while (true) {
                discovery.check();
                if (journal.failure() != null) fail(journal.failure());
                List<NativeClient.Call> cancel = List.of();
                synchronized (this) {
                    if (!begun && !stopped && System.nanoTime() >= prepareDeadline) fail("Timed out waiting for begin");
                    if (producerDone && queue.isEmpty() && active.isEmpty()) break;
                    if (producerDone && System.nanoTime() >= drainDeadline) {
                        fail("Client drain timed out");
                        cancel = List.copyOf(active.values());
                    } else if (begun && !stopped && !queue.isEmpty() && active.size() < plan.concurrency()) {
                        var item = queue.remove();
                        var entry = item.entry();
                        var input = entry.input().toBuilder().setRequestId(item.requestId());
                        input.getGenerateConfigBuilder().setReuseCache(Boolean.TRUE.equals(plan.reuse_cache()))
                            .setMaxNewTokens(item.outputLimit()).setIgnoreEos(entry.observedOutputTokens() > 0)
                            .setCanUsePdSeparation(true).setReturnIncremental(true).setIsStreaming(true);
                        var generateInput = input.build();
                        var request = FlexlbScheduleRequestPB.newBuilder().setRequestId(item.requestId())
                            .setCacheKeyBlockSize(blockSize).setSeqLen(input.getTokenIdsCount())
                            .setMaxNewTokens(input.getGenerateConfig().getMaxNewTokens()).setPriority(entry.priority())
                            .setGenerateInput(generateInput.toByteString()).build();
                        var call = client.send(request, generateInput, Duration.ofMillis(plan.request_timeout_ms()), entry.expectedImageTokens());
                        active.put(item.requestId(), call); issued++; metrics.issued();
                        call.completion().whenComplete((result, error) -> {
                            synchronized (LoadClient.this) {
                                active.remove(item.requestId()); finished++;
                                if (error != null) fail("Request execution failed: " + error);
                                else { metrics.finished(result); outcomes.merge(result.outcome().name(), 1L, Long::sum); }
                                LockSupport.unpark(dispatcher);
                            }
                        });
                        continue;
                    }
                }
                // Native completion callbacks acquire this object's monitor; never cancel while holding it.
                cancel.forEach(NativeClient.Call::cancel);
                LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(100));
                if (Thread.interrupted()) throw new InterruptedException();
            }
        } catch (InterruptedException error) {
            fail("Client dispatcher interrupted"); Thread.currentThread().interrupt();
        } catch (RuntimeException error) { fail("Client dispatcher failed: " + error); }
        finally {
            if (player != null) try { player.join(1000); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            client.close(); discovery.close();
            try { journal.finish(issued, finished); }
            catch (java.io.IOException error) { fail(error.getMessage()); }
            synchronized (this) { completion.complete(snapshot(true)); LockSupport.unpark(dispatcher); }
        }
    }
    private State snapshot(boolean done) {
        return new State(done, stopped, produced.get(), issued, finished, plan.count() - issued,
            queue.size(), active.size(), Map.copyOf(outcomes), List.copyOf(errors));
    }
    synchronized State state() { return completion.isDone() ? completion.join() : snapshot(false); }
    State await(long timeoutMillis) throws Exception { return completion.get(timeoutMillis, TimeUnit.MILLISECONDS); }
    @Override public void close() {
        halt();
        List<NativeClient.Call> calls;
        synchronized (this) { calls = List.copyOf(active.values()); }
        calls.forEach(NativeClient.Call::cancel);
        try { completion.get(8, TimeUnit.SECONDS); }
        catch (Exception error) { dispatcher.interrupt(); }
    }
}

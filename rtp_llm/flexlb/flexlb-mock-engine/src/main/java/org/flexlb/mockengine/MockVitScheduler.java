package org.flexlb.mockengine;

import io.grpc.Status;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;

/** MMScheduler queue, work-budget and continuation semantics; CPU metadata forward. */
final class MockVitScheduler implements AutoCloseable {
    record Item(Callable<MockVitImage.Result> forward, long images, MockVitWorkEstimate estimate) {
        Item { if (images < 0) throw new IllegalArgumentException("negative image count"); }
    }
    private static final class ForwardFuture<R> extends CompletableFuture<R> {
        private boolean running;
        private volatile boolean cancelled;
        synchronized boolean claim() {
            if (cancelled || isCancelled() || isCompletedExceptionally()) return false;
            running = true;
            return true;
        }
        @Override public synchronized boolean cancel(boolean interrupt) {
            // Scheduler timeout sets its request event even if Future.cancel
            // cannot cancel an already RUNNING forward. No continuation follows.
            if (running) return false;
            cancelled = true;
            return super.cancel(false);
        }
    }
    private static final class Request<R> {
        final ForwardFuture<R> result = new ForwardFuture<>();
        final List<MockVitImage.Result> outputs = new ArrayList<>();
        final Function<List<MockVitImage.Result>, R> merge;
        final long started = System.nanoTime();
        List<Chunk> chunks;
        int next = 1;
        Request(Function<List<MockVitImage.Result>, R> merge) { this.merge = merge; }
        void finish() { result.complete(merge.apply(outputs)); }
    }
    private static final class Chunk {
        final Request<?> request;
        final List<Item> items;
        final long images;
        final MockVitWorkEstimate work;
        long enqueuedAt;
        Chunk(Request<?> request, List<Item> items) {
            this.request = request; this.items = List.copyOf(items);
            images = items.stream().mapToLong(Item::images).reduce(0, Math::addExact);
            work = items.stream().allMatch(item -> item.estimate != null)
                    ? items.stream().map(Item::estimate).reduce(MockVitWorkEstimate.ZERO, MockVitWorkEstimate::plus) : null;
        }
    }
    // Only new requests are capped. Continuations bypass the cap, as real's
    // sole consumer must never block while enqueueing its next chunk.
    private final LinkedBlockingQueue<Chunk> waiting = new LinkedBlockingQueue<>();
    private final int queueSize, batchSize;
    private final long maxImages, waitMs, forwardMs;
    private final MockVitWorkEstimate budget;
    private final Thread worker;
    private volatile boolean closed;
    private volatile Chunk pending;
    private volatile Consumer<Map<String, Number>> metrics = values -> {};
    private long batches;

    MockVitScheduler(int queueSize, int batchSize, long maxImages, long waitMs, long forwardMs) {
        this(queueSize, batchSize, maxImages, waitMs, forwardMs, null);
    }
    MockVitScheduler(int queueSize, int batchSize, long maxImages, long waitMs, long forwardMs, MockVitWorkEstimate budget) {
        if (queueSize < 1 || batchSize < 1 || maxImages < 1 || waitMs < 0 || forwardMs < 0)
            throw new IllegalArgumentException("invalid ViT scheduler limits");
        this.queueSize = queueSize; this.batchSize = batchSize; this.maxImages = maxImages;
        this.waitMs = waitMs; this.forwardMs = forwardMs; this.budget = budget;
        worker = Thread.ofPlatform().daemon().name("mock-vit-gpu-scheduler").start(this::run);
    }
    void cancelRequest(CompletableFuture<?> future) {
        if (future instanceof ForwardFuture<?> forward) forward.cancelled = true;
        future.cancel(false);
    }
    void setMetrics(Consumer<Map<String, Number>> metrics) { this.metrics = metrics; }
    private void report(String name, Number value) {
        try { metrics.accept(Map.of(name, value)); }
        catch (RuntimeException error) { /* monitoring cannot change scheduler ownership */ }
    }
    private void depth() { report("py_rtp_vit_embedding_queue_size", queued()); }
    CompletableFuture<MockVitImage.Result> submit(Callable<MockVitImage.Result> forward) {
        return submit(List.of(new Item(forward, 1, null)), outputs -> outputs.getFirst());
    }
    CompletableFuture<List<MockVitImage.Result>> submitItems(List<Item> items) {
        return submit(items, ArrayList::new);
    }
    private synchronized <R> CompletableFuture<R> submit(List<Item> items, Function<List<MockVitImage.Result>, R> merge) {
        var request = new Request<R>(merge);
        try {
            request.chunks = buildChunks(request, items);
            if (closed) throw Status.UNAVAILABLE.asRuntimeException();
            if (waiting.size() >= queueSize) {
                report("py_rtp_vit_embedding_overload_qps", 1);
                throw MockRpcErrors.error(Status.RESOURCE_EXHAUSTED, 903, "[MM_PROCESS_ERROR] ViT GPU waiting queue is full");
            }
            enqueue(request.chunks.getFirst());
        } catch (RuntimeException error) { request.result.completeExceptionally(error); }
        return request.result;
    }
    private List<Chunk> buildChunks(Request<?> request, List<Item> items) {
        if (items.isEmpty()) throw new IllegalArgumentException("ViT requires at least one work item");
        List<Chunk> result = new ArrayList<>();
        List<Item> chunk = new ArrayList<>();
        long images = 0;
        var work = MockVitWorkEstimate.ZERO;
        for (var item : items) {
            if (item.images > maxImages) throw MockRpcErrors.error(Status.INVALID_ARGUMENT, 902, "[MM_WRONG_FORMAT_ERROR] indivisible ViT work item exceeds image limit");
            if (budget != null && item.estimate == null) throw new IllegalArgumentException("cost-aware ViT work item requires a model work estimate");
            var next = budget == null ? work : work.plus(item.estimate);
            if (budget != null && !chunk.isEmpty() && (images + item.images > maxImages || !next.fits(budget))) {
                result.add(new Chunk(request, chunk)); chunk.clear(); images = 0; work = MockVitWorkEstimate.ZERO;
                next = item.estimate;
            }
            chunk.add(item); images = Math.addExact(images, item.images); work = next;
        }
        if (budget == null && images > maxImages)
            throw MockRpcErrors.error(Status.INVALID_ARGUMENT, 902, "[MM_WRONG_FORMAT_ERROR] ViT request exceeds image limit");
        result.add(new Chunk(request, chunk));
        return result;
    }
    private void enqueue(Chunk chunk) {
        chunk.enqueuedAt = System.nanoTime(); waiting.add(chunk); depth();
    }
    private Chunk pull(long nanos) throws InterruptedException {
        Chunk chunk = waiting.poll(nanos, TimeUnit.NANOSECONDS); if (chunk != null) depth(); return chunk;
    }
    private boolean cancelled(Chunk chunk) { return chunk.request.result.cancelled || chunk.request.result.isDone(); }
    private List<Chunk> collect() throws InterruptedException {
        Chunk first;
        do {
            if (closed) return List.of();
            first = pending;
            if (first != null) { pending = null; depth(); }
            else first = pull(TimeUnit.MILLISECONDS.toNanos(10));
        } while (first == null || cancelled(first));
        var batch = new ArrayList<Chunk>(); batch.add(first);
        long images = first.images;
        var work = first.work;
        long until = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(waitMs);
        while (!closed && batch.size() < batchSize && images < maxImages) {
            long remaining = until - System.nanoTime();
            if (remaining <= 0) break;
            var next = pull(Math.min(remaining, TimeUnit.MILLISECONDS.toNanos(10)));
            if (next == null || cancelled(next)) continue;
            var combined = work == null || next.work == null ? null : work.plus(next.work);
            if (images + next.images > maxImages || (budget != null && !combined.fits(budget))) {
                pending = next; depth(); break;
            }
            batch.add(next); images += next.images; work = combined;
        }
        return batch;
    }
    private void run() {
        List<Chunk> batch = List.of();
        try {
            while (!closed) {
                batch = collect();
                synchronized (this) {
                    if (closed) break;
                    batch.removeIf(chunk -> !chunk.request.result.claim());
                }
                if (batch.isEmpty()) continue;
                report("py_rtp_vit_embedding_batch_size", batch.size());
                long started = System.nanoTime();
                for (var chunk : batch) report("py_rtp_vit_embedding_queue_wait_rt", (started - chunk.enqueuedAt) / 1e6);
                try {
                    if (forwardMs > 0) Thread.sleep(forwardMs);
                    synchronized (this) { batches++; }
                    List<List<MockVitImage.Result>> results = new ArrayList<>();
                    for (var chunk : batch) {
                        List<MockVitImage.Result> values = new ArrayList<>();
                        for (var item : chunk.items) values.add(item.forward.call());
                        results.add(values);
                    }
                    report("py_rtp_vit_embedding_forward_rt", (System.nanoTime() - started) / 1e6);
                    for (int i = 0; i < batch.size(); i++) {
                        var request = batch.get(i).request;
                        request.outputs.addAll(results.get(i));
                        synchronized (this) {
                            if (!request.result.cancelled && request.next < request.chunks.size() && !closed)
                                enqueue(request.chunks.get(request.next++));
                            else if (request.result.cancelled) request.finish();
                            else if (request.next < request.chunks.size())
                                request.result.completeExceptionally(Status.UNAVAILABLE.asRuntimeException());
                            else request.finish();
                        }
                        if (request.result.isDone() && !request.result.cancelled) report("py_rtp_vit_embedding_rt", (System.nanoTime() - request.started) / 1e6);
                    }
                } catch (Throwable error) {
                    for (var chunk : batch) chunk.request.result.completeExceptionally(error);
                }
                batch = List.of();
            }
        } catch (InterruptedException stopped) { Thread.currentThread().interrupt(); }
        finally {
            for (var chunk : batch) failClosed(chunk);
            if (pending != null) { failClosed(pending); pending = null; }
            Chunk chunk;
            while ((chunk = waiting.poll()) != null) failClosed(chunk);
            depth();
        }
    }
    private void failClosed(Chunk chunk) { chunk.request.result.completeExceptionally(Status.UNAVAILABLE.asRuntimeException()); }
    long maxRequestImages() { return budget == null ? maxImages : Long.MAX_VALUE; }
    synchronized long batches() { return batches; }
    int queued() { return waiting.size() + (pending == null ? 0 : 1); }
    @Override public void close() {
        synchronized (this) { closed = true; }
        try {
            worker.join(5000);
            if (worker.isAlive()) throw new IllegalStateException("ViT forward did not finish during close");
        }
        catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
    }
}

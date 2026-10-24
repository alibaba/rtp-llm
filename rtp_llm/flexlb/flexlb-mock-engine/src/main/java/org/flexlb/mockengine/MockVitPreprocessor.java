package org.flexlb.mockengine;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.*;
import java.util.function.Consumer;

/** Logical spawn workers: isolated URL caches and MultiprocessPreprocessExecutor recovery. */
final class MockVitPreprocessor implements AutoCloseable {
    private final boolean local;
    private final int count;
    private ExecutorService workers;
    private final Set<CompletableFuture<?>> pending = ConcurrentHashMap.newKeySet();
    private final Map<Thread, MockVitMediaCache> media = new ConcurrentHashMap<>();
    private MockVitMediaCache configuredLocalMedia = new MockVitMediaCache(0);
    private static final class PreparedFuture<T> extends CompletableFuture<T> {
        Map<String, Number> timing = Map.of();
    }
    private int timeouts;
    private long generation;
    private final ThreadLocal<MockVitMediaCache> workerMedia = new ThreadLocal<>();
    private volatile boolean closed;
    private volatile Consumer<Map<String, Number>> metrics = ignored -> {};
    // Failure injection at the IPC submission boundary, without process or timing simulation.
    java.util.function.BiConsumer<ExecutorService, Runnable> dispatch = ExecutorService::execute;

    MockVitPreprocessor(boolean local, int count) {
        this.local = local; this.count = count;
        if (!local) workers = newWorkers();
    }
    private ExecutorService newWorkers() {
        long epoch = ++generation;
        return Executors.newFixedThreadPool(count, task -> {
            var thread = new Thread(() -> {
                var cache = new MockVitMediaCache(10); workerMedia.set(cache);
                synchronized (this) { if (epoch == generation) media.put(Thread.currentThread(), cache); }
                try { task.run(); } finally { media.remove(Thread.currentThread(), cache); workerMedia.remove(); }
            }, "mock-vit-preprocess"); thread.setDaemon(true); return thread;
        });
    }
    void configureMediaCache(int items) { configuredLocalMedia = new MockVitMediaCache(items); }
    MockVitMediaCache mediaCache() {
        // real spawn imports multimodal_util's default 10-item cache. Parent resize is process-local.
        return local ? configuredLocalMedia : java.util.Objects.requireNonNull(workerMedia.get(), "preprocessing worker cache");
    }
    long mediaCacheItems() { return local ? configuredLocalMedia.size() : media.values().stream().mapToLong(MockVitMediaCache::size).sum(); }
    void setMetrics(Consumer<Map<String, Number>> reporter) { metrics = reporter; reportQueue(); }
    int pending() { return pending.size(); }
    private void report(Map<String, Number> values) {
        try { metrics.accept(values); } catch (RuntimeException ignored) { /* metrics do not own work */ }
    }
    private void reportQueue() { report(Map.of("py_rtp_vit_preprocess_queue_size", pending())); }
    private static boolean broken(Throwable error) {
        return error instanceof java.io.IOException || error instanceof java.io.UncheckedIOException;
    }
    private synchronized void rebuild() {
        workers.shutdownNow(); workers = newWorkers();
        pending.clear(); media.clear(); reportQueue();
    }
    <T> CompletableFuture<T> submit(Callable<T> prepare) {
        var result = new PreparedFuture<T>();
        Runnable task = () -> {
            long started = System.nanoTime();
            try {
                T value = prepare.call();
                double total = (System.nanoTime() - started) / 1e6;
                double download = value instanceof MockVitImage.Prepared image ? image.downloadMs() : 0;
                result.timing = Map.of("py_rtp_vit_preprocess_rt", total, "py_rtp_vit_download_rt", download,
                        "py_rtp_vit_preprocess_other_rt", Math.max(0, total - download));
                result.complete(value);
            } catch (Throwable error) { result.completeExceptionally(error); }
            finally { if (!local && pending.remove(result)) reportQueue(); }
        };
        synchronized (this) {
            if (closed) return CompletableFuture.failedFuture(new RejectedExecutionException("preprocessor closed"));
            if (!local) {
                pending.add(result); reportQueue();
                try { dispatch.accept(workers, task); }
                catch (RuntimeException error) {
                    if (broken(error)) {
                        rebuild(); pending.add(result); reportQueue();
                        try { dispatch.accept(workers, task); }
                        catch (RuntimeException retry) { pending.remove(result); reportQueue(); result.completeExceptionally(retry); }
                    } else { pending.remove(result); reportQueue(); result.completeExceptionally(error); }
                }
                return result;
            }
        }
        task.run(); return result;
    }
    <T> T await(CompletableFuture<T> result, long timeoutMs) throws Exception {
        try {
            T value = result.get(timeoutMs, TimeUnit.MILLISECONDS);
            if (result instanceof PreparedFuture<?> prepared) report(prepared.timing);
            synchronized (this) { timeouts = 0; }
            return value;
        } catch (TimeoutException error) {
            synchronized (this) { if (!local && !closed && ++timeouts >= count) { rebuild(); timeouts = 0; } }
            throw error;
        } catch (ExecutionException error) {
            synchronized (this) { if (!local && !closed && broken(error.getCause())) rebuild(); }
            throw error;
        }
    }
    @Override public synchronized void close() {
        closed = true;
        if (workers != null) workers.shutdownNow();
        pending.forEach(task -> task.cancel(false)); pending.clear(); media.clear(); configuredLocalMedia.clear(); reportQueue();
    }
}

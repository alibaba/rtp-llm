package org.flexlb.mockengine;

import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

/** LocalPreprocessExecutor / MultiprocessPreprocessExecutor logical queue, using JVM workers. */
final class MockVitPreprocessor implements AutoCloseable {
    private final ExecutorService workers;
    private final java.util.Set<CompletableFuture<?>> pending = ConcurrentHashMap.newKeySet();
    private volatile boolean closed;
    private volatile Consumer<Map<String, Number>> metrics = ignored -> {};

    MockVitPreprocessor(boolean local, int count) {
        workers = local ? null : Executors.newFixedThreadPool(count, task -> {
            var thread = new Thread(task, "mock-vit-preprocess"); thread.setDaemon(true); return thread;
        });
    }
    void setMetrics(Consumer<Map<String, Number>> reporter) { metrics = reporter; reportQueue(); }
    int pending() { return pending.size(); }
    private void report(Map<String, Number> values) {
        try { metrics.accept(values); } catch (RuntimeException ignored) { /* metrics do not own work */ }
    }
    private void reportQueue() { report(Map.of("py_rtp_vit_preprocess_queue_size", pending())); }
    <T> CompletableFuture<T> submit(Callable<T> prepare) {
        var result = new CompletableFuture<T>();
        if (closed) return CompletableFuture.failedFuture(new java.util.concurrent.RejectedExecutionException("preprocessor closed"));
        if (workers != null) { pending.add(result); reportQueue(); }
        Runnable task = () -> {
            long started = System.nanoTime();
            try {
                T value = prepare.call();
                double total = (System.nanoTime() - started) / 1e6;
                double download = value instanceof MockVitImage.Prepared image ? image.downloadMs() : 0;
                report(Map.of("py_rtp_vit_preprocess_rt", total, "py_rtp_vit_download_rt", download,
                        "py_rtp_vit_preprocess_other_rt", Math.max(0, total - download)));
                result.complete(value);
            } catch (Throwable error) { result.completeExceptionally(error); }
            finally { if (workers != null) { pending.remove(result); reportQueue(); } }
        };
        if (workers == null) task.run();
        else try { workers.execute(task); }
        catch (RuntimeException error) { pending.remove(result); reportQueue(); result.completeExceptionally(error); }
        return result;
    }
    @Override public void close() {
        closed = true;
        if (workers != null) workers.shutdownNow();
        pending.forEach(task -> task.cancel(false)); pending.clear(); reportQueue();
    }
}

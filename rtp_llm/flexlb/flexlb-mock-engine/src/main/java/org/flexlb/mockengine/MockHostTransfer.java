package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

/** Logical single-rank business/transfer queues. No tensor copy or bandwidth model. */
final class MockHostTransfer implements AutoCloseable {
    record Policy(int workers, int businessQueue, int transferQueue, int descriptors, long timeoutMs) {
        static final String[] FIELDS = {"block_tree_transfer_worker_count", "block_tree_business_queue_max_size",
                "block_tree_transfer_queue_max_size", "memory_cache_max_descriptors_per_transfer_batch", "memory_cache_sync_timeout_ms"};
        static Policy parse(JsonNode config) {
            return new Policy(MockEngineConfig.integer(config, FIELDS[0], 1), MockEngineConfig.integer(config, FIELDS[1], 0),
                    MockEngineConfig.integer(config, FIELDS[2], 0), MockEngineConfig.integer(config, FIELDS[3], 1),
                    MockEngineConfig.integer(config, FIELDS[4], 1));
        }
    }
    private final Policy policy;
    private final MockCacheTaskPool business, load, background;
    private final java.util.Set<CompletableFuture<Void>> active = java.util.concurrent.ConcurrentHashMap.newKeySet();
    MockHostTransfer(Policy policy) {
        this.policy = policy;
        business = new MockCacheTaskPool(4, policy.businessQueue() == 0 ? 0 : Math.addExact(policy.businessQueue(), 64));
        load = new MockCacheTaskPool(policy.workers(), policy.transferQueue());
        background = new MockCacheTaskPool(policy.workers(), policy.transferQueue());
    }
    private boolean stopped;
    synchronized CompletableFuture<Void> submit(boolean loading, int blocks, double delayMs) {
        if (stopped) return CompletableFuture.failedFuture(new java.util.concurrent.RejectedExecutionException("host transfer stopped"));
        var result = new CompletableFuture<Void>();
        active.add(result); result.whenComplete((ignored, error) -> active.remove(result));
        long deadline = MockPdPolicy.deadline(System.nanoTime(), policy.timeoutMs());
        Runnable timeout = () -> result.completeExceptionally(new java.util.concurrent.TimeoutException("host cache sync timeout"));
        Runnable begin = () -> {
            if (result.isDone()) return;
            var batches = new ArrayList<CompletableFuture<Void>>();
            for (int start = 0; start < blocks; start += policy.descriptors()) {
                var copied = new CompletableFuture<Void>(); batches.add(copied);
                Runnable transfer = () -> {
                    if (result.isDone()) { copied.cancel(false); return; }
                    try {
                        if (delayMs > 0) java.util.concurrent.TimeUnit.NANOSECONDS.sleep(Math.min(
                                Math.round(delayMs * 1e6), Math.max(0, deadline - System.nanoTime())));
                        if (System.nanoTime() >= deadline) copied.completeExceptionally(new java.util.concurrent.TimeoutException("host transfer timeout"));
                        else copied.complete(null);
                    } catch (InterruptedException error) { Thread.currentThread().interrupt(); copied.completeExceptionally(error); }
                };
                if (!(loading ? load : background).submit(loading, transfer, deadline,
                        () -> copied.completeExceptionally(new java.util.concurrent.TimeoutException("host transfer queue timeout"))))
                    copied.completeExceptionally(new java.util.concurrent.RejectedExecutionException("host transfer queue full"));
            }
            CompletableFuture.allOf(batches.toArray(CompletableFuture[]::new)).whenComplete((ignored, error) -> {
                Runnable settle = () -> { if (error == null) result.complete(null); else result.completeExceptionally(error); };
                if (!business.complete(settle)) settle.run();
            });
        };
        if (!business.submit(loading, begin, deadline, timeout))
            result.completeExceptionally(new java.util.concurrent.RejectedExecutionException("host business queue full"));
        return result;
    }
    java.util.Map<String, Object> snapshot() {
        return java.util.Map.of("business", business.sizes(), "load_transfer", load.sizes(),
                "background_transfer", background.sizes(), "in_flight", active.size());
    }
    @Override public void close() {
        synchronized (this) { if (stopped) return; stopped = true; }
        business.stopAdmission();
        business.waitForIdle(); // queued bodies may still submit transfers
        load.stopAdmission(); background.stopAdmission();
        load.waitForIdle(); background.waitForIdle();
        business.waitForIdle(); // transfer completions enqueue settlement here
        load.close(); background.close(); business.close();
    }
}

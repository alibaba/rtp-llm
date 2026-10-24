package org.flexlb.mockengine;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BiConsumer;
import java.util.function.Function;

/** MMEmbeddingCache state/allocator semantics, with immutable Java payloads in place of tensors. */
final class MockVitEmbeddingCache<V> implements AutoCloseable {
    enum Device { GPU, CPU }
    enum Copy { STORE, READ, DETACH }
    record Tensor(Object identity, long bytes, int elementSize, Device device) {
        Tensor {
            Objects.requireNonNull(identity); Objects.requireNonNull(device);
            if (bytes < 0 || elementSize < 1) throw new IllegalArgumentException("invalid tensor layout");
        }
    }
    record Bytes(long gpu, long cpu) {}
    record Hit<V>(V value, String generation) {}
    private static final class Entry<V> {
        final V value;
        final List<Tensor> layout;
        final String generation;
        MockTensorPool.Block gpuBlock, cpuBlock;
        long gpuBytes, cpuBytes;
        boolean offloaded, retired, ready;
        int readers;
        Entry(V value, List<Tensor> layout, String generation) {
            this.value = value; this.layout = List.copyOf(layout); this.generation = generation;
        }
    }
    private final Map<String, Entry<V>> entries = new LinkedHashMap<>();
    private final LinkedHashMap<String, Entry<V>> gpuLru = new LinkedHashMap<>(), cpuLru = new LinkedHashMap<>();
    private final MockTensorPool gpu, cpu;
    private final ReentrantLock transfer = new ReentrantLock();
    private long gpuBytes, cpuBytes, evictions, promotions, demotions, transferErrors;
    private boolean closed;
    // Harness-only completion/failure controls. No sleeps or bandwidth/performance model.
    BiConsumer<Copy, V> copier = (operation, value) -> {};
    Function<V, List<CompletableFuture<Void>>> releaseEvents = value -> List.of();

    MockVitEmbeddingCache(long gpuBytes, long cpuBytes) {
        gpu = new MockTensorPool(gpuBytes); cpu = new MockTensorPool(cpuBytes);
    }
    static Bytes bytes(List<Tensor> tensors, boolean offload) {
        // Match (tensor identity, target device), rather than deduplicating equal contents.
        var seen = new IdentityHashMap<Object, EnumSet<Device>>();
        long gpu = 0, cpu = 0;
        for (var tensor : tensors) {
            Device target = offload ? Device.CPU : tensor.device;
            if (!seen.computeIfAbsent(tensor.identity, ignored -> EnumSet.noneOf(Device.class)).add(target)) continue;
            if (target == Device.GPU) gpu = Math.addExact(MockTensorPool.align(gpu, tensor.elementSize), tensor.bytes);
            else cpu = Math.addExact(MockTensorPool.align(cpu, tensor.elementSize), tensor.bytes);
        }
        return new Bytes(gpu, cpu);
    }
    synchronized boolean contains(String key) { var entry = entries.get(key); return entry != null && entry.ready; }
    synchronized String tier(String key) {
        var entry = entries.get(key); return entry == null || !entry.ready || entry.gpuBytes + entry.cpuBytes == 0 ? null : entry.gpuBytes > 0 ? "gpu" : "cpu";
    }
    synchronized List<String> keys(Device tier, int limit) {
        return (tier == Device.GPU ? gpuLru : cpuLru).reversed().keySet().stream().limit(limit).toList();
    }
    Hit<V> get(String key) { return get(key, -1); }

    Hit<V> get(String key, long timeoutMs) {
        var read = claim(key, timeoutMs);
        return read == null ? null : read.get();
    }

    java.util.function.Supplier<Hit<V>> claim(String key, long timeoutMs) {
        Entry<V> entry;
        long started = System.nanoTime();
        synchronized (this) {
            entry = entries.get(key);
            if (entry == null || !entry.ready) return null;
            entry.readers++;
            if (entry.gpuBytes + entry.cpuBytes > 0) {
                var lru = entry.gpuBytes > 0 ? gpuLru : cpuLru;
                lru.remove(key); lru.put(key, entry);
            }
        }
        return () -> {
            try {
                long remaining = timeoutMs < 0 ? -1 : Math.max(0, timeoutMs
                        - java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started));
                return read(key, entry, remaining);
            } finally { releaseReader(entry); }
        };
    }

    private Hit<V> read(String key, Entry<V> entry, long timeoutMs) {
        boolean offloaded;
        synchronized (this) { offloaded = entry.offloaded; }
        // The claim keeps the entry alive even if a concurrent eviction removes its index.
        // Unoffloaded reads use request-owned copies and do not serialize behind unrelated tier transfers.
        if (!offloaded) { read(entry); return new Hit<>(entry.value, entry.generation); }
        try {
            if (timeoutMs < 0) transfer.lockInterruptibly();
            else if (!transfer.tryLock(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS))
                throw MockRpcErrors.error(io.grpc.Status.DEADLINE_EXCEEDED, 903, "[MM_PROCESS_ERROR] Waiting for embedding cache transfer timed out");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw io.grpc.Status.CANCELLED.withDescription("embedding cache read interrupted").asRuntimeException();
        }
        try {
            read(entry);
            if (entry.offloaded && bytes(entry.layout, false).gpu > 0) {
                var original = bytes(entry.layout, false);
                synchronized (this) {
                    if (entries.get(key) != entry || original.gpu > gpu.stats().capacity() || original.cpu > cpu.stats().capacity())
                        return new Hit<>(entry.value, entry.generation);
                    uncharge(key, entry);
                }
                release(entry);
                if (store(key, entry, false)) { synchronized (this) { promotions++; } }
            }
            return new Hit<>(entry.value, entry.generation);
        } finally { transfer.unlock(); }
    }
    private void read(Entry<V> entry) {
        try { copier.accept(Copy.READ, entry.value); }
        catch (RuntimeException error) { synchronized (this) { transferErrors++; } throw error; }
    }
    void put(String key, V value, List<Tensor> layout, String generation) {
        transfer.lock();
        try {
            synchronized (this) { if (closed) return; removeLocked(key, false); }
            var entry = new Entry<>(value, layout, generation);
            synchronized (this) { entries.put(key, entry); }
            var required = bytes(layout, false);
            store(key, entry, required.gpu > gpu.stats().capacity());
        } finally { transfer.unlock(); }
    }
    private boolean store(String key, Entry<V> entry, boolean offload) {
        var required = bytes(entry.layout, offload);
        if (required.gpu > gpu.stats().capacity() || required.cpu > cpu.stats().capacity()
                || required.gpu == 0 && required.cpu == 0) {
            synchronized (this) { removeLocked(key, true); }
            return false;
        }
        makeRoom(Device.GPU, required.gpu); makeRoom(Device.CPU, required.cpu);
        makePoolRoom(Device.GPU, required.gpu); makePoolRoom(Device.CPU, required.cpu);
        MockTensorPool.Block gpuBlock = null, cpuBlock = null;
        try {
            for (int attempt = 0; ; attempt++) {
                try {
                    gpuBlock = gpu.reserve(required.gpu); cpuBlock = cpu.reserve(required.cpu);
                    break;
                } catch (MockTensorPool.CapacityException error) {
                    gpu.release(gpuBlock, List.of()); gpuBlock = null;
                    cpu.release(cpuBlock, List.of()); cpuBlock = null;
                    if (attempt == 1) throw error;
                    makePoolRoom(Device.GPU, required.gpu); makePoolRoom(Device.CPU, required.cpu);
                }
            }
            copier.accept(Copy.STORE, entry.value);
        } catch (RuntimeException error) {
            gpu.release(gpuBlock, List.of()); cpu.release(cpuBlock, List.of());
            synchronized (this) { transferErrors++; removeLocked(key, true); }
            return false; // Optional insertion cannot invalidate the successful forward/read.
        }
        synchronized (this) {
            entry.gpuBlock = gpuBlock; entry.cpuBlock = cpuBlock;
            entry.gpuBytes = required.gpu; entry.cpuBytes = required.cpu; entry.offloaded = offload; entry.ready = true;
            gpuBytes += required.gpu; cpuBytes += required.cpu;
            (required.gpu > 0 ? gpuLru : cpuLru).put(key, entry);
            if (offload) demotions++;
        }
        return true;
    }
    private void makeRoom(Device device, long required) {
        if (required == 0) return;
        for (;;) {
            synchronized (this) {
                long used = device == Device.GPU ? gpuBytes : cpuBytes;
                if (used + required <= pool(device).stats().capacity()) return;
            }
            evictOne(device);
        }
    }
    private void makePoolRoom(Device device, long required) {
        while (required > 0) {
            var stats = pool(device).stats();
            if (stats.largest() >= required || stats.pending() > 0) return;
            if (!evictOne(device)) return;
        }
    }
    private boolean evictOne(Device device) {
        String key;
        Entry<V> entry;
        synchronized (this) {
            var lru = device == Device.GPU ? gpuLru : !cpuLru.isEmpty() ? cpuLru : gpuLru;
            if (lru.isEmpty()) return false;
            key = lru.firstEntry().getKey(); entry = lru.firstEntry().getValue();
            if (device == Device.CPU) { removeLocked(key, true); return true; }
            uncharge(key, entry);
        }
        // Demotion retains the logical entry and generation, unlike eviction.
        // Store the CPU copy before releasing the old GPU backing.
        var oldGpu = entry.gpuBlock; var oldCpu = entry.cpuBlock;
        entry.gpuBlock = null; entry.cpuBlock = null;
        store(key, entry, true);
        gpu.release(oldGpu, releaseEvents.apply(entry.value));
        cpu.release(oldCpu, releaseEvents.apply(entry.value));
        return true;
    }
    private MockTensorPool pool(Device device) { return device == Device.GPU ? gpu : cpu; }
    private void uncharge(String key, Entry<V> entry) {
        gpuLru.remove(key); cpuLru.remove(key);
        gpuBytes -= entry.gpuBytes; cpuBytes -= entry.cpuBytes;
        entry.gpuBytes = entry.cpuBytes = 0;
    }
    private void release(Entry<V> entry) {
        gpu.release(entry.gpuBlock, releaseEvents.apply(entry.value)); entry.gpuBlock = null;
        cpu.release(entry.cpuBlock, releaseEvents.apply(entry.value)); entry.cpuBlock = null;
    }
    private void removeLocked(String key, boolean eviction) {
        var entry = entries.remove(key);
        if (entry == null) return;
        uncharge(key, entry);
        if (eviction) evictions++;
        try { copier.accept(Copy.DETACH, entry.value); }
        catch (RuntimeException error) {
            if (entry.readers > 0) { entry.retired = true; return; }
        }
        release(entry);
    }
    void remove(String key) {
        transfer.lock();
        try { synchronized (this) { removeLocked(key, false); } }
        finally { transfer.unlock(); }
    }
    private synchronized void releaseReader(Entry<V> entry) {
        if (--entry.readers == 0 && entry.retired) release(entry);
    }
    synchronized Map<String, Long> stats() {
        var result = new LinkedHashMap<String, Long>();
        result.put("gpu_embedding_bytes", gpuBytes); result.put("embedding_bytes", cpuBytes);
        result.put("cache_evictions", evictions); result.put("cache_promotions", promotions);
        result.put("cache_demotions", demotions); result.put("cache_transfer_errors", transferErrors);
        for (var device : Device.values()) {
            var stats = pool(device).stats(); String prefix = device == Device.GPU ? "gpu_pool_" : "cpu_pool_";
            result.put(prefix + "capacity_bytes", stats.capacity()); result.put(prefix + "used_bytes", stats.used());
            result.put(prefix + "pending_bytes", stats.pending()); result.put(prefix + "free_bytes", stats.free());
            result.put(prefix + "largest_free_block_bytes", stats.largest());
        }
        return result;
    }
    @Override public void close() {
        transfer.lock();
        try { synchronized (this) { closed = true; for (String key : List.copyOf(entries.keySet())) removeLocked(key, false); } }
        finally { transfer.unlock(); }
    }
}

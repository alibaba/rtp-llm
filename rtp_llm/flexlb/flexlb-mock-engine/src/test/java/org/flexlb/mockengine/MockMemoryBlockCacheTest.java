package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicLong;
import org.flexlb.engine.grpc.EngineRpcService;
import static org.junit.jupiter.api.Assertions.*;

class MockMemoryBlockCacheTest {
    @TempDir Path dir;

    @Test void lruTouchPreservesCreationAgeAndMissStopsPrefix() {
        var time = new AtomicLong();
        var ages = new ArrayList<Double>();
        var cache = new MockMemoryBlockCache(2, ages::add, time::get);
        cache.write(List.of(1L, 2L));
        time.set(10_000_000);
        assertEquals(1, cache.match(List.of(1L, 9L, 2L), 0));
        cache.write(List.of(3L)); // key 2 is now oldest
        assertEquals(List.of(1L, 3L), cache.keys());
        time.set(20_000_000);
        cache.write(List.of(1L)); // touches without restarting age
        cache.write(List.of(4L));
        time.set(30_000_000);
        cache.write(List.of(5L));
        assertEquals(List.of(10.0, 10.0, 30.0), ages);
        assertEquals(3, cache.evictions());
    }

    private MockPerformanceModel model(String config) throws Exception {
        Path p = dir.resolve("perf.json"), master = dir.resolve("master.json");
        Files.writeString(p, "{\"block_size\":64,\"prefill\":{\"memory_cache\":" + config + "}}");
        MockMasterConfig.writeWithPrefillExpression(master, "1 + sum(computeTokens)");
        return MockPerformanceModel.load(p.toString(), master.toString());
    }

    private Object field(Object object, String name) throws Exception {
        var f = object.getClass().getDeclaredField(name); f.setAccessible(true); return f.get(object);
    }

    @Test void hostHitReducesComputeButDoesNotExpandGpuAdmissionAndIsAdvertised() throws Exception {
        var model = model("{\"enabled\":true,\"capacity_blocks\":10}");
        try (var cluster = MockEngineTestCluster.create(model, 61000, 1, 1)) {
            var p = cluster.prefill(0);
            var mem = (MockMemoryBlockCache) field(p, "memoryCache");
            var gpu = (MockLruBlockCache) field(p, "cache");
            mem.write(List.of(1L, 2L));
            var shape = new MockPerformanceModel.RequestShape(EngineRpcService.GenerateInputPB.getDefaultInstance(),
                    192, 1, List.of(1L, 2L, 3L), 0, 0, false);
            var hot = p.matchPrefillMemory(shape);
            assertEquals(128, hot.hitTokens());
            assertEquals(2, hot.memoryHitBlocks());
            assertEquals(65, model.prefillMs(List.of(hot)));
            assertEquals(0, gpu.lruKeyBlocks());
            var acquired = p.getClass().getDeclaredMethod("acquireBlockLease", long.class, MockPerformanceModel.RequestShape.class);
            acquired.setAccessible(true);
            var busy = gpu.acquire(gpu.totalBlocks() - gpu.reserveBlocks(), List.of());
            assertNotNull(busy);
            assertNull(acquired.invoke(p, 7L, hot), "host hits cannot admit into a full GPU");
            gpu.release(busy);
            assertNotNull(acquired.invoke(p, 7L, hot));
            assertEquals(3, gpu.heldBlocks(), "host hits still allocate all three GPU blocks");
            var result = new java.util.concurrent.atomic.AtomicReference<EngineRpcService.CacheStatusPB>();
            p.getCacheStatus(EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true).build(),
                    new io.grpc.stub.StreamObserver<EngineRpcService.CacheStatusPB>() {
                        public void onNext(EngineRpcService.CacheStatusPB x) { result.set(x); }
                        public void onError(Throwable t) { fail(t); }
                        public void onCompleted() { }
                    });
            assertEquals(java.util.Set.of(1L, 2L), result.get().getCacheKeysMap().keySet());
            assertEquals(p.getTotalKvTokens(), result.get().getTotalKvCache());
            assertNull(field(cluster.decode(0), "memoryCache"));
        }
    }

    @Test void completedPrefillPopulatesHostAndReusesAfterGpuEviction() throws Exception {
        var model = model("{\"enabled\":true,\"capacity_blocks\":10}");
        try (var cluster = MockEngineTestCluster.create(model, 61020, 1, 0)) {
            var p = cluster.prefill(0);
            for (int round = 0; round < 2; round++) {
                var request = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(100 + round)
                        .addAllTokenIds(java.util.Collections.nCopies(192, 123))
                        .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(1)
                                .setUniqueKey("{\"input_len\":192,\"output_len\":1,\"block_cache_keys\":[1,2,3]}"))
                        .build();
                var done = new java.util.concurrent.CountDownLatch(1);
                var error = new java.util.concurrent.atomic.AtomicReference<Throwable>();
                p.generateStreamCall(request, new io.grpc.stub.StreamObserver<EngineRpcService.GenerateOutputsPB>() {
                    public void onNext(EngineRpcService.GenerateOutputsPB x) {
                        if (x.hasErrorInfo()) error.set(new AssertionError(x.getErrorInfo().toString()));
                    }
                    public void onError(Throwable t) { error.set(t); done.countDown(); }
                    public void onCompleted() { done.countDown(); }
                });
                assertTrue(done.await(3, java.util.concurrent.TimeUnit.SECONDS));
                assertNull(error.get());
                var mem = (MockMemoryBlockCache) field(p, "memoryCache");
                assertEquals(3, mem.size());
                if (round == 0) ((MockLruBlockCache) field(p, "cache")).clear();
            }
            assertEquals(3, p.whaleMetrics().get("mock_memory_cache_read_blocks_total").longValue());
        }
    }

    @Test void optInAndConfigurationValidation() throws Exception {
        assertEquals(0, model("{}").memoryCacheBlocks);
        var copy = model("{\"enabled\":true,\"capacity_blocks\":10,\"read_ms_per_block\":0.2}").forEngine();
        assertEquals(10, copy.memoryCacheBlocks);
        assertEquals(.2, copy.memoryReadMsPerBlock);
        assertThrows(IllegalStateException.class, () -> model("{\"enabled\":true,\"capacity_blocks\":0}"));
        assertThrows(IllegalStateException.class, () -> model("{\"enabled\":true,\"capacity_blocks\":2,\"read_ms_per_block\":-1}"));
    }
    @Test void pinnedReadsSurvivePressureAndReleaseExactlyOnce() {
        var c = new MockMemoryBlockCache(2, ignored -> {});
        c.write(List.of(1L, 2L));
        var a = c.pinRead(List.of(1L, 2L), 0);
        var b = c.pinRead(List.of(1L), 0);
        assertEquals(0, c.availableBlocks());
        assertNull(c.beginWrite(List.of(3L)));
        a.close(); a.close();
        assertEquals(1, c.availableBlocks());
        var write = c.beginWrite(List.of(3L));
        assertNotNull(write);
        assertEquals(List.of(1L), c.keys(), "pinned key survives eviction");
        assertEquals(1, c.pendingBlocks());
        assertEquals(0, c.peekMatch(List.of(3L), 0), "uncommitted data is invisible");
        assertTrue(write.commit());
        b.close();
        assertEquals(2, c.availableBlocks());
    }

    @Test void duplicateWriteDoesNotTouchLruAndAbortOrClearCannotResurrectKeys() {
        var c = new MockMemoryBlockCache(2, ignored -> {});
        c.write(List.of(1L, 2L));
        c.beginWrite(List.of(1L)).commit();
        var w = c.beginWrite(List.of(3L));
        assertEquals(List.of(2L), c.keys(), "duplicate writes must not refresh key 1");
        w.close(); w.close(); assertFalse(w.commit());
        assertEquals(0, c.pendingBlocks());
        var old = c.beginWrite(List.of(4L));
        c.clear();
        c.beginWrite(List.of(4L)).commit();
        assertFalse(old.commit(), "a pre-clear callback cannot publish into a new incarnation");
        assertEquals(List.of(4L), c.keys());
    }

    @Test void birthTimeStartsAtCommitAndCopyConfigIsPreserved() throws Exception {
        var t = new AtomicLong(); var ages = new ArrayList<Double>();
        var c = new MockMemoryBlockCache(1, ages::add, t::get);
        var w = c.beginWrite(List.of(1L));
        t.set(20_000_000); w.commit();
        t.set(25_000_000); c.beginWrite(List.of(2L)).commit();
        assertEquals(List.of(5.0), ages);
        var m = model("{\"enabled\":true,\"capacity_blocks\":10,\"copy_lifecycle\":true,\"write_ms_per_block\":2}").forEngine();
        assertTrue(m.memoryCopyLifecycle); assertEquals(2.0, m.memoryWriteMsPerBlock);
        assertThrows(IllegalStateException.class, () -> model("{\"enabled\":true,\"capacity_blocks\":2,\"write_ms_per_block\":-1}"));
        assertThrows(IllegalStateException.class, () -> model("{\"enabled\":true,\"capacity_blocks\":2,\"copy_lifecycle\":1}"));
    }

    @Test void asyncWriteRetainsGpuUntilCommitAndCancellationReleasesBothPools() throws Exception {
        var m = model("{\"enabled\":true,\"capacity_blocks\":10,\"copy_lifecycle\":true,\"write_ms_per_block\":200}");
        try (var cluster = MockEngineTestCluster.create(m, 61040, 1, 0)) {
            var p = cluster.prefill(0);
            var mem = (MockMemoryBlockCache) field(p, "memoryCache");
            var gpu = (MockLruBlockCache) field(p, "cache");
            for (int round = 0; round < 2; round++) {
                long id = 800 + round;
                String keys = round == 0 ? "[1,2,3]" : "[4,5,6]";
                var req = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id)
                        .addAllTokenIds(java.util.Collections.nCopies(192, 123))
                        .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(1)
                            .setUniqueKey("{\"input_len\":192,\"output_len\":1,\"block_cache_keys\":" + keys + "}"))
                        .build();
                var done = new java.util.concurrent.CountDownLatch(1);
                var error = new java.util.concurrent.atomic.AtomicReference<Throwable>();
                p.generateStreamCall(req, new io.grpc.stub.StreamObserver<EngineRpcService.GenerateOutputsPB>() {
                    public void onNext(EngineRpcService.GenerateOutputsPB value) {
                        if (value.hasErrorInfo()) error.set(new AssertionError(value.getErrorInfo()));
                    }
                    public void onError(Throwable t) { error.set(t); done.countDown(); }
                    public void onCompleted() { done.countDown(); }
                });
                assertTrue(done.await(3, java.util.concurrent.TimeUnit.SECONDS)); assertNull(error.get());
                assertEquals(3, mem.pendingBlocks());
                assertTrue(gpu.referencedKeyBlocks() >= 3, "GPU copy source remains pinned after P terminal");
                if (round == 1) p.cancel(id);
                long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(2);
                while (mem.pendingBlocks() != 0 && System.nanoTime() < deadline) Thread.sleep(5);
                assertEquals(0, mem.pendingBlocks());
                // Copy completion releases the GPU reference just after publishing host keys.
                while (gpu.referencedKeyBlocks() != 0 && System.nanoTime() < deadline) Thread.sleep(5);
                assertEquals(0, gpu.referencedKeyBlocks());
            }
            assertEquals(java.util.Set.of(1L, 2L, 3L), new java.util.HashSet<>(mem.keys()));
        }
    }

}

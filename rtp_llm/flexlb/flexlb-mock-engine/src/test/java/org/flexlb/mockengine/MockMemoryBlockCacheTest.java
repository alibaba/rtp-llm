package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
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
        cache.match(List.of(1L), 0); // a read touches without restarting age
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
        assertThrows(IllegalStateException.class, () -> model("{\"enabled\":true,\"capacity_blocks\":0}"));
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
        var c = new MockMemoryBlockCache(2, false, ignored -> {}, System::nanoTime);
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
        assertTrue(m.memoryCopyLifecycle);
        assertThrows(IllegalStateException.class, () -> model("{\"enabled\":true,\"capacity_blocks\":2,\"copy_lifecycle\":1}"));
    }

    @Test void copyCommitsWithoutLegacyDelayAndGpuEvictionPreservesMemory() throws Exception {
        var m = model("{\"enabled\":true,\"capacity_blocks\":10,\"copy_lifecycle\":true,\"write_ms_per_block\":60000,\"read_ms_per_block\":60000}");
        try (var cluster = MockEngineTestCluster.create(m, 61040, 1, 0)) {
            var p = cluster.prefill(0);
            var mem = (MockMemoryBlockCache) field(p, "memoryCache");
            var gpu = (MockLruBlockCache) field(p, "cache");
            for (int round = 0; round < 2; round++) {
                long id = 800 + round;
                String keys = "[1,2,3]";
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
                assertEquals(0, mem.pendingBlocks(), "copy commits immediately despite legacy delay config");
                assertEquals(0, mem.pinnedBlocks());
                assertEquals(3, mem.size());
                long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(2);
                while (gpu.referencedKeyBlocks() != 0 && System.nanoTime() < deadline) Thread.sleep(5);
                assertEquals(0, gpu.referencedKeyBlocks());
                if (round == 0) gpu.clear(); // next request must read from Memory

            }
            assertEquals(java.util.Set.of(1L, 2L, 3L), new java.util.HashSet<>(mem.keys()));
            assertEquals(3, p.whaleMetrics().get("mock_memory_cache_read_blocks_total").longValue());
        }
    }

    @Test void holeCopiesExistingSuffixAndCommitTouchesWithoutReplacingPinnedEntry() {
        var time = new AtomicLong();
        var ages = new ArrayList<Double>();
        var c = new MockMemoryBlockCache(6, ages::add, time::get);
        c.write(List.of(1L, 3L, 4L));
        var read = c.pinRead(List.of(3L), 0);
        var copy = c.beginWrite(List.of(1L, 2L, 3L));
        assertEquals(List.of(2L, 3L), copy.keys(), "only the continuous prefix skips copying");
        assertEquals(2, c.pendingBlocks(), "existing suffix still needs a temporary backing");
        time.set(10_000_000);
        copy.commit();
        assertEquals(List.of(1L, 4L, 3L, 2L), c.keys(),
                "duplicate commit must not refresh the existing suffix entry");
        assertEquals(1, c.pinnedBlocks(), "duplicate commit must preserve existing readers");
        assertEquals(0, c.pendingBlocks());
        read.close();
        assertEquals(0, c.pinnedBlocks());
        time.set(20_000_000);
        c.write(List.of(5L, 6L, 7L, 8L, 9L, 10L));
        assertEquals(List.of(20.0, 20.0, 10.0, 20.0), ages,
                "suffix refresh preserves the original birth time of key 3");
    }

    @Test void failedWriteKeepsPriorEvictionsButReleasesItsReservations() {
        var c = new MockMemoryBlockCache(3, ignored -> {});
        c.write(List.of(1L, 2L, 3L));
        var read = c.pinRead(List.of(1L), 0);
        assertNull(c.beginWrite(List.of(4L, 5L, 6L)));
        assertEquals(List.of(1L), c.keys());
        assertEquals(2, c.evictions(), "failed write does not restore already evicted keys");
        assertEquals(0, c.pendingBlocks());
        assertEquals(2, c.availableBlocks());
        assertEquals(1, c.writeRejected());
        read.close();
        c.write(List.of(4L, 5L, 6L));
        assertEquals(List.of(4L, 5L, 6L), c.keys());
    }

    @Test void prefixTreeEvictsOnlyLeavesAndPromotesTheirParents() {
        var c = new MockMemoryBlockCache(4, ignored -> {});
        c.write(List.of(1L, 2L, 3L));
        c.write(List.of(1L, 4L));

        c.write(List.of(5L));
        assertEquals(Set.of(1L, 2L, 4L, 5L), Set.copyOf(c.keys()));
        assertFalse(c.keys().contains(3L), "oldest leaf is evicted first");
        assertTrue(c.keys().contains(1L), "an internal prefix cannot be selected as a victim");

        c.write(List.of(6L));
        assertFalse(c.keys().contains(2L), "parent becomes an eligible leaf after its child is removed");
        assertTrue(c.keys().contains(1L), "branch point remains structurally protected");
    }

    @Test void residentAndPinnedLeavesAreNotEvictable() {
        var c = new MockMemoryBlockCache(2, ignored -> {});
        c.writeResident(List.of(1L));
        c.write(List.of(2L));
        var read = c.pinRead(List.of(2L), 0);
        assertNull(c.beginWrite(List.of(3L)));
        assertEquals(Set.of(1L, 2L), Set.copyOf(c.keys()));
        read.close();
        c.write(List.of(3L));
        assertEquals(Set.of(1L, 3L), Set.copyOf(c.keys()));
    }

    @Test void successfulReadConsumesHostEntriesButCancelledReadKeepsThem() {
        var c = new MockMemoryBlockCache(4, ignored -> {});
        c.write(List.of(1L, 2L, 3L));
        var successful = c.pinRead(List.of(1L, 2L), 0);
        assertEquals(2, successful.consume());
        assertEquals(List.of(3L), c.keys());
        assertEquals(0, c.pinnedBlocks());

        var cancelled = c.pinRead(List.of(3L), 0);
        cancelled.close();
        assertEquals(List.of(3L), c.keys(), "failed/cancelled H2D only releases in-flight protection");
    }

    @Test void memoryPrefixTreeDefaultsOnAndCanBeDisabled() throws Exception {
        assertTrue(model("{\"enabled\":true,\"capacity_blocks\":10}").memoryPrefixTree);
        assertFalse(model("{\"enabled\":true,\"capacity_blocks\":10,\"enable_prefix_tree\":false}")
                .memoryPrefixTree);
        assertThrows(IllegalStateException.class,
                () -> model("{\"enabled\":true,\"capacity_blocks\":10,\"enable_prefix_tree\":1}"));
    }

    @Test void concurrentCopiesOwnSeparateBackingsAndCommitCannotResetReaders() {
        var c = new MockMemoryBlockCache(3, ignored -> {});
        var first = c.beginWrite(List.of(1L));
        var second = c.beginWrite(List.of(1L));
        assertEquals(2, c.pendingBlocks());
        first.commit();
        var read = c.pinRead(List.of(1L), 0);
        assertFalse(second.commit(), "duplicate backing is discarded");
        assertEquals(1, c.size());
        assertEquals(1, c.pinnedBlocks());
        assertEquals(0, c.pendingBlocks());
        read.close();
        assertEquals(3, c.availableBlocks());
    }

    @Test void nativePrefillCompletionHonorsDeviceAndMemoryWriteFlags() throws Exception {
        var m = model("{\"enabled\":true,\"capacity_blocks\":10,\"copy_lifecycle\":true}");
        m.nativeTokenCacheKeys = true;
        try (var cluster = MockEngineTestCluster.create(m, 61060, 1, 0)) {
            var p = cluster.prefill(0);
            var gpu = (MockLruBlockCache) field(p, "cache");
            var mem = (MockMemoryBlockCache) field(p, "memoryCache");
            for (int mode = 0; mode < 4; mode++) {
                gpu.clear(); mem.clear();
                var cfg = EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(1);
                // Descriptors preserve compatibility with the pinned master's proto jar.
                for (String name : List.of("reuse_cache", "enable_device_cache", "enable_memory_cache")) {
                    boolean enabled = name.equals("reuse_cache") ? mode != 0
                            : name.equals("enable_device_cache") ? mode == 1 || mode == 3 : mode >= 2;
                    cfg.setField(cfg.getDescriptorForType().findFieldByName(name), enabled);
                }
                var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(900 + mode)
                        .addAllTokenIds(java.util.Collections.nCopies(128, 19 + mode)).setGenerateConfig(cfg).build();
                var done = new java.util.concurrent.CountDownLatch(1);
                var error = new java.util.concurrent.atomic.AtomicReference<Throwable>();
                p.generateStreamCall(input, new io.grpc.stub.StreamObserver<EngineRpcService.GenerateOutputsPB>() {
                    public void onNext(EngineRpcService.GenerateOutputsPB v) {
                        if (v.hasErrorInfo()) error.set(new AssertionError(v.getErrorInfo()));
                    }
                    public void onError(Throwable t) { error.set(t); done.countDown(); }
                    public void onCompleted() { done.countDown(); }
                });
                assertTrue(done.await(3, java.util.concurrent.TimeUnit.SECONDS));
                assertNull(error.get());
                long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(2);
                while (gpu.availableBlocks() != gpu.totalBlocks() && System.nanoTime() < deadline) Thread.sleep(5);
                assertEquals(gpu.totalBlocks(), gpu.availableBlocks());
                assertEquals(mode == 1 || mode == 3 ? 2 : 0, gpu.lruKeyBlocks());
                assertEquals(mode >= 2 ? 2 : 0, mem.size());
                assertEquals(0, mem.pendingBlocks());
            }
        }
    }

    @Test void cancelledPrefillCannotPublishCacheFromItsLateForwardCallback() throws Exception {
        var m = model("{\"enabled\":true,\"capacity_blocks\":10,\"copy_lifecycle\":true}");
        m.setOverrideFixedPrefillMs(100.0);
        try (var cluster = MockEngineTestCluster.create(m, 61080, 1, 0)) {
            var p = cluster.prefill(0);
            var gpu = (MockLruBlockCache) field(p, "cache");
            var mem = (MockMemoryBlockCache) field(p, "memoryCache");
            var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(950)
                    .addAllTokenIds(java.util.Collections.nCopies(128, 17))
                    .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(1)
                            .setUniqueKey("{\"input_len\":128,\"output_len\":1,\"block_cache_keys\":[91,92]}"))
                    .build();
            p.generateStreamCall(input, new io.grpc.stub.StreamObserver<EngineRpcService.GenerateOutputsPB>() {
                public void onNext(EngineRpcService.GenerateOutputsPB v) { }
                public void onError(Throwable t) { }
                public void onCompleted() { }
            });
            p.cancel(950);
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(3);
            var active = (java.util.concurrent.atomic.AtomicInteger) field(p, "activePrefillBatches");
            while (active.get() != 0 && System.nanoTime() < deadline) Thread.sleep(5);
            assertEquals(0, active.get(), "the late forward callback must have run");
            assertEquals(0, gpu.lruKeyBlocks());
            assertEquals(gpu.totalBlocks(), gpu.availableBlocks());
            assertEquals(0, mem.size());
            assertEquals(0, mem.pendingBlocks());
        }
    }

}

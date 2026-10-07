package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class WhaleDecodeCachePolicyTest {
    @TempDir Path dir;

    @Test void gpuEvictionModeIsRoleSpecificAndCopiedToEngines() throws Exception {
        Path performance = dir.resolve("flat.json");
        Path master = dir.resolve("master.json");
        Files.writeString(master, "{}");
        Files.writeString(performance,
                "{\"block_size\":64,\"prefill\":{\"formula\":\"1\",\"enable_gpu_prefix_tree\":false},\"decode\":{}}");
        var model = MockPerformanceModel.load(performance.toString(), master.toString());
        assertFalse(model.forEngine().prefillGpuPrefixTree);
        assertTrue(model.forEngine().decodeGpuPrefixTree);
        try (var cluster = MockEngineTestCluster.create(model, 62000, 1, 1)) {
            assertFalse(cache(cluster.prefills().get(0)).prefixTreeEnabled());
            assertTrue(cache(cluster.decodes().get(0)).prefixTreeEnabled());
            assertEquals(false, cluster.prefills().get(0).getSnapshot().get("gpu_prefix_tree_enabled"));
        }
        Files.writeString(performance,
                "{\"prefill\":{\"formula\":\"1\",\"enable_gpu_prefix_tree\":\"false\"},\"decode\":{}}");
        assertThrows(IllegalStateException.class,
                () -> MockPerformanceModel.load(performance.toString(), master.toString()));
    }

    private MockPerformanceModel model(String decode) throws Exception {
        Path path = dir.resolve("performance.json");
        Files.writeString(path, "{\"block_size\":64,\"prefill\":{\"formula\":\"1\"},\"decode\":" + decode + "}");
        Path master = dir.resolve("master.json");
        Files.writeString(master, "{}");
        return MockPerformanceModel.load(path.toString(), master.toString());
    }

    private MockLruBlockCache cache(JavaMockEngineCluster.FastRpcService engine) throws Exception {
        var field = engine.getClass().getDeclaredField("cache");
        field.setAccessible(true);
        return (MockLruBlockCache) field.get(engine);
    }

    @Test void disabledReuseReleasesCompletionAndIgnoresExistingKeys() throws Exception {
        var model = model("{\"reuse_cache\":false,\"reserve_block_ratio\":8}");
        try (var cluster = MockEngineTestCluster.create(model, 62000, 1, 1)) {
            var d = cluster.decodes().get(0);
            var cache = cache(d);
            // Even a pre-existing key must not reduce demand when reuse is disabled.
            var seed = cache.acquire(1, List.of(7L));
            cache.admit(seed, List.of(7L));
            var shape = new MockPerformanceModel.RequestShape(
                    EngineRpcService.GenerateInputPB.getDefaultInstance(), 64, 1, List.of(7L), 0, 0, false);
            assertTrue(d.reserveDecodeLease(1L, shape));
            assertEquals(1, cache.heldBlocks());
            assertEquals(0, cache.referencedKeyBlocks());
            var finish = d.getClass().getDeclaredMethod("admitBlockLease", long.class, MockPerformanceModel.RequestShape.class);
            finish.setAccessible(true);
            assertEquals(false, finish.invoke(d, 1L, shape));
            assertEquals(0, cache.heldBlocks());
            assertEquals(cache.totalBlocks(), cache.availableBlocks());
            assertEquals(java.util.Set.of(7L), cache.snapshotKeys());
            // A fresh prefix is not inserted on successful completion either.
            var fresh = new MockPerformanceModel.RequestShape(shape.input(), 64, 1, List.of(8L), 0, 0, false);
            assertTrue(d.reserveDecodeLease(2L, fresh));
            finish.invoke(d, 2L, fresh);
            assertEquals(java.util.Set.of(7L), cache.snapshotKeys());
            assertEquals((int) (cache.totalBlocks() * 8L / 100), cache.reserveBlocks());
            assertEquals((int) Math.ceil(cache(cluster.prefills().get(0)).totalBlocks() * .05),
                    cache(cluster.prefills().get(0)).reserveBlocks());
        }
    }

    @Test void disabledReuseStillAccountsGeneratedKvWhileDecodeIsRunning() throws Exception {
        var model = model("{\"reuse_cache\":false,\"reserve_block_ratio\":0}");
        try (var cluster = MockEngineTestCluster.create(model, 62000, 1, 1)) {
            var d = cluster.decodes().get(0);
            var cache = cache(d);
            var shape = new MockPerformanceModel.RequestShape(
                    EngineRpcService.GenerateInputPB.getDefaultInstance(),
                    64, 129, List.of(), 0, 0, false);
            assertTrue(d.reserveDecodeLease(1L, shape));
            assertEquals(1, cache.heldBlocks());

            // Decode owns the generated KV while the stream is alive even when
            // completed Decode prefixes are not retained for later reuse.
            var grow = d.getClass().getDeclaredMethod("growDecodeLeaseLocked", long.class, int.class);
            grow.setAccessible(true);
            assertNull(grow.invoke(d, 1L, 193));
            assertEquals(4, cache.heldBlocks());

            var finish = d.getClass().getDeclaredMethod(
                    "admitBlockLease", long.class, MockPerformanceModel.RequestShape.class);
            finish.setAccessible(true);
            assertEquals(false, finish.invoke(d, 1L, shape));
            assertEquals(0, cache.heldBlocks());
            assertTrue(cache.snapshotKeys().isEmpty());
        }
    }

    @Test void explicitWatermarkUsesProductionFloorAndBoundary() {
        var cache = new MockLruBlockCache(26, .08, true);
        assertEquals(2, cache.reserveBlocks());
        assertNull(cache.acquire(25, List.of()));
        assertNotNull(cache.acquire(24, List.of()));
        assertEquals(2, cache.availableBlocks());
        assertEquals(2, new MockLruBlockCache(26, .05).reserveBlocks());
    }

    @Test void absentPolicyPreservesLegacyAndBadConfigurationFails() throws Exception {
        var model = model("{}").forEngine();
        assertTrue(model.decodeReuseCache);
        assertNull(model.decodeReserveBlockRatio);
        assertThrows(IllegalStateException.class, () -> model("{\"reuse_cache\":\"false\"}"));
        assertThrows(IllegalStateException.class, () -> model("{\"reserve_block_ratio\":-1}"));
        assertThrows(IllegalStateException.class, () -> model("{\"reserve_block_ratio\":80}"));
    }
}

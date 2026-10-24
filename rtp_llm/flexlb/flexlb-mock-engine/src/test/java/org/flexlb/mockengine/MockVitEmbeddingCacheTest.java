package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mockengine.MockVitEmbeddingCache.Device.*;

class MockVitEmbeddingCacheTest {
    private static List<MockVitEmbeddingCache.Tensor> layout(long gpu, long cpu) {
        return List.of(new MockVitEmbeddingCache.Tensor(new Object(), gpu, 4, GPU),
                new MockVitEmbeddingCache.Tensor(new Object(), cpu, 4, CPU));
    }
    private static void put(MockVitEmbeddingCache<String> cache, String key, long gpu, long cpu) {
        cache.put(key, key, layout(gpu, cpu), key + "-generation");
    }
    @Test void tensorIdentityAlignmentAndCpuMetadataDetermineTierBudgets() {
        var object = new Object();
        assertEquals(new MockVitEmbeddingCache.Bytes(16, 0), MockVitEmbeddingCache.bytes(List.of(
                new MockVitEmbeddingCache.Tensor(object, 16, 4, GPU),
                new MockVitEmbeddingCache.Tensor(object, 16, 4, GPU)), false));
        assertEquals(new MockVitEmbeddingCache.Bytes(0, 12), MockVitEmbeddingCache.bytes(List.of(
                new MockVitEmbeddingCache.Tensor(new Object(), 1, 1, CPU),
                new MockVitEmbeddingCache.Tensor(new Object(), 8, 4, CPU)), false));
        try (var cache = new MockVitEmbeddingCache<String>(64, 8)) {
            put(cache, "a", 16, 8); put(cache, "b", 16, 8);
            assertFalse(cache.contains("a")); assertTrue(cache.contains("b"));
            assertEquals(16, cache.stats().get("gpu_embedding_bytes"));
            assertEquals(8, cache.stats().get("embedding_bytes"));
        }
    }
    @Test void oversizedInsertionBypassesWithoutFlushingUnrelatedResults() {
        try (var cache = new MockVitEmbeddingCache<String>(16, 32)) {
            put(cache, "small", 8, 0); put(cache, "huge", 64, 0);
            assertTrue(cache.contains("small")); assertFalse(cache.contains("huge"));
            assertEquals(8, cache.stats().get("gpu_embedding_bytes"));
            put(cache, "cpu", 24, 0);
            assertEquals("cpu", cache.tier("cpu")); assertEquals(24, cache.stats().get("embedding_bytes"));
        }
    }
    @Test void fragmentedPoolEvictsMoreLruThanLogicalByteBudgetRequires() {
        try (var cache = new MockVitEmbeddingCache<String>(0, 32)) {
            for (String key : List.of("a", "b", "c", "d")) put(cache, key, 0, 8);
            cache.remove("b"); cache.remove("d");
            assertEquals(16, cache.stats().get("cpu_pool_free_bytes"));
            assertEquals(8, cache.stats().get("cpu_pool_largest_free_block_bytes"));
            put(cache, "large", 0, 16);
            assertFalse(cache.contains("a")); assertTrue(cache.contains("c")); assertTrue(cache.contains("large"));
            assertEquals(24, cache.stats().get("cpu_pool_used_bytes"));
        }
    }
    @Test void promotionDropsOldCpuReservationBeforeDemotingGpuVictim() {
        try (var cache = new MockVitEmbeddingCache<String>(16, 16)) {
            put(cache, "a", 16, 0); put(cache, "b", 16, 0);
            assertEquals("cpu", cache.tier("a")); assertEquals("gpu", cache.tier("b"));
            assertEquals("a-generation", cache.get("a").generation());
            assertEquals("gpu", cache.tier("a")); assertEquals("cpu", cache.tier("b"));
            assertEquals(16, cache.stats().get("gpu_pool_used_bytes"));
            assertEquals(16, cache.stats().get("cpu_pool_used_bytes"));
        }
    }
    @Test void optionalPoolCopyFailurePreservesValueButReadFailureIsAnError() {
        try (var cache = new MockVitEmbeddingCache<String>(16, 32)) {
            cache.copier = (operation, value) -> { if (operation == MockVitEmbeddingCache.Copy.STORE) throw new IllegalStateException("copy failed"); };
            assertDoesNotThrow(() -> put(cache, "a", 16, 0));
            assertFalse(cache.contains("a")); assertEquals(0, cache.stats().get("gpu_pool_used_bytes"));
            cache.copier = (operation, value) -> {}; put(cache, "a", 16, 0); put(cache, "b", 16, 0);
            cache.copier = (operation, value) -> { if (operation == MockVitEmbeddingCache.Copy.READ) throw new IllegalStateException("restore failed"); };
            assertThrows(io.grpc.StatusRuntimeException.class, () -> cache.get("a"));
            assertTrue(cache.contains("a")); assertEquals("cpu", cache.tier("a"));
            cache.copier = (operation, value) -> { if (operation == MockVitEmbeddingCache.Copy.STORE) throw new IllegalStateException("promotion failed"); };
            assertEquals("a", cache.get("a").value());
            assertFalse(cache.contains("a"));
        }
    }
    @Test void detachedValueSurvivesReplacementAndPendingStorageCannotBeReusedEarly() throws Exception {
        var cache = new MockVitEmbeddingCache<String>(0, 16);
        put(cache, "a", 0, 16); var returned = cache.get("a");
        var event = new CompletableFuture<Void>(); cache.releaseEvents = value -> List.of(event);
        cache.remove("a");
        assertEquals(0, cache.stats().get("embedding_bytes"));
        assertEquals(16, cache.stats().get("cpu_pool_pending_bytes"));
        var replacement = CompletableFuture.runAsync(() -> put(cache, "b", 0, 16));
        assertFalse(replacement.isDone()); event.complete(null); replacement.get(2, TimeUnit.SECONDS);
        assertEquals("a", returned.value()); assertEquals("b", cache.get("b").value());
        cache.close(); assertEquals(0, cache.stats().get("cpu_pool_used_bytes"));
    }
    @Test void failedDetachKeepsReservationUntilOutstandingWaiterReleasesIt() throws Exception {
        try (var cache = new MockVitEmbeddingCache<String>(0, 16)) {
            put(cache, "a", 0, 16); var waiter = cache.claim("a", 1000);
            cache.copier = (operation, value) -> { if (operation == MockVitEmbeddingCache.Copy.DETACH) throw new IllegalStateException("detach failed"); };
            cache.remove("a");
            assertFalse(cache.contains("a")); assertEquals(16, cache.stats().get("cpu_pool_used_bytes"));
            put(cache, "b", 0, 16);
            assertFalse(cache.contains("b"), "no reusable block: bypass cache without rejecting successful compute");
            assertEquals("a", waiter.get().value());
            assertEquals(0, cache.stats().get("cpu_pool_used_bytes"));
            put(cache, "b", 0, 16); assertTrue(cache.contains("b"));
        }
    }
    @Test void closedCacheCannotAdmitLateCompletion() {
        var cache = new MockVitEmbeddingCache<String>(16, 16);
        put(cache, "a", 16, 0); cache.close(); put(cache, "late", 16, 0);
        assertFalse(cache.contains("late")); assertEquals(0, cache.stats().get("gpu_pool_used_bytes"));
    }
    @Test void coldReadHonorsTransferTimeoutWhileHotReadBypassesTransfer() throws Exception {
        try (var cache = new MockVitEmbeddingCache<String>(32, 64)) {
            put(cache, "a", 16, 0); put(cache, "b", 16, 0); put(cache, "c", 16, 0);
            var copying = new CompletableFuture<Void>();
            var finish = new CompletableFuture<Void>();
            cache.copier = (operation, value) -> {
                if (operation == MockVitEmbeddingCache.Copy.STORE && value.equals("d")) {
                    copying.complete(null); finish.join();
                }
            };
            var transfer = CompletableFuture.runAsync(() -> put(cache, "d", 8, 0));
            try {
                copying.get(2, TimeUnit.SECONDS);
                var error = assertThrows(io.grpc.StatusRuntimeException.class, () -> cache.get("a", 20));
                assertEquals(io.grpc.Status.Code.DEADLINE_EXCEEDED, error.getStatus().getCode());
                assertEquals("c", cache.get("c", 20).value());
                assertEquals("cpu", cache.tier("a"));
            } finally { finish.complete(null); transfer.get(2, TimeUnit.SECONDS); }
            assertEquals("a", cache.get("a", 1000).value());
            assertEquals("gpu", cache.tier("a"));
        }
    }
}

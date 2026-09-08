package org.flexlb.constraint;

import org.flexlb.constraint.source.SidBucketClient;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class BucketSidReaderTest {
    static BucketSidReader.Settings settings(int buckets, int concurrency, int limit, int retries) {
        return new BucketSidReader.Settings("pool_", buckets, concurrency, limit,
                Duration.ofMillis(100), Duration.ofSeconds(10), retries);
    }

    static List<List<SidBucketClient.Row>> data(int buckets, int items) {
        return data(buckets, items, 100);
    }

    static List<List<SidBucketClient.Row>> data(int buckets, int items, int distinctSids) {
        var result = new ArrayList<List<SidBucketClient.Row>>();
        for (int i = 0; i < buckets; i++) { result.add(new ArrayList<>()); }
        for (int i = 0; i < items; i++) {
            String id = Integer.toString(i);
            int bucket = BucketSidReader.bucketForItem(id, buckets);
            int sid = i % distinctSids;
            result.get(bucket).add(new SidBucketClient.Row("pool_" + bucket, id,
                    "C" + (sid / 2048) + "C" + (sid % 2048)));
        }
        return result;
    }

    static int index(String key) { return Integer.parseInt(key.substring("pool_".length())); }

    @Test
    void readsAllBucketsWithBoundedAsyncConcurrencyAndDeduplicatesSid() throws Exception {
        var data = data(32, 500);
        var live = new AtomicInteger();
        var peak = new AtomicInteger();
        var calls = new AtomicInteger();
        var executor = Executors.newScheduledThreadPool(4);
        try {
            SidBucketClient client = (key, limit, timeout) -> {
                assertEquals(101, limit);
                calls.incrementAndGet();
                peak.accumulateAndGet(live.incrementAndGet(), Math::max);
                var future = new CompletableFuture<List<SidBucketClient.Row>>();
                executor.schedule(() -> {
                    live.decrementAndGet();
                    future.complete(data.get(index(key)));
                }, 5, TimeUnit.MILLISECONDS);
                return future;
            };
            var result = new BucketSidReader(client, settings(32, 4, 100, 0)).read(() -> true);
            assertEquals(500, result.itemCount());
            assertEquals(100, result.sids().size());
            assertEquals(32, calls.get());
            assertTrue(peak.get() > 1 && peak.get() <= 4);
        } finally { executor.shutdownNow(); }
    }

    @Test
    void retriesOnlyTheFailedBucket() throws Exception {
        var data = data(4, 10);
        var calls = new AtomicInteger();
        SidBucketClient client = (key, limit, timeout) -> {
            if (index(key) == 2 && calls.incrementAndGet() == 1) {
                return CompletableFuture.failedFuture(new IllegalStateException("temporary source error"));
            }
            return CompletableFuture.completedFuture(data.get(index(key)));
        };
        var result = new BucketSidReader(client, settings(4, 4, 100, 1)).read(() -> true);
        assertEquals(10, result.itemCount());
        assertEquals(2, calls.get());
    }

    @Test
    void failureCancelsOutstandingWindowAndDoesNotReturnPartialInput() {
        var outstanding = new CompletableFuture<List<SidBucketClient.Row>>();
        SidBucketClient client = (key, limit, timeout) -> index(key) == 0
                ? CompletableFuture.failedFuture(new IllegalStateException("failed")) : outstanding;
        assertThrows(IllegalStateException.class,
                () -> new BucketSidReader(client, settings(4, 2, 100, 0)).read(() -> true));
        assertTrue(outstanding.isCancelled());
    }

    @Test
    void timeoutCancelsFuture() {
        var outstanding = new CompletableFuture<List<SidBucketClient.Row>>();
        var config = new BucketSidReader.Settings("pool_", 1, 1, 10,
                Duration.ofMillis(5), Duration.ofSeconds(1), 0);
        assertThrows(IllegalStateException.class,
                () -> new BucketSidReader((k, l, t) -> outstanding, config).read(() -> true));
        assertTrue(outstanding.isCancelled());
    }

    @Test
    void rejectsOverflowEmptyInvalidAndDuplicateRows() {
        var config = settings(1, 1, 2, 0);
        var row = new SidBucketClient.Row("pool_0", "1", "C1C2");
        for (var rows : List.of(List.<SidBucketClient.Row>of(), List.of(row, row, row), List.of(row, row),
                List.of(new SidBucketClient.Row("pool_0", "1", "invalid")),
                List.of(new SidBucketClient.Row("wrong_key", "1", "C1C2")))) {
            assertThrows(IllegalStateException.class, () -> new BucketSidReader(
                    (k, l, t) -> CompletableFuture.completedFuture(rows), config).read(() -> true));
        }
    }

    @Test
    void deletingOneOfTwoItemsWithTheSameSidKeepsTheSid() throws Exception {
        var rows = new ArrayList<>(List.of(new SidBucketClient.Row("pool_0", "1", "C1C2"),
                new SidBucketClient.Row("pool_0", "2", "C1C2")));
        var reader = new BucketSidReader((k, l, t) -> CompletableFuture.completedFuture(List.copyOf(rows)),
                settings(1, 1, 10, 0));
        assertEquals(List.of("C1C2"), reader.read(() -> true).sids());
        rows.remove(0);
        assertEquals(List.of("C1C2"), reader.read(() -> true).sids());
    }

    @Test
    void leadershipLossAbortsBeforeIssuingQueries() {
        var calls = new AtomicInteger();
        var reader = new BucketSidReader((k, l, t) -> {
            calls.incrementAndGet();
            return CompletableFuture.completedFuture(List.of());
        }, settings(1, 1, 10, 0));
        assertThrows(IllegalStateException.class, () -> reader.read(() -> false));
        assertEquals(0, calls.get());
    }

    @Test
    void validatesConfigAndCrossLanguageHashVector() {
        assertEquals(0xcbf43926L % 4096, BucketSidReader.bucketForItem("123456789", 4096));
        assertThrows(IllegalArgumentException.class, () -> settings(1, 0, 100, 0));
        assertThrows(IllegalArgumentException.class, () -> settings(1, 1, 50_000, 0));
        assertThrows(IllegalArgumentException.class, () -> new BucketSidReader.Settings("pool:", 1, 1, 100,
                Duration.ofSeconds(1), Duration.ofSeconds(1), 0));
    }

    @Test
    void twoMillionSyntheticItemsAcross4096Buckets() throws Exception {
        assumeTrue("1".equals(System.getenv("CONSTRAINT_TREE_RUN_IGRAPH_SCALE_TEST")), "opt-in synthetic scale test");
        var data = data(4096, 2_000_000, 2_000_000);
        int max = data.stream().mapToInt(List::size).max().orElseThrow();
        var config = new BucketSidReader.Settings("pool_", 4096, 16, 2000,
                Duration.ofSeconds(5), Duration.ofMinutes(5), 0);
        var result = new BucketSidReader((k, l, t) -> CompletableFuture.completedFuture(data.get(index(k))), config)
                .read(() -> true);
        assertEquals(2_000_000, result.itemCount());
        assertEquals(2_000_000, result.sids().size());
        assertTrue(max < 2000);
        System.out.println("SYNTHETIC_IGRAPH_READER items=" + result.itemCount() + " buckets=" + result.bucketCount()
                + " maxBucket=" + max + " mergeMs=" + result.elapsedMillis()
                + " (in-memory source; NOT iGraph network RT or GPU/CSR benchmark)");
        var tokens = new HashMap<String, Integer>();
        for (int i = 0; i < 2048; i++) { tokens.put("C" + i, 170_000 + i); }
        var mapping = ConstraintTreeSidMappingTest.mapping(tokens).validated();
        var input = new ConstraintTreeModels.BuildRequest(1, "gul_item", null, null, null, null, result.sids());
        long started = System.nanoTime();
        var identity = ConstraintTreeRequestIdentity.fingerprint(input);
        var converted = mapping.convert(input);
        long convertedAt = System.nanoTime();
        try (var builder = new ConstraintTreeBuilder()) {
            var tree = builder.build(converted);
            long builtAt = System.nanoTime();
            byte[] payload = ConstraintTreeCsrCodec.encode(tree, mapping.fingerprint(), identity);
            var decoded = ConstraintTreeCsrCodec.decode(payload);
            assertEquals(2_000_000, decoded.sidCount());
            assertEquals(mapping.fingerprint(), decoded.mappingFingerprint());
            assertEquals(identity, decoded.contentSha256());
            System.out.println("SYNTHETIC_IGRAPH_CSR identityAndMappingMs="
                    + TimeUnit.NANOSECONDS.toMillis(convertedAt - started) + " buildMs="
                    + TimeUnit.NANOSECONDS.toMillis(builtAt - convertedAt) + " payloadBytes=" + payload.length
                    + " (synthetic C-token mapping; no iGraph server, Worker or GPU in this scale case)");
        }
    }
}

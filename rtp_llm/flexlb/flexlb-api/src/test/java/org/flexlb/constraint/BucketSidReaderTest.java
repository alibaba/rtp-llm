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
    @Test
    void failureSummaryIncludesRootCauseWithoutSdkRequestContext() {
        var reader = new BucketSidReader((k, l, t) -> CompletableFuture.failedFuture(
                new RuntimeException("wait response exception with requestContext [query=private]",
                        new java.io.IOException("Too many connections 4"))), numericSettings(4000, 2000));
        var error = assertThrows(IllegalStateException.class, () -> reader.read(() -> true));
        assertEquals("failed reading bucket 0: IOException: Too many connections 4", error.getMessage());
        assertNotNull(error.getCause());
        assertFalse(BucketSidReader.failureSummary(new RuntimeException("failed with requestContext [query=private]"))
                .contains("private"));
        assertEquals("TimeoutException", BucketSidReader.failureSummary(new java.util.concurrent.TimeoutException()));
        assertTrue(BucketSidReader.failureSummary(new java.io.IOException("x".repeat(1000))).length() < 300);
    }

    static BucketSidReader.Settings skipEmptySettings(int buckets, int sourceLimit) {
        return new BucketSidReader.Settings("", buckets, 4, 2000, Duration.ofSeconds(5),
                Duration.ofMinutes(5), 0, BucketSidReader.BucketAlgorithm.ITEM_ID_MOD, sourceLimit,
                BucketSidReader.EmptySidPolicy.SKIP);
    }

    @Test
    void skipsOnlyExplicitEmptyMappingsAndCountsBeforeSidDeduplication() throws Exception {
        var rows = List.of(new SidBucketClient.Row("0", "0", ""),
                new SidBucketClient.Row("0", "1", "C1C2"), new SidBucketClient.Row("0", "2", "C1C2"));
        var result = new BucketSidReader((k, l, t) -> CompletableFuture.completedFuture(rows),
                skipEmptySettings(1, 2000)).read(() -> true);
        assertEquals(3, result.itemCount());
        assertEquals(1, result.skippedEmptySids());
        assertEquals(2, result.eligibleItems());
        assertEquals(3, result.maxBucketRows());
        assertEquals(List.of("C1C2"), result.sids());
        assertThrows(IllegalStateException.class, () -> new BucketSidReader(
                (k, l, t) -> CompletableFuture.completedFuture(rows), numericSettings(1, 2000)).read(() -> true));
    }

    @Test
    void skipModeStillRejectsMalformedRowsAndDuplicateItems() {
        var valid = new SidBucketClient.Row("0", "0", "C1C2");
        for (var bad : List.of(new SidBucketClient.Row("0", "2", null),
                new SidBucketClient.Row("0", "2", " "), new SidBucketClient.Row("0", "2", "C1oops"),
                new SidBucketClient.Row("1", "2", ""), new SidBucketClient.Row("0", "1", ""),
                new SidBucketClient.Row("0", "", ""), new SidBucketClient.Row("0", "0", ""))) {
            assertThrows(IllegalStateException.class, () -> new BucketSidReader((k, l, t) ->
                    CompletableFuture.completedFuture(List.of(valid, bad)), skipEmptySettings(2, 2000)).read(() -> true));
        }
        var empty = new SidBucketClient.Row("0", "0", "");
        assertTrue(assertThrows(IllegalStateException.class, () -> new BucketSidReader((k, l, t) ->
                CompletableFuture.completedFuture(List.of(empty, valid)), skipEmptySettings(1, 2000))
                .read(() -> true)).getMessage().contains("duplicate item"));
    }

    @Test
    void sourceCapIsCheckedBeforeSkippingEmptyMappings() {
        var rows = new ArrayList<SidBucketClient.Row>();
        for (int i = 0; i < 2000; i++) { rows.add(new SidBucketClient.Row("0", "" + i, "")); }
        rows.set(0, new SidBucketClient.Row("0", "0", "C1C2"));
        assertTrue(assertThrows(IllegalStateException.class, () -> new BucketSidReader((k, l, t) ->
                CompletableFuture.completedFuture(rows), skipEmptySettings(1, 2000)).read(() -> true))
                .getMessage().contains("possible truncation"));
    }

    @Test
    void allEmptyMappingsRejectPublicationWithDiagnosticCounts() {
        var calls = new AtomicInteger();
        var error = assertThrows(IllegalStateException.class, () -> new BucketSidReader((k, l, t) -> {
            calls.incrementAndGet();
            return CompletableFuture.completedFuture(List.of(new SidBucketClient.Row(k, k, "")));
        }, skipEmptySettings(4000, 2000)).read(() -> true));
        assertEquals(4000, calls.get());
        assertTrue(error.getMessage().contains("skippedEmptySids=4000"));
        assertTrue(error.getMessage().contains("retaining existing tree"));
    }

    static BucketSidReader.Settings numericSettings(int buckets, int sourceLimit) {
        return new BucketSidReader.Settings("", buckets, 4, 2000, Duration.ofSeconds(5),
                Duration.ofMinutes(5), 0, BucketSidReader.BucketAlgorithm.ITEM_ID_MOD, sourceLimit);
    }

    @Test
    void numericModuloIsExactAndStrictAndLegacyCrcIsUnchanged() {
        for (String id : List.of("0", "3999", "4000", "000123", "9007199254740993",
                "9223372036854775808", "123456789012345678901234567890123456789")) {
            assertEquals(new java.math.BigInteger(id).mod(java.math.BigInteger.valueOf(4000)).intValue(),
                    BucketSidReader.bucketForItem(id, 4000, BucketSidReader.BucketAlgorithm.ITEM_ID_MOD));
        }
        for (String id : List.of("", "-1", "+1", " 1", "1 ", "1.0", "1e3", "１２３", "item1")) {
            assertThrows(IllegalArgumentException.class,
                    () -> BucketSidReader.bucketForItem(id, 4000, BucketSidReader.BucketAlgorithm.ITEM_ID_MOD));
        }
        assertEquals("0", numericSettings(4000, 2000).key(0));
        assertEquals("3999", numericSettings(4000, 2000).key(3999));
        assertEquals(0xcbf43926L % 4096, BucketSidReader.bucketForItem("123456789", 4096));
        assertThrows(IllegalArgumentException.class, () -> new BucketSidReader.Settings(null, 1, 1, 10,
                Duration.ofSeconds(1), Duration.ofSeconds(1), 0));
    }

    @Test
    void rejectsWrongModuloBucketAndExactServerCap() throws Exception {
        var config = numericSettings(4000, 2000);
        assertThrows(IllegalStateException.class, () -> new BucketSidReader((k, l, t) ->
                CompletableFuture.completedFuture(k.equals("0")
                        ? List.of(new SidBucketClient.Row(k, "1", "C1C2")) : List.of()), config).read(() -> true));
        var rows = new ArrayList<SidBucketClient.Row>();
        for (int i = 0; i < 2000; i++) { rows.add(new SidBucketClient.Row("0", "" + (i * 4000), "C1C2")); }
        var reader = new BucketSidReader((k, l, t) -> CompletableFuture.completedFuture(
                k.equals("0") ? rows : List.of()), config);
        assertTrue(assertThrows(IllegalStateException.class, () -> reader.read(() -> true))
                .getMessage().contains("possible truncation"));
        rows.remove(rows.size() - 1);
        assertEquals(1999, reader.read(() -> true).itemCount());
    }

    @Test
    void numeric4000BucketsRemainBoundedAndMergeSharedSids() throws Exception {
        var live = new AtomicInteger();
        var peak = new AtomicInteger();
        var calls = new AtomicInteger();
        var executor = Executors.newScheduledThreadPool(4);
        try {
            var reader = new BucketSidReader((key, limit, timeout) -> {
                calls.incrementAndGet();
                peak.accumulateAndGet(live.incrementAndGet(), Math::max);
                var future = new CompletableFuture<List<SidBucketClient.Row>>();
                executor.schedule(() -> {
                    live.decrementAndGet();
                    future.complete(List.of(new SidBucketClient.Row(key, key, "C1C2")));
                }, 1, TimeUnit.MILLISECONDS);
                return future;
            }, numericSettings(4000, 2000));
            var result = reader.read(() -> true);
            assertEquals(4000, calls.get());
            assertEquals(4000, result.itemCount());
            assertEquals(1, result.sids().size());
            assertTrue(peak.get() > 1 && peak.get() <= 4);
        } finally { executor.shutdownNow(); }
    }

    @Test
    void twoPointFiveMillionNumericItemsAcross4000Buckets() throws Exception {
        assumeTrue("1".equals(System.getenv("CONSTRAINT_TREE_RUN_IGRAPH_SCALE_TEST")), "opt-in synthetic scale test");
        var data = new ArrayList<List<SidBucketClient.Row>>();
        for (int bucket = 0; bucket < 4000; bucket++) {
            var rows = new ArrayList<SidBucketClient.Row>();
            for (int i = bucket; i < 2_500_000; i += 4000) {
                rows.add(new SidBucketClient.Row("" + bucket, "" + i, "C" + (i / 2048) + "C" + (i % 2048)));
            }
            data.add(rows);
        }
        var result = new BucketSidReader((k, l, t) -> CompletableFuture.completedFuture(
                data.get(Integer.parseInt(k))), numericSettings(4000, 2000)).read(() -> true);
        assertEquals(2_500_000, result.itemCount());
        assertEquals(2_500_000, result.sids().size());
        assertEquals(625, result.maxBucketRows());
        System.out.println("SYNTHETIC_NUMERIC_IGRAPH items=" + result.itemCount() + " buckets=" + result.bucketCount()
                + " maxBucket=" + result.maxBucketRows() + " mergeMs=" + result.elapsedMillis()
                + " (sequential synthetic IDs, NOT real distribution or network RT)");
    }

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

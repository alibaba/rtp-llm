package org.flexlb.constraint;

import org.flexlb.constraint.source.SidBucketClient;

import java.time.Duration;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.regex.Pattern;
import java.util.zip.CRC32;

/** Bounded asynchronous point queries over a known keyspace; this is NOT a storage snapshot scan. */
public final class BucketSidReader {
    private static final Pattern SID = Pattern.compile("(?:C[0-9]+)+");
    private final SidBucketClient client;
    private final Settings settings;

    public record Settings(String keyPrefix, int bucketCount, int concurrency, int maxRowsPerBucket,
                           Duration queryTimeout, Duration roundTimeout, int retries) {
        public Settings {
            if (keyPrefix == null || !keyPrefix.matches("[A-Za-z0-9_.-]+")
                    || bucketCount < 1 || bucketCount > 1_000_000 || concurrency < 1 || concurrency > 256
                    || maxRowsPerBucket < 1 || maxRowsPerBucket >= 50_000 || retries < 0 || retries > 3
                    || queryTimeout == null || queryTimeout.toMillis() < 1
                    || roundTimeout == null || roundTimeout.toMillis() < 1) {
                throw new IllegalArgumentException("invalid bucket reader settings");
            }
        }

        public String key(int bucket) { return keyPrefix + bucket; }
    }

    public record Result(List<String> sids, long itemCount, int bucketCount, long elapsedMillis) { }

    public BucketSidReader(SidBucketClient client, Settings settings) {
        this.client = client;
        this.settings = settings;
    }

    public Result read(BooleanSupplier mayContinue) throws Exception {
        long started = System.nanoTime();
        long deadline = started + settings.roundTimeout().toNanos();
        var sids = new HashSet<String>();
        long items = 0;
        // Only one window is live at a time. Parsing/merging happens on the caller's background thread.
        for (int first = 0; first < settings.bucketCount(); first += settings.concurrency()) {
            check(mayContinue, deadline);
            List<CompletableFuture<List<SidBucketClient.Row>>> pending = new ArrayList<>();
            try {
                int end = Math.min(settings.bucketCount(), first + settings.concurrency());
                for (int bucket = first; bucket < end; bucket++) {
                    pending.add(start(settings.key(bucket)));
                }
                for (int i = 0; i < pending.size(); i++) {
                    String key = settings.key(first + i);
                    List<SidBucketClient.Row> rows = await(pending.get(i), key, mayContinue, deadline);
                    if (rows == null || rows.size() > settings.maxRowsPerBucket()) {
                        throw new IllegalStateException("bucket " + key + " exceeds row limit or has invalid response");
                    }
                    Map<String, String> uniqueItems = new HashMap<>();
                    for (var row : rows) {
                        if (row == null || !key.equals(row.pkey()) || row.itemId() == null || row.itemId().isBlank()
                                || row.sid() == null || !SID.matcher(row.sid()).matches()) {
                            throw new IllegalStateException("invalid item/SID in bucket " + key);
                        }
                        if (!key.equals(settings.key(bucketForItem(row.itemId(), settings.bucketCount())))) {
                            throw new IllegalStateException("item is in the wrong hash bucket " + key);
                        }
                        String previous = uniqueItems.putIfAbsent(row.itemId(), row.sid());
                        if (previous != null) {
                            throw new IllegalStateException("duplicate item in bucket " + key);
                        }
                        sids.add(row.sid());
                    }
                    items += uniqueItems.size();
                }
            } finally {
                pending.forEach(future -> future.cancel(true));
            }
        }
        check(mayContinue, deadline);
        if (sids.isEmpty()) {
            throw new IllegalStateException("empty source; retaining existing tree (empty-tree publication unsupported)");
        }
        return new Result(List.copyOf(sids), items, settings.bucketCount(),
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started));
    }

    private CompletableFuture<List<SidBucketClient.Row>> start(String key) {
        try {
            // Fetch one extra row: an overflowing bucket must never silently become a truncated tree.
            return client.readAsync(key, settings.maxRowsPerBucket() + 1, settings.queryTimeout());
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    private List<SidBucketClient.Row> await(CompletableFuture<List<SidBucketClient.Row>> initial, String key,
                                           BooleanSupplier mayContinue, long deadline) throws Exception {
        var future = initial;
        try {
            for (int attempt = 0; ; attempt++) {
                check(mayContinue, deadline);
                try {
                    long remaining = Math.min(settings.queryTimeout().toNanos(), deadline - System.nanoTime());
                    return future.get(Math.max(1, remaining), TimeUnit.NANOSECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw e;
                } catch (Exception e) {
                    future.cancel(true);
                    if (attempt >= settings.retries()) {
                        throw new IllegalStateException("failed reading bucket " + key, e);
                    }
                    check(mayContinue, deadline);
                    future = start(key);
                }
            }
        } finally {
            future.cancel(true);
        }
    }

    private static void check(BooleanSupplier mayContinue, long deadline) {
        if (Thread.currentThread().isInterrupted() || !mayContinue.getAsBoolean()) {
            throw new IllegalStateException("bucket read cancelled or leadership lost");
        }
        if (System.nanoTime() >= deadline) {
            throw new IllegalStateException("bucket read round timed out");
        }
    }

    /** Writer contract: unsigned CRC32 of the exact UTF-8 item_id string, modulo a fixed bucket count. */
    public static int bucketForItem(String itemId, int bucketCount) {
        if (itemId == null || itemId.isBlank() || bucketCount < 1) {
            throw new IllegalArgumentException("item id and bucket count are required");
        }
        CRC32 crc = new CRC32();
        crc.update(itemId.getBytes(StandardCharsets.UTF_8));
        return (int) (crc.getValue() % bucketCount);
    }
}

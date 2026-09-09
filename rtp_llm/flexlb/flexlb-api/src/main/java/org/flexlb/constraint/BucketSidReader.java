package org.flexlb.constraint;

import org.flexlb.constraint.source.SidBucketClient;

import java.time.Duration;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
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

    public enum BucketAlgorithm { CRC32, ITEM_ID_MOD }
    public enum EmptySidPolicy { REJECT, SKIP }

    public record Settings(String keyPrefix, int bucketCount, int concurrency, int maxRowsPerBucket,
                           Duration queryTimeout, Duration roundTimeout, int retries,
                           BucketAlgorithm bucketAlgorithm, int sourceRowLimit, EmptySidPolicy emptySidPolicy) {
        public Settings(String keyPrefix, int bucketCount, int concurrency, int maxRowsPerBucket,
                        Duration queryTimeout, Duration roundTimeout, int retries,
                        BucketAlgorithm bucketAlgorithm, int sourceRowLimit) {
            this(keyPrefix, bucketCount, concurrency, maxRowsPerBucket, queryTimeout, roundTimeout, retries,
                    bucketAlgorithm, sourceRowLimit, EmptySidPolicy.REJECT);
        }

        public Settings(String keyPrefix, int bucketCount, int concurrency, int maxRowsPerBucket,
                        Duration queryTimeout, Duration roundTimeout, int retries) {
            this(keyPrefix, bucketCount, concurrency, maxRowsPerBucket, queryTimeout, roundTimeout, retries,
                    BucketAlgorithm.CRC32, 0);
        }

        public Settings {
            if (keyPrefix == null || !keyPrefix.matches("[A-Za-z0-9_.-]*")
                    || bucketAlgorithm == null || emptySidPolicy == null || sourceRowLimit < 0
                    || bucketCount < 1 || bucketCount > 1_000_000 || concurrency < 1 || concurrency > 256
                    || maxRowsPerBucket < 1 || maxRowsPerBucket >= 50_000 || retries < 0 || retries > 3
                    || queryTimeout == null || queryTimeout.toMillis() < 1
                    || roundTimeout == null || roundTimeout.toMillis() < 1) {
                throw new IllegalArgumentException("invalid bucket reader settings");
            }
        }

        public String key(int bucket) { return keyPrefix + bucket; }
    }

    public record Result(List<String> sids, long itemCount, int bucketCount, long elapsedMillis,
                         int maxBucketRows, long skippedEmptySids) {
        public long eligibleItems() { return itemCount - skippedEmptySids; }
    }

    public BucketSidReader(SidBucketClient client, Settings settings) {
        this.client = client;
        this.settings = settings;
    }

    public Result read(BooleanSupplier mayContinue) throws Exception {
        long started = System.nanoTime();
        long deadline = started + settings.roundTimeout().toNanos();
        var sids = new HashSet<String>();
        long items = 0;
        long skippedEmptySids = 0;
        int maxBucketRows = 0;
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
                    if (settings.sourceRowLimit() > 0 && rows.size() >= settings.sourceRowLimit()) {
                        throw new IllegalStateException("bucket " + key + " reached source row limit; possible truncation");
                    }
                    maxBucketRows = Math.max(maxBucketRows, rows.size());
                    var uniqueItems = new HashSet<String>();
                    for (var row : rows) {
                        if (row == null || !key.equals(row.pkey()) || row.itemId() == null || row.itemId().isBlank()
                                || row.sid() == null) {
                            throw new IllegalStateException("invalid item/SID in bucket " + key);
                        }
                        if (!key.equals(settings.key(bucketForItem(row.itemId(), settings.bucketCount(),
                                settings.bucketAlgorithm())))) {
                            throw new IllegalStateException("item is in the wrong hash bucket " + key);
                        }
                        if (!uniqueItems.add(row.itemId())) {
                            throw new IllegalStateException("duplicate item in bucket " + key);
                        }
                        // Only the explicitly confirmed empty mapping is skippable. Missing fields,
                        // malformed SIDs and wrong/duplicate items remain errors in SKIP mode too.
                        if (row.sid().isEmpty() && settings.emptySidPolicy() == EmptySidPolicy.SKIP) {
                            skippedEmptySids++;
                            continue;
                        }
                        if (!SID.matcher(row.sid()).matches()) {
                            throw new IllegalStateException("invalid SID in bucket " + key);
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
            throw new IllegalStateException("empty source; retaining existing tree (empty-tree publication unsupported); "
                    + "items=" + items + "; skippedEmptySids=" + skippedEmptySids);
        }
        return new Result(List.copyOf(sids), items, settings.bucketCount(),
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started), maxBucketRows, skippedEmptySids);
    }

    private CompletableFuture<List<SidBucketClient.Row>> start(String key) {
        try {
            // Detect client-limit overflow. Hidden server/index truncation still requires source reconciliation.
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
                        throw new IllegalStateException("failed reading bucket " + key + ": " + failureSummary(e), e);
                    }
                    check(mayContinue, deadline);
                    future = start(key);
                }
            }
        } finally {
            future.cancel(true);
        }
    }

    /** Keep status actionable without copying the SDK's full request context/query into it. */
    static String failureSummary(Throwable error) {
        var seen = new HashSet<Throwable>();
        while (error.getCause() != null && seen.add(error) && !seen.contains(error.getCause())) {
            error = error.getCause();
        }
        String message = error.getMessage();
        if (message == null || message.isBlank()) { return error.getClass().getSimpleName(); }
        int context = message.indexOf("requestContext");
        if (context >= 0) { message = message.substring(0, context); }
        message = message.replace('\n', ' ').replace('\r', ' ').strip();
        return error.getClass().getSimpleName() + ": " + message.substring(0, Math.min(message.length(), 240));
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
        return bucketForItem(itemId, bucketCount, BucketAlgorithm.CRC32);
    }

    public static int bucketForItem(String itemId, int bucketCount, BucketAlgorithm algorithm) {
        if (itemId == null || itemId.isBlank() || bucketCount < 1) {
            throw new IllegalArgumentException("item id and bucket count are required");
        }
        if (algorithm == null) { throw new IllegalArgumentException("bucket algorithm is required"); }
        if (algorithm == BucketAlgorithm.ITEM_ID_MOD) {
            // Decimal streaming remainder: exact even beyond signed long / floating point precision.
            long remainder = 0;
            for (int i = 0; i < itemId.length(); i++) {
                char digit = itemId.charAt(i);
                if (digit < '0' || digit > '9') {
                    throw new IllegalArgumentException("ITEM_ID_MOD requires an unsigned decimal item_id");
                }
                remainder = (remainder * 10 + digit - '0') % bucketCount;
            }
            return (int) remainder;
        }
        CRC32 crc = new CRC32();
        crc.update(itemId.getBytes(StandardCharsets.UTF_8));
        return (int) (crc.getValue() % bucketCount);
    }
}

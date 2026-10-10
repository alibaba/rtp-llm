package org.flexlb.constraint.source;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/** Internal storage adapters must return errors, never a successful partial/failed query. */
public interface SidBucketClient {
    CompletableFuture<List<Row>> readAsync(String pkey, int limit, Duration timeout);

    record Row(String pkey, String itemId, String sid) { }
}

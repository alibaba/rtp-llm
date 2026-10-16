package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import java.util.Map;

/** Explicit synthetic cache-status payload; independent from inference KV capacity. */
final class CacheSyncFixture {
    private final int worker, workers, unique, shared;
    private long offset, version = 1, calls, entries, bytes, changedResponses, changedKeys;
    private int delta, changeEvery;
    private EngineRpcService.CacheStatusPB payload;

    CacheSyncFixture(int worker, int workers, int unique, int shared, long initialVersion) {
        if (initialVersion <= 0 || workers < 3 || workers > 4096 || worker < 0 || worker >= workers || unique < 1
                || shared < 0 || (long) unique + 2L * shared > 1_000_000) {
            throw new IllegalArgumentException("invalid ring cache fixture geometry");
        }
        this.version = initialVersion;
        this.worker = worker; this.workers = workers;
        this.unique = unique; this.shared = shared;
        rebuild();
    }

    private static long hash(long x) {
        x = (x ^ (x >>> 30)) * 0xbf58476d1ce4e5b9L;
        x = (x ^ (x >>> 27)) * 0x94d049bb133111ebL;
        return x ^ (x >>> 31);
    }

    private void rebuild() {
        var builder = EngineRpcService.CacheStatusPB.newBuilder().setVersion(version);
        // Disjoint signed-int64 namespaces; each adjacent pair owns one shared segment.
        for (int i = 0; i < unique; i++)
            builder.putCacheKeys(hash(((long) worker << 40) + offset + i), true);
        for (int segment : new int[]{worker, (worker + workers - 1) % workers})
            for (int i = 0; i < shared; i++)
                builder.putCacheKeys(hash(Long.MIN_VALUE + ((long) segment << 40) + i), true);
        payload = builder.build();
        // Once per changed version, never walk the protobuf map merely to observe a response.
        payload.getSerializedSize();
    }

    synchronized void churn(int delta, int every) {
        if (delta < 0 || delta > unique || every < 0 || (delta == 0) != (every == 0))
            throw new IllegalArgumentException("churn requires bounded delta and positive response cadence");
        this.delta = delta; this.changeEvery = every;
    }

    synchronized EngineRpcService.CacheStatusPB response() {
        calls++;
        if (changeEvery > 0 && calls % changeEvery == 0) {
            if (offset + delta + unique >= (1L << 40))
                throw new IllegalStateException("fixture key namespace exhausted");
            offset += delta; version++; changedResponses++; changedKeys += delta; rebuild();
        }
        entries += payload.getCacheKeysCount();
        bytes += payload.getSerializedSize();
        return payload;
    }

    synchronized long version() { return version; }

    synchronized Map<String, Object> snapshot() {
        return Map.of("keys", payload.getCacheKeysCount(), "version", version,
                "responses", calls, "entries", entries, "key_payload_bytes", bytes,
                "changed_responses", changedResponses,
                "added_keys", changedKeys, "removed_keys", changedKeys);
    }
}

package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import java.util.Map;
import java.util.function.LongSupplier;

/** Controlled sync-only snapshot, deliberately separate from inference cache capacity. */
final class CacheSyncFixture {
    private final int worker, workers, unique, shared;
    private final LongSupplier clock;
    private long offset, version, calls, entries, bytes, changedVersions, changedKeys;
    private long churnStart, appliedEpoch, intervalNanos;
    private int delta;
    private CacheFixtureWire.Keys wireKeys;
    private EngineRpcService.CacheStatusPB compatibilityPayload;

    CacheSyncFixture(int worker, int workers, int unique, int shared, long initialVersion) {
        this(worker, workers, unique, shared, initialVersion, System::nanoTime);
    }

    CacheSyncFixture(int worker, int workers, int unique, int shared, long initialVersion, LongSupplier clock) {
        if (initialVersion <= 0 || workers < 1 || workers > 4096 || worker < 0 || worker >= workers
                || unique < 1 || shared < 0 || (long) unique + shared > 1_000_000)
            throw new IllegalArgumentException("invalid global-shared cache fixture geometry");
        this.worker = worker; this.workers = workers; this.unique = unique; this.shared = shared;
        this.version = initialVersion; this.clock = clock;
        rebuild();
    }

    private void rebuild() {
        wireKeys = CacheFixtureWire.Keys.create(worker, workers, unique, shared, offset, version);
        compatibilityPayload = null;
    }

    /** Changes only private keys; polling frequency never controls mutation rate. */
    synchronized void churn(int delta, int intervalMs) {
        if (delta < 0 || delta > unique || intervalMs <= 0)
            throw new IllegalArgumentException("churn requires bounded delta and positive interval_ms");
        advance(); // Preserve elapsed changes before pausing or changing the rate.
        this.delta = delta;
        intervalNanos = Math.multiplyExact((long) intervalMs, 1_000_000L);
        churnStart = clock.getAsLong(); appliedEpoch = 0;
    }

    private void advance() {
        if (delta == 0) return;
        long epoch = Math.max(0L, clock.getAsLong() - churnStart) / intervalNanos;
        long steps = epoch - appliedEpoch;
        if (steps <= 0) return;
        long moved = Math.multiplyExact((long) delta, steps);
        if (offset + moved + unique >= (1L << 40))
            throw new IllegalStateException("fixture key namespace exhausted");
        offset += moved; version += steps; changedVersions += steps; changedKeys += moved;
        appliedEpoch = epoch;
        rebuild();
    }

    synchronized EngineRpcService.CacheStatusPB response() { return response(-1L); }

    synchronized EngineRpcService.CacheStatusPB response(long latestVersion) {
        var frame = wireResponse(latestVersion);
        if (compatibilityPayload == null) compatibilityPayload = frame.toProto();
        return compatibilityPayload;
    }

    synchronized CacheFixtureWire.Frame wireResponse(long latestVersion) {
        advance(); calls++;
        var header = EngineRpcService.CacheStatusPB.newBuilder().setVersion(version).build();
        entries += wireKeys.count(); bytes += header.getSerializedSize() + wireKeys.encodedSize();
        // Real engines send full snapshots even when the requested version matches.
        return new CacheFixtureWire.Frame(header, wireKeys);
    }

    synchronized CacheFixtureWire.Keys keys() { return wireKeys; }

    synchronized long version() { advance(); return version; }

    synchronized Map<String, Object> snapshot() {
        advance();
        Map<String, Object> result = new java.util.LinkedHashMap<>();
        result.put("mode", "controlled"); result.put("keys", wireKeys.count()); result.put("version", version);
        result.put("unique_keys", unique); result.put("shared_keys", shared);
        result.put("replacement_keys_per_second", delta == 0 ? 0.0 : delta * 1e9 / intervalNanos);
        result.put("responses", calls); result.put("entries", entries); result.put("key_payload_bytes", bytes);
        result.put("changed_versions", changedVersions); result.put("added_keys", changedKeys);
        result.put("removed_keys", changedKeys); result.put("full_responses", calls);
        result.put("unchanged_responses", 0L);
        result.put("sum", Long.toUnsignedString(wireKeys.sum())); result.put("xor", Long.toUnsignedString(wireKeys.xor()));
        return result;
    }
}

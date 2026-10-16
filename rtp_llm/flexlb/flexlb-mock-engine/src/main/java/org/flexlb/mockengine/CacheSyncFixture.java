package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import java.util.Map;

/** Explicit synthetic cache-status payload; independent from inference KV capacity. */
final class CacheSyncFixture {
    private final int worker, workers;
    private int unique, shared;
    private long offset, version = 1, calls, entries, bytes, changedResponses, changedKeys;
    private long fullResponses, unchangedResponses, sum, xor, churnStart, appliedEpoch, grownKeys;
    private final boolean omitUnchanged;
    private long intervalNanos;
    private final java.util.function.LongSupplier clock;
    private int delta, changeEvery;
    private CacheFixtureWire.Keys wireKeys;
    // Used only by direct Java callers; the gRPC path never materializes a key map.
    private EngineRpcService.CacheStatusPB compatibilityPayload;

    CacheSyncFixture(int worker, int workers, int unique, int shared, long initialVersion) {
        this(worker, workers, unique, shared, initialVersion,
                Boolean.getBoolean("flexlb.mock.cache.omit-unchanged"),
                Long.getLong("flexlb.mock.cache.churn-interval-ms", 0L) * 1_000_000L, System::nanoTime);
    }

    CacheSyncFixture(int worker, int workers, int unique, int shared, long initialVersion,
            boolean omitUnchanged, long intervalNanos, java.util.function.LongSupplier clock) {
        this.omitUnchanged = omitUnchanged;
        this.intervalNanos = intervalNanos;
        this.clock = clock;
        if (intervalNanos < 0) throw new IllegalArgumentException("negative churn interval");
        if (initialVersion <= 0 || workers < 3 || workers > 4096 || worker < 0 || worker >= workers || unique < 1
                || shared < 0 || (long) unique + 2L * shared > 1_000_000) {
            throw new IllegalArgumentException("invalid ring cache fixture geometry");
        }
        this.version = initialVersion;
        this.worker = worker; this.workers = workers;
        this.unique = unique; this.shared = shared;
        rebuild();
    }

    private void rebuild() {
        wireKeys = CacheFixtureWire.Keys.create(worker, workers, unique, shared, offset, version);
        sum = wireKeys.sum(); xor = wireKeys.xor();
        compatibilityPayload = null;
    }

    synchronized void grow(int unique, int shared) {
        if (changeEvery != 0 || offset != 0 || unique < this.unique || shared < this.shared
                || (long) unique + 2L * shared > 1_000_000)
            throw new IllegalArgumentException("growth requires monotonic bounded geometry before churn");
        if (unique == this.unique && shared == this.shared) return;
        grownKeys += (long) unique + 2L * shared - this.unique - 2L * this.shared;
        this.unique = unique; this.shared = shared; version++;
        rebuild();
    }

    synchronized void clockInterval(int intervalMs) {
        if (intervalMs <= 0) throw new IllegalArgumentException("positive clock interval required");
        intervalNanos = Math.multiplyExact((long) intervalMs, 1_000_000L);
        churnStart = clock.getAsLong(); appliedEpoch = 0;
    }

    synchronized void churn(int delta, int every) {
        if (delta < 0 || delta > unique || every < 0 || (delta == 0) != (every == 0))
            throw new IllegalArgumentException("churn requires bounded delta and positive response cadence");
        this.delta = delta; this.changeEvery = every;
        churnStart = clock.getAsLong(); appliedEpoch = 0;
    }

    synchronized EngineRpcService.CacheStatusPB response() { return response(-1L); }

    synchronized EngineRpcService.CacheStatusPB response(long latestVersion) {
        var frame = wireResponse(latestVersion);
        if (frame.keys() == null) return frame.header();
        if (compatibilityPayload == null) compatibilityPayload = frame.toProto();
        return compatibilityPayload;
    }

    synchronized CacheFixtureWire.Frame wireResponse(long latestVersion) {
        calls++;
        long epoch = intervalNanos > 0 ? Math.max(0L, clock.getAsLong() - churnStart) / intervalNanos : 0L;
        long steps = changeEvery == 0 ? 0L : intervalNanos > 0 ? epoch - appliedEpoch
                : calls % changeEvery == 0 ? 1L : 0L;
        if (steps > 0) {
            long moved = Math.multiplyExact((long) delta, steps);
            if (offset + moved + unique >= (1L << 40))
                throw new IllegalStateException("fixture key namespace exhausted");
            offset += moved; version += steps; changedResponses++; changedKeys += moved;
            appliedEpoch = epoch; rebuild();
        }
        var header = EngineRpcService.CacheStatusPB.newBuilder().setVersion(version).build();
        CacheFixtureWire.Keys keys = wireKeys;
        if (omitUnchanged && latestVersion >= 0 && latestVersion == version) {
            keys = null;
            unchangedResponses++;
        } else {
            fullResponses++;
        }
        entries += keys == null ? 0 : keys.count();
        bytes += header.getSerializedSize() + (keys == null ? 0 : keys.encodedSize());
        return new CacheFixtureWire.Frame(header, keys);
    }

    synchronized long version() { return version; }

    synchronized Map<String, Object> snapshot() {
        Map<String, Object> result = new java.util.LinkedHashMap<>();
        result.put("keys", wireKeys.count()); result.put("version", version);
        result.put("responses", calls); result.put("entries", entries); result.put("key_payload_bytes", bytes);
        result.put("changed_responses", changedResponses); result.put("added_keys", changedKeys + grownKeys);
        result.put("removed_keys", changedKeys); result.put("full_responses", fullResponses);
        result.put("unchanged_responses", unchangedResponses);
        result.put("sum", Long.toUnsignedString(sum)); result.put("xor", Long.toUnsignedString(xor));
        return result;
    }
}

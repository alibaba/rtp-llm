package org.flexlb.mockengine;

import java.util.*;
import java.util.function.LongSupplier;

/** Bounded, opt-in counterfactual observations; never changes routing or real LRU order. */
final class MockCacheDiagnostics {
    private final Map<String, Set<Long>> local = new HashMap<>();
    private final Set<Long> global = new HashSet<>();
    private final LongSupplier clock;
    private final long deadline;
    private final int maxKeys, sampleEvery;
    private long storedKeys, seen, samples, inputTokens, actualTokens, localTokens, bestTokens, globalTokens;
    private String stopped = "";
    private final ArrayDeque<Map<String, Object>> rows = new ArrayDeque<>();

    MockCacheDiagnostics(int seconds, int maxKeys, int sampleEvery, LongSupplier clock) {
        if (seconds < 1 || seconds > 300 || maxKeys < 1 || maxKeys > 8_000_000
                || sampleEvery < 1) throw new IllegalArgumentException("invalid diagnostic bounds");
        this.clock = clock;
        this.deadline = clock.getAsLong() + seconds * 1000L;
        this.maxKeys = maxKeys;
        this.sampleEvery = sampleEvery;
    }

    synchronized boolean active() {
        if (stopped.isEmpty() && clock.getAsLong() >= deadline) stopped = "duration_limit";
        return stopped.isEmpty();
    }

    synchronized void stop() { stopped = "stopped"; }

    // Seed from resident keys; subsequently call ONLY after successful P computation.
    synchronized void completed(String engine, Collection<Long> keys) {
        if (!active()) return;
        Set<Long> history = local.computeIfAbsent(engine, k -> new HashSet<>());
        for (Long key : keys) {
            long additional = (history.contains(key) ? 0 : 1) + (global.contains(key) ? 0 : 1);
            if (storedKeys + additional > maxKeys) { stopped = "key_limit"; return; }
            history.add(key);
            global.add(key);
            storedKeys += additional;
        }
    }

    synchronized boolean select() { return active() && seen++ % sampleEvery == 0; }

    static int prefix(List<Long> keys, Set<Long> cache) {
        int n = 0;
        while (n < keys.size() && cache.contains(keys.get(n))) n++;
        return n;
    }

    synchronized void observe(String engine, List<Long> keys, int blockSize, long input,
                              long actual, int bestBlocks, String bestEngine) {
        if (!active()) return;
        long own = Math.min(input, (long) prefix(keys, local.getOrDefault(engine, Set.of())) * blockSize);
        long all = Math.min(input, (long) prefix(keys, global) * blockSize);
        long best = Math.min(input, (long) bestBlocks * blockSize);
        samples++; inputTokens += input; actualTokens += actual;
        localTokens += own; globalTokens += all; bestTokens += best;
        var row = new LinkedHashMap<String, Object>();
        row.put("engine", engine); row.put("best_engine", bestEngine);
        row.put("input_tokens", input); row.put("actual_tokens", actual);
        row.put("local_no_eviction_tokens", own); row.put("best_resident_tokens", best);
        row.put("global_no_eviction_tokens", all);
        rows.addLast(row);
        if (rows.size() > 128) rows.removeFirst();
    }

    synchronized Map<String, Object> snapshot() {
        boolean running = active();
        var result = new LinkedHashMap<String, Object>();
        result.put("active", running); result.put("stop_reason", stopped);
        result.put("stored_key_entries", storedKeys); result.put("sample_every", sampleEvery);
        result.put("requests_seen", seen); result.put("samples", samples);
        result.put("input_tokens", inputTokens); result.put("actual_tokens", actualTokens);
        result.put("local_no_eviction_tokens", localTokens);
        result.put("best_resident_tokens", bestTokens);
        result.put("global_no_eviction_tokens", globalTokens);
        result.put("rows", new ArrayList<>(rows));
        result.put("scope", "P admission observations; includes later rejected requests; resident seed plus successful P completions; peer reads are non-atomic; global history is a shared-cache oracle, not a routable single-engine cache");
        return result;
    }
}

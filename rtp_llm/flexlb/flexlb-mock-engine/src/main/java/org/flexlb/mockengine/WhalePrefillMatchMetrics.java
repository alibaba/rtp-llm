package org.flexlb.mockengine;

import java.util.*;

/** Token-caliber counterpart of PrefillCacheHitMetricsReporter/RecentCacheKeyWindow. */
final class WhalePrefillMatchMetrics {
    private record Entry(long time, List<Long> keys) {}
    private final long windowMs;
    private final Deque<Entry> entries = new ArrayDeque<>();
    private final Map<Long, Integer> counts = new HashMap<>();
    private long retained, allHit, allInput;
    WhalePrefillMatchMetrics(long windowMs) { this.windowMs = windowMs > 0 ? windowMs : 1800000; }
    static long configuredWindow() {
        for (String key : List.of("PREFILL_CACHE_HIT_TIME_WINDOW_MS", "CACHE_HIT_TIME_WINDOW_MS")) {
            try { String s = System.getenv(key); if (s != null && !s.isBlank()) return Long.parseLong(s); }
            catch (NumberFormatException ignored) {}
        }
        return 1800000;
    }
    synchronized Map<String, Number> record(List<Long> keys, long input, long actualHit, int blockSize, long now) {
        while (!entries.isEmpty() && entries.peekFirst().time() <= now - windowMs) {
            for (Long key : entries.removeFirst().keys()) {
                counts.compute(key, (k,v) -> v == 1 ? null : v-1); retained--;
            }
        }
        long hits = keys.stream().filter(counts::containsKey).count();
        if (!keys.isEmpty()) {
            entries.addLast(new Entry(now, List.copyOf(keys)));
            for (Long key : keys) { counts.merge(key,1,Integer::sum); retained++; }
        }
        long tokens = Math.min(Math.max(0,input), hits * blockSize);
        allHit += tokens; allInput += Math.max(0,input);
        return Map.ofEntries(
            Map.entry("rtp_llm_prefill_worker_recent_cache_key_hit_count",tokens),
            Map.entry("rtp_llm_prefill_worker_recent_cache_key_total_count",Math.max(0,input)),
            Map.entry("rtp_llm_prefill_worker_recent_cache_key_hit_ratio",input>0 ? (double)tokens/input : 0),
            Map.entry("rtp_llm_prefill_worker_recent_cache_key_retained_occurrences",retained),
            Map.entry("rtp_llm_prefill_worker_recent_cache_key_retained_unique_cache_keys",counts.size()),
            Map.entry("rtp_llm_prefill_worker_recent_cache_key_time_window_ms",windowMs),
            Map.entry("rtp_llm_prefill_worker_theory_cache_all_hit_tokens",allHit),
            Map.entry("rtp_llm_prefill_worker_theory_cache_all_input_tokens",allInput),
            Map.entry("rtp_llm_prefill_worker_theory_cache_all_hit_ratio",allInput>0 ? (double)allHit/allInput : 0),
            Map.entry("mock_prefill_kv_match_ratio",input>0 ? (double)Math.min(actualHit,input)/input : 0));
    }
}

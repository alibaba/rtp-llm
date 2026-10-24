package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.engine.grpc.EngineRpcService;
import java.util.*;

/** MMHashKeyCache: independent CPU pool and real-runtime metadata accounting. */
final class MockVitHashCache implements AutoCloseable {
    record Entry(EngineRpcService.TensorPB hashes, int rows, String generation,
                 long charge, MockTensorPool.Block block) {}
    private final LinkedHashMap<String, Entry> entries = new LinkedHashMap<>();
    private final JsonNode accounting;
    private final MockTensorPool pool;
    private long used;
    private boolean closed;

    MockVitHashCache(long capacity, JsonNode accounting) {
        validateAccounting(accounting);
        this.accounting = accounting.deepCopy(); pool = new MockTensorPool(capacity);
    }
    static void validateAccounting(JsonNode accounting) {
        var fields = new String[]{"entry_overhead", "ascii_base", "latin1_base", "bmp_base", "ucs4_base",
                "hash_list_bytes", "tensor_object_bytes", "record_bytes", "integer_bytes"};
        MockEngineConfig.fields(accounting, fields);
        for (String name : fields) {
            JsonNode value = accounting.path(name);
            if (!value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() < 1)
                throw new IllegalArgumentException("missing real hash-cache accounting: " + name);
        }
    }
    private long stringBytes(String value) {
        int max = value.codePoints().max().orElse(0);
        int width = max <= 255 ? 1 : max <= 65535 ? 2 : 4;
        String base = max <= 127 ? "ascii_base" : max <= 255 ? "latin1_base" : max <= 65535 ? "bmp_base" : "ucs4_base";
        return Math.addExact(accounting.path(base).longValue(), Math.multiplyExact((long)value.codePointCount(0, value.length()), width));
    }
    long charge(String key, long tensorBytes, String generation) {
        long total = Math.addExact(tensorBytes, Math.addExact(stringBytes(key), stringBytes(generation)));
        for (String field : List.of("entry_overhead", "hash_list_bytes", "tensor_object_bytes", "record_bytes", "integer_bytes"))
            total = Math.addExact(total, accounting.path(field).longValue());
        return total;
    }
    synchronized void put(String key, MockVitImage.Result result, String generation) {
        if (closed || pool.stats().capacity() == 0 || key.isEmpty()) return;
        long tensorBytes = result.hashBytes();
        if (tensorBytes > pool.stats().capacity()) { remove(key); return; }
        long charge = charge(key, tensorBytes, generation);
        remove(key); // Oversized replacement must discard its stale generation only.
        if (charge > pool.stats().capacity()) return;
        while (used + charge > pool.stats().capacity()) remove(entries.firstEntry().getKey());
        MockTensorPool.Block block;
        try { block = pool.reserve(tensorBytes); }
        catch (MockTensorPool.CapacityException error) { return; }
        entries.put(key, new Entry(result.hashes(), result.rows(), generation, charge, block));
        used += charge;
    }
    synchronized Entry peek(String key) { return entries.get(key); }
    synchronized Entry consume(String key, String generation) {
        var entry = entries.get(key);
        if (entry == null || !entry.generation.equals(generation)) return null;
        entries.remove(key); entries.put(key, entry); return entry;
    }
    synchronized List<String> keys(int limit) {
        var keys = new ArrayList<>(entries.reversed().keySet().stream().limit(limit).toList());
        Collections.reverse(keys); return keys;
    }
    synchronized void remove(String key) {
        var old = entries.remove(key);
        if (old != null) { used -= old.charge; pool.release(old.block, List.of()); }
    }
    synchronized Map<String, Long> stats() {
        var stats = pool.stats();
        return Map.of("hash_bytes", used, "hash_pool_capacity_bytes", stats.capacity(),
                "hash_pool_used_bytes", stats.used(), "hash_pool_free_bytes", stats.free(),
                "hash_pool_largest_free_block_bytes", stats.largest());
    }
    @Override public synchronized void close() { closed = true; for (String key : List.copyOf(entries.keySet())) remove(key); }
}

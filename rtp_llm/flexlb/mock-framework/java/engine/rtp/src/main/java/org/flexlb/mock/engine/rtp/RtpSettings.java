package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.*;
import org.flexlb.mock.api.model.Types.Role;

/** Recognized settings and deferred behavior; requested values are never presented as effective ones. */
final class RtpSettings {
    private static final Set<String> COMMON = Set.of(
        "test_block_num", "reserve_block_ratio", "device_eviction_policy", "seq_size_per_block", "reuse_cache",
        "enable_device_cache", "disable_decode_first_malloc_device_reuse", "max_seq_len", "max_batch_tokens_size",
        "max_generate_batch_size", "tp_size", "cp_size", "cp_enabled", "dp_size", "dp_rank", "compute",
        "enable_memory_cache", "enable_disk_cache", "enable_remote_cache", "max_rpc_timeout_ms",
        "memory_cache_blocks", "memory_eviction_policy", "block_tree_memory_evict_low_watermark_ratio",
        "block_tree_memory_evict_high_watermark_ratio",
        "block_tree_device_evict_low_watermark_ratio", "block_tree_device_evict_high_watermark_ratio",
        "block_tree_business_queue_max_size", "block_tree_transfer_queue_max_size");
    private static final Set<String> PREFILL = Set.of(
        "prefill_retry_times", "prefill_retry_timeout_ms", "prefill_max_wait_timeout_ms",
        "max_batch_tokens_without_cache", "max_inited_kv_cache_streams", "memory_cache_sync_timeout_ms");
    private static final Set<String> DECODE = Set.of("load_cache_timeout_ms");
    private static final Set<String> PENDING_INTS = Set.of(
        "prefill_retry_timeout_ms", "prefill_max_wait_timeout_ms", "max_batch_tokens_without_cache",
        "max_inited_kv_cache_streams", "memory_cache_sync_timeout_ms", "max_rpc_timeout_ms",
        "block_tree_business_queue_max_size", "block_tree_transfer_queue_max_size");

    static ObjectNode normalize(JsonNode config, Role role) {
        var result = ((ObjectNode) config).objectNode();
        config.fields().forEachRemaining(entry -> {
            String key = entry.getKey().toLowerCase(Locale.ROOT);
            if ((!entry.getKey().equals(key) && !entry.getKey().equals(key.toUpperCase(Locale.ROOT)))
                    || !COMMON.contains(key) && !(role == Role.prefill ? PREFILL : DECODE).contains(key))
                throw new IllegalArgumentException("Unknown, removed, or misplaced RTP " + role + " setting: " + entry.getKey());
            if (result.has(key)) throw new IllegalArgumentException("Duplicate native setting: " + key);
            if (entry.getValue().isNull()) throw new IllegalArgumentException("Null RTP setting: " + key);
            result.set(key, entry.getValue());
        });
        return result;
    }

    static ObjectNode pending(JsonNode config) {
        var result = ((ObjectNode) config).objectNode();
        for (String key : new TreeSet<>(PENDING_INTS)) {
            if (!config.has(key)) continue;
            int value = RtpConfig.nativeInt(config, key, null, 0, Integer.MAX_VALUE);
            if (value != 0) result.put(key, value);
        }
        int retries = RtpConfig.nativeInt(config, "prefill_retry_times", 0, 0, Integer.MAX_VALUE);
        if (retries != 0) result.put("prefill_retry_times", retries);
        return result;
    }

    private RtpSettings() {}
}

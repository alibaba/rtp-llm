package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.cache.DeviceCache.EvictionPolicy;
import org.flexlb.mock.api.model.DecodeSpec;

/** Native RTP names/units stay native; resolved pool capacity excludes the native null block. */
public record RtpConfig(String id, Role role, int blockSize, int deviceBlocks, int reserveBlocks, EvictionPolicy evictionPolicy,
                        boolean reuseCache, boolean deviceCache, boolean disableDecodeFirstReuse, int maxSeqLen, int maxBatchTokens,
                        int maxGenerateBatchSize, int cpSize, boolean cpEnabled, int tpSize, int dpSize, int loadTimeoutMs,
                        JsonNode business, ComputeModel compute) {
    public static final int REQUEST_LIMIT = 1024;
    private static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    public RtpConfig {
        if (id == null || id.isBlank() || (role != Role.prefill && role != Role.decode)
                || blockSize < 1 || deviceBlocks < 1 || reserveBlocks < 0 || reserveBlocks > deviceBlocks
                || maxSeqLen < 1 || maxBatchTokens < 0 || maxGenerateBatchSize < 1 || cpSize < 1 || tpSize < 1 || dpSize < 1 || loadTimeoutMs < 1)
            throw new IllegalArgumentException("Invalid RTP node configuration");
        business = business.deepCopy();
        Objects.requireNonNull(evictionPolicy); Objects.requireNonNull(compute);
    }

    @Override public JsonNode business() { return business.deepCopy(); }

    public static RtpConfig load(Path path) throws IOException {
        JsonNode root = JSON.readTree(path.toFile());
        String component = root.path("component_role").asText();
        Role role;
        try { role = Role.valueOf(component); } catch (RuntimeException error) { throw new IllegalArgumentException("Missing/invalid RTP component role", error); }
        JsonNode selected = root.path("component");
        if (!selected.path("engine").asText().equals("rtp")) throw new IllegalArgumentException("RTP node requires engine: rtp");
        var config = selected.path("config");
        if (!config.isObject()) throw new IllegalArgumentException("RTP config must be an object");
        if (config.has("capacity") || config.has("capacity.device_blocks") || config.has("capacity.reserve_blocks"))
            throw new IllegalArgumentException("Removed Mock capacity settings: use test_block_num and reserve_block_ratio");
        config = RtpSettings.normalize(config, role);
        var pending = RtpSettings.pending(config);
        // Native pools reserve block zero; test_block_num counts that physical block too.
        int blocks = nativeInt(config, "test_block_num", null, 2, Integer.MAX_VALUE) - 1;
        int ratio = nativeInt(config, "reserve_block_ratio", 5, 0, 100);
        int reserve = (int) ((long) blocks * ratio / 100);
        for (String tier : List.of("enable_memory_cache", "enable_disk_cache", "enable_remote_cache"))
            if (nativeBool(config, tier, false)) throw new IllegalArgumentException(tier + " is not connected in the RTP Mock yet");
        var policyValue = config.get("device_eviction_policy");
        EvictionPolicy policy;
        try {
            if (policyValue != null && !policyValue.isTextual()) throw new IllegalArgumentException("Expected a string");
            policy = EvictionPolicy.valueOf(policyValue == null ? "LRU" : policyValue.textValue().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException error) { throw new IllegalArgumentException("Invalid device_eviction_policy: expected lru, lfu, or fifo", error); }
        int cpSize = nativeInt(config, "cp_size", 1, 1, Integer.MAX_VALUE);
        int tpSize = nativeInt(config, "tp_size", 1, 1, Integer.MAX_VALUE);
        int dpSize = nativeInt(config, "dp_size", 1, 1, Integer.MAX_VALUE);
        boolean cpEnabled = nativeBool(config, "cp_enabled", false);
        if (dpSize != 1 || nativeInt(config, "dp_rank", 0, 0, Integer.MAX_VALUE) != 0)
            throw new IllegalArgumentException("RTP node currently supports dp_size=1, dp_rank=0 only");
        if (root.path("environment").has("compute")) throw new IllegalArgumentException("Removed environment.compute: use component.config.compute");
        var compute = ComputeModel.fromJson(config.path("compute"), Map.of(
            "cpSize", (double) cpSize, "cpEnabled", cpEnabled ? 1.0 : 0.0, "tpSize", (double) tpSize, "dpSize", (double) dpSize));
        if (role == Role.prefill && !compute.configuration().decode().equals(new DecodeSpec(null, null, null, null)))
            throw new IllegalArgumentException("Decode progress settings belong in Decode compute");
        if (!root.path("id").isTextual()) throw new IllegalArgumentException("Node id must be a string");
        if (!pending.isEmpty()) System.err.println("RTP " + root.path("id").asText() + " settings pending implementation (not effective): " + pending);
        return new RtpConfig(root.path("id").textValue(), role,
            nativeInt(config, "seq_size_per_block", null, 1, Integer.MAX_VALUE), blocks, reserve, policy,
            nativeBool(config, "reuse_cache", false), nativeBool(config, "enable_device_cache", true),
            nativeBool(config, "disable_decode_first_malloc_device_reuse", false),
            nativeInt(config, "max_seq_len", null, 1, Integer.MAX_VALUE),
            nativeInt(config, "max_batch_tokens_size", 0, 0, Integer.MAX_VALUE),
            nativeInt(config, "max_generate_batch_size", 1, 1, REQUEST_LIMIT),
            cpSize, cpEnabled, tpSize, dpSize,
            loadTimeout(config), config, compute);
    }

    public ObjectNode effective() {
        ObjectNode result = JSON.createObjectNode();
        result.put("id", id).put("role", role.name()).put("seq_size_per_block", blockSize)
            .put("reserve_block_ratio", nativeInt(business, "reserve_block_ratio", 5, 0, 100))
            .put("test_block_num", deviceBlocks + 1).put("device_blocks", deviceBlocks).put("reserve_blocks", reserveBlocks)
            .put("device_eviction_policy", evictionPolicy.name().toLowerCase(Locale.ROOT))
            .put("reuse_cache", reuseCache).put("enable_device_cache", deviceCache)
            .put("disable_decode_first_malloc_device_reuse", disableDecodeFirstReuse)
            .put("load_cache_timeout_ms", loadTimeoutMs).put("max_seq_len", maxSeqLen).put("max_batch_tokens_size", maxBatchTokens)
            .put("dp_size", dpSize).put("dp_rank", 0).put("tp_size", tpSize).put("cp_size", cpSize).put("cp_enabled", cpEnabled)
            .put("enable_memory_cache", false).put("enable_disk_cache", false).put("enable_remote_cache", false);
        if (role == Role.prefill) { result.remove("load_cache_timeout_ms"); result.put("prefill_retry_times", 0); }
        result.put("max_generate_batch_size", maxGenerateBatchSize).put("local_request_limit", REQUEST_LIMIT);
        result.set("requested_config", business()); result.set("pending_config", RtpSettings.pending(business));
        result.set("compute", compute.effectiveConfiguration());
        return result;
    }

    private static int loadTimeout(JsonNode config) {
        int value = nativeInt(config, "load_cache_timeout_ms", 5000, Integer.MIN_VALUE, Integer.MAX_VALUE);
        return value > 0 ? value : 5000;
    }
    static int nativeInt(JsonNode config, String key, Integer fallback, int min, int max) {
        var value = config.get(key);
        if (value == null && fallback != null) return fallback;
        return integer(value, key, min, max);
    }
    private static int integer(JsonNode value, String key, int min, int max) {
        // Whale env strings are accepted as argparse integers; fractional/coerced booleans are not.
        if (value != null && (value.isIntegralNumber() || value.isTextual() && value.textValue().matches("[+-]?\\d+"))) {
            try {
                long number = Long.parseLong(value.asText());
                if (number >= min && number <= max) return (int) number;
            } catch (NumberFormatException ignored) { }
        }
        throw new IllegalArgumentException("Invalid/missing integer setting: " + key);
    }
    private static boolean nativeBool(JsonNode config, String key, boolean fallback) {
        var value = config.get(key);
        if (value == null) return fallback;
        if (value.isBoolean()) return value.booleanValue();
        if (value.isTextual()) {
            String word = value.textValue().toLowerCase(Locale.ROOT);
            if (Set.of("yes", "true", "t", "1", "on").contains(word)) return true;
            if (Set.of("no", "false", "f", "0", "off").contains(word)) return false;
        }
        throw new IllegalArgumentException("Invalid boolean setting: " + key);
    }
}

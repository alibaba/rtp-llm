package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.ComputeModels;

/** Native RTP names/units stay native; only simulated capacity is added by this loader. */
public record RtpConfig(String id, Role role, int blockSize, int deviceBlocks, int reserveBlocks,
                        boolean reuseCache, boolean deviceCache, boolean disableDecodeFirstReuse, int maxSeqLen, int maxBatchTokens,
                        int tpSize, int loadTimeoutMs, JsonNode business, ComputeModels compute) {
    private static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    public RtpConfig {
        if (id == null || id.isBlank() || (role != Role.prefill && role != Role.decode)
                || blockSize < 1 || deviceBlocks < 1 || reserveBlocks < 0 || reserveBlocks > deviceBlocks
                || maxSeqLen < 1 || maxBatchTokens < 0 || tpSize < 1 || loadTimeoutMs < 1)
            throw new IllegalArgumentException("Invalid RTP node configuration");
        business = business.deepCopy();
        Objects.requireNonNull(compute);
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
        var capacity = config.path("capacity");
        if (!capacity.isObject()) throw new IllegalArgumentException("Missing capacity.device_blocks");
        capacity.fieldNames().forEachRemaining(key -> {
            if (!Set.of("device_blocks", "reserve_blocks").contains(key)) throw new IllegalArgumentException("Unknown Mock capacity field: " + key);
        });
        int blocks = integer(capacity.get("device_blocks"), "capacity.device_blocks", 1, Integer.MAX_VALUE);
        int ratio = nativeInt(config, "reserve_block_ratio", 5, 0, 100);
        int reserve = capacity.has("reserve_blocks") ? integer(capacity.get("reserve_blocks"), "capacity.reserve_blocks", 0, blocks)
            : (int) Math.ceil(blocks * ratio / 100.0);
        if (nativeInt(config, "dp_size", 1, 1, Integer.MAX_VALUE) != 1 || nativeInt(config, "dp_rank", 0, 0, Integer.MAX_VALUE) != 0)
            throw new IllegalArgumentException("RTP node currently supports dp_size=1, dp_rank=0 only");
        if (nativeInt(config, "max_context_batch_size", 1, 1, Integer.MAX_VALUE) != 1)
            throw new IllegalArgumentException("Local prefill currently supports max_context_batch_size=1 only");
        var compute = ComputeModels.fromJson(root.path("environment").path("compute"));
        if ((role == Role.prefill ? compute.configuration().prefill() : compute.configuration().decode()) == null)
            throw new IllegalArgumentException("Missing compute model for " + role);
        if (!root.path("id").isTextual()) throw new IllegalArgumentException("Node id must be a string");
        return new RtpConfig(root.path("id").textValue(), role,
            nativeInt(config, "seq_size_per_block", null, 1, Integer.MAX_VALUE), blocks, reserve,
            nativeBool(config, "reuse_cache", false), nativeBool(config, "enable_device_cache", true),
            nativeBool(config, "disable_decode_first_malloc_device_reuse", false),
            nativeInt(config, "max_seq_len", null, 1, Integer.MAX_VALUE),
            nativeInt(config, "max_batch_tokens_size", 0, 0, Integer.MAX_VALUE),
            nativeInt(config, "tp_size", 1, 1, Integer.MAX_VALUE),
            loadTimeout(config), config, compute);
    }

    public ObjectNode effective() {
        ObjectNode result = JSON.createObjectNode();
        result.put("id", id).put("role", role.name()).put("seq_size_per_block", blockSize)
            .put("device_blocks", deviceBlocks).put("reserve_blocks", reserveBlocks)
            .put("reuse_cache", reuseCache).put("enable_device_cache", deviceCache)
            .put("disable_decode_first_malloc_device_reuse", disableDecodeFirstReuse)
            .put("load_cache_timeout_ms", loadTimeoutMs).put("max_seq_len", maxSeqLen).put("max_batch_tokens_size", maxBatchTokens)
            .put("dp_size", 1).put("dp_rank", 0).put("tp_size", tpSize);
        result.put("max_context_batch_size", 1).put("local_request_limit", LocalPrefill.LIMIT);
        result.set("config", business()); result.set("compute", compute.effectiveConfiguration());
        return result;
    }

    private static int loadTimeout(JsonNode config) {
        int value = nativeInt(config, "load_cache_timeout_ms", 5000, Integer.MIN_VALUE, Integer.MAX_VALUE);
        return value > 0 ? value : 5000;
    }
    private static JsonNode nativeValue(JsonNode config, String key) {
        String env = key.toUpperCase(Locale.ROOT);
        if (config.has(key) && config.has(env)) throw new IllegalArgumentException("Duplicate native setting: " + key + "/" + env);
        return config.has(key) ? config.get(key) : config.get(env);
    }
    private static int nativeInt(JsonNode config, String key, Integer fallback, int min, int max) {
        var value = nativeValue(config, key);
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
        var value = nativeValue(config, key);
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

package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache.EvictionPolicy;
import org.flexlb.mock.api.model.DecodeSpec;
import org.flexlb.mock.api.model.StoppingSpec;

/** Native RTP names/units stay native; resolved pool capacity excludes the native null block. */
public record RtpConfig(String id, Role role, int blockSize, int deviceBlocks, int reserveBlocks, EvictionPolicy evictionPolicy,
                        boolean reuseCache, boolean deviceCache, boolean disableDecodeFirstReuse, int maxSeqLen, int maxBatchTokens,
                        int maxContextBatchSize, int maxGenerateBatchSize, int tpSize, int loadTimeoutMs, JsonNode business, ComputeModel compute, DecodeStepModel decodeModel, StoppingModel stoppingModel) {
    private static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private static final ObjectMapper MODELS_JSON = JSON.copy().disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
        .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS);

    public RtpConfig {
        if (id == null || id.isBlank() || (role != Role.prefill && role != Role.decode)
                || blockSize < 1 || deviceBlocks < 1 || reserveBlocks < 0 || reserveBlocks > deviceBlocks
                || maxSeqLen < 1 || maxBatchTokens < 0 || maxContextBatchSize < 1 || maxGenerateBatchSize < 1 || tpSize < 1 || loadTimeoutMs < 1)
            throw new IllegalArgumentException("Invalid RTP node configuration");
        business = business.deepCopy();
        Objects.requireNonNull(evictionPolicy); Objects.requireNonNull(compute); Objects.requireNonNull(decodeModel); Objects.requireNonNull(stoppingModel);
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
        // Native pools reserve block zero; test_block_num counts that physical block too.
        int blocks = nativeInt(config, "test_block_num", null, 2, Integer.MAX_VALUE) - 1;
        int ratio = nativeInt(config, "reserve_block_ratio", 5, 0, 100);
        int reserve = (int) ((long) blocks * ratio / 100);
        for (String tier : List.of("enable_memory_cache", "enable_disk_cache", "enable_remote_cache"))
            if (nativeBool(config, tier, false)) throw new IllegalArgumentException(tier + " is not connected in the RTP Mock yet");
        var policyValue = nativeValue(config, "device_eviction_policy");
        EvictionPolicy policy;
        try {
            if (policyValue != null && !policyValue.isTextual()) throw new IllegalArgumentException("Expected a string");
            policy = EvictionPolicy.valueOf(policyValue == null ? "LRU" : policyValue.textValue().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException error) { throw new IllegalArgumentException("Invalid device_eviction_policy: expected lru, lfu, or fifo", error); }
        if (nativeInt(config, "dp_size", 1, 1, Integer.MAX_VALUE) != 1 || nativeInt(config, "dp_rank", 0, 0, Integer.MAX_VALUE) != 0)
            throw new IllegalArgumentException("RTP node currently supports dp_size=1, dp_rank=0 only");
        if (root.path("environment").has("compute")) throw new IllegalArgumentException("Removed environment.compute: use component.config.compute");
        var compute = ComputeModel.fromJson(config.path("compute"));
        if (role == Role.prefill && config.has("decode")) throw new IllegalArgumentException("decode model belongs in Decode config");
        if (role == Role.decode && config.has("generation")) throw new IllegalArgumentException("generation model belongs in Prefill config; D uses the transferred stop length");
        var decode = new DecodeStepModel(model(config.path("decode"), DecodeSpec.class));
        var stopping = new StoppingModel(model(config.path("generation"), StoppingSpec.class));
        if (!root.path("id").isTextual()) throw new IllegalArgumentException("Node id must be a string");
        return new RtpConfig(root.path("id").textValue(), role,
            nativeInt(config, "seq_size_per_block", null, 1, Integer.MAX_VALUE), blocks, reserve, policy,
            nativeBool(config, "reuse_cache", false), nativeBool(config, "enable_device_cache", true),
            nativeBool(config, "disable_decode_first_malloc_device_reuse", false),
            nativeInt(config, "max_seq_len", null, 1, Integer.MAX_VALUE),
            nativeInt(config, "max_batch_tokens_size", 0, 0, Integer.MAX_VALUE),
            nativeInt(config, "max_context_batch_size", 1, 1, RtpScheduler.LIMIT),
            nativeInt(config, "max_generate_batch_size", 1, 1, RtpScheduler.LIMIT),
            nativeInt(config, "tp_size", 1, 1, Integer.MAX_VALUE),
            loadTimeout(config), config, compute, decode, stopping);
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
            .put("dp_size", 1).put("dp_rank", 0).put("tp_size", tpSize);
        result.put("max_context_batch_size", maxContextBatchSize).put("max_generate_batch_size", maxGenerateBatchSize).put("local_request_limit", LocalPrefill.LIMIT);
        result.set("config", business()); result.set("compute", compute.effectiveConfiguration());
        if (role == Role.prefill) result.set("generation", MODELS_JSON.valueToTree(stoppingModel.configuration()));
        else result.set("decode", MODELS_JSON.valueToTree(decodeModel.configuration()));
        return result;
    }

    private static <T> T model(JsonNode node, Class<T> type) throws IOException {
        if (node.isMissingNode()) node = MODELS_JSON.createObjectNode();
        if (!node.isObject()) throw new IllegalArgumentException("Model config must be an object");
        for (var field : node) if (field.isNull()) throw new IllegalArgumentException("Model config cannot contain null");
        return MODELS_JSON.treeToValue(node, type);
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

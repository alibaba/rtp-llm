package org.flexlb.mock.engine.config;

import org.flexlb.mock.protocol.ConfigInput;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.forward.ComputeModel;

/**
 * Loads and validates effective configuration for an RTP mock engine.
 */
public record RtpConfig(String id, Role role, int physicalBlockSize, int deviceBlocks, int totalCacheBlocks, int reserveBlocks,
                        boolean reuseCache, boolean deviceCache, boolean disableDecodeFirstReuse, int maxSeqLen, int maxBatchTokens,
                        int maxGenerateBatchSize, int maxInitedKvStreams, int maxUncachedBatchTokens, int cpSize,
                        JsonNode business, ComputeModel compute, int imageToken, long cacheKeySeed) {
    private static final String PREFILL_OVERHEAD_MS = "mock.engine.prefill.overhead.ms";
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Map<String, String> KVCM_ENV_BINDINGS = Map.of(
        "RECO_ENABLE_VIPSERVER", "kvcm_enable_vipserver",
        "RECO_VIPSERVER_DOMAIN", "kvcm_vipserver_domain",
        "RECO_SERVER_ADDRESS", "kvcm_server_address");
    private static final Set<String> COMMON = Set.of(
        "prefill_cp_size", "image_token_id", "cache_key_hash_seed", "test_block_num", "reserve_block_ratio", "seq_size_per_block", "reuse_cache",
        "enable_device_cache", "disable_decode_first_malloc_device_reuse", "max_seq_len", "max_batch_tokens_size", "max_context_batch_size", "cp_force_single_prefill",
        "kv_cache_event_publisher_type", "kv_cache_event_manager_endpoint", "kv_cache_event_instance_id",
        "kv_cache_event_instance_group", "kv_cache_event_host_ip_port", "kv_cache_event_snapshot_interval_ms",
        "kvcm_enable_vipserver", "kvcm_vipserver_domain", "kvcm_server_address",
        "enable_remote_cache", "model_name", "concurrency_limit", "enable_memory_cache");

    public RtpConfig {
        if (id == null || id.isBlank() || (role != Role.prefill && role != Role.decode)
                || physicalBlockSize < 1 || deviceBlocks < 1 || totalCacheBlocks < deviceBlocks || reserveBlocks < 0 || reserveBlocks > deviceBlocks
                || maxSeqLen < 1 || maxBatchTokens < 0 || maxGenerateBatchSize < 1 || maxInitedKvStreams < 0 || maxUncachedBatchTokens < 0 || cpSize < 1)
            throw new IllegalArgumentException("Invalid RTP node configuration");
        business = business.deepCopy();
        Objects.requireNonNull(compute);
    }

    @Override public JsonNode business() { return business.deepCopy(); }

    public static RtpConfig fromEnvironment(String id, Role role, Map<String, String> env) throws IOException {
        var root = JSON.createObjectNode().put("id", id).put("component_role", role.name());
        var component = root.putObject("component").put("engine", "rtp");
        component.set("config", environment(env, role));
        return load(root);
    }

    public static RtpConfig load(Path path) throws IOException {
        return load(ConfigInput.read(path));
    }

    public static RtpConfig load(JsonNode root) throws IOException {
        String component = root.path("component_role").asText();
        Role role;
        try { role = Role.valueOf(component); } catch (RuntimeException error) { throw new IllegalArgumentException("Missing/invalid RTP component role", error); }
        JsonNode selected = root.path("component");
        if (!selected.path("engine").asText().equals("rtp")) throw new IllegalArgumentException("RTP node requires engine: rtp");
        var config = selected.path("config");
        if (!config.isObject()) throw new IllegalArgumentException("RTP config must be an object");
        if (config.has("capacity") || config.has("capacity.device_blocks") || config.has("capacity.reserve_blocks"))
            throw new IllegalArgumentException("Removed Mock capacity settings: use test_block_num and reserve_block_ratio");
        if (config.has("device_eviction_policy") || config.has("DEVICE_EVICTION_POLICY"))
            throw new IllegalArgumentException("Removed device_eviction_policy: Mock uses LRU");
        config = normalize(config, role);
        if (config.has(PREFILL_OVERHEAD_MS)) {
            if (role != Role.prefill) {
                throw new IllegalArgumentException(PREFILL_OVERHEAD_MS + " is only supported for Prefill");
            }
            nativeInt(config, PREFILL_OVERHEAD_MS, 0, 0, Integer.MAX_VALUE);
        }
        if (nativeBool(config, "enable_memory_cache", false) || nativeBool(config, "enable_remote_cache", false))
            throw new IllegalArgumentException("Mock RTP supports only local KV; remove Host and remote KV settings");
        nativeInt(config, "kv_cache_event_snapshot_interval_ms", 600000, 1, Integer.MAX_VALUE);
        finalizeSettings((ObjectNode) config);
        int physicalBlockSize = config.path("seq_size_per_block").intValue();
        int maxSeqLen = config.path("max_seq_len").intValue();
        int maxBatchTokens = config.path("max_batch_tokens_size").intValue();
        var scheduler = config.path("mock.scheduler.mode");
        if (!scheduler.isMissingNode() && (!scheduler.isTextual() || !Set.of("fifo", "group").contains(scheduler.textValue())))
            throw new IllegalArgumentException("Invalid mock.scheduler.mode: expected fifo or group");
        // Native pools reserve block zero; test_block_num counts that physical block too.
        int blocks = nativeInt(config, "test_block_num", null, 2, Integer.MAX_VALUE) - 1;
        int ratio = nativeInt(config, "reserve_block_ratio", 5, 0, 100);
        int reserve = (int) ((long) blocks * ratio / 100);
        int totalBlocks = nativeInt(config, "mock.cache.total.blocks", blocks, blocks, Integer.MAX_VALUE);
        int prefillCpSize = nativeInt(config, "prefill_cp_size", 1, 0, Integer.MAX_VALUE);
        int cpSize = role == Role.prefill ? Math.max(1, prefillCpSize) : 1;
        Math.multiplyExact(physicalBlockSize, cpSize);
        ((ObjectNode) config).put("prefill_cp_size", cpSize);
        var compute = ComputeModel.fromSettings(config, Map.of(
            "cpSize", (double) cpSize, "cpEnabled", cpSize > 1 ? 1.0 : 0.0));
        if (role == Role.prefill && compute.hasCustomDecodeStep())
            throw new IllegalArgumentException("Decode progress settings belong in Decode compute");
        if (!root.path("id").isTextual()) throw new IllegalArgumentException("Node id must be a string");
        return new RtpConfig(root.path("id").textValue(), role,
            physicalBlockSize, blocks, totalBlocks, reserve,
            nativeBool(config, "reuse_cache", false), nativeBool(config, "enable_device_cache", true),
            nativeBool(config, "disable_decode_first_malloc_device_reuse", false),
            maxSeqLen, maxBatchTokens,
            nativeInt(config, "concurrency_limit", 32, 1, Integer.MAX_VALUE),
            nativeInt(config, "max_inited_kv_cache_streams", 0, 0, Integer.MAX_VALUE),
            nativeInt(config, "max_batch_tokens_without_cache", 0, 0, Integer.MAX_VALUE),
            cpSize,
            config, compute, nativeInt(config, "image_token_id", org.flexlb.mock.protocol.MultimodalProtocol.IMAGE_TOKEN, 0, Integer.MAX_VALUE),
            ConfigInput.signedLong(config, "cache_key_hash_seed", 0));
    }

    public static int nativeInt(JsonNode config, String key, Integer fallback, int min, int max) {
        var value = config.get(key);
        if (value == null && fallback != null) return fallback;
        return integer(value, key, min, max);
    }

    public static boolean nativeBool(JsonNode config, String key, boolean fallback) {
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

    public boolean forceSinglePrefill() { return cpSize > 1 && business.path("cp_force_single_prefill").asBoolean(true); }

    public ObjectNode effective() {
        ObjectNode result = JSON.createObjectNode();
        result.put("id", id).put("role", role.name()).put("seq_size_per_block", physicalBlockSize)
            .put("reserve_block_ratio", nativeInt(business, "reserve_block_ratio", 5, 0, 100))
            .put("test_block_num", deviceBlocks + 1).put("device_blocks", deviceBlocks).put("reserve_blocks", reserveBlocks)
            .put("mock.cache.total.blocks", totalCacheBlocks)
            .put("reuse_cache", reuseCache).put("enable_device_cache", deviceCache)
            .put("disable_decode_first_malloc_device_reuse", disableDecodeFirstReuse)
            .put("max_seq_len", maxSeqLen).put("max_batch_tokens_size", maxBatchTokens)
            .put("max_context_batch_size", business.path("max_context_batch_size").asInt())
            .put("cp_force_single_prefill", business.path("cp_force_single_prefill").asBoolean())
            .put("prefill_cp_size", cpSize)
            .put("cache_key_block_size", tokensPerCacheKey());
        if (role == Role.prefill) {
            result.put("max_inited_kv_cache_streams", maxInitedKvStreams).put("max_batch_tokens_without_cache", maxUncachedBatchTokens); }
        if (business.has("image_token_id")) result.put("image_token_id", imageToken);
        result.put("concurrency_limit", maxGenerateBatchSize).put("cache_key_hash_seed", cacheKeySeed);
        String events = business.path("kv_cache_event_publisher_type").asText("none");
        if (!events.equals("none")) {
            for (String key : List.of("kv_cache_event_manager_endpoint","kv_cache_event_instance_id","kv_cache_event_instance_group","kv_cache_event_host_ip_port", "kvcm_enable_vipserver", "kvcm_vipserver_domain", "kvcm_server_address"))
                if (business.has(key)) result.set(key,business.get(key));
            result.put("kv_cache_event_publisher_type",events)
                .put("kv_cache_event_snapshot_interval_ms",nativeInt(business,"kv_cache_event_snapshot_interval_ms",600000,Integer.MIN_VALUE,Integer.MAX_VALUE));
            result.putObject("kvcm_capabilities").put("payload_io",false).put("hbm_events",events.equals("kvcm"));
        }
        if (role == Role.prefill) {
            result.put("mock.scheduler.mode", groupScheduling() ? "group" : "fifo")
                .put(PREFILL_OVERHEAD_MS, prefillOverheadMs());
        }
        result.setAll((ObjectNode) compute.effectiveSettings());
        return result;
    }

    public int tokensPerCacheKey() { return Math.multiplyExact(physicalBlockSize, cpSize); }

    public boolean groupScheduling() {
        return role == Role.prefill && business.path("mock.scheduler.mode").asText("fifo").equals("group");
    }

    public int prefillOverheadMs() {
        return nativeInt(business, PREFILL_OVERHEAD_MS, 0, 0, Integer.MAX_VALUE);
    }

    static ObjectNode environment(Map<String, String> env, Role role) {
        var result = ConfigInput.environment(env, environmentNames(role), mockNames());
        // Real ENV bindings differ from their native config attribute names.
        KVCM_ENV_BINDINGS.forEach((variable, setting) -> {
            String value = env.get(variable);
            if (value != null) result.put(setting, value);
        });
        return result;
    }

    static Set<String> environmentNames(Role role) {
        var names = new HashSet<>(nativeNames(role));
        // These are real internal attributes, without public server ENV bindings.
        names.remove("disable_decode_first_malloc_device_reuse");
        names.removeAll(KVCM_ENV_BINDINGS.values());
        return Set.copyOf(names);
    }

    static Set<String> nativeNames(Role role) {
        var names = new HashSet<>(COMMON);
        if (role == Role.prefill) names.addAll(Set.of("max_batch_tokens_without_cache", "max_inited_kv_cache_streams"));
        return Set.copyOf(names);
    }

    static Set<String> mockNames() {
        var names = new HashSet<>(ComputeModel.settingNames());
        names.addAll(Set.of("mock.scheduler.mode", "mock.cache.total.blocks", PREFILL_OVERHEAD_MS));
        return Set.copyOf(names);
    }

    static ObjectNode normalize(JsonNode config, Role role) {
        return ConfigInput.normalize(config, nativeNames(role), mockNames());
    }

    /** Resolve native automatic values once, before constructing any execution component. */
    private static void finalizeSettings(ObjectNode config) {
        int physicalBlockSize = nativeInt(config, "seq_size_per_block", 0, 0, Integer.MAX_VALUE);
        if (physicalBlockSize == 0) physicalBlockSize = 64; // Native ordinary-device default; no GPU probing on a Mock host.
        int maxSeqLen = nativeInt(config, "max_seq_len", 0, 0, Integer.MAX_VALUE);
        if (maxSeqLen == 0) maxSeqLen = 8192;
        int contextBatch = nativeInt(config, "max_context_batch_size", 1, 1, Integer.MAX_VALUE);
        int maxBatchTokens = nativeInt(config, "max_batch_tokens_size", 0, 0, Integer.MAX_VALUE);
        if (maxBatchTokens == 0) {
            long derived = (long) contextBatch * maxSeqLen;
            if (derived > Integer.MAX_VALUE) throw new IllegalArgumentException("Derived max_batch_tokens_size exceeds supported integer range");
            maxBatchTokens = (int) derived;
        }
        config.put("seq_size_per_block", physicalBlockSize).put("max_seq_len", maxSeqLen)
            .put("max_context_batch_size", contextBatch).put("max_batch_tokens_size", maxBatchTokens)
            .put("cp_force_single_prefill", nativeBool(config, "cp_force_single_prefill", true));
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
}

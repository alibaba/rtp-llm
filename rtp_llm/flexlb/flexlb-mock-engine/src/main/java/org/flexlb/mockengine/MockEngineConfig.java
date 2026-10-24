package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.Set;

/** Real finalized behavior and allocator observations; no performance defaults. */
final class MockEngineConfig {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final JsonNode document;

    private MockEngineConfig(JsonNode document) {
        fields(document, "schema_version", "source", "roles");
        if (integer(document, "schema_version", 1) != 1) fail("unsupported engine config schema");
        fields(document.path("source"), "kind", "revision");
        if (!Set.of("real", "synthetic").contains(document.path("source").path("kind").asText()))
            fail("source.kind must be real or synthetic");
        if (!document.path("source").path("revision").isTextual() || document.path("source").path("revision").asText().isBlank()) fail("source revision is required");
        fields(document.path("roles"), "prefill", "decode", "vit");
        if (document.path("roles").isEmpty()) fail("at least one role is required");
        this.document = document.deepCopy();
        if (document.path("roles").has("vit")) validateVit(role("vit"));
        for (String name : new String[]{"prefill", "decode"})
            if (document.path("roles").has(name)) {
                validateRole(role(name));
                JsonNode capacity = role(name).path("capacity");
                if (capacity.path("kind").asText().equals("real")
                        && !capacity.path("source_revision").asText().equals(document.path("source").path("revision").asText()))
                    fail("allocator observation and effective config use different source revisions");
                if (name.equals("decode") && role(name).path("kv_cache").path("enable_memory_cache").booleanValue())
                    fail("decode memory cache is unsupported");
            }
    }

    static MockEngineConfig load(String path) throws IOException {
        return new MockEngineConfig(JSON.readTree(Path.of(path).toFile()));
    }

    static MockEngineConfig parse(JsonNode node) { return new MockEngineConfig(node); }
    JsonNode snapshot() { return document.deepCopy(); }

    JsonNode role(String name) {
        JsonNode value = document.path("roles").path(name);
        if (!value.isObject()) fail("missing engine role: " + name);
        return value;
    }

    private static void validateRole(JsonNode role) {
        fields(role, "scheduler", "model", "kv_cache", "capacity", "provenance", "pd_sep");
        MockPdPolicy.parse(role.path("pd_sep"));
        JsonNode scheduler = role.path("scheduler"), model = role.path("model"), kv = role.path("kv_cache");
        fields(scheduler, "type", "max_generate_batch_size", "max_batch_tokens_size",
                "max_batch_tokens_without_cache", "max_inited_kv_cache_streams", "cp_force_single_prefill");
        if (!scheduler.path("type").asText().equals("fifo")) fail("only real FIFO scheduling is supported");
        integer(scheduler, "max_generate_batch_size", 1);
        integer(scheduler, "max_batch_tokens_size", 1);
        integer(scheduler, "max_batch_tokens_without_cache", 0);
        integer(scheduler, "max_inited_kv_cache_streams", 0);
        bool(scheduler, "cp_force_single_prefill");
        fields(model, "max_seq_len", "cp_size", "cp_enabled", "tp_size", "dp_size");
        integer(model, "tp_size", 1);
        if (integer(model, "dp_size", 1) != 1) fail("one logical engine supports dp_size=1 only");
        integer(model, "max_seq_len", 1);
        if (!bool(model, "cp_enabled") && integer(model, "cp_size", 1) != 1) fail("disabled CP requires cp_size=1");
        integer(model, "cp_size", 1);
        fields(kv, "seq_size_per_block", "reuse_cache", "enable_device_cache", "enable_memory_cache",
                "enable_disk_cache", "enable_remote_cache", "reserve_block_ratio", "device_eviction_policy", "memory_eviction_policy",
                "block_tree_transfer_worker_count", "block_tree_business_queue_max_size", "block_tree_transfer_queue_max_size",
                "memory_cache_max_descriptors_per_transfer_batch", "memory_cache_sync_timeout_ms",
                "block_tree_device_evict_low_watermark_ratio", "block_tree_device_evict_high_watermark_ratio",
                "block_tree_memory_evict_low_watermark_ratio", "block_tree_memory_evict_high_watermark_ratio");
        MockCacheWatermarks.parse(kv);
        MockHostTransfer.Policy.parse(kv);
        for (String name : new String[]{"device_eviction_policy", "memory_eviction_policy"})
            if (!kv.path(name).asText().equals("lru")) fail(name + " must be lru; other eviction policies are unsupported");
        integer(kv, "seq_size_per_block", 1);
        bool(kv, "reuse_cache"); bool(kv, "enable_device_cache"); bool(kv, "enable_memory_cache");
        if (bool(kv, "enable_remote_cache")) fail("remote KV cache is unsupported");
        if (bool(kv, "enable_disk_cache")) fail("disk cache is unsupported");
        JsonNode ratio = kv.path("reserve_block_ratio");
        if (!ratio.isNumber() || !Double.isFinite(ratio.doubleValue()) || ratio.doubleValue() < 0 || ratio.doubleValue() > 50)
            fail("reserve_block_ratio must be a percentage in [0,50]");
        JsonNode capacity = role.path("capacity");
        fields(capacity, "kind", "source", "source_revision", "layout", "device_blocks", "memory_blocks", "reserve_step", "reserve_blocks", "seq_size_per_block",
                "disable_decode_first_malloc_device_reuse", "enable_gpu_prefix_tree", "enable_memory_prefix_tree");
        if (!Set.of("real", "synthetic").contains(capacity.path("kind").asText()) || capacity.path("source").asText().isBlank())
            fail("capacity requires kind and observation source");
        if (!capacity.path("layout").asText().equals("single_pool"))
            fail("only single_pool capacity is supported; multi-group capacity cannot be flattened");
        integer(capacity, "device_blocks", 1); integer(capacity, "memory_blocks", 0);
        integer(capacity, "reserve_step", 0);
        if (integer(capacity, "reserve_blocks", 0) > capacity.path("device_blocks").intValue()) fail("reserve blocks exceed pool");
        if (integer(capacity, "seq_size_per_block", 1) != kv.path("seq_size_per_block").intValue())
            fail("allocator block size differs from effective KV block size");
        bool(capacity, "disable_decode_first_malloc_device_reuse");
        bool(capacity, "enable_gpu_prefix_tree"); bool(capacity, "enable_memory_prefix_tree");
        if (!kv.path("enable_memory_cache").booleanValue() && capacity.path("memory_blocks").intValue() != 0)
            fail("disabled memory cache must have zero memory blocks");
    }

    private static void validateVit(JsonNode vit) {
        fields(vit, "vit_concurrency", "vit_max_queue_size", "gpu_max_batch_size", "gpu_max_batch_images",
                "gpu_batch_wait_ms", "mm_max_queue_size", "mm_cache_gpu_max_bytes", "mm_cache_cpu_max_bytes", "mm_hash_key_cache_max_bytes", "mm_remote_cache_enable", "mm_transport_mode", "work_budget", "mm_timeout_ms", "mm_image_max_file_size_kb", "use_local_preprocess", "mm_preprocess_max_workers", "download_headers");
        if (!vit.has("work_budget")) fail("vit.work_budget is required (nullable)");
        vitWorkBudget(vit);
        if (bool(vit, "mm_remote_cache_enable")) fail("remote embedding cache is unsupported");
        if (!vit.path("mm_transport_mode").asText().equals("grpc")) fail("only inline grpc embedding transport is supported");
        for (String field : new String[]{"vit_concurrency", "gpu_max_batch_size", "mm_max_queue_size"})
            integer(vit, field, 1);
        JsonNode images = vit.path("gpu_max_batch_images");
        if (!images.isIntegralNumber() || !images.canConvertToLong() || images.longValue() < 1)
            fail("gpu_max_batch_images must be a positive integer");
        bool(vit, "use_local_preprocess"); integer(vit, "mm_preprocess_max_workers", 1);
        vitDownloadHeaders(vit);
        integer(vit, "mm_timeout_ms", 1);
        vitImageBytes(vit);
        integer(vit, "vit_max_queue_size", 0); integer(vit, "gpu_batch_wait_ms", 0);
        if (vit.path("gpu_max_batch_size").intValue() == 1
                && (vit.path("gpu_batch_wait_ms").longValue() != 0 || images.longValue() != Long.MAX_VALUE))
            fail("serial ViT requires resolved batch_wait_ms=0 and max_batch_images=sys.maxsize");
        for (String field : new String[]{"mm_cache_gpu_max_bytes", "mm_cache_cpu_max_bytes", "mm_hash_key_cache_max_bytes"}) {
            JsonNode value = vit.path(field);
            if (!value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() < 0) fail(field + " must be nonnegative bytes");
        }
    }

    static java.util.Map<String, String> vitDownloadHeaders(JsonNode vit) {
        var value = vit.path("download_headers");
        if (!value.isTextual()) throw new IllegalArgumentException("download_headers must be a JSON string");
        if (value.textValue().isEmpty()) return java.util.Map.of(
                "User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
                "Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7");
        try {
            JsonNode headers = new com.fasterxml.jackson.databind.ObjectMapper().readTree(value.textValue());
            if (!headers.isObject()) throw new IllegalArgumentException("download_headers must contain an object");
            var result = new java.util.LinkedHashMap<String, String>();
            headers.fields().forEachRemaining(entry -> {
                if (!entry.getValue().isTextual()) throw new IllegalArgumentException("download header values must be strings");
                result.put(entry.getKey(), entry.getValue().textValue());
            });
            return java.util.Map.copyOf(result);
        } catch (java.io.IOException error) { throw new IllegalArgumentException("invalid download_headers JSON", error); }
    }

    static int vitImageBytes(JsonNode vit) {
        int kilobytes = integer(vit, "mm_image_max_file_size_kb", 1);
        if (kilobytes > Integer.MAX_VALUE / 1024) fail("image byte limit exceeds JVM supported size");
        return kilobytes * 1024;
    }

    static MockVitWorkEstimate vitWorkBudget(JsonNode vit) {
        var value = vit.path("work_budget");
        if (value.isNull() || value.isMissingNode()) return null;
        String[] names = {"input_patches", "output_tokens", "estimated_workspace_bytes", "max_attention_segment", "attention_work"};
        fields(value, names);
        long[] dimensions = new long[5];
        for (int index = 0; index < names.length; index++) {
            var dimension = value.path(names[index]);
            if (!dimension.isIntegralNumber() || !dimension.canConvertToLong() || dimension.longValue() < 0)
                fail("vit.work_budget." + names[index] + " must be a nonnegative integer");
            dimensions[index] = dimension.longValue();
        }
        return new MockVitWorkEstimate(dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
    }

    int blocks(String name) { return role(name).path("capacity").path("device_blocks").intValue(); }
    int concurrency(String name) { return role(name).path("scheduler").path("max_generate_batch_size").intValue(); }

    MockPerformanceModel configure(String name, MockPerformanceModel calibration) {
        var result = calibration.forEngine();
        JsonNode role = role(name), kv = role.path("kv_cache"), model = role.path("model"), capacity = role.path("capacity");
        JsonNode scheduler = role.path("scheduler");
        result.pdPolicy = MockPdPolicy.parse(role.path("pd_sep"));
        result.hostTransferPolicy = MockHostTransfer.Policy.parse(role.path("kv_cache"));
        result.cacheWatermarks = MockCacheWatermarks.parse(role.path("kv_cache"));
        result.setBlockSize(kv.path("seq_size_per_block").intValue());
        result.prefillReuseCache = result.decodeReuseCache = kv.path("reuse_cache").booleanValue();
        result.prefillDeviceCache = kv.path("enable_device_cache").booleanValue();
        result.prefillReserveBlockRatio = result.decodeReserveBlockRatio = kv.path("reserve_block_ratio").doubleValue();
        result.prefillGpuPrefixTree = result.decodeGpuPrefixTree = capacity.path("enable_gpu_prefix_tree").booleanValue();
        result.memoryPrefixTree = capacity.path("enable_memory_prefix_tree").booleanValue();
        result.memoryCacheBlocks = name.equals("prefill") ? capacity.path("memory_blocks").intValue() : 0;
        result.decodeDeviceCache = kv.path("enable_device_cache").booleanValue();
        result.disableDecodeFirstReuse = !kv.path("enable_device_cache").booleanValue() || capacity.path("disable_decode_first_malloc_device_reuse").booleanValue();
        result.observedReserveBlocks = capacity.path("reserve_blocks").intValue();
        result.setDecodeReserveStep(capacity.path("reserve_step").intValue());
        result.setPrefillBatchPolicy(new MockPrefillBatchPolicy(scheduler.path("max_generate_batch_size").intValue(), scheduler.path("max_batch_tokens_size").longValue(), model.path("max_seq_len").longValue(), model.path("cp_size").intValue(), scheduler.path("cp_force_single_prefill").booleanValue(), scheduler.path("max_batch_tokens_without_cache").longValue(), scheduler.path("max_inited_kv_cache_streams").intValue(), model.path("cp_enabled").booleanValue()));
        return result;
    }

    static int integer(JsonNode node, String name, int minimum) {
        JsonNode value = node.path(name);
        if (!value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() < minimum)
            fail(name + " must be an integer >= " + minimum);
        return value.intValue();
    }
    static boolean bool(JsonNode node, String name) {
        if (!node.path(name).isBoolean()) fail(name + " must be boolean");
        return node.path(name).booleanValue();
    }
    static void fields(JsonNode node, String... allowed) {
        if (node == null || !node.isObject()) fail("expected config object");
        Set<String> names = Set.of(allowed);
        for (Iterator<String> it = node.fieldNames(); it.hasNext();) {
            String name = it.next();
            if (!names.contains(name)) fail("unknown engine config field: " + name);
        }
    }
    private static void fail(String message) { throw new IllegalArgumentException(message); }
}

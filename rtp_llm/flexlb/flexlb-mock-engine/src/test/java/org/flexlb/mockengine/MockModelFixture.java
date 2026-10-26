package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.io.IOException;
import java.util.List;

/** Isolated scheduler test vectors; executable configurations use MockEngineConfig. */
final class MockModelFixture {
    private static final MockPrefillBatchPolicy DEFAULT = new MockPrefillBatchPolicy(32, 1_048_576, 1_048_576, 1, false, 0, 0, false);

    static MockPrefillBatchPolicy scheduler(JsonNode node) {
        if (node.isMissingNode() || node.isNull()) return DEFAULT;
        MockEngineConfig.fields(node, "max_generate_batch_size", "max_batch_tokens_size", "max_seq_len",
                "cp_size", "cp_force_single_prefill", "max_batch_tokens_without_cache", "max_inited_kv_cache_streams", "cp_enabled");
        MockPrefillBatchPolicy policy = new MockPrefillBatchPolicy(
                node.path("max_generate_batch_size").asInt(0), node.path("max_batch_tokens_size").asLong(0),
                node.path("max_seq_len").asLong(0),
                node.path("cp_size").asInt(1), node.path("cp_force_single_prefill").asBoolean(true),
                node.path("max_batch_tokens_without_cache").asLong(0),
                node.path("max_inited_kv_cache_streams").asInt(0),
                node.path("cp_enabled").asBoolean(node.path("cp_size").asInt(1) > 1));
        if (policy.maxRequests() <= 0 || policy.maxTokens() <= 0
                || policy.maxSeqLen() <= 0 || policy.cpSize() <= 0 || policy.uncachedStop() < 0
                || policy.maxInitedKvStreams() < 0
                || (!policy.cpEnabled() && policy.cpSize() != 1)) {
            throw new IllegalArgumentException("FIFO requires positive max_generate_batch_size, "
                    + "max_batch_tokens_size, max_seq_len, cp_size and nonnegative remaining limits");
        }
        return policy;
    }

    static MockPerformanceModel load(String file, String master) throws IOException {
        var json = new ObjectMapper();
        JsonNode input = json.readTree(Path.of(file).toFile());
        ObjectNode calibration = json.createObjectNode();
        if (input.has("calibration_sha256")) calibration.set("calibration_sha256", input.get("calibration_sha256"));
        ObjectNode mock = calibration.putObject("mock");
        mock.putObject("timing").put("time_scale", input.path("sleep_scale").asDouble(1));
        ObjectNode forward = mock.putObject("forward");
        JsonNode pre = input.path("prefill"), dec = input.path("decode");
        for (String obsolete : new String[]{"max_waiting_batches", "direct_batch_size_max", "max_batch_tokens", "max_batch_requests"})
            if (pre.has(obsolete)) throw new IllegalArgumentException("removed scheduler parameter: " + obsolete);
        if (dec.has("report_queued_as_kv_allocated")) throw new IllegalArgumentException("removed decode reporting mode");
        ObjectNode timing = forward.putObject("prefill");
        for (String field : new String[]{"fixed_ms", "min_ms", "scale", "jitter", "pause"})
            if (pre.has(field)) timing.set(field, pre.get(field));
        var defaults = MockPerformanceModel.loadCalibration(json.createObjectNode());
        timing.put("expression", MockMasterConfig.readExpression(master, defaults.prefillExpressionState().get("expression").toString()));
        ObjectNode decode = forward.putObject("decode");
        for (String field : new String[]{"scale", "jitter", "pause", "step_ms_by_batch", "step_base_ms", "step_per_running_ms", "tokens_per_step", "eos"})
            if (dec.has(field)) decode.set(field, dec.get(field));
        var model = MockPerformanceModel.loadCalibration(calibration);
        model.setBlockSize(input.path("block_size").asInt(1024));
        model.setDecodeReserveStep(dec.path("reserve_step").asInt(0));
        model.prefillReuseCache = pre.path("reuse_cache").asBoolean(true);
        model.prefillDeviceCache = pre.path("enable_device_cache").asBoolean(true);
        model.decodeReuseCache = dec.path("reuse_cache").asBoolean(true);
        for (JsonNode role : List.of(pre, dec))
            if (role.has("enable_gpu_prefix_tree") && !role.get("enable_gpu_prefix_tree").isBoolean())
                throw new IllegalStateException("enable_gpu_prefix_tree must be boolean");
        if (dec.has("reuse_cache") && !dec.get("reuse_cache").isBoolean()) throw new IllegalStateException("decode.reuse_cache must be boolean");
        model.prefillGpuPrefixTree = pre.path("enable_gpu_prefix_tree").asBoolean(true);
        model.decodeGpuPrefixTree = dec.path("enable_gpu_prefix_tree").asBoolean(true);
        if (pre.has("reserve_block_ratio")) model.prefillReserveBlockRatio = pre.get("reserve_block_ratio").doubleValue();
        if (dec.has("reserve_block_ratio")) model.decodeReserveBlockRatio = dec.get("reserve_block_ratio").doubleValue();
        var memory = pre.path("memory_cache");
        for (String field : new String[]{"read_ms_per_block", "write_ms_per_block", "copy_lifecycle"})
            if (memory.has(field)) throw new IllegalStateException("unsupported memory fixture field: " + field);
        if (memory.has("enabled") && !memory.get("enabled").isBoolean()) throw new IllegalStateException("enabled must be boolean");
        if (memory.path("enabled").asBoolean(false)) {
            if (!memory.path("capacity_blocks").isIntegralNumber() || memory.path("capacity_blocks").intValue() <= 0)
                throw new IllegalStateException("memory capacity must be positive");
            model.memoryCacheBlocks = memory.path("capacity_blocks").intValue();
            if (memory.has("enable_prefix_tree") && !memory.get("enable_prefix_tree").isBoolean())
                throw new IllegalStateException("enable_prefix_tree must be boolean");
            model.memoryPrefixTree = memory.path("enable_prefix_tree").asBoolean(true);
        }
        model.setPrefillBatchPolicy(scheduler(pre.path("fifo")));
        return model;
    }
}

package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.io.IOException;
import java.util.List;

/** Isolated scheduler test vectors; executable configurations use MockEngineConfig. */
final class MockModelFixture {
    static MockPerformanceModel load(String file, String master) throws IOException {
        var json = new ObjectMapper();
        JsonNode input = json.readTree(Path.of(file).toFile());
        ObjectNode calibration = json.createObjectNode();
        for (String field : new String[]{"sleep_scale", "jitter_pct", "calibration_sha256"})
            if (input.has(field)) calibration.set(field, input.get(field));
        JsonNode pre = input.path("prefill"), dec = input.path("decode");
        ObjectNode timing = calibration.putObject("prefill");
        for (String field : new String[]{"fixed_ms", "min_ms", "scale", "noise"})
            if (pre.has(field)) timing.set(field, pre.get(field));
        // Synthetic vectors explicitly select the predictor expression used by that test.
        var defaults = MockPerformanceModel.loadCalibration(json.createObjectNode());
        timing.put("expression", MockMasterConfig.readExpression(master, defaults.prefillExpressionState().get("expression").toString()));
        ObjectNode decode = calibration.putObject("decode");
        for (String field : new String[]{"scale", "noise", "step_ms_by_batch", "step_base_ms", "step_per_running_ms", "tokens_per_step", "eos"})
            if (dec.has(field)) decode.set(field, dec.get(field));
        var model = MockPerformanceModel.loadCalibration(calibration);
        model.setBlockSize(input.path("block_size").asInt(1024));
        model.maxWaitingPrefillBatches = pre.path("max_waiting_batches").asInt(0);
        model.directBatchSizeMax = pre.path("direct_batch_size_max").asInt(32);
        model.maxBatchTokens = pre.path("max_batch_tokens").asInt(MockPerformanceModel.DEFAULT_MAX_BATCH_TOKENS);
        model.maxBatchRequests = pre.path("max_batch_requests").asInt(MockPerformanceModel.DEFAULT_MAX_BATCH_REQUESTS);
        model.reportQueuedAsKvAllocated = dec.path("report_queued_as_kv_allocated").asBoolean(false);
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
        if (pre.has("fifo") && (pre.has("direct_batch_size_max") || pre.has("max_batch_tokens") || pre.has("max_batch_requests")))
            throw new IllegalStateException("FIFO test vector cannot mix batch budgets");
        model.setPrefillBatchPolicy(MockPrefillBatchPolicy.load(pre.path("fifo")));
        return model;
    }
}

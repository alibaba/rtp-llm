package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.engine.grpc.EngineRpcService;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * L1 mock-side theoretical-value calculator (thin main, test scope).
 *
 * <p>Evaluates the exact timing entry points of {@link MockPerformanceModel}
 * — the same code the mock engine sleeps by — for every cell of an L1 grid
 * plan produced by {@code l1_grid_runner.py plan}, WITHOUT any network or
 * running engine. Keeping this in the same package (test sources) is what
 * makes the formulas impossible to drift: the reference IS the production
 * mock model, invoked directly.
 *
 * <ul>
 *   <li>prefill cells: builds {@code batch_size} identical
 *       {@link MockPerformanceModel.RequestShape}s through the REAL shape
 *       construction path — a {@code GenerateInputPB} carrying the
 *       {@code flexlb_eval:} unique-key metadata (the same JSON envelope
 *       JavaLoadClient writes) plus a {@link MockLruBlockCache} that is
 *       either empty (zero cells) or pre-populated with the cell's block
 *       keys via {@code admit} (warm cells) — then prices the batch with
 *       {@code prefillMs(List&lt;RequestShape&gt;)}.</li>
 *   <li>decode cells: {@code decodeMs(outputLen, batchSize)} (engine
 *       step-budget caliber: ceil(ol/tokensPerStep) steps) plus the
 *       first-token-excluded caliber
 *       {@code decodeSteps(max(1, ol-1)) * decodeStepDelayMs(batch)} used
 *       when comparing against client {@code total_ms - ttft_ms}.</li>
 * </ul>
 *
 * <p>CLI: {@code --plan <l1_grid_plan.json> [--performance <perf.json>]
 * [--master <master.json>] --out <l1_mock_reference.json>}. When
 * --performance/--master are omitted, minimal defaults are written to a
 * temp dir: sleep_scale=1, no scales, no jitter, no step declaration (the
 * production DSv4 fits apply — prefill expression from the master config
 * or the built-in DSv4 fit, decode 19.5 + 0.175 x running, 2.6 tok/step).
 * Pass the SAME performance/master JSON the remote mock runs with to keep
 * the reference aligned with the deployment under test.
 */
public final class L1MockReferenceMain {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static void main(String[] args) throws Exception {
        String planFile = null;
        String performanceFile = null;
        String masterConfigFile = null;
        String outFile = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--plan" -> planFile = args[++i];
                case "--performance" -> performanceFile = args[++i];
                case "--master" -> masterConfigFile = args[++i];
                case "--out" -> outFile = args[++i];
                default -> {
                    System.err.println("unknown argument: " + args[i]);
                    System.exit(2);
                }
            }
        }
        if (planFile == null || outFile == null) {
            System.err.println("usage: L1MockReferenceMain --plan <l1_grid_plan.json>"
                    + " [--performance <perf.json>] [--master <master.json>]"
                    + " --out <l1_mock_reference.json>");
            System.exit(2);
        }

        Path tempDir = Files.createTempDirectory("l1-mock-reference");
        boolean defaultPerformance = performanceFile == null;
        if (defaultPerformance) {
            performanceFile = tempDir.resolve("performance-default.json").toString();
            MAPPER.writeValue(Path.of(performanceFile).toFile(),
                    Map.of("block_size", 1024, "sleep_scale", 1.0));
        }
        boolean defaultMaster = masterConfigFile == null;
        if (defaultMaster) {
            masterConfigFile = tempDir.resolve("master-default.json").toString();
            MAPPER.writeValue(Path.of(masterConfigFile).toFile(),
                    Map.of("zone_process_setting",
                            Map.of("process_info", Map.of("envs", List.of()))));
        }

        MockPerformanceModel model = MockPerformanceModel.load(
                performanceFile, masterConfigFile);
        JsonNode plan = MAPPER.readTree(Path.of(planFile).toFile());

        ObjectNode out = MAPPER.createObjectNode();
        out.put("tool", "l1_mock_reference");
        out.put("version", 1);
        ObjectNode cfg = out.putObject("model_config");
        cfg.put("performance_file", performanceFile);
        cfg.put("performance_default", defaultPerformance);
        cfg.put("master_config_file", masterConfigFile);
        cfg.put("master_config_default", defaultMaster);
        cfg.put("block_size", model.blockSize());
        cfg.put("decode_tokens_per_step", model.tokensPerStep());

        long[] cacheTotalBlocks = {0};
        for (JsonNode cell : plan.path("cells")) {
            String axis = cell.path("axis").asText();
            if ("prefill".equals(axis)) {
                // A pool far larger than any cell's key count: warm cells
                // must never evict their own prefix while populating.
                cacheTotalBlocks[0] += cell.path("block_keys").size() + 8;
            }
        }
        int cacheBlocks = Math.max(64, (int) Math.min(cacheTotalBlocks[0], 1 << 20));

        ArrayNode cellsOut = out.putArray("cells");
        for (JsonNode cell : plan.path("cells")) {
            String axis = cell.path("axis").asText();
            ObjectNode row = cellsOut.addObject();
            row.put("grid_id", cell.path("grid_id").asText());
            row.put("axis", axis);
            row.put("input_len", cell.path("input_len").asInt());
            row.put("output_len", cell.path("output_len").asInt());
            row.put("batch_size", cell.path("batch_size").asInt());
            row.put("cache_mode", cell.path("cache_mode").asText());
            row.put("repeats", cell.path("repeats").asInt());

            if ("prefill".equals(axis)) {
                int inputLen = cell.path("input_len").asInt();
                int outputLen = Math.max(1, cell.path("output_len").asInt());
                int batchSize = cell.path("batch_size").asInt();
                String cacheMode = cell.path("cache_mode").asText();
                List<Long> blockKeys = new ArrayList<>();
                for (JsonNode key : cell.path("block_keys")) {
                    blockKeys.add(key.bigIntegerValue().longValue());
                }

                MockLruBlockCache cache = new MockLruBlockCache(cacheBlocks);
                if ("warm".equals(cacheMode) && !blockKeys.isEmpty()) {
                    cache.admit(blockKeys);
                }
                EngineRpcService.GenerateInputPB input = generateInput(
                        cell.path("grid_id").asText(), inputLen, outputLen, blockKeys);
                MockPerformanceModel.RequestShape shape = model.shape(input, cache);

                List<MockPerformanceModel.RequestShape> batch = new ArrayList<>();
                for (int i = 0; i < batchSize; i++) {
                    batch.add(shape);
                }
                long prefillMs = model.prefillMs(batch);

                row.put("hit_tokens", shape.hitTokens());
                row.put("hit_blocks", shape.hitBlocks());
                row.put("compute_tokens", Math.max(0, inputLen - shape.hitTokens()));
                row.put("mock_prefill_ms", prefillMs);
                if (prefillMs <= 0) {
                    row.put("warning", "non-positive prefill formula output");
                }
            } else if ("decode".equals(axis)) {
                int outputLen = cell.path("output_len").asInt();
                int batchSize = cell.path("batch_size").asInt();
                long stepMs = model.decodeStepDelayMs(batchSize);
                row.put("mock_decode_ms", model.decodeMs(outputLen, batchSize));
                row.put("mock_decode_steps", model.decodeSteps(outputLen));
                row.put("mock_decode_step_ms", stepMs);
                int stepsFirstExcl = model.decodeSteps(Math.max(1, outputLen - 1));
                row.put("mock_decode_steps_first_excl", stepsFirstExcl);
                row.put("mock_decode_ms_first_excl", stepsFirstExcl * stepMs);
            } else {
                row.put("warning", "unknown axis " + axis + " — skipped");
            }
        }

        Path outPath = Path.of(outFile);
        if (outPath.getParent() != null) {
            Files.createDirectories(outPath.getParent());
        }
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(outPath.toFile(), out);
        System.out.println("mock reference written: " + outPath
                + " (" + cellsOut.size() + " cells)");
    }

    /**
     * Builds the protobuf input exactly the way JavaLoadClient does: token
     * ids of the declared length, max_new_tokens, and the
     * {@code flexlb_eval:} unique key carrying input_len / output_len /
     * block_cache_keys — the metadata channel MockPerformanceModel.shape()
     * parses. Token VALUES are irrelevant to the timing formulas (only the
     * count and the declared lengths matter), so a fixed pattern suffices.
     */
    private static EngineRpcService.GenerateInputPB generateInput(
            String gridId, int inputLen, int outputLen, List<Long> blockKeys) {
        ObjectNode meta = MAPPER.createObjectNode();
        meta.put("rid", gridId);
        meta.put("input_len", inputLen);
        meta.put("output_len", outputLen);
        ArrayNode keys = meta.putArray("block_cache_keys");
        for (long key : blockKeys) {
            keys.add(key);
        }
        String uniqueKey = "flexlb_eval:" + meta;

        EngineRpcService.GenerateConfigPB.Builder config =
                EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(outputLen)
                        .setUniqueKey(uniqueKey);
        EngineRpcService.GenerateInputPB.Builder input =
                EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(1L)
                        .setGenerateConfig(config.build());
        for (int token = 0; token < inputLen; token++) {
            input.addTokenIds(token & 0x7F);
        }
        return input.build();
    }

    private L1MockReferenceMain() {
    }
}

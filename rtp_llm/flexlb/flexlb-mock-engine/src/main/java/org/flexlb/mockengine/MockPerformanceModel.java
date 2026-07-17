package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.strategy.PrefillTimeFormula;
import org.flexlb.engine.grpc.EngineRpcService;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

final class MockPerformanceModel {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final int blockSize;
    private final double sleepScale;
    private final double prefillScale;
    private final Double fixedPrefillMs;
    private final PrefillTimeFormula prefillFormula;
    private final List<DecodePoint> decodePoints;
    private final double decodeScale;

    private MockPerformanceModel(int blockSize,
                                 double sleepScale,
                                 double prefillScale,
                                 Double fixedPrefillMs,
                                 PrefillTimeFormula prefillFormula,
                                 List<DecodePoint> decodePoints,
                                 double decodeScale) {
        this.blockSize = blockSize;
        this.sleepScale = sleepScale;
        this.prefillScale = prefillScale;
        this.fixedPrefillMs = fixedPrefillMs;
        this.prefillFormula = prefillFormula;
        this.decodePoints = decodePoints;
        this.decodeScale = decodeScale;
    }

    static MockPerformanceModel load(String performanceFile, String masterConfigFile) throws IOException {
        JsonNode performance = MAPPER.readTree(Path.of(performanceFile).toFile());
        int blockSize = performance.path("block_size").asInt(1024);
        double sleepScale = performance.path("sleep_scale").asDouble(1.0);
        JsonNode prefill = performance.path("prefill");
        double prefillScale = prefill.path("scale").asDouble(1.0);
        Double fixedPrefillMs = prefill.has("fixed_ms") ? prefill.get("fixed_ms").asDouble() : null;

        String formulaSource = loadPrefillFormula(masterConfigFile);
        PrefillTimeFormula formula = formulaSource == null ? null : PrefillTimeFormula.parse(formulaSource);

        JsonNode decode = performance.path("decode");
        List<DecodePoint> points = new ArrayList<>();
        for (JsonNode pair : decode.path("step_ms_by_batch")) {
            if (pair.isArray() && pair.size() >= 2) {
                points.add(new DecodePoint(pair.get(0).asInt(), pair.get(1).asDouble()));
            }
        }
        if (points.isEmpty()) {
            for (int batch : new int[]{1, 2, 4, 8, 16, 32, 64, 128, 256}) {
                points.add(new DecodePoint(batch, 1.0));
            }
        }
        points.sort(Comparator.comparingInt(DecodePoint::batchSize));
        return new MockPerformanceModel(blockSize, sleepScale, prefillScale, fixedPrefillMs,
                formula, List.copyOf(points), decode.path("scale").asDouble(1.0));
    }

    private static String loadPrefillFormula(String masterConfigFile) throws IOException {
        JsonNode root = MAPPER.readTree(Path.of(masterConfigFile).toFile());
        JsonNode envs = root.path("zone_process_setting").path("process_info").path("envs");
        for (JsonNode item : envs) {
            if (item.isArray() && item.size() >= 2
                    && "PREFILL_TIME_FORMULA".equals(item.get(0).asText())) {
                return item.get(1).asText();
            }
        }
        return null;
    }

    RequestShape shape(EngineRpcService.GenerateInputPB input, MockLruBlockCache cache) {
        int inputLen = input.getTokenIdsCount();
        int outputLen = Math.max(1, input.getGenerateConfig().getMaxNewTokens());
        List<Long> blockKeys = new ArrayList<>();
        String uniqueKey = input.getGenerateConfig().getUniqueKey();
        if (uniqueKey.startsWith("flexlb_eval:")) {
            uniqueKey = uniqueKey.substring("flexlb_eval:".length());
        }
        if (!uniqueKey.isBlank()) {
            try {
                JsonNode meta = MAPPER.readTree(uniqueKey);
                inputLen = meta.path("input_len").asInt(inputLen);
                outputLen = meta.path("output_len").asInt(outputLen);
                for (JsonNode key : meta.path("block_cache_keys")) {
                    blockKeys.add(key.bigIntegerValue().longValue());
                }
            } catch (IOException ignored) {
                // Fall back to protobuf lengths when metadata is absent or malformed.
            }
        }
        long hitTokens = (long) cache.prefixHitBlocks(blockKeys) * blockSize;
        hitTokens = Math.min(hitTokens, inputLen);
        return new RequestShape(input, inputLen, Math.max(1, outputLen), List.copyOf(blockKeys), hitTokens);
    }

    long prefillMs(List<RequestShape> requests) {
        if (requests.isEmpty()) {
            return 0;
        }
        double latency;
        if (prefillFormula != null) {
            double[] batchVars = new double[5];
            batchVars[0] = requests.size();
            List<double[]> itemVars = new ArrayList<>(requests.size());
            for (RequestShape request : requests) {
                double[] vars = new double[5];
                vars[0] = requests.size();
                vars[1] = request.inputLen;
                vars[2] = request.hitTokens;
                vars[3] = Math.max(0, request.inputLen - request.hitTokens);
                vars[4] = request.hitTokens > 0 ? 1 : 0;
                itemVars.add(vars);
            }
            latency = prefillFormula.evaluate(batchVars, itemVars);
        } else if (fixedPrefillMs != null) {
            latency = fixedPrefillMs;
        } else {
            latency = 300.0;
        }
        return scaledMs(latency * prefillScale);
    }

    long decodeMs(int outputLen, int activeBatchSize) {
        return scaledMs(outputLen * interpolateStepMs(activeBatchSize) * decodeScale);
    }

    private double interpolateStepMs(int activeBatchSize) {
        if (activeBatchSize <= decodePoints.get(0).batchSize) {
            return decodePoints.get(0).stepMs;
        }
        DecodePoint last = decodePoints.get(decodePoints.size() - 1);
        if (activeBatchSize >= last.batchSize) {
            return last.stepMs;
        }
        for (int i = 0; i < decodePoints.size() - 1; i++) {
            DecodePoint left = decodePoints.get(i);
            DecodePoint right = decodePoints.get(i + 1);
            if (activeBatchSize <= right.batchSize) {
                double ratio = (activeBatchSize - left.batchSize)
                        / (double) (right.batchSize - left.batchSize);
                return left.stepMs + ratio * (right.stepMs - left.stepMs);
            }
        }
        return last.stepMs;
    }

    private long scaledMs(double latencyMs) {
        return Math.max(1L, Math.round(Math.max(0.0, latencyMs) * sleepScale));
    }

    int blockSize() {
        return blockSize;
    }

    record RequestShape(EngineRpcService.GenerateInputPB input,
                        int inputLen,
                        int outputLen,
                        List<Long> blockKeys,
                        long hitTokens) {
    }

    private record DecodePoint(int batchSize, double stepMs) {
    }
}

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
import java.util.concurrent.ThreadLocalRandom;

final class MockPerformanceModel {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Sentinel for "no cap" on queued prefill batches: JSON
     * "prefill.max_waiting_batches" absent or negative means unbounded queue
     * (legacy behavior). A value of 0 is the strictest cap: zero-waiting
     * fail-fast — a batch is only accepted when the engine is idle
     * (a prefill concurrency slot is free), otherwise rejected immediately.
     */
    static final int UNBOUNDED_MAX_WAITING_PREFILL_BATCHES = -1;

    private volatile int blockSize;
    private final double sleepScale;
    private final double prefillScale;
    private final Double fixedPrefillMs;
    // Floor (ms) for the final post-scale prefill sleep from JSON "prefill.min_ms".
    // Guards against sleep_scale making prefill unrealistically fast. Null signals
    // "absent in JSON → no floor".
    private final Double prefillMinMs;
    // Cap on queued (not running) prefill batches from JSON "prefill.max_waiting_batches".
    // >= 0 is enforced (0 = zero-waiting fail-fast); absent or negative = unbounded.
    private final int maxWaitingPrefillBatches;
    private final PrefillTimeFormula prefillFormula;
    private final List<DecodePoint> decodePoints;
    private final double decodeScale;
    // Fixed per-token decode latency (ms) from JSON "decode.per_token_ms".
    // When non-null, decodeMs uses outputLen * perTokenMs instead of the
    // step_ms_by_batch curve. Null signals "absent in JSON → use curve fallback".
    private final Double perTokenMs;
    private volatile double jitterPct;
    private volatile double cacheAdmissionRate;
    private volatile Double overrideFixedPrefillMs;
    private volatile Double overrideDecodeStepMs;
    // Python /set_perf compatibility: decode_scale overrides the config-file
    // decode scale (Python-compat /set_perf -> performance.decode_scale).
    private volatile Double overrideDecodeScale;

    private MockPerformanceModel(int blockSize,
                                 double sleepScale,
                                 double prefillScale,
                                 Double fixedPrefillMs,
                                 Double prefillMinMs,
                                 int maxWaitingPrefillBatches,
                                 PrefillTimeFormula prefillFormula,
                                 List<DecodePoint> decodePoints,
                                 double decodeScale,
                                 Double perTokenMs,
                                 double jitterPct,
                                 double cacheAdmissionRate) {
        this.blockSize = blockSize;
        this.sleepScale = sleepScale;
        this.prefillScale = prefillScale;
        this.fixedPrefillMs = fixedPrefillMs;
        this.prefillMinMs = prefillMinMs;
        this.maxWaitingPrefillBatches = maxWaitingPrefillBatches;
        this.prefillFormula = prefillFormula;
        this.decodePoints = decodePoints;
        this.decodeScale = decodeScale;
        this.perTokenMs = perTokenMs;
        this.jitterPct = jitterPct;
        this.cacheAdmissionRate = cacheAdmissionRate;
    }

    static MockPerformanceModel load(String performanceFile, String masterConfigFile) throws IOException {
        JsonNode performance = MAPPER.readTree(Path.of(performanceFile).toFile());
        int blockSize = performance.path("block_size").asInt(1024);
        double sleepScale = performance.path("sleep_scale").asDouble(1.0);
        JsonNode prefill = performance.path("prefill");
        double prefillScale = prefill.path("scale").asDouble(1.0);
        Double fixedPrefillMs = prefill.has("fixed_ms") ? prefill.get("fixed_ms").asDouble() : null;
        Double prefillMinMs = prefill.has("min_ms") ? prefill.get("min_ms").asDouble() : null;
        // Absent field = unbounded (backward compat for configs written before the
        // cap existed); the shipped performance JSONs set 0 (zero-waiting fail-fast).
        int maxWaitingPrefillBatches = prefill.has("max_waiting_batches")
                ? prefill.get("max_waiting_batches").asInt()
                : UNBOUNDED_MAX_WAITING_PREFILL_BATCHES;

        String formulaSource = loadPrefillFormula(masterConfigFile);
        PrefillTimeFormula formula = formulaSource == null ? null : PrefillTimeFormula.parse(formulaSource);

        JsonNode decode = performance.path("decode");
        Double perTokenMs = decode.has("per_token_ms") ? decode.get("per_token_ms").asDouble() : null;
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
        double jitterPct = performance.path("jitter_pct").asDouble(0.0);
        double cacheAdmissionRate = performance.path("cache_admission_rate").asDouble(1.0);
        return new MockPerformanceModel(blockSize, sleepScale, prefillScale, fixedPrefillMs,
                prefillMinMs, maxWaitingPrefillBatches, formula, List.copyOf(points),
                decode.path("scale").asDouble(1.0), perTokenMs, jitterPct, cacheAdmissionRate);
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
        if (overrideFixedPrefillMs != null) {
            latency = overrideFixedPrefillMs;
        } else if (prefillFormula != null) {
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
        long result = scaledMs(latency * prefillScale);
        // Clamp on the final (post-scale) value: min_ms is the actual-sleep floor.
        return prefillMinMs != null ? Math.max(result, Math.round(prefillMinMs)) : result;
    }

    void setOverrideFixedPrefillMs(Double ms) {
        this.overrideFixedPrefillMs = ms;
    }

    void setOverrideDecodeStepMs(Double ms) {
        this.overrideDecodeStepMs = ms;
    }

    /** Python /set_perf {@code decode_scale}: replace the decode latency scale. */
    void setOverrideDecodeScale(Double scale) {
        this.overrideDecodeScale = scale;
    }

    /** Python launcher {@code --block-size}: override the block size from the perf config. */
    void setBlockSize(int blockSize) {
        this.blockSize = blockSize;
    }

    /**
     * Cap on queued (not running) prefill batches per engine
     * (JSON "prefill.max_waiting_batches"). >= 0 is enforced — 0 means
     * zero-waiting fail-fast (accept only when a concurrency slot is free);
     * absent in JSON or negative means unbounded (legacy behavior).
     */
    int maxWaitingPrefillBatches() {
        return maxWaitingPrefillBatches;
    }

    void setJitterPct(double pct) {
        this.jitterPct = pct;
    }

    void setCacheAdmissionRate(double rate) {
        this.cacheAdmissionRate = rate;
    }

    long decodeMs(int outputLen, int activeBatchSize) {
        double stepMs;
        if (overrideDecodeStepMs != null) {
            // Runtime override (Python /set_perf decode_step_ms): fixed per-token semantics.
            stepMs = overrideDecodeStepMs;
        } else if (perTokenMs != null) {
            // JSON "decode.per_token_ms": fixed per-token latency (e.g. 45ms ≈ DeepSeek V3 ~22 tok/s).
            stepMs = perTokenMs;
        } else {
            // Fallback: step_ms_by_batch curve interpolation (backward compat for
            // configs/tests that only set step_ms_by_batch without per_token_ms).
            stepMs = interpolateStepMs(activeBatchSize);
        }
        double effectiveScale = overrideDecodeScale != null ? overrideDecodeScale : decodeScale;
        return scaledMs(outputLen * stepMs * effectiveScale);
    }

    boolean shouldAdmitCache() {
        if (cacheAdmissionRate >= 1.0) {
            return true;
        }
        if (cacheAdmissionRate <= 0.0) {
            return false;
        }
        return ThreadLocalRandom.current().nextDouble() < cacheAdmissionRate;
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
        double scaled = Math.max(0.0, latencyMs) * sleepScale;
        if (jitterPct > 0) {
            double factor = 1.0 + ThreadLocalRandom.current().nextDouble(-jitterPct, jitterPct);
            scaled = scaled * factor;
        }
        return Math.max(1L, Math.round(scaled));
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

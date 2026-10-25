package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.prediction.PrefillTimeFormula;
import org.flexlb.engine.grpc.EngineRpcService;

import java.io.IOException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

final class MockPerformanceModel {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    static final int DEFAULT_DECODE_RESERVE_STEP = 0;

    private record Calibration(JsonNode data, String sha256) {}

    /** The bundled calibration is shared with online_eval and marked legacy_unverified. */
    private static Calibration loadCalibration() throws IOException {
        try (var stream = MockPerformanceModel.class.getClassLoader()
                .getResourceAsStream("dsv4_l20_mock_calibration.json")) {
            if (stream == null) {
                throw new IOException("missing packaged mock calibration: dsv4_l20_mock_calibration.json");
            }
            byte[] bytes = stream.readAllBytes();
            JsonNode data = MAPPER.readTree(bytes);
            JsonNode decode = data.path("decode");
            if (data.path("schema_version").asInt() != 1
                    || !"dsv4_l20_legacy_mock".equals(data.path("id").asText())
                    || !"legacy_unverified".equals(data.path("status").asText())
                    || data.path("model").asText().isBlank()
                    || data.path("hardware").asText().isBlank()
                    || data.path("source").asText().isBlank()
                    || data.path("prefill_expression").asText().isBlank()
                    || !decode.path("step_base_ms").isNumber()
                    || !decode.path("step_per_running_ms").isNumber()
                    || !decode.path("tokens_per_step").isNumber()
                    || !Double.isFinite(decode.path("step_base_ms").asDouble())
                    || !Double.isFinite(decode.path("step_per_running_ms").asDouble())
                    || !Double.isFinite(decode.path("tokens_per_step").asDouble())
                    || decode.path("step_base_ms").asDouble() < 0
                    || decode.path("step_per_running_ms").asDouble() < 0
                    || decode.path("tokens_per_step").asDouble() <= 0) {
                throw new IOException("invalid packaged mock calibration");
            }
            try {
                String sha256 = HexFormat.of().formatHex(
                        MessageDigest.getInstance("SHA-256").digest(bytes));
                return new Calibration(data, sha256);
            } catch (NoSuchAlgorithmException e) {
                throw new IllegalStateException("SHA-256 is unavailable", e);
            }
        }
    }

    private volatile int blockSize;
    private JsonNode calibrationConfig;
    private final java.util.concurrent.atomic.AtomicLong configGeneration = new java.util.concurrent.atomic.AtomicLong();
    private MockEosModel eosModel;
    boolean nativeTokenCacheKeys;
    boolean prefillReuseCache = false;
    boolean prefillDeviceCache = true;
    boolean decodeDeviceCache = true;
    long hostLoadMs;
    MockHostTransfer.Policy hostTransferPolicy;
    MockCacheWatermarks cacheWatermarks;
    MockPdPolicy pdPolicy = MockPdPolicy.DEFAULT;
    private final double sleepScale;
    private final double prefillScale;
    // Floor (ms) for the final post-scale prefill sleep from JSON "prefill.min_ms".
    // Guards against sleep_scale making prefill unrealistically fast. Null signals
    // "absent in JSON → no floor".
    private final Double prefillMinMs;
    private final PrefillTimeFormula prefillFormula;
    private record RuntimePrefill(String expression, PrefillTimeFormula formula) {}
    private record RuntimeDecode(double stepBaseMs, double stepPerRunningMs,
                                 double tokensPerStep) {}
    private String configuredPrefillExpression = "";
    // Decode step-latency sources, exactly one active per model:
    //   - explicit step_ms_by_batch curve (decodePoints non-empty; legacy
    //     declared channel, kept for suites that price steps themselves), or
    //   - the configured linear fit stepBaseMs + stepPerRunningMs * running
    //     (decodePoints empty; coefficients come from the packaged calibration
    //     unless the performance JSON declares them).
    // The runtime override (setOverrideDecodeStepMs) beats both.
    private final List<DecodePoint> decodePoints;
    private final double stepBaseMs;
    private final double stepPerRunningMs;
    // MTP acceptance fold: tokens advanced per running stream per step
    // (JSON "decode.tokens_per_step", default from packaged calibration).
    private final double tokensPerStep;
    // Decode KV reserve-step window in tokens (JSON "decode.reserve_step",
    // default DEFAULT_DECODE_RESERVE_STEP = 0 = disabled): see the constant's
    // javadoc for the production speculative-decode anchor.
    private int decodeReserveStep;
    boolean disableDecodeFirstReuse;
    Integer observedReserveBlocks;
    boolean decodeReuseCache = false;
    boolean prefillGpuPrefixTree = true;
    boolean decodeGpuPrefixTree = true;
    Double decodeReserveBlockRatio; // Explicit percentage; null preserves legacy case rounding.
    Double prefillReserveBlockRatio; // Explicit percentage; null preserves legacy case rounding.
    private final double decodeScale;
    private volatile double jitterPct;
    // Opt-in, bounded residual noise. Keep the legacy percentage jitter for old suites.
    private NoiseSpec prefillNoise = NoiseSpec.DISABLED;
    private NoiseSpec decodeNoise = NoiseSpec.DISABLED;

    record NoiseSpec(double baseStdMs, double variancePerUnitMs2,
                     double maxStdMs, double maxAbsMs) {
        static final NoiseSpec DISABLED = new NoiseSpec(0, 0, 0, 0);

        static NoiseSpec read(JsonNode node, String path) {
            if (node.isMissingNode() || node.isNull()) return DISABLED;
            if (!node.isObject()) throw new IllegalStateException(path + " must be an object");
            double base = number(node, "base_std_ms", path);
            double slope = number(node, "variance_per_unit_ms2", path);
            double stdCap = number(node, "max_std_ms", path);
            double absCap = number(node, "max_abs_ms", path);
            if ((base > 0 || slope > 0) && (stdCap <= 0 || absCap <= 0))
                throw new IllegalStateException(path + " requires positive max_std_ms and max_abs_ms");
            return new NoiseSpec(base, slope, stdCap, absCap);
        }

        private static double number(JsonNode node, String key, String path) {
            JsonNode value = node.path(key);
            if (value.isMissingNode()) return 0;
            double number = value.asDouble(Double.NaN);
            if (!value.isNumber() || !Double.isFinite(number) || number < 0)
                throw new IllegalStateException(path + "." + key + " must be finite and nonnegative");
            return number;
        }

        double stdMs(double units) {
            return Math.min(maxStdMs, Math.sqrt(baseStdMs * baseStdMs
                    + variancePerUnitMs2 * Math.max(0, units)));
        }

        double sampleMs(double units) {
            if (maxAbsMs == 0) return 0;
            double z = Math.max(-3, Math.min(3, ThreadLocalRandom.current().nextGaussian()));
            return Math.max(-maxAbsMs, Math.min(maxAbsMs, z * stdMs(units)));
        }
    }
    // Explicit performance-JSON "prefill.fixed_ms": a declared flat prefill
    // for duration-blind suites (chaos/elastic). Null = not declared ->
    // formula-driven. This is an explicit configuration channel, NOT the
    // removed silent fallback (a missing key never invents a duration).
    private final Double configuredFixedPrefillMs;
    /** One immutable publication for all runtime performance controls. */
    private record RuntimeSettings(RuntimePrefill prefill, RuntimeDecode decode, MockEosModel eos,
                                   Double fixedPrefillMs, Double decodeStepMs, Double decodeScale) {}
    private volatile RuntimeSettings controls = new RuntimeSettings(null, null, null, null, null, null);

    private MockPrefillBatchPolicy prefillBatchPolicy;

    void setPrefillBatchPolicy(MockPrefillBatchPolicy policy) { prefillBatchPolicy = java.util.Objects.requireNonNull(policy); }
    void setDecodeReserveStep(int steps) { decodeReserveStep = steps; }

    static MockPerformanceModel loadCalibration(String file) throws IOException {
        return loadCalibration(MAPPER.readTree(Path.of(file).toFile()));
    }

    static MockPerformanceModel loadCalibration(JsonNode node) throws IOException {
        MockEngineConfig.fields(node, "sleep_scale", "jitter_pct", "prefill", "decode", "vit",
                "calibration_id", "calibration_model", "calibration_hardware", "calibration_status", "calibration_sha256");
        JsonNode prefill = node.path("prefill"), decode = node.path("decode");
        if (node.has("prefill")) MockEngineConfig.fields(prefill, "expression", "fixed_ms", "min_ms", "scale", "noise", "host_load_ms");
        if (node.has("decode")) MockEngineConfig.fields(decode, "scale", "noise", "step_ms_by_batch",
                "step_base_ms", "step_per_running_ms", "tokens_per_step", "eos");
        Calibration calibration = loadCalibration();
        if (node.has("calibration_sha256") && !calibration.sha256().equals(node.path("calibration_sha256").asText()))
            throw new IllegalStateException("calibration_sha256 disagrees with packaged mock calibration");
        JsonNode defaults = calibration.data().path("decode");
        String expression = prefill.path("expression").asText(calibration.data().path("prefill_expression").asText());
        var formula = validatePrefillExpression(expression);
        List<DecodePoint> points = new ArrayList<>();
        if (decode.has("step_ms_by_batch")) {
            if (!decode.get("step_ms_by_batch").isArray() || decode.get("step_ms_by_batch").isEmpty())
                throw new IllegalArgumentException("step_ms_by_batch must be a nonempty curve");
            for (JsonNode pair : decode.get("step_ms_by_batch")) {
                if (!pair.isArray() || pair.size() != 2 || !pair.get(0).isIntegralNumber()
                        || !pair.get(0).canConvertToInt() || pair.get(0).intValue() < 1
                        || !pair.get(1).isNumber() || !Double.isFinite(pair.get(1).doubleValue()) || pair.get(1).doubleValue() < 0)
                    throw new IllegalArgumentException("invalid decode curve point");
                points.add(new DecodePoint(pair.get(0).intValue(), pair.get(1).doubleValue()));
            }
            points.sort(Comparator.comparingInt(DecodePoint::batchSize));
            if (points.stream().map(DecodePoint::batchSize).distinct().count() != points.size())
                throw new IllegalArgumentException("duplicate decode batch size");
            if (decode.has("step_base_ms") || decode.has("step_per_running_ms"))
                throw new IllegalArgumentException("decode curve and linear coefficients are mutually exclusive");
        }
        double tokens = number(decode, "tokens_per_step", defaults.path("tokens_per_step").doubleValue());
        if (tokens == 0) throw new IllegalArgumentException("tokens_per_step must be positive");
        double jitter = number(node, "jitter_pct", 0);
        NoiseSpec prefillNoise = NoiseSpec.read(prefill.path("noise"), "prefill.noise");
        NoiseSpec decodeNoise = NoiseSpec.read(decode.path("noise"), "decode.noise");
        if (jitter > 0 && (prefillNoise.maxAbsMs() > 0 || decodeNoise.maxAbsMs() > 0))
            throw new IllegalStateException("jitter_pct and role noise are mutually exclusive");
        var model = new MockPerformanceModel(1, number(node, "sleep_scale", 1), number(prefill, "scale", 1),
                prefill.has("min_ms") ? number(prefill, "min_ms", 0) : null,
                prefill.has("fixed_ms") ? number(prefill, "fixed_ms", 0) : null,
                formula, List.copyOf(points),
                number(decode, "step_base_ms", defaults.path("step_base_ms").doubleValue()),
                number(decode, "step_per_running_ms", defaults.path("step_per_running_ms").doubleValue()),
                tokens, 0, number(decode, "scale", 1), jitter);
        model.configuredPrefillExpression = expression;
        double hostLoadMs = number(prefill, "host_load_ms", 0);
        if (hostLoadMs > Long.MAX_VALUE) throw new IllegalArgumentException("host_load_ms is too large");
        model.hostLoadMs = (long)Math.ceil(hostLoadMs);
        model.prefillNoise = prefillNoise; model.decodeNoise = decodeNoise;
        model.eosModel = MockEosModel.load(decode.path("eos"));
        model.calibrationConfig = node.deepCopy();
        return model;
    }

    private static double number(JsonNode node, String key, double fallback) {
        JsonNode value = node.get(key);
        if (value == null) return fallback;
        if (!value.isNumber() || !Double.isFinite(value.doubleValue()) || value.doubleValue() < 0)
            throw new IllegalArgumentException(key + " must be finite and nonnegative");
        return value.doubleValue();
    }

    MockPrefillBatchPolicy prefillBatchPolicy() {
        if (prefillBatchPolicy == null) throw new IllegalStateException("engine-config must supply scheduler policy");
        return prefillBatchPolicy;
    }

    private MockPerformanceModel(int blockSize,
                                 double sleepScale,
                                 double prefillScale,
                                 Double prefillMinMs,
                                 Double configuredFixedPrefillMs,
                                 PrefillTimeFormula prefillFormula,
                                 List<DecodePoint> decodePoints,
                                 double stepBaseMs,
                                 double stepPerRunningMs,
                                 double tokensPerStep,
                                 int decodeReserveStep,
                                 double decodeScale,
                                 double jitterPct) {
        this.blockSize = blockSize;
        this.sleepScale = sleepScale;
        this.prefillScale = prefillScale;
        this.prefillMinMs = prefillMinMs;
        this.configuredFixedPrefillMs = configuredFixedPrefillMs;
        this.prefillFormula = prefillFormula;
        this.decodePoints = decodePoints;
        this.stepBaseMs = stepBaseMs;
        this.stepPerRunningMs = stepPerRunningMs;
        this.tokensPerStep = tokensPerStep;
        this.decodeReserveStep = Math.max(0, decodeReserveStep);
        this.decodeScale = decodeScale;
        this.jitterPct = jitterPct;
    }

    /** Give each engine its own mutable controls, including dynamically added engines. */
    MockPerformanceModel forEngine() {
        MockPerformanceModel copy = new MockPerformanceModel(
                blockSize, sleepScale, prefillScale, prefillMinMs, configuredFixedPrefillMs,
                prefillFormula, List.copyOf(decodePoints), stepBaseMs, stepPerRunningMs,
                tokensPerStep, decodeReserveStep, decodeScale, jitterPct);
        // Explicit overrides installed before startup are part of that engine's initial settings.
        copy.calibrationConfig = calibrationConfig;
        copy.eosModel = eosModel;
        copy.controls = controls;
        copy.decodeReuseCache = decodeReuseCache;
        copy.disableDecodeFirstReuse = disableDecodeFirstReuse;
        copy.observedReserveBlocks = observedReserveBlocks;
        copy.prefillGpuPrefixTree = prefillGpuPrefixTree;
        copy.decodeGpuPrefixTree = decodeGpuPrefixTree;
        copy.memoryCacheBlocks = memoryCacheBlocks;
        copy.memoryPrefixTree = memoryPrefixTree;
        copy.decodeReserveBlockRatio = decodeReserveBlockRatio;
        copy.prefillReserveBlockRatio = prefillReserveBlockRatio;
        copy.prefillBatchPolicy = prefillBatchPolicy;
        copy.nativeTokenCacheKeys = nativeTokenCacheKeys;
        copy.prefillReuseCache = prefillReuseCache;
        copy.prefillDeviceCache = prefillDeviceCache;
        copy.decodeDeviceCache = decodeDeviceCache;
        copy.hostLoadMs = hostLoadMs;
        copy.pdPolicy = pdPolicy;
        copy.hostTransferPolicy = hostTransferPolicy;
        copy.cacheWatermarks = cacheWatermarks;
        copy.prefillNoise = prefillNoise;
        copy.decodeNoise = decodeNoise;
        copy.configuredPrefillExpression = configuredPrefillExpression;
        return copy;
    }

    RequestShape shape(EngineRpcService.GenerateInputPB input, MockLruBlockCache cache) {
        int inputLen = input.getTokenIdsCount();
        int outputLen = Math.max(1, input.getGenerateConfig().getMaxNewTokens());
        boolean explicitOutputLength = false;
        boolean explicitCacheKeys = false;
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
                explicitOutputLength = meta.has("output_len");
                explicitCacheKeys = meta.has("block_cache_keys");
                for (JsonNode key : meta.path("block_cache_keys")) {
                    blockKeys.add(key.bigIntegerValue().longValue());
                }
            } catch (IOException ignored) {
                // Fall back to protobuf lengths when metadata is absent or malformed.
            }
        }
        boolean nativeKeys = nativeTokenCacheKeys && !explicitCacheKeys && inputLen == input.getTokenIdsCount();
        if (nativeKeys) {
            // HashUtil.h / KVCacheHashUtil.cc: signed rolling Jenkins hash.
            // Publish only complete blocks; partial blocks are not reusable.
            int physicalBlockSize = blockSize;
            blockKeys.addAll(MockImageCacheKeys.routeBlocks(input, physicalBlockSize, blockSize, MockImageCacheKeys.seed(input)));
        }
        // hitBlocks carries the RAW prefix-match run length (key count) — the
        // mock key-level cache-hit counter recorded by the engine at this admission hit
        // computation point. Unlike hitTokens it is NOT clamped to inputLen, so
        // a trace whose bh keys exceed the request's own block count keeps an
        // honest requested/hit key pair.
        int hitBlocks = prefillDeviceCache && MockCachePolicy.reuse(input, nativeKeys, prefillReuseCache)
                ? cache.prefixHitBlocks(MockCachePolicy.matchKeys(blockKeys, inputLen, blockSize, nativeKeys)) : 0;
        if (nativeKeys && input.hasV41Inputs()) hitBlocks = MockImageCacheKeys.validHitBlocks(input, hitBlocks, blockSize);
        long hitTokens = (long) hitBlocks * blockSize;
        hitTokens = Math.min(hitTokens, inputLen);
        MockEosModel activeEos = controls.eos();
        outputLen = (activeEos == null ? eosModel : activeEos)
                .outputLength(input, Math.max(1, outputLen), explicitOutputLength);
        if (prefillBatchPolicy != null) {
            outputLen = input.getGenerateConfig().getMaxNewTokens() == 0 ? 0
                    : (int)Math.min(outputLen, Math.max(0, prefillBatchPolicy.maxSeqLen() - decodeReserveStep - inputLen));
        }
        return new RequestShape(input, inputLen, outputLen, List.copyOf(blockKeys),
                hitTokens, hitBlocks, nativeKeys);
    }

    synchronized void setEosModel(MockEosModel model) {
        RuntimeSettings settings = controls;
        controls = new RuntimeSettings(settings.prefill(), settings.decode(), java.util.Objects.requireNonNull(model), settings.fixedPrefillMs(), settings.decodeStepMs(), settings.decodeScale());
    }

    Map<String, Object> outputLengthState() {
        MockEosModel active = controls.eos();
        return Map.of("eos", (active == null ? eosModel : active).configuration(),
                "runtime_override", active != null);
    }

    long prefillMs(List<RequestShape> requests) {
        RuntimeSettings settings = controls;
        RuntimePrefill runtime = settings.prefill();
        if (requests.isEmpty()) {
            return 0;
        }
        double latency;
        if (runtime == null && settings.fixedPrefillMs() != null) {
            // Runtime override (Python /set_perf prefill_fixed_ms): explicit
            // test-time control, length-blind by design.
            latency = settings.fixedPrefillMs();
        } else if (runtime == null && configuredFixedPrefillMs != null) {
            // Explicit performance-JSON "prefill.fixed_ms": the declared flat
            // prefill for duration-blind suites. Declared explicitly, so it
            // wins over the formula (priority: runtime > JSON > formula).
            latency = configuredFixedPrefillMs;
        } else {
            // The only prefill source: the expression resolved in
            // loadPrefillExpression (explicit FORMULA or the packaged calibration).
            double[] batchVars = new double[10];
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
                // PrefillTimeFormula batch variables occupy slots 5..9.
                batchVars[5] += vars[1];
                batchVars[6] += vars[2];
                batchVars[7] += vars[3];
                batchVars[8] = Math.max(batchVars[8], vars[1]);
                batchVars[9] = Math.max(batchVars[9], vars[3]);
            }
            latency = (runtime == null ? prefillFormula : runtime.formula()).evaluate(batchVars, itemVars);
        }
        double computeKTokens = requests.stream()
                .mapToDouble(r -> Math.max(0, r.inputLen - r.hitTokens)).sum() / 1024.0;
        long result = scaledMs(latency * (runtime == null ? prefillScale : 1.0),
                prefillNoise, computeKTokens);
        // Clamp on the final (post-scale) value: min_ms is the actual-sleep floor.
        return prefillMinMs != null ? Math.max(result, Math.round(prefillMinMs)) : result;
    }

    static PrefillTimeFormula validatePrefillExpression(String expression) {
        if (expression == null || expression.isBlank() || expression.length() > 32768)
            throw new IllegalArgumentException("expression must contain 1..32768 characters");
        PrefillTimeFormula formula = PrefillTimeFormula.parse(expression);
        for (double[] values : List.of(new double[]{1, 512, 0, 512, 0, 512, 0, 512, 512, 512},
                new double[]{1, 512, 512, 0, 1, 512, 512, 0, 512, 0}, new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0})) {
            double result = formula.evaluateAsDouble(values, List.of(values));
            if (!Double.isFinite(result) || result < 0)
                throw new IllegalArgumentException("formula must return finite nonnegative milliseconds");
        }
        return formula;
    }

    synchronized void setPrefillExpression(String expression) {
        RuntimeSettings settings = controls;
        controls = new RuntimeSettings(new RuntimePrefill(expression, validatePrefillExpression(expression)), settings.decode(), settings.eos(), settings.fixedPrefillMs(), settings.decodeStepMs(), settings.decodeScale());
        configGeneration.incrementAndGet();
    }

    Map<String, Object> calibrationState() {
        var state = new java.util.LinkedHashMap<String, Object>();
        state.put("generation", configGeneration.get());
        state.put("initial", calibrationConfig == null ? Map.of() : calibrationConfig);
        state.put("prefill", prefillExpressionState());
        state.put("decode", decodeModelState());
        state.put("output_length", outputLengthState());
        state.put("prefill_fixed_ms", controls.fixedPrefillMs() == null ? configuredFixedPrefillMs : controls.fixedPrefillMs());
        state.put("decode_scale", effectiveDecodeScale(controls));
        return state;
    }

    Map<String, Object> prefillExpressionState() {
        RuntimeSettings settings = controls;
        RuntimePrefill runtime = settings.prefill();
        return Map.of("runtime_override", runtime != null,
                "expression", runtime == null ? configuredPrefillExpression : runtime.expression(),
                "scale", runtime == null ? prefillScale : 1.0);
    }

    synchronized void setOverrideFixedPrefillMs(Double ms) {
        if (ms == null || !Double.isFinite(ms) || ms < 0) throw new IllegalArgumentException("prefill_fixed_ms must be finite and nonnegative");
        RuntimeSettings settings = controls;
        controls = new RuntimeSettings(settings.prefill(), settings.decode(), settings.eos(), ms, settings.decodeStepMs(), settings.decodeScale());
        configGeneration.incrementAndGet();
    }

    synchronized void setOverrideDecodeStepMs(Double ms) {
        RuntimeSettings settings = controls;
        controls = new RuntimeSettings(settings.prefill(), settings.decode(), settings.eos(), settings.fixedPrefillMs(), ms, settings.decodeScale());
    }

    /** Python /set_perf {@code decode_scale}: replace the decode latency scale. */
    synchronized void setOverrideDecodeScale(Double scale) {
        if (scale == null || !Double.isFinite(scale) || scale < 0) throw new IllegalArgumentException("decode_scale must be finite and nonnegative");
        RuntimeSettings settings = controls;
        controls = new RuntimeSettings(settings.prefill(), settings.decode(), settings.eos(), settings.fixedPrefillMs(), settings.decodeStepMs(), scale);
        configGeneration.incrementAndGet();
    }

    /** Set finalized allocator geometry before engine construction. */
    void setBlockSize(int blockSize) {
        this.blockSize = blockSize;
    }

    /**
     * Total decode duration for {@code outputLen} tokens at the given running
     * batch size (external semantics unchanged: "time to produce outputLen
     * tokens"). Internally per-step: steps = ceil(outputLen / tokensPerStep)
     * (MTP fold), each step priced by {@link #decodeStepDelayMs}.
     */
    long decodeMs(int outputLen, int activeBatchSize) {
        var settings = controls;
        int steps = (int) Math.ceil(outputLen / tokensPerStep(settings));
        double step = stepMs(activeBatchSize, settings) * effectiveDecodeScale(settings);
        if (decodeNoise.maxAbsMs() == 0) return scaledMs(steps * step);
        long duration = 0;
        for (int i = 0; i < steps; i++) duration += scaledMs(step, decodeNoise, Math.max(1, activeBatchSize));
        return duration;
    }

    /**
     * MTP fold: number of decode steps needed to produce {@code outputLen}
     * tokens at the configured tokens_per_step (ceil — the final partial step
     * still costs a full step, like a real engine's last draft round).
     */
    int decodeSteps(int outputLen) {
        return (int) Math.ceil(outputLen / tokensPerStep());
    }

    /** Tokens produced per running stream per decode step (MTP acceptance fold). */
    double tokensPerStep() {
        return tokensPerStep(controls);
    }

    private double tokensPerStep(RuntimeSettings settings) {
        return settings.decode() == null ? tokensPerStep : settings.decode().tokensPerStep();
    }

    synchronized void applyOverrides(JsonNode patch) {
        MockEngineConfig.fields(patch, "engine", "port", "prefill_fixed_ms", "decode_scale");
        for (String name : List.of("prefill_fixed_ms", "decode_scale")) {
            var value = patch.get(name);
            if (value != null && (!value.isNumber() || !Double.isFinite(value.doubleValue()) || value.doubleValue() < 0))
                throw new IllegalArgumentException(name + " must be finite and nonnegative");
        }
        var settings = controls;
        controls = new RuntimeSettings(settings.prefill(), settings.decode(), settings.eos(),
                patch.has("prefill_fixed_ms") ? Double.valueOf(patch.path("prefill_fixed_ms").asDouble()) : settings.fixedPrefillMs(),
                settings.decodeStepMs(), patch.has("decode_scale") ? Double.valueOf(patch.path("decode_scale").asDouble()) : settings.decodeScale());
        configGeneration.incrementAndGet();
    }

    synchronized void setRuntimeDecode(double baseMs, double perRunningMs, double tokensPerStep) {
        if (!Double.isFinite(baseMs) || baseMs <= 0 || !Double.isFinite(perRunningMs)
                || perRunningMs < 0 || !Double.isFinite(tokensPerStep) || tokensPerStep <= 0)
            throw new IllegalArgumentException("decode coefficients must be finite; base/tokens positive and slope nonnegative");
        RuntimeSettings settings = controls;
        controls = new RuntimeSettings(settings.prefill(), new RuntimeDecode(baseMs, perRunningMs, tokensPerStep), settings.eos(), settings.fixedPrefillMs(), settings.decodeStepMs(), settings.decodeScale());
        configGeneration.incrementAndGet();
    }

    Map<String, Object> decodeModelState() {
        RuntimeSettings settings = controls;
        RuntimeDecode current = settings.decode();
        return Map.of("runtime_override", current != null,
                "step_base_ms", current == null ? stepBaseMs : current.stepBaseMs(),
                "step_per_running_ms", current == null ? stepPerRunningMs : current.stepPerRunningMs(),
                "tokens_per_step", tokensPerStep(settings), "scale", effectiveDecodeScale(settings));
    }

    /**
     * Decode KV reserve-step window in tokens (JSON "decode.reserve_step",
     * default 0 = disabled): when > 0, decode block demand prices
     * {@code ceil((seq_len + reserve_step) / spb)} — initial admission and
     * per-step growth alike (the constant look-ahead window production's
     * speculative config front-loads; see DEFAULT_DECODE_RESERVE_STEP).
     */
    int decodeReserveStep() {
        return decodeReserveStep;
    }

    /** Effective decode scale (runtime /set_perf override > JSON config). */
    private double effectiveDecodeScale(RuntimeSettings settings) {
        return settings.decodeScale() != null ? settings.decodeScale() : decodeScale;
    }

    /**
     * Raw (pre-scale) per-step decode latency at the given running batch
     * size, one source: runtime override > explicit step_ms_by_batch curve >
     * configured linear fit (stepBaseMs + stepPerRunningMs × running).
     */
    private double stepMs(int activeBatchSize, RuntimeSettings settings) {
        RuntimeDecode current = settings.decode();
        if (current != null) return current.stepBaseMs() + current.stepPerRunningMs() * activeBatchSize;
        if (settings.decodeStepMs() != null) {
            // Runtime override (Python /set_perf decode_step_ms): fixed
            // per-STEP semantics (one step emits tokens_per_step tokens).
            return settings.decodeStepMs();
        }
        if (!decodePoints.isEmpty()) {
            return interpolateStepMs(activeBatchSize);
        }
        return stepBaseMs + stepPerRunningMs * activeBatchSize;
    }

    /**
     * Per-step decode delay for the continuous-batching decode loop (production
     * FIFOScheduler semantics): the step unit WITHOUT output-length
     * multiplication, resolved with the same source priority as
     * {@link #decodeMs} (runtime override > step_ms_by_batch curve at the
     * CURRENT running batch size > configured linear fit), then scaled by
     * decode scale + sleep scale and jittered (same formula as
     * {@code scaledMs}). Returns >= 1 ms. Each step advances every running
     * stream by {@link #tokensPerStep()} tokens — the MTP fold the per-step
     * loop needs on top of the step duration.
     */
    long decodeStepDelayMs(int activeBatchSize) {
        var settings = controls;
        return scaledMs(stepMs(activeBatchSize, settings) * effectiveDecodeScale(settings),
                decodeNoise, Math.max(1, activeBatchSize));
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
        return scaledMs(latencyMs, NoiseSpec.DISABLED, 0);
    }

    private long scaledMs(double latencyMs, NoiseSpec noise, double units) {
        double scaled = Math.max(0.0, latencyMs) * sleepScale;
        if (jitterPct > 0) {
            double factor = 1.0 + ThreadLocalRandom.current().nextDouble(-jitterPct, jitterPct);
            scaled = scaled * factor;
        }
        scaled += noise.sampleMs(units);
        return Math.max(1L, Math.round(scaled));
    }

    int blockSize() {
        return blockSize;
    }

    int memoryCacheBlocks;
    boolean memoryPrefixTree = true;

    record RequestShape(EngineRpcService.GenerateInputPB input,
                        int inputLen,
                        int outputLen,
                        List<Long> blockKeys,
                        long hitTokens,
                        int hitBlocks, boolean nativeKeys, int memoryHitBlocks) {
        /** GenerateStream::batchSize(0): beam search starts with one sequence. */
        int prefillSequenceCount() {
            var config = input.getGenerateConfig();
            int maxBeams = config.getVariableNumBeamsCount() > 0
                    ? config.getVariableNumBeamsList().stream().mapToInt(Integer::intValue).max().orElse(1)
                    : config.getNumBeams();
            return maxBeams > 1 ? 1 : Math.max(config.getNumReturnSequences(), 1);
        }

        RequestShape(EngineRpcService.GenerateInputPB input, int inputLen, int outputLen,
                     List<Long> blockKeys, long hitTokens, int hitBlocks, boolean nativeKeys) {
            this(input, inputLen, outputLen, blockKeys, hitTokens, hitBlocks, nativeKeys, 0);
        }
    }

    private record DecodePoint(int batchSize, double stepMs) {
    }
}

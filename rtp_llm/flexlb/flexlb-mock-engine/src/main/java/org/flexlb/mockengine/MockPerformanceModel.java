package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.prediction.PrefillTimeFormula;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig.EstimatorType;
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
import java.util.concurrent.atomic.AtomicBoolean;

final class MockPerformanceModel {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final AtomicBoolean CALIBRATION_LOGGED = new AtomicBoolean();

    /**
     * Default cap on queued (not running) prefill batches per engine, JSON
     * "prefill.max_waiting_batches". Derivation for the recommended value 4:
     * prefill batches run FIFO, so the k-th queued batch waits k × batch_ms
     * before it starts. With a 1000 ms target latency and a prefill execution
     * of ~150 ms, the wait allowance is about 850 ms; n = 4 bounds the deepest
     * wait at 4 × 150 = 600 ms (750 ms total), leaving ~25% headroom. Rule of
     * thumb: n ≈ target_latency_ms / batch_ms − 1.
     *
     * <p>The default here is 0 (unbounded, legacy behavior): the Auto-TPM E2E
     * suites deliberately build deep prefill queues (queue-evict scenarios), so
     * the cap is opt-in via the performance JSON — the online_eval dsv4 profiles
     * set 4 explicitly for realistic backpressure.
     */
    static final int DEFAULT_MAX_WAITING_PREFILL_BATCHES = 0;

    /**
     * Default per-execution-batch prefill TOKEN budget, JSON
     * "prefill.max_batch_tokens" (engine-internal regroup, 20260903 #8).
     * The default mirrors the exact figure the mock reports to the master in
     * WorkerStatus.max_batch_tokens_size (1_048_576): one number, two uses —
     * the master clamps its decision-group token capacity against it
     * (BatcherContext prefers the worker-reported limit), the engine's own
     * regroup budget holds the same ceiling, so the two layers can never
     * disagree about what the engine will accept. Production reference:
     * FIFOScheduler.cc evaluateWaitingStreams — a token budget stops admitting
     * new streams into the running batch once cumulative cost reaches it
     * (mock caliber per the approved spec: sum(computeTokens + hitTokens),
     * i.e. the FULL logical input length INCLUDING cache hits; the first
     * member always admits, the budget binds from the second member on).
     * 0 disables the token dimension.
     */
    static final int DEFAULT_MAX_BATCH_TOKENS = 1_048_576;

    /**
     * Default per-execution-batch prefill REQUEST count cap, JSON
     * "prefill.max_batch_requests" (engine-internal regroup, 20260903 #8).
     * Default 32 keeps the same anchor as prefill.direct_batch_size_max
     * ("matching the master FIXED_WINDOW maxRequests") — the production
     * max_generate_batch_size sequence-count constraint (FIFOScheduler.cc
     * evaluateRunningBatch, inclusive: running + admitted + 1 <= cap)
     * surfacing at the engine's own admission layer. 0 disables the
     * request-count dimension.
     *
     * <p>Both dimensions at 0 = regroup fully OFF: master-composed batches
     * execute verbatim (the legacy behavior, reproducible on demand).
     */
    static final int DEFAULT_MAX_BATCH_REQUESTS = 32;

    /**
     * Default decode KV reserve-step window in TOKENS, JSON
     * "decode.reserve_step" (0 = disabled, the non-speculative production
     * default). Production reference (Zola forensics): the speculative
     * decode config sets {@code reserve_step_ = gen_num_per_circle + 1}
     * (tokens proposed per step + 1, typically a single-digit number) — a
     * CONSTANT look-ahead window front-loaded at initKVBlock: the initial
     * allocation prices {@code ceil((seq_len + reserve_step) / spb)} blocks,
     * and the front-loaded part is CONSUMED as seq_len grows (incrKVBlock may
     * spend reserved blocks), adding one block only at each block boundary —
     * constant front-load, never an accumulating append. MTP-style
     * speculative stress profiles set it to {@code tokens_per_step + 1}; the
     * default 0 keeps every existing behavior bit-identical.
     */
    static final int DEFAULT_DECODE_RESERVE_STEP = 0;

    private record Calibration(JsonNode data, String sha256) {}

    /** The file is packaged from the same audited calibration read by online_eval. */
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
    private MockEosModel eosModel;
    private volatile MockEosModel runtimeEosModel;
    boolean nativeTokenCacheKeys;
    private final double sleepScale;
    private final double prefillScale;
    // Floor (ms) for the final post-scale prefill sleep from JSON "prefill.min_ms".
    // Guards against sleep_scale making prefill unrealistically fast. Null signals
    // "absent in JSON → no floor".
    private final Double prefillMinMs;
    // Cap on queued (not running) prefill batches from JSON "prefill.max_waiting_batches".
    // <= 0 disables the cap; defaults to DEFAULT_MAX_WAITING_PREFILL_BATCHES when absent.
    private final int maxWaitingPrefillBatches;
    // Cap on the number of requests coalesced into ONE prefill batch on the direct
    // (generate_stream / NON_BATCH) path, JSON "prefill.direct_batch_size_max"
    // (default 32, matching the master FIXED_WINDOW maxRequests). Production
    // engines run continuous batching on the prefill side, so per-engine drain
    // scales with batch size instead of being capped at 1 request per batch —
    // without coalescing the mock's direct-path drain rate is ~batch_ms per
    // SINGLE request, several times below production. 1 restores the legacy
    // one-request-per-batch behaviour.
    private final int directBatchSizeMax;
    // Engine-internal regroup budget (20260903 #8): per-execution-batch token
    // ceiling and request-count cap. Both 0 = regroup off (legacy verbatim
    // master batches). See the DEFAULT_* constants above for the anchors.
    private final int maxBatchTokens;
    private final int maxBatchRequests;
    private final PrefillTimeFormula prefillFormula;
    private record RuntimePrefill(String expression, PrefillTimeFormula formula) {}
    private volatile RuntimePrefill runtimePrefill;
    private record RuntimeDecode(double stepBaseMs, double stepPerRunningMs,
                                 double tokensPerStep) {}
    private volatile RuntimeDecode runtimeDecode;
    private String configuredPrefillExpression = "";
    // Decode step-latency sources, exactly one active per model:
    //   - explicit step_ms_by_batch curve (decodePoints non-empty; legacy
    //     declared channel, kept for suites that price steps themselves), or
    //   - the linear production fit stepBaseMs + stepPerRunningMs * running
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
    private final int decodeReserveStep;
    boolean decodeReuseCache = true;
    boolean prefillGpuPrefixTree = true;
    boolean decodeGpuPrefixTree = true;
    Double decodeReserveBlockRatio; // Explicit percentage; null preserves legacy case rounding.
    Double prefillReserveBlockRatio; // Explicit percentage; null preserves legacy case rounding.
    private final double decodeScale;
    // Opt-in accepted-layer visibility window, JSON
    // "decode.report_queued_as_kv_allocated" (default false = current
    // behavior, zero change). When true, decode requests parked in the
    // pending queue (admitted, not yet running) are reported in WorkerStatus
    // as TASK_PHASE_KV_ALLOCATED instead of TASK_PHASE_RUNNING — mirroring a
    // real engine where KV_ALLOCATED is exactly "KV reserved, not running
    // yet". The decode hard concurrency gate (park overflow in the engine-side
    // waiting queue) is unconditional and needs no switch.
    private final boolean reportQueuedAsKvAllocated;
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
    private volatile Double overrideFixedPrefillMs;
    private volatile Double overrideDecodeStepMs;
    // Python /set_perf compatibility: decode_scale overrides the config-file
    // decode scale (Python-compat /set_perf -> performance.decode_scale).
    private volatile Double overrideDecodeScale;
    // Python /set_perf compatibility: max_waiting_batches overrides the
    // config-file prefill waiting-queue cap (Python-compat /set_perf ->
    // performance prefill.max_waiting_batches). Null = not overridden;
    // the override value follows the same semantics as the JSON field
    // (0 = unbounded, > 0 = cap on queued prefill batches).
    private volatile Integer overrideMaxWaitingPrefillBatches;
    private MockPrefillBatchPolicy prefillBatchPolicy;

    MockPrefillBatchPolicy prefillBatchPolicy() { return prefillBatchPolicy; }

    private MockPerformanceModel(int blockSize,
                                 double sleepScale,
                                 double prefillScale,
                                 Double prefillMinMs,
                                 Double configuredFixedPrefillMs,
                                 int maxWaitingPrefillBatches,
                                 int directBatchSizeMax,
                                 int maxBatchTokens,
                                 int maxBatchRequests,
                                 PrefillTimeFormula prefillFormula,
                                 List<DecodePoint> decodePoints,
                                 double stepBaseMs,
                                 double stepPerRunningMs,
                                 double tokensPerStep,
                                 int decodeReserveStep,
                                 double decodeScale,
                                 boolean reportQueuedAsKvAllocated,
                                 double jitterPct) {
        this.blockSize = blockSize;
        this.sleepScale = sleepScale;
        this.prefillScale = prefillScale;
        this.prefillMinMs = prefillMinMs;
        this.configuredFixedPrefillMs = configuredFixedPrefillMs;
        this.maxWaitingPrefillBatches = maxWaitingPrefillBatches;
        this.directBatchSizeMax = Math.max(1, directBatchSizeMax);
        this.maxBatchTokens = maxBatchTokens;
        this.maxBatchRequests = maxBatchRequests;
        this.prefillFormula = prefillFormula;
        this.decodePoints = decodePoints;
        this.stepBaseMs = stepBaseMs;
        this.stepPerRunningMs = stepPerRunningMs;
        this.tokensPerStep = tokensPerStep;
        this.decodeReserveStep = Math.max(0, decodeReserveStep);
        this.decodeScale = decodeScale;
        this.reportQueuedAsKvAllocated = reportQueuedAsKvAllocated;
        this.jitterPct = jitterPct;
    }

    /** Give each engine its own mutable controls, including dynamically added engines. */
    MockPerformanceModel forEngine() {
        MockPerformanceModel copy = new MockPerformanceModel(
                blockSize, sleepScale, prefillScale, prefillMinMs, configuredFixedPrefillMs,
                maxWaitingPrefillBatches, directBatchSizeMax, maxBatchTokens, maxBatchRequests,
                prefillFormula, List.copyOf(decodePoints), stepBaseMs, stepPerRunningMs,
                tokensPerStep, decodeReserveStep, decodeScale, reportQueuedAsKvAllocated, jitterPct);
        // Explicit overrides installed before startup are part of that engine's initial settings.
        copy.eosModel = eosModel;
        copy.runtimeEosModel = runtimeEosModel;
        copy.decodeReuseCache = decodeReuseCache;
        copy.prefillGpuPrefixTree = prefillGpuPrefixTree;
        copy.decodeGpuPrefixTree = decodeGpuPrefixTree;
        copy.memoryCacheBlocks = memoryCacheBlocks;
        copy.memoryCopyLifecycle = memoryCopyLifecycle;
        copy.memoryPrefixTree = memoryPrefixTree;
        copy.decodeReserveBlockRatio = decodeReserveBlockRatio;
        copy.prefillReserveBlockRatio = prefillReserveBlockRatio;
        copy.prefillBatchPolicy = prefillBatchPolicy;
        copy.nativeTokenCacheKeys = nativeTokenCacheKeys;
        copy.overrideFixedPrefillMs = overrideFixedPrefillMs;
        copy.runtimePrefill = runtimePrefill;
        copy.prefillNoise = prefillNoise;
        copy.decodeNoise = decodeNoise;
        copy.configuredPrefillExpression = configuredPrefillExpression;
        copy.overrideDecodeStepMs = overrideDecodeStepMs;
        copy.overrideDecodeScale = overrideDecodeScale;
        copy.overrideMaxWaitingPrefillBatches = overrideMaxWaitingPrefillBatches;
        return copy;
    }

    static MockPerformanceModel load(String performanceFile, String masterConfigFile) throws IOException {
        Calibration calibration = loadCalibration();
        JsonNode defaultDecode = calibration.data().path("decode");
        JsonNode performance = MAPPER.readTree(Path.of(performanceFile).toFile());
        if (performance.has("calibration_sha256")
                && !calibration.sha256().equals(performance.path("calibration_sha256").asText())) {
            throw new IllegalStateException("Performance JSON '" + performanceFile
                    + "': calibration_sha256 disagrees with packaged mock calibration");
        }
        if (CALIBRATION_LOGGED.compareAndSet(false, true)) {
            System.out.printf("mock calibration available: id=%s model=%s hardware=%s status=%s sha256=%s%n",
                    calibration.data().path("id").asText(),
                    calibration.data().path("model").asText(),
                    calibration.data().path("hardware").asText(),
                    calibration.data().path("status").asText(), calibration.sha256());
        }
        int blockSize = performance.path("block_size").asInt(1024);
        double sleepScale = performance.path("sleep_scale").asDouble(1.0);
        JsonNode prefill = performance.path("prefill");
        double prefillScale = prefill.path("scale").asDouble(1.0);
        // "fixed_ms" is an explicit opt-in for duration-blind suites
        // (chaos/elastic): when the JSON declares it, mock prefill is flat.
        // Absent (the normal path) -> formula-driven, keeping mock execution
        // time and master routing predictions on one expression. What was
        // removed is the SILENT fallback, not this explicit channel.
        Double prefillFixedMs = prefill.has("fixed_ms") ? prefill.get("fixed_ms").asDouble() : null;
        Double prefillMinMs = prefill.has("min_ms") ? prefill.get("min_ms").asDouble() : null;
        int maxWaitingPrefillBatches = prefill.path("max_waiting_batches")
                .asInt(DEFAULT_MAX_WAITING_PREFILL_BATCHES);
        int directBatchSizeMax = prefill.path("direct_batch_size_max").asInt(32);
        int maxBatchTokens = prefill.path("max_batch_tokens")
                .asInt(DEFAULT_MAX_BATCH_TOKENS);
        int maxBatchRequests = prefill.path("max_batch_requests")
                .asInt(DEFAULT_MAX_BATCH_REQUESTS);

        String prefillExpression = loadPrefillExpression(masterConfigFile,
                calibration.data().path("prefill_expression").asText());
        PrefillTimeFormula formula = PrefillTimeFormula.parse(prefillExpression);

        JsonNode decode = performance.path("decode");
        // per_token_ms is REMOVED (task #69, wrong-version-deleted-clean rule):
        // it was a fixed per-token latency (V3-era no-MTP single-stream caliber)
        // that overstated low-batch decode ~5.5x and full-batch ~2.8x versus
        // production. Fail fast with a migration hint instead of silently
        // reinterpreting it.
        if (decode.has("per_token_ms")) {
            throw new IllegalStateException("Performance JSON '" + performanceFile
                    + "': decode.per_token_ms is removed — decode is now priced per STEP"
                    + ". Remove per_token_ms and declare decode step timing in the"
                    + " performance file, or use the packaged mock calibration.");
        }
        boolean reportQueuedAsKvAllocated =
                decode.path("report_queued_as_kv_allocated").asBoolean(false);
        List<DecodePoint> points = new ArrayList<>();
        for (JsonNode pair : decode.path("step_ms_by_batch")) {
            if (pair.isArray() && pair.size() >= 2) {
                points.add(new DecodePoint(pair.get(0).asInt(), pair.get(1).asDouble()));
            }
        }
        boolean hasLinearCoeffs = decode.has("step_base_ms") || decode.has("step_per_running_ms");
        if (!points.isEmpty() && hasLinearCoeffs) {
            // Two explicit step-latency declarations are a config conflict;
            // picking one silently would violate the least-surprise rule.
            throw new IllegalStateException("Performance JSON '" + performanceFile
                    + "': decode.step_ms_by_batch and decode.step_base_ms/step_per_running_ms"
                    + " are mutually exclusive — declare exactly one step-latency source.");
        }
        points.sort(Comparator.comparingInt(DecodePoint::batchSize));
        // A missing decode declaration uses the identified calibration packed
        // into this jar. Explicit curves or coefficients override it.
        double stepBaseMs = decode.path("step_base_ms")
                .asDouble(defaultDecode.path("step_base_ms").asDouble());
        double stepPerRunningMs = decode.path("step_per_running_ms")
                .asDouble(defaultDecode.path("step_per_running_ms").asDouble());
        double tokensPerStep = decode.path("tokens_per_step")
                .asDouble(defaultDecode.path("tokens_per_step").asDouble());
        if (tokensPerStep <= 0) {
            throw new IllegalStateException("Performance JSON '" + performanceFile
                    + "': decode.tokens_per_step must be > 0 (got " + tokensPerStep + ")");
        }
        int decodeReserveStep = decode.path("reserve_step").asInt(DEFAULT_DECODE_RESERVE_STEP);
        if (decodeReserveStep < 0) {
            throw new IllegalStateException("Performance JSON '" + performanceFile
                    + "': decode.reserve_step must be >= 0 (got " + decodeReserveStep + ")");
        }
        double jitterPct = performance.path("jitter_pct").asDouble(0.0);
        NoiseSpec prefillNoise = NoiseSpec.read(prefill.path("noise"), "prefill.noise");
        NoiseSpec decodeNoise = NoiseSpec.read(decode.path("noise"), "decode.noise");
        if (jitterPct > 0 && (prefillNoise.maxAbsMs() > 0 || decodeNoise.maxAbsMs() > 0))
            throw new IllegalStateException("jitter_pct and role noise are mutually exclusive");
        MockPerformanceModel model = new MockPerformanceModel(blockSize, sleepScale, prefillScale,
                prefillMinMs, prefillFixedMs, maxWaitingPrefillBatches, directBatchSizeMax,
                maxBatchTokens, maxBatchRequests, formula,
                List.copyOf(points), stepBaseMs, stepPerRunningMs, tokensPerStep,
                decodeReserveStep,
                decode.path("scale").asDouble(1.0),
                reportQueuedAsKvAllocated, jitterPct);
        model.prefillNoise = prefillNoise;
        model.decodeNoise = decodeNoise;
        for (var role : List.of(prefill, decode)) {
            if (role.has("enable_gpu_prefix_tree") && !role.get("enable_gpu_prefix_tree").isBoolean())
                throw new IllegalStateException("enable_gpu_prefix_tree must be boolean");
        }
        model.configuredPrefillExpression = prefillExpression;
        model.prefillGpuPrefixTree = prefill.path("enable_gpu_prefix_tree").asBoolean(true);
        model.decodeGpuPrefixTree = decode.path("enable_gpu_prefix_tree").asBoolean(true);
        if (decode.has("reuse_cache")) {
            if (!decode.get("reuse_cache").isBoolean()) {
                throw new IllegalStateException("decode.reuse_cache must be boolean");
            }
            model.decodeReuseCache = decode.get("reuse_cache").booleanValue();
        }
        if (decode.has("reserve_block_ratio")) {
            JsonNode ratio = decode.get("reserve_block_ratio");
            double value = ratio.asDouble(Double.NaN);
            if (!ratio.isNumber() || !Double.isFinite(value) || value < 0 || value > 50) {
                throw new IllegalStateException("decode.reserve_block_ratio must be a percentage in [0, 50]");
            }
            model.decodeReserveBlockRatio = value;
        }
        if (prefill.has("reserve_block_ratio")) {
            JsonNode ratio = prefill.get("reserve_block_ratio");
            double value = ratio.asDouble(Double.NaN);
            if (!ratio.isNumber() || !Double.isFinite(value) || value < 0 || value > 50) {
                throw new IllegalStateException("prefill.reserve_block_ratio must be a percentage in [0, 50]");
            }
            model.prefillReserveBlockRatio = value;
        }
        JsonNode memory = prefill.path("memory_cache");
        if (memory.has("enabled") && !memory.get("enabled").isBoolean())
            throw new IllegalStateException("prefill.memory_cache.enabled must be boolean");
        if (memory.path("enabled").asBoolean(false)) {
            JsonNode blocks = memory.path("capacity_blocks");
            if (!blocks.isIntegralNumber() || !blocks.canConvertToInt() || blocks.asInt() <= 0)
                throw new IllegalStateException("prefill.memory_cache.capacity_blocks must be a positive integer");
            model.memoryCacheBlocks = blocks.asInt();
            if (memory.has("enable_prefix_tree") && !memory.get("enable_prefix_tree").isBoolean())
                throw new IllegalStateException("prefill.memory_cache.enable_prefix_tree must be boolean");
            model.memoryPrefixTree = memory.path("enable_prefix_tree").asBoolean(true);
            // Legacy read/write latency settings are ignored: copies are instantaneous.
            if (memory.has("copy_lifecycle") && !memory.get("copy_lifecycle").isBoolean())
                throw new IllegalStateException("prefill.memory_cache.copy_lifecycle must be boolean");
            model.memoryCopyLifecycle = memory.path("copy_lifecycle").asBoolean(false);

        }
        model.eosModel = MockEosModel.load(decode.path("eos"));
        model.prefillBatchPolicy = MockPrefillBatchPolicy.load(prefill.path("fifo"));
        return model;
    }

    /**
     * Resolve the prefill duration formula — exactly one source, never a
     * silent hard-coded fallback:
     * <ol>
     *   <li>an explicit FORMULA estimator in the master config's FLEXLB_CONFIG
     *       (blank expression = misconfiguration, fail fast);</li>
     *   <li>otherwise the packaged, legacy-unverified DSv4 mock calibration.
     *       It is a static approximation for LEARNING, which this mock cannot
     *       replay.</li>
     * </ol>
     * Explicit FORMULA keeps mock execution time and master predictions on
     * the same expression; the legacy silent fixed_ms / 300 ms fallbacks are gone
     * (an explicit performance-JSON "prefill.fixed_ms" declaration still
     * wins over the formula — see prefillMs).
     */
    private static String loadPrefillExpression(String masterConfigFile,
                                               String defaultExpression) throws IOException {
        JsonNode root = MAPPER.readTree(Path.of(masterConfigFile).toFile());
        JsonNode envs = root.path("zone_process_setting").path("process_info").path("envs");
        for (JsonNode item : envs) {
            if (item.isArray() && item.size() >= 2
                    && "FLEXLB_CONFIG".equals(item.get(0).asText())) {
                FlexlbConfig config = ConfigService.parse(item.get(1).asText());
                var estimator = config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator();
                if (estimator != null
                        && estimator.getType() == EstimatorType.FORMULA) {
                    String expression = estimator.getExpression();
                    if (expression == null || expression.isBlank()) {
                        throw new IllegalStateException("Master config " + masterConfigFile
                                + ": router.roles.prefill.executionTimeEstimator is FORMULA"
                                + " with a blank expression — set the expression explicitly or"
                                + " omit the estimator to use the packaged mock calibration");
                    }
                    return expression;
                }
                break;  // LEARNING uses the file-backed mock approximation.
            }
        }
        return defaultExpression;
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
            long hash = 0;
            for (int i = 0; i < inputLen; i++) {
                hash ^= (long) input.getTokenIds(i) + 0x9e3779b97f4a7c15L
                        + (hash << 12) + (hash >> 32);
                if ((i + 1) % blockSize == 0) blockKeys.add(hash);
            }
        }
        // hitBlocks carries the RAW prefix-match run length (key count) — the
        // mock key-level cache-hit counter recorded by the engine at this admission hit
        // computation point. Unlike hitTokens it is NOT clamped to inputLen, so
        // a trace whose bh keys exceed the request's own block count keeps an
        // honest requested/hit key pair.
        int hitBlocks = cache.prefixHitBlocks(blockKeys);
        long hitTokens = (long) hitBlocks * blockSize;
        hitTokens = Math.min(hitTokens, inputLen);
        MockEosModel activeEos = runtimeEosModel;
        outputLen = (activeEos == null ? eosModel : activeEos)
                .outputLength(input, Math.max(1, outputLen), explicitOutputLength);
        return new RequestShape(input, inputLen, outputLen, List.copyOf(blockKeys),
                hitTokens, hitBlocks, nativeKeys);
    }

    void setEosModel(MockEosModel model) {
        runtimeEosModel = java.util.Objects.requireNonNull(model);
    }

    Map<String, Object> outputLengthState() {
        MockEosModel active = runtimeEosModel;
        return Map.of("eos", (active == null ? eosModel : active).configuration(),
                "runtime_override", active != null);
    }

    long prefillMs(List<RequestShape> requests) {
        RuntimePrefill runtime = runtimePrefill;
        if (requests.isEmpty()) {
            return 0;
        }
        double latency;
        if (runtime == null && overrideFixedPrefillMs != null) {
            // Runtime override (Python /set_perf prefill_fixed_ms): explicit
            // test-time control, length-blind by design.
            latency = overrideFixedPrefillMs;
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

    void setPrefillExpression(String expression) {
        runtimePrefill = new RuntimePrefill(expression, validatePrefillExpression(expression));
    }

    Map<String, Object> prefillExpressionState() {
        RuntimePrefill runtime = runtimePrefill;
        return Map.of("runtime_override", runtime != null,
                "expression", runtime == null ? configuredPrefillExpression : runtime.expression(),
                "scale", runtime == null ? prefillScale : 1.0);
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

    /**
     * Python /set_perf {@code max_waiting_batches}: replace the prefill
     * waiting-queue cap (same semantics as the JSON field: 0 = unbounded,
     * > 0 = cap on queued batches). Null restores the JSON-configured value.
     */
    void setOverrideMaxWaitingPrefillBatches(Integer cap) {
        this.overrideMaxWaitingPrefillBatches = cap;
    }

    /** Python launcher {@code --block-size}: override the block size from the perf config. */
    void setBlockSize(int blockSize) {
        this.blockSize = blockSize;
    }

    /**
     * Cap on queued (not running) prefill batches per engine
     * (JSON "prefill.max_waiting_batches", default 0 = unbounded).
     * Runtime /set_perf override (max_waiting_batches) beats the JSON value,
     * same priority chain as prefillMs (runtime > JSON).
     */
    int maxWaitingPrefillBatches() {
        return overrideMaxWaitingPrefillBatches != null
                ? overrideMaxWaitingPrefillBatches : maxWaitingPrefillBatches;
    }

    /**
     * Cap on requests coalesced into one direct-path prefill batch
     * (JSON "prefill.direct_batch_size_max", default 32, minimum 1).
     */
    int directBatchSizeMax() {
        return directBatchSizeMax;
    }

    /**
     * Per-execution-batch prefill token budget for the engine-internal
     * regroup (JSON "prefill.max_batch_tokens", default
     * DEFAULT_MAX_BATCH_TOKENS; 0 = dimension disabled). Caliber:
     * sum(computeTokens + hitTokens) over batch members.
     */
    int maxBatchTokens() {
        return maxBatchTokens;
    }

    /**
     * Per-execution-batch prefill request-count cap for the engine-internal
     * regroup (JSON "prefill.max_batch_requests", default
     * DEFAULT_MAX_BATCH_REQUESTS; 0 = dimension disabled).
     */
    int maxBatchRequests() {
        return maxBatchRequests;
    }

    /**
     * Opt-in accepted-layer window (performance JSON
     * "decode.report_queued_as_kv_allocated", default false): report queued
     * decode requests as TASK_PHASE_KV_ALLOCATED in WorkerStatus.
     */
    boolean reportQueuedAsKvAllocated() {
        return reportQueuedAsKvAllocated;
    }

    /**
     * Total decode duration for {@code outputLen} tokens at the given running
     * batch size (external semantics unchanged: "time to produce outputLen
     * tokens"). Internally per-step: steps = ceil(outputLen / tokensPerStep)
     * (MTP fold), each step priced by {@link #decodeStepDelayMs}.
     */
    long decodeMs(int outputLen, int activeBatchSize) {
        int steps = decodeSteps(outputLen);
        if (decodeNoise.maxAbsMs() == 0)
            return scaledMs(steps * stepMs(activeBatchSize) * effectiveDecodeScale());
        long duration = 0;
        for (int i = 0; i < steps; i++) duration += decodeStepDelayMs(activeBatchSize);
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
        RuntimeDecode current = runtimeDecode;
        return current == null ? tokensPerStep : current.tokensPerStep();
    }

    void setRuntimeDecode(double baseMs, double perRunningMs, double tokensPerStep) {
        if (!Double.isFinite(baseMs) || baseMs <= 0 || !Double.isFinite(perRunningMs)
                || perRunningMs < 0 || !Double.isFinite(tokensPerStep) || tokensPerStep <= 0)
            throw new IllegalArgumentException("decode coefficients must be finite; base/tokens positive and slope nonnegative");
        runtimeDecode = new RuntimeDecode(baseMs, perRunningMs, tokensPerStep);
    }

    Map<String, Object> decodeModelState() {
        RuntimeDecode current = runtimeDecode;
        return Map.of("runtime_override", current != null,
                "step_base_ms", current == null ? stepBaseMs : current.stepBaseMs(),
                "step_per_running_ms", current == null ? stepPerRunningMs : current.stepPerRunningMs(),
                "tokens_per_step", tokensPerStep(), "scale", effectiveDecodeScale());
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
    private double effectiveDecodeScale() {
        return overrideDecodeScale != null ? overrideDecodeScale : decodeScale;
    }

    /**
     * Raw (pre-scale) per-step decode latency at the given running batch
     * size, one source: runtime override > explicit step_ms_by_batch curve >
     * linear production fit (stepBaseMs + stepPerRunningMs × running).
     */
    private double stepMs(int activeBatchSize) {
        RuntimeDecode current = runtimeDecode;
        if (current != null) return current.stepBaseMs() + current.stepPerRunningMs() * activeBatchSize;
        if (overrideDecodeStepMs != null) {
            // Runtime override (Python /set_perf decode_step_ms): fixed
            // per-STEP semantics (one step emits tokens_per_step tokens).
            return overrideDecodeStepMs;
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
     * CURRENT running batch size > linear production fit), then scaled by
     * decode scale + sleep scale and jittered (same formula as
     * {@code scaledMs}). Returns >= 1 ms. Each step advances every running
     * stream by {@link #tokensPerStep()} tokens — the MTP fold the per-step
     * loop needs on top of the step duration.
     */
    long decodeStepDelayMs(int activeBatchSize) {
        return scaledMs(stepMs(activeBatchSize) * effectiveDecodeScale(),
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
    boolean memoryCopyLifecycle;
    boolean memoryPrefixTree = true;

    record RequestShape(EngineRpcService.GenerateInputPB input,
                        int inputLen,
                        int outputLen,
                        List<Long> blockKeys,
                        long hitTokens,
                        int hitBlocks, boolean nativeKeys, int memoryHitBlocks) {
        RequestShape(EngineRpcService.GenerateInputPB input, int inputLen, int outputLen,
                     List<Long> blockKeys, long hitTokens, int hitBlocks, boolean nativeKeys) {
            this(input, inputLen, outputLen, blockKeys, hitTokens, hitBlocks, nativeKeys, 0);
        }
    }

    private record DecodePoint(int batchSize, double stepMs) {
    }
}

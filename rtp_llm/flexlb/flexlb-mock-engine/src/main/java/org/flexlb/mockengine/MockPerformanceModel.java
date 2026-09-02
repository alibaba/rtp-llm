package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.prediction.PrefillTimeFormula;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig.FormulaEstimatorConfig;
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
     * Default per-step decode latency intercept, JSON "decode.step_base_ms":
     * the production DSv4 fit step_ms = 19.5 + 0.175 x running (task #68
     * measurement, R^2 = 0.82 on the step caliber). Code default mirrors the
     * prefill fallback pattern: absent config boots on the production fit,
     * explicit JSON overrides it.
     */
    static final double DEFAULT_DECODE_STEP_BASE_MS = 19.5;

    /**
     * Default per-step decode latency slope per running stream, JSON
     * "decode.step_per_running_ms" (production fit, task #68).
     */
    static final double DEFAULT_DECODE_STEP_PER_RUNNING_MS = 0.175;

    /**
     * Default MTP acceptance fold, JSON "decode.tokens_per_step": tokens
     * produced per running stream per decode step. Production DSv4 accepts
     * 2.54-2.88 tokens/step (slightly lower at full batch; task #68), so the
     * per-step model advances each stream by this many tokens per step --
     * per-token pricing overstated low-batch latency ~5.5x and full-batch
     * ~2.8x because the fixed per_token_ms caliber ignored both MTP folding
     * and batch amortisation. Must be > 0.
     */
    static final double DEFAULT_TOKENS_PER_STEP = 2.6;

    /**
     * Production DSv4 prefill execution-time fit, verbatim from the
     * RoutingConfig.FormulaEstimatorConfig.DEFAULT_EXPRESSION constant as it
     * existed on the intake3 test line (commit 6980b3d508..91498cfa4f, where
     * it briefly served as the production code default). The mock is a test
     * process, not bound by the production default: it keeps the production
     * fit as its own built-in fallback so a master config that omits the
     * estimator still boots on realistic prefill durations. Explicit FORMULA
     * expressions in the master config (harness.py and
     * data/config/master_fixed_window.json inject this same expression)
     * always win over this fallback.
     */
    private static final String DSV4_PREFILL_FIT_EXPRESSION =
            "max(196, -68.612174288157 + 0.993068319341 * (max(0, 287.3980926717 + 2.30134977837751 * batchSize + "
            + "0.158123254797307 * sum(hitCacheTokens / 1024.) + 0.575522710053703 * sum(computeTokens / 1024.) + "
            + "0.0517623430739831 * sum(computeTokens / 1024. * computeTokens / 1024.) + 0.0395308136993267 * "
            + "sum(hitCacheTokens / 1024. * computeTokens / 1024.) + 0.0104363634681015 * sum(hitCacheTokens / 1024. * "
            + "hitCacheTokens / 1024.) + 0.575522710053703 * max(sum(computeTokens / 1024.) - 16, 0) + 2.82077211814514 "
            + "* max(sum(computeTokens / 1024.) - 32, 0) - 0.0254671429192862 * max(sum(computeTokens / 1024.) - 64, 0) "
            + "+ 2.15779213792494 * max(sum(computeTokens / 1024.) - 96, 0) + 0.247806025472364 * "
            + "max(sum(hitCacheTokens / 1024.) - 32, 0) - 0.444522654549492 * max(sum(hitCacheTokens / 1024.) - 64, 0) "
            + "- 0.427317020061895 * max(sum(hitCacheTokens / 1024.) - 128, 0) + 0.347029077528455 * "
            + "max(sum(hitCacheTokens / 1024.) - 256, 0) - 0.298742307762735 * max(sum(hitCacheTokens / 1024.) - 384, "
            + "0) + 2.30134977837751 * max(batchSize - 8, 0) - 3.54884859699154 * max(batchSize - 16, 0) - "
            + "11.3438560779984 * max(batchSize - 24, 0) + 0.879751992138183 * sum(max(computeTokens / 1024. - 2, 0)) + "
            + "0.636364578079591 * sum(max(computeTokens / 1024. - 4, 0)) - 0.0513345988517118 * sum(max(computeTokens "
            + "/ 1024. - 8, 0)) - 0.332584389129357 * sum(max(hitCacheTokens / 1024. - 2, 0)) + 0.305819761192588 * "
            + "sum(max(hitCacheTokens / 1024. - 4, 0)) - 0.287610979974721 * sum(max(hitCacheTokens / 1024. - 8, 0)) + "
            + "0.191310200712013 * sum(max(hitCacheTokens / 1024. - 12, 0)) + 0.0130251644478961 * max(batchSize - 8, "
            + "0) * sum(hitCacheTokens / 1024.) + 0.00981382840761646 * max(batchSize - 16, 0) * sum(hitCacheTokens / "
            + "1024.) - 0.0299132587297009 * max(batchSize - 24, 0) * sum(hitCacheTokens / 1024.) + 0.0447455122487382 "
            + "* max(batchSize - 8, 0) * sum(computeTokens / 1024.) + 0.0104635312001851 * max(batchSize - 16, 0) * "
            + "sum(computeTokens / 1024.) + 0.0542737877321807 * max(batchSize - 24, 0) * sum(computeTokens / 1024.))))";

    private volatile int blockSize;
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
    private final PrefillTimeFormula prefillFormula;
    // Decode step-latency sources, exactly one active per model:
    //   - explicit step_ms_by_batch curve (decodePoints non-empty; legacy
    //     declared channel, kept for suites that price steps themselves), or
    //   - the linear production fit stepBaseMs + stepPerRunningMs * running
    //     (decodePoints empty; coefficients default to the production DSv4
    //     fit, JSON-declared values override).
    // The runtime override (setOverrideDecodeStepMs) beats both.
    private final List<DecodePoint> decodePoints;
    private final double stepBaseMs;
    private final double stepPerRunningMs;
    // MTP acceptance fold: tokens advanced per running stream per step
    // (JSON "decode.tokens_per_step", default DEFAULT_TOKENS_PER_STEP).
    private final double tokensPerStep;
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

    private MockPerformanceModel(int blockSize,
                                 double sleepScale,
                                 double prefillScale,
                                 Double prefillMinMs,
                                 Double configuredFixedPrefillMs,
                                 int maxWaitingPrefillBatches,
                                 int directBatchSizeMax,
                                 PrefillTimeFormula prefillFormula,
                                 List<DecodePoint> decodePoints,
                                 double stepBaseMs,
                                 double stepPerRunningMs,
                                 double tokensPerStep,
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
        this.prefillFormula = prefillFormula;
        this.decodePoints = decodePoints;
        this.stepBaseMs = stepBaseMs;
        this.stepPerRunningMs = stepPerRunningMs;
        this.tokensPerStep = tokensPerStep;
        this.decodeScale = decodeScale;
        this.reportQueuedAsKvAllocated = reportQueuedAsKvAllocated;
        this.jitterPct = jitterPct;
    }

    static MockPerformanceModel load(String performanceFile, String masterConfigFile) throws IOException {
        JsonNode performance = MAPPER.readTree(Path.of(performanceFile).toFile());
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

        PrefillTimeFormula formula = PrefillTimeFormula.parse(loadPrefillExpression(masterConfigFile));

        JsonNode decode = performance.path("decode");
        // per_token_ms is REMOVED (task #69, wrong-version-deleted-clean rule):
        // it was a fixed per-token latency (V3-era no-MTP single-stream caliber)
        // that overstated low-batch decode ~5.5x and full-batch ~2.8x versus
        // production. Fail fast with a migration hint instead of silently
        // reinterpreting it.
        if (decode.has("per_token_ms")) {
            throw new IllegalStateException("Performance JSON '" + performanceFile
                    + "': decode.per_token_ms is removed — decode is now priced per STEP"
                    + " (production fit step_base_ms=" + DEFAULT_DECODE_STEP_BASE_MS
                    + " + step_per_running_ms=" + DEFAULT_DECODE_STEP_PER_RUNNING_MS
                    + " × running, " + DEFAULT_TOKENS_PER_STEP + " tokens/step by default)."
                    + " Remove per_token_ms to get the production-fit defaults, or declare"
                    + " decode.step_ms_by_batch / decode.step_base_ms explicitly.");
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
        // No decode latency declaration at all -> the linear production fit
        // (same pattern as the prefill fallback: the code default IS the
        // production DSv4 fit, so an absent decode section boots on real
        // numbers instead of the former fail-fast). An explicit curve
        // overrides the linear fit; explicit coefficients override its
        // default intercept/slope.
        double stepBaseMs = decode.path("step_base_ms").asDouble(DEFAULT_DECODE_STEP_BASE_MS);
        double stepPerRunningMs = decode.path("step_per_running_ms")
                .asDouble(DEFAULT_DECODE_STEP_PER_RUNNING_MS);
        double tokensPerStep = decode.path("tokens_per_step").asDouble(DEFAULT_TOKENS_PER_STEP);
        if (tokensPerStep <= 0) {
            throw new IllegalStateException("Performance JSON '" + performanceFile
                    + "': decode.tokens_per_step must be > 0 (got " + tokensPerStep + ")");
        }
        double jitterPct = performance.path("jitter_pct").asDouble(0.0);
        return new MockPerformanceModel(blockSize, sleepScale, prefillScale,
                prefillMinMs, prefillFixedMs, maxWaitingPrefillBatches, directBatchSizeMax, formula,
                List.copyOf(points), stepBaseMs, stepPerRunningMs, tokensPerStep,
                decode.path("scale").asDouble(1.0),
                reportQueuedAsKvAllocated, jitterPct);
    }

    /**
     * Resolve the prefill duration formula — exactly one source, never a
     * silent hard-coded fallback:
     * <ol>
     *   <li>an explicit FORMULA estimator in the master config's FLEXLB_CONFIG
     *       (blank expression = misconfiguration, fail fast);</li>
     *   <li>otherwise {@link #DSV4_PREFILL_FIT_EXPRESSION} — the production
     *       DSv4 fit the mock keeps as its own built-in default, and the
     *       static approximation for a LEARNING estimator the mock cannot
     *       replay.</li>
     * </ol>
     * This keeps mock execution time and master routing predictions on the
     * same expression; the legacy silent fixed_ms / 300 ms fallbacks are gone
     * (an explicit performance-JSON "prefill.fixed_ms" declaration still
     * wins over the formula — see prefillMs).
     */
    private static String loadPrefillExpression(String masterConfigFile) throws IOException {
        JsonNode root = MAPPER.readTree(Path.of(masterConfigFile).toFile());
        JsonNode envs = root.path("zone_process_setting").path("process_info").path("envs");
        for (JsonNode item : envs) {
            if (item.isArray() && item.size() >= 2
                    && "FLEXLB_CONFIG".equals(item.get(0).asText())) {
                FlexlbConfig config = ConfigService.parse(item.get(1).asText());
                var estimator = config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator();
                if (estimator instanceof FormulaEstimatorConfig formula) {
                    String expression = formula.getExpression();
                    if (expression == null || expression.isBlank()) {
                        throw new IllegalStateException("Master config " + masterConfigFile
                                + ": router.roles.prefill.executionTimeEstimator is FORMULA"
                                + " with a blank expression — set the expression explicitly or"
                                + " omit the estimator to use the built-in DSv4 production fit"
                                + " (MockPerformanceModel.DSV4_PREFILL_FIT_EXPRESSION)");
                    }
                    return expression;
                }
                break;  // LEARNING estimator: fall through to the production-fit default
            }
        }
        return DSV4_PREFILL_FIT_EXPRESSION;
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
        // hitBlocks carries the RAW prefix-match run length (key count) — the
        // key-level cache-hit caliber (production recent_cache_key_hit_count /
        // total_count analogue) recorded by the engine at this admission hit
        // computation point. Unlike hitTokens it is NOT clamped to inputLen, so
        // a trace whose bh keys exceed the request's own block count keeps an
        // honest requested/hit key pair.
        int hitBlocks = cache.prefixHitBlocks(blockKeys);
        long hitTokens = (long) hitBlocks * blockSize;
        hitTokens = Math.min(hitTokens, inputLen);
        return new RequestShape(input, inputLen, Math.max(1, outputLen), List.copyOf(blockKeys),
                hitTokens, hitBlocks);
    }

    long prefillMs(List<RequestShape> requests) {
        if (requests.isEmpty()) {
            return 0;
        }
        double latency;
        if (overrideFixedPrefillMs != null) {
            // Runtime override (Python /set_perf prefill_fixed_ms): explicit
            // test-time control, length-blind by design.
            latency = overrideFixedPrefillMs;
        } else if (configuredFixedPrefillMs != null) {
            // Explicit performance-JSON "prefill.fixed_ms": the declared flat
            // prefill for duration-blind suites. Declared explicitly, so it
            // wins over the formula (priority: runtime > JSON > formula).
            latency = configuredFixedPrefillMs;
        } else {
            // The only prefill source: the expression resolved in
            // loadPrefillExpression (explicit FORMULA or the production fit).
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
     * (JSON "prefill.max_waiting_batches", default 0 = unbounded).
     */
    int maxWaitingPrefillBatches() {
        return maxWaitingPrefillBatches;
    }

    /**
     * Cap on requests coalesced into one direct-path prefill batch
     * (JSON "prefill.direct_batch_size_max", default 32, minimum 1).
     */
    int directBatchSizeMax() {
        return directBatchSizeMax;
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
        return scaledMs(decodeSteps(outputLen) * stepMs(activeBatchSize) * effectiveDecodeScale());
    }

    /**
     * MTP fold: number of decode steps needed to produce {@code outputLen}
     * tokens at the configured tokens_per_step (ceil — the final partial step
     * still costs a full step, like a real engine's last draft round).
     */
    int decodeSteps(int outputLen) {
        return (int) Math.ceil(outputLen / tokensPerStep);
    }

    /** Tokens produced per running stream per decode step (MTP acceptance fold). */
    double tokensPerStep() {
        return tokensPerStep;
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
        return scaledMs(stepMs(activeBatchSize) * effectiveDecodeScale());
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
                        long hitTokens,
                        int hitBlocks) {
    }

    private record DecodePoint(int batchSize, double stepMs) {
    }
}

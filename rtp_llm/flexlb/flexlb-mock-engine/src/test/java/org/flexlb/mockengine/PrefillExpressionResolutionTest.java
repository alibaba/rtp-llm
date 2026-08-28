package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.prediction.PrefillTimeFormula;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Resolution priority of the mock engine's prefill duration expression —
 * exactly one source, never a silent fallback:
 * <ol>
 *   <li>an explicit FORMULA estimator in the master config's FLEXLB_CONFIG;</li>
 *   <li>otherwise the performance-JSON {@code prefill.expression} direct
 *       supply (the NAVI_BATCH channel, for when the master config cannot
 *       provide an expression — e.g. the LEARNING estimator);</li>
 *   <li>otherwise {@link MockPerformanceModel#DSV4_PREFILL_FIT_EXPRESSION} —
 *       the production DSv4 fit the mock keeps as its built-in default.</li>
 * </ol>
 */
class PrefillExpressionResolutionTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String LEARNING_CONFIG =
            "{\"router\":{\"roles\":{\"prefill\":{\"executionTimeEstimator\":{\"type\":\"LEARNING\"}}}}}";
    private static final int INPUT_TOKENS = 1024;

    @TempDir
    Path tempDir;

    @Test
    void learningEstimatorFallsBackToPerformanceJsonExpression() throws Exception {
        // LEARNING master config + performance prefill.expression "1234": the
        // direct-supply channel must win over the code default.
        MockPerformanceModel model = load(LEARNING_CONFIG, Map.of("expression", "1234"));
        assertEquals(1234, prefillMs(model),
                "prefill.expression direct supply must drive the mock prefill duration");
    }

    @Test
    void learningEstimatorWithoutPerformanceExpressionUsesCodeDefault() throws Exception {
        // LEARNING master config, no prefill.expression: the production DSv4
        // fit (DSV4_PREFILL_FIT_EXPRESSION) is the formula — verified against
        // an independent oracle evaluation of the same constant.
        MockPerformanceModel model = load(LEARNING_CONFIG, Map.of());
        long oracle = PrefillTimeFormula.parse(MockPerformanceModel.DSV4_PREFILL_FIT_EXPRESSION)
                .evaluate(batchVars(), itemVars());
        assertEquals(oracle, prefillMs(model),
                "absent direct supply must fall back to the production DSv4 fit");
    }

    @Test
    void explicitFormulaEstimatorWinsOverPerformanceExpression() throws Exception {
        // FORMULA "77" in the master config outranks the performance-JSON
        // "1234" direct supply (priority chain unchanged by the merge).
        String formulaConfig = "{\"router\":{\"roles\":{\"prefill\":{\"executionTimeEstimator\":"
                + "{\"type\":\"FORMULA\",\"expression\":\"77\"}}}}}";
        MockPerformanceModel model = load(formulaConfig, Map.of("expression", "1234"));
        assertEquals(77, prefillMs(model),
                "an explicit FORMULA estimator must outrank the performance-JSON supply");
    }

    // ──────────── helpers ────────────

    private long prefillMs(MockPerformanceModel model) {
        MockPerformanceModel.RequestShape shape = new MockPerformanceModel.RequestShape(
                null, INPUT_TOKENS, 8, List.of(), 0L);
        return model.prefillMs(List.of(shape));
    }

    /** Mirrors the batch vars MockPerformanceModel builds for one request. */
    private double[] batchVars() {
        return new double[] {1, 0, 0, 0, 0};
    }

    /** Mirrors the per-item vars for a single all-miss 1024-token request. */
    private List<double[]> itemVars() {
        return List.of(new double[] {1, INPUT_TOKENS, 0, INPUT_TOKENS, 0});
    }

    private MockPerformanceModel load(String flexlbConfig, Map<String, Object> prefill)
            throws IOException {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> performanceJson = new LinkedHashMap<>();
        performanceJson.put("sleep_scale", 1.0);
        performanceJson.put("jitter_pct", 0.0);
        // decode pricing is intentionally absent: per_token_ms is removed in
        // the intake3 schema (priced per STEP) and this suite only asserts
        // prefill expression resolution — the production-fit decode defaults
        // are irrelevant here.
        performanceJson.put("prefill", prefill);
        MAPPER.writeValue(performance.toFile(), performanceJson);
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("FLEXLB_CONFIG", flexlbConfig))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }
}

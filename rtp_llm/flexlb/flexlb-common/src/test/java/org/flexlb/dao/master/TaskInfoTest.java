package org.flexlb.dao.master;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TaskInfoTest {
    private static final String SMOKE_TEST_FORMULA = "(tokens - hitCacheTokens) * 2 + hitCacheTokens / 2";

    @Test
    void should_use_default_prefill_time_formula() {
        assertEquals(130L, TaskInfo.estimatePrefillTimeMs(200, 100));
    }

    @Test
    void should_calculate_prefill_time_with_configured_formula() {
        PrefillTimeFormula formula = TaskInfo.readPrefillTimeFormula(Map.of(
                TaskInfo.PREFILL_TIME_ESTIMATE_FORMULA_ENV,
                SMOKE_TEST_FORMULA));

        assertEquals(250L, formula.estimate(200, 100));
    }

    @ParameterizedTest
    @CsvSource({
            "'tokens', 200",
            "'tokens - hitCacheTokens', 100",
            "'tokens * 1.5 - hitCacheTokens * 0.25', 275",
            "'(tokens - hitCacheTokens) * 3', 300",
            "'tokens / 2 + hitCacheTokens / 4', 125",
            "'-hitCacheTokens + tokens', 100",
            "'1e2 + tokens / 2', 200",
            "'inputLength - prefixLength * 0.5', 150",
            "'input_length + hit_cache_tokens / 10', 210"
    })
    void should_calculate_supported_formula_shapes(String expression, long expected) {
        PrefillTimeFormula formula = TaskInfo.readPrefillTimeFormula(Map.of(
                TaskInfo.PREFILL_TIME_ESTIMATE_FORMULA_ENV,
                expression));

        assertEquals(expected, formula.estimate(200, 100));
    }

    @Test
    void should_fallback_to_default_when_environment_formula_is_invalid() {
        PrefillTimeFormula formula = TaskInfo.readPrefillTimeFormula(Map.of(
                TaskInfo.PREFILL_TIME_ESTIMATE_FORMULA_ENV,
                "tokens -"));

        assertEquals(130L, formula.estimate(200, 100));
    }

    @Test
    void should_fallback_to_default_when_environment_formula_uses_unsupported_variable() {
        PrefillTimeFormula formula = TaskInfo.readPrefillTimeFormula(Map.of(
                TaskInfo.PREFILL_TIME_ESTIMATE_FORMULA_ENV,
                "tokens - unknownTokens"));

        assertEquals(130L, formula.estimate(200, 100));
    }

    @Test
    void should_initialize_prefill_time_formula_from_environment() {
        String formula = System.getenv(TaskInfo.PREFILL_TIME_ESTIMATE_FORMULA_ENV);
        if (formula == null || formula.trim().isEmpty()) {
            return;
        }

        assertEquals(PrefillTimeFormula.parse(formula).estimate(200, 100), TaskInfo.estimatePrefillTimeMs(200, 100));
    }

    @Test
    void should_initialize_known_prefill_time_formula_from_environment() {
        String formula = System.getenv(TaskInfo.PREFILL_TIME_ESTIMATE_FORMULA_ENV);
        if (!SMOKE_TEST_FORMULA.equals(formula)) {
            return;
        }

        assertEquals(250L, TaskInfo.estimatePrefillTimeMs(200, 100));
    }
}

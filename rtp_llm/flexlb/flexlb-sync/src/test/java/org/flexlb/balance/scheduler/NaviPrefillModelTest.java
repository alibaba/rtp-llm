package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Golden-value tests for {@link NaviPrefillModel}, verified against the navi C++
 * {@code NonLinearPrefillModelR} implementation with initial prior weights.
 */
class NaviPrefillModelTest {

    /** Navi initial prior: {@code INITIAL_PARAMETERS} from NonLinearPrefillModelR.cpp L17-19. */
    private static final double[] NAVI_PARAMS =
            {-4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 1.4, 1.0, -4.0};

    // ==================== Constants verification ====================

    @Test
    @DisplayName("TOKEN_SCALE matches navi constexpr")
    void tokenScaleMatchesNavi() {
        assertEquals(1024.0, NaviPrefillModel.TOKEN_SCALE);
    }

    @Test
    @DisplayName("COEFFICIENT_1 matches navi constexpr")
    void coefficient1MatchesNavi() {
        assertEquals(0.005, NaviPrefillModel.COEFFICIENT_1);
    }

    @Test
    @DisplayName("COEFFICIENT_2 matches navi constexpr")
    void coefficient2MatchesNavi() {
        assertEquals(0.02, NaviPrefillModel.COEFFICIENT_2);
    }

    @Test
    @DisplayName("COEFFICIENT_3 matches navi constexpr")
    void coefficient3MatchesNavi() {
        assertEquals(320.0, NaviPrefillModel.COEFFICIENT_3);
    }

    // ==================== calculateRequestLinearCost ====================

    @Test
    @DisplayName("calculateRequestLinearCost: standard case (totalTokens=2048, cacheHit=512)")
    void linearCostStandardCase() {
        // Hand-computed from navi formula:
        // cacheMiss=1536, reuse=0.5, compute=1.5, compute^2=2.25, reuse*compute=0.75
        // cost = 1*10 + 0.5*1.4 + 1.5*20 + 2.25*0.1 + 0.75*0.09 = 40.9925
        double expected = 40.9925;
        double actual = NaviPrefillModel.calculateRequestLinearCost(NAVI_PARAMS, 2048, 512);
        assertEquals(expected, actual, 1e-12, "standard case golden mismatch");
    }

    @Test
    @DisplayName("calculateRequestLinearCost: all miss (cacheHitTokens=0)")
    void linearCostAllMiss() {
        // cacheMiss=2048, reuse=0, compute=2.0, compute^2=4.0, reuse*compute=0
        // cost = 1*10 + 0 + 2.0*20 + 4.0*0.1 + 0 = 10+40+0.4 = 50.4
        double expected = 50.4;
        double actual = NaviPrefillModel.calculateRequestLinearCost(NAVI_PARAMS, 2048, 0);
        assertEquals(expected, actual, 1e-12, "all-miss golden mismatch");
    }

    @Test
    @DisplayName("calculateRequestLinearCost: near full hit (cacheHit=1920, totalTokens=2048)")
    void linearCostNearFullHit() {
        // cacheMiss=128, reuse=1920/1024=1.875, compute=128/1024=0.125
        // compute^2=0.015625, reuse*compute=0.234375
        // cost = 1*10 + 1.875*1.4 + 0.125*20 + 0.015625*0.1 + 0.234375*0.09
        double expected = 15.14765625;
        double actual = NaviPrefillModel.calculateRequestLinearCost(NAVI_PARAMS, 2048, 1920);
        assertEquals(expected, actual, 1e-12, "near-full-hit golden mismatch");
    }

    @Test
    @DisplayName("calculateRequestLinearCost: minimal token (totalTokens=1, cacheHit=0)")
    void linearCostMinimalToken() {
        double compute = 1.0 / 1024.0;
        double expected = 10.0 + compute * 20.0 + compute * compute * 0.1;
        double actual = NaviPrefillModel.calculateRequestLinearCost(NAVI_PARAMS, 1, 0);
        assertEquals(expected, actual, 1e-12, "minimal-token golden mismatch");
    }

    @Test
    @DisplayName("calculateRequestLinearCost: null params returns 0")
    void linearCostNullParams() {
        assertEquals(0.0, NaviPrefillModel.calculateRequestLinearCost(null, 2048, 512));
    }

    @Test
    @DisplayName("calculateRequestLinearCost: short params returns 0")
    void linearCostShortParams() {
        assertEquals(0.0,
                NaviPrefillModel.calculateRequestLinearCost(new double[]{1.0, 2.0}, 2048, 512));
    }

    // ==================== calculateLatencyAndDerivative ====================

    @Test
    @DisplayName("calculateLatencyAndDerivative: golden values from navi prior + requestCostSum=40.9925")
    void latencyAndDerivativeStandardCase() {
        double requestCostSum = 40.9925;
        // Traced from navi evaluateNonLinear (NonLinearPrefillModelR.cpp L300-311):
        double linear = (requestCostSum + NAVI_PARAMS[0]) / 320.0;
        double linearOffset = linear + NAVI_PARAMS[8] + 1.0;
        double squareRoot = Math.sqrt(linearOffset * linearOffset + 4.0);
        double expectedLatency = 280.0 + 50.0 * (linearOffset + squareRoot);
        double expectedDerivative = 50.0 * (1.0 + linearOffset / squareRoot) / 320.0;

        double[] result = NaviPrefillModel.calculateLatencyAndDerivative(requestCostSum, NAVI_PARAMS);
        assertEquals(expectedLatency, result[0], 1e-10, "latency golden mismatch");
        assertEquals(expectedDerivative, result[1], 1e-10, "derivative golden mismatch");
    }

    @Test
    @DisplayName("calculateLatencyAndDerivative: zero costSum")
    void latencyAndDerivativeZeroCost() {
        double requestCostSum = 0.0;
        double linear = (0.0 + NAVI_PARAMS[0]) / 320.0;
        double linearOffset = linear + NAVI_PARAMS[8] + 1.0;
        double squareRoot = Math.sqrt(linearOffset * linearOffset + 4.0);
        double expectedLatency = 280.0 + 50.0 * (linearOffset + squareRoot);
        double expectedDerivative = 50.0 * (1.0 + linearOffset / squareRoot) / 320.0;

        double[] result = NaviPrefillModel.calculateLatencyAndDerivative(requestCostSum, NAVI_PARAMS);
        assertEquals(expectedLatency, result[0], 1e-10);
        assertEquals(expectedDerivative, result[1], 1e-10);
    }

    @Test
    @DisplayName("calculateLatencyAndDerivative: derivative is always positive for valid params")
    void derivativePositive() {
        for (double costSum : new double[]{0.0, 10.0, 50.0, 100.0, 500.0}) {
            double[] result = NaviPrefillModel.calculateLatencyAndDerivative(costSum, NAVI_PARAMS);
            assertEquals(true, result[1] > 0.0,
                    "derivative must be positive for costSum=" + costSum);
        }
    }
}

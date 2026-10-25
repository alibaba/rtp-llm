package org.flexlb.mockengine;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class MockDecodeStepMetricsTest {
    @Test void mixedFullAndTerminalStepsUseActualAcceptedTokens() {
        // Two participating streams produce 4 and 1 tokens; not the configured 4 each.
        var metrics = MockDecodeStepMetrics.values(40, 5, 2);
        assertEquals(16000.0, metrics.get("rtp_llm_sp_estimate_tpot_us"));
        assertEquals(2.5, metrics.get("rtp_llm_sp_avg_accept_token_num"));
        assertEquals(40000.0, metrics.get("rtp_llm_model_forward_us"));
    }
    @Test void emptyStepDoesNotInventTpotOrAcceptanceRate() {
        var metrics = MockDecodeStepMetrics.values(40, 0, 0);
        assertFalse(metrics.containsKey("rtp_llm_sp_estimate_tpot_us"));
        assertFalse(metrics.containsKey("rtp_llm_sp_avg_accept_rate"));
    }
}

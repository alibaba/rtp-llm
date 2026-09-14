package org.flexlb.mockengine;

import java.util.LinkedHashMap;
import java.util.Map;

/** Step-level metrics; no proposed-token or acceptance-rate claims without a draft model. */
final class MockDecodeStepMetrics {
    static Map<String, Number> values(long stepMs, int acceptedTokens, int streams) {
        Map<String, Number> values = new LinkedHashMap<>();
        values.put("rtp_llm_model_forward_us", stepMs * 1000.0);
        values.put("rtp_llm_sp_step_latency_us", stepMs * 1000.0);
        if (streams > 0 && acceptedTokens > 0) {
            double average = (double) acceptedTokens / streams;
            values.put("rtp_llm_sp_total_accepted_token_num", acceptedTokens);
            values.put("rtp_llm_sp_avg_accept_token_num", average);
            values.put("rtp_llm_sp_estimate_tpot_us", stepMs * 1000.0 / average);
        }
        return values;
    }
}

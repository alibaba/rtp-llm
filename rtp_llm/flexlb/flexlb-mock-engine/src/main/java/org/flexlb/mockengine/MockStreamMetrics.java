package org.flexlb.mockengine;

import java.util.LinkedHashMap;
import java.util.Map;

/** Engine-local successful stream observations, as in GenerateStream::reportStreamMetrics.
 * C++ REPORT_GAUGE omits zero observations; absence must not dilute positive sample averages. */
final class MockStreamMetrics {
    static Map<String, Number> completed(long input, long output, long reuse, long firstUs, long residenceUs) {
        Map<String, Number> values = new LinkedHashMap<>();
        positive(values, "rtp_llm_input_token_length", input);
        positive(values, "rtp_llm_output_token_length", output);
        positive(values, "rtp_llm_reuse_length", reuse);
        positive(values, "rtp_llm_effective_context_length", Math.max(0L, input - reuse));
        if (output > 0) positive(values, "rtp_llm_first_token_latency_us", firstUs);
        positive(values, "rtp_llm_latency_us", residenceUs);
        return values;
    }

    private static void positive(Map<String, Number> values, String name, long value) {
        if (value > 0) values.put(name, value);
    }
}

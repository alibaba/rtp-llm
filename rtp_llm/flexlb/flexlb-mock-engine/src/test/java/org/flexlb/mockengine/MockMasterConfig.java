package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Writes the process-config envelope consumed by the Java mock engine tests. */
final class MockMasterConfig {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private MockMasterConfig() {
    }

    static void writeWithPrefillExpression(Path target, String expression) throws IOException {
        String flexlbConfig = MAPPER.writeValueAsString(Map.of(
                "router", Map.of(
                        "roles", Map.of(
                                "prefill", Map.of(
                                        "executionTimeEstimator", Map.of(
                                                "type", "FORMULA",
                                                "expression", expression))))));
        MAPPER.writeValue(target.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("FLEXLB_CONFIG", flexlbConfig))))));
    }
}

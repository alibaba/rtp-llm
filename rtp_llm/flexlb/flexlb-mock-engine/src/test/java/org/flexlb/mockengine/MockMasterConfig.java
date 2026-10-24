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
                "schemaVersion", 3,
                "requestLifecycle", Map.of(
                        "request", Map.of("timeoutMs", 60000),
                        "decision", Map.of("lifetime", 2.0)),
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
    static String readExpression(String masterConfigFile,
                                               String defaultExpression) throws java.io.IOException {
        com.fasterxml.jackson.databind.JsonNode root = new com.fasterxml.jackson.databind.ObjectMapper().readTree(java.nio.file.Path.of(masterConfigFile).toFile());
        // The standalone launcher writes the schema-v3 configuration directly;
        // bundle deployments retain the historical zone/env envelope.
        if (root.has("schemaVersion")) {
            var estimator = org.flexlb.config.ConfigService.parse(root.toString()).getRouter().getRoles()
                    .getPrefill().getExecutionTimeEstimator();
            if (estimator != null && estimator.getType() == org.flexlb.config.RoutingConfig.EstimatorType.FORMULA) {
                return estimator.getExpression();
            }
            return defaultExpression;
        }
        com.fasterxml.jackson.databind.JsonNode envs = root.path("zone_process_setting").path("process_info").path("envs");
        for (com.fasterxml.jackson.databind.JsonNode item : envs) {
            if (item.isArray() && item.size() >= 2
                    && "FLEXLB_CONFIG".equals(item.get(0).asText())) {
                org.flexlb.config.FlexlbConfig config = org.flexlb.config.ConfigService.parse(item.get(1).asText());
                var estimator = config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator();
                if (estimator != null
                        && estimator.getType() == org.flexlb.config.RoutingConfig.EstimatorType.FORMULA) {
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

}

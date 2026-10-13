package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

/** Explicit workload values for configuration contract tests. */
final class ConfigTestFixtures {
    private static final ObjectMapper JSON = new ObjectMapper();
    static final String REQUIRED = """
            {
              "schemaVersion": 3,
              "requestLifecycle": {
                "request": {"timeoutMs": 60000},
                "decision": {"lifetime": 2.0}
              }
            }
            """;

    static FlexlbConfig parse(String patch) {
        return ConfigService.parse(document(patch));
    }

    static String document(String patch) {
        try {
            ObjectNode root = (ObjectNode) JSON.readTree(REQUIRED);
            merge(root, (ObjectNode) JSON.readTree(patch));
            return root.toString();
        } catch (Exception error) {
            throw new IllegalArgumentException(error);
        }
    }

    private static void merge(ObjectNode target, ObjectNode patch) {
        patch.fields().forEachRemaining(entry -> {
            JsonNode existing = target.get(entry.getKey());
            if (existing instanceof ObjectNode object && entry.getValue() instanceof ObjectNode update) {
                merge(object, update);
            } else {
                target.set(entry.getKey(), entry.getValue());
            }
        });
    }

    private ConfigTestFixtures() { }
}

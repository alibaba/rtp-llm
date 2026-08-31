package org.flexlb.service.config.merger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.config.ConfigValidationException;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.FlexlbConfigValidator;
import org.flexlb.util.JsonUtils;

import java.util.Iterator;
import java.util.Map;

public final class FlexlbConfigMerger {

    private static final String FLEXLB_CONFIG = "FLEXLB_CONFIG";

    private FlexlbConfigMerger() {}

    public static FlexlbConfig mergeWithDefaults(String content) {
        return merge(new FlexlbConfig(), content, FLEXLB_CONFIG);
    }

    public static FlexlbConfig merge(FlexlbConfig baseConfig, String content, String sourceName) {
        ObjectNode overrides = parseOverrides(content, sourceName);
        if (overrides.isEmpty()) {
            return baseConfig;
        }
        ObjectNode merged = JsonUtils.strictValueToTree(baseConfig);
        deepMerge(merged, overrides);
        return parseMergedConfig(merged, sourceName);
    }

    private static FlexlbConfig parseMergedConfig(ObjectNode merged, String sourceName) {
        try {
            FlexlbConfig config = JsonUtils.strictTreeToValue(merged, FlexlbConfig.class);
            FlexlbConfigValidator.validate(config);
            return config;
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(sourceName, "Invalid FlexLB configuration: " + error.getMessage(), error);
        }
    }

    private static ObjectNode parseOverrides(String content, String sourceName) {
        if (content == null || content.isBlank()) {
            return JsonUtils.createObjectNode();
        }
        try {
            JsonNode parsed = JsonUtils.readStrictTree(content);
            JsonUtils.rejectJsonNull(parsed, "$", sourceName);
            if (!(parsed instanceof ObjectNode overrides)) {
                throw new ConfigValidationException(sourceName, "configuration must be a JSON object");
            }
            return overrides;
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(sourceName, "Invalid configuration JSON: " + error.getMessage(), error);
        }
    }

    private static void deepMerge(ObjectNode target, ObjectNode overrides) {
        Iterator<Map.Entry<String, JsonNode>> fields = overrides.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            JsonNode current = target.get(field.getKey());
            JsonNode replacement = field.getValue();
            if (current instanceof ObjectNode currentObject && replacement instanceof ObjectNode replacementObject
                    && hasCompatibleType(currentObject, replacementObject)) {
                deepMerge(currentObject, replacementObject);
            } else {
                target.set(field.getKey(), replacement.deepCopy());
            }
        }
    }

    private static boolean hasCompatibleType(ObjectNode current, ObjectNode replacement) {
        JsonNode replacementType = replacement.get("type");
        return replacementType == null || replacementType.equals(current.get("type"));
    }
}

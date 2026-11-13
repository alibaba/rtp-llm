package org.flexlb.config.loading.parser;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.config.loading.ConfigValidationException;
import org.flexlb.config.service.ServiceRoute;
import org.flexlb.util.JsonUtils;

/**
 * Parses and validates model service endpoint topology before it is published to readers.
 */
public final class ModelServiceConfigParser {

    private static final String CONFIG_NAME = "MODEL_SERVICE_CONFIG";

    private ModelServiceConfigParser() {}

    public static ServiceRoute parse(String document) {
        try {
            JsonNode tree = JsonUtils.readStrictTree(document);
            org.flexlb.config.loading.ConfigJsonValidation.rejectJsonNull(tree, "$", CONFIG_NAME);
            org.flexlb.config.loading.ConfigJsonValidation.rejectModelBehaviorFields(
                    tree, CONFIG_NAME);
            return JsonUtils.strictTreeToValue(tree, ServiceRoute.class);
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(
                    CONFIG_NAME, "Invalid MODEL_SERVICE_CONFIG JSON: " + error.getMessage(), error);
        }
    }
}

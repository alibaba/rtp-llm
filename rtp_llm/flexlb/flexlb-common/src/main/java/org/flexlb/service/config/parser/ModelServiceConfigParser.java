package org.flexlb.service.config.parser;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.config.ConfigValidationException;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.JsonUtils;

public final class ModelServiceConfigParser {

    private static final String CONFIG_NAME = "MODEL_SERVICE_CONFIG";

    private ModelServiceConfigParser() {}

    public static ServiceRoute parse(String document) {
        try {
            JsonNode tree = JsonUtils.readStrictTree(document);
            JsonUtils.rejectJsonNull(tree, "$", CONFIG_NAME);
            JsonUtils.rejectModelBehaviorFields(tree, CONFIG_NAME);
            return JsonUtils.strictTreeToValue(tree, ServiceRoute.class);
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(CONFIG_NAME, "Invalid MODEL_SERVICE_CONFIG JSON: " + error.getMessage(), error);
        }
    }
}

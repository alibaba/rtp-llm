package org.flexlb.config;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;

/** Loads the single, strict FLEXLB_CONFIG JSON document. */
@Slf4j
@Component
public class ConfigService {

    static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";

    private static final ObjectMapper STRICT_MAPPER = JsonMapper.builder()
            .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
            .enable(DeserializationFeature.FAIL_ON_NUMBERS_FOR_ENUMS)
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
            .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS)
            .build();

    private final FlexlbConfig flexlbConfig;

    public ConfigService() {
        this(System.getenv());
    }

    ConfigService(Map<String, String> environment) {
        String document = environment.get(FLEXLB_CONFIG_ENV);
        this.flexlbConfig = document == null ? new FlexlbConfig() : parse(document);
        FlexlbConfigValidator.validate(flexlbConfig);
        logEffectiveConfig(flexlbConfig);
    }

    public static FlexlbConfig parse(String document) {
        try {
            JsonNode tree = STRICT_MAPPER.readTree(document);
            rejectJsonNull(tree, "$");
            FlexlbConfigValidator.validateDocumentShape(tree);
            FlexlbConfig config = STRICT_MAPPER.treeToValue(
                    tree, FlexlbConfig.class);
            FlexlbConfigValidator.validate(config);
            return config;
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(FLEXLB_CONFIG_ENV,
                    "Invalid FLEXLB_CONFIG JSON: " + error.getMessage(), error);
        }
    }

    public FlexlbConfig loadBalanceConfig() {
        return flexlbConfig;
    }

    public synchronized void updateTrafficPolicy(TrafficPolicyConfig groupSelector) {
        if (groupSelector == null) {
            throw new IllegalArgumentException("groupSelector cannot be null");
        }
        TrafficPolicyConfig.validate(groupSelector);
        flexlbConfig.getRouter().setGroupSelector(groupSelector);
        log.info("Group selector updated: rules={}", groupSelector.getRules().size());
    }

    private static void rejectJsonNull(JsonNode node, String path) {
        if (node == null || node.isNull()) {
            throw new ConfigValidationException(FLEXLB_CONFIG_ENV,
                    "JSON null is not allowed at " + path);
        }
        if (node.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                rejectJsonNull(field.getValue(), path + "." + field.getKey());
            }
        } else if (node.isArray()) {
            for (int index = 0; index < node.size(); index++) {
                rejectJsonNull(node.get(index), path + "[" + index + "]");
            }
        }
    }

    private static void logEffectiveConfig(FlexlbConfig config) {
        String scheduler = config.isDirect() ? "DIRECT" : "QUEUE";
        String ordering = config.isDirect() ? "N/A"
                : config.isPriorityOrdering() ? "PRIORITY" : "FIFO";
        String decision = config.isDirect() ? "N/A"
                : config.isFixedWindowDecision() ? "FIXED_WINDOW" : "SINGLE";
        String dispatcher = config.getDispatcher().typeName();
        log.info("FlexLB config loaded: schemaVersion={}, scheduler={}, ordering={}, decision={}, "
                        + "dispatcher={}, prefillCandidateChoice={}, groupRules={}",
                config.getSchemaVersion(), scheduler, ordering, decision, dispatcher,
                config.getRouter().getRoles().getPrefill()
                        .getCandidateChoice().getType(),
                config.getRouter().getGroupSelector() == null ? 0
                        : config.getRouter().getGroupSelector().getRules().size());
    }
}

package org.flexlb.config;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;

/**
 * Loads Master behavior and model topology from their startup documents.
 */
@Slf4j
@Component
public class ConfigService {
    static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";
    static final String MODEL_SERVICE_CONFIG_ENV = "MODEL_SERVICE_CONFIG";
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
    private final ServiceRoute modelServiceConfig;

    public ConfigService() {
        this(System.getenv());
    }

    ConfigService(Map<String, String> environment) {
        this(environment.get(FLEXLB_CONFIG_ENV), environment.get(MODEL_SERVICE_CONFIG_ENV));
    }

    ConfigService(String document) {
        this(document, null);
    }

    private ConfigService(String document, String topologyDocument) {
        if (document == null || document.isBlank()) {
            throw new ConfigValidationException(FLEXLB_CONFIG_ENV,
                    "is required; configure requestLifecycle.request.timeoutMs");
        }
        this.flexlbConfig = parse(document);
        this.modelServiceConfig = topologyDocument == null ? null : parseModelServiceConfig(topologyDocument);
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

    public ServiceRoute modelServiceConfig() {
        if (modelServiceConfig == null) {
            throw new IllegalStateException("master load balancer env MODEL_SERVICE_CONFIG is empty");
        }
        return modelServiceConfig;
    }

    public synchronized void updateTrafficPolicy(TrafficPolicyConfig groupSelector) {
        if (groupSelector == null) {
            throw new IllegalArgumentException("groupSelector cannot be null");
        }
        TrafficPolicyConfig.validate(groupSelector);
        flexlbConfig.getRouter().setGroupSelector(groupSelector);
        log.info("Group selector updated: rules={}", groupSelector.getRules().size());
    }

    static ServiceRoute parseModelServiceConfig(String document) {
        if (document == null || document.isBlank()) {
            throw new IllegalStateException("master load balancer env MODEL_SERVICE_CONFIG is empty");
        }
        return JsonUtils.toObject(document, ServiceRoute.class);
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
                        + "dispatcher={}, prefillSelection=BEST_ONLY, "
                        + "decodeCostExpression={}, groupRules={}",
                config.getSchemaVersion(), scheduler, ordering, decision, dispatcher,
                config.getRouter().getRoles().getDecode().getCostEstimator().getExpression(),
                config.getRouter().getGroupSelector() == null ? 0
                        : config.getRouter().getGroupSelector().getRules().size());
    }
}

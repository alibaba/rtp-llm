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
import java.util.Set;

/** Loads the single, strict FLEXLB_CONFIG JSON document. */
@Slf4j
@Component
public class ConfigService {

    static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";

    private static final Set<String> REMOVED_LEGACY_ENV_VARS = Set.copyOf(
            java.util.Arrays.asList("""
                    LOAD_BALANCE_STRATEGY DECODE_LOAD_BALANCE_STRATEGY VIT_LOAD_BALANCE_STRATEGY
                    WEIGHTED_CACHE_DECAY_FACTOR CACHE_HIT_TIME_WINDOW_MS CACHE_HIT_MAX_CACHE_KEYS
                    CACHE_HIT_WINDOW_WRITE_ENABLED CACHE_HIT_METRIC_REPORT_ENABLED
                    CACHE_HIT_TRACE_LOG_ENABLED CACHE_HIT_THEORY_LOG_ENABLED MAX_QUEUE_SIZE
                    MAX_RETRY_COUNT PREFILL_QUEUE_SIZE_THRESHOLD DECODE_AVAILABLE_MEMORY_THRESHOLD
                    DECODE_CONCURRENCY_LIMIT HYSTERESIS_BIAS_PERCENT SCHEDULE_WORKER_SIZE
                    RESOURCE_CHECK_INTERVAL_MS MAX_PREFILL_QUEUE_SIZE DECODE_FULL_SPEED_THRESHOLD
                    DECODE_STOP_THRESHOLD NETTY_SELECT_THREAD_MULTIPLIER NETTY_WORKER_THREAD_MULTIPLIER
                    GRPC_CLIENT_EXECUTOR_CORE_SIZE GRPC_CLIENT_EXECUTOR_MAX_SIZE
                    GRPC_CLIENT_EXECUTOR_QUEUE_SIZE GRPC_CLIENT_EVENT_LOOP_THREADS
                    GRPC_SERVER_WORKER_EVENT_LOOP_THREADS HTTP_NETTY_EVENT_LOOP_THREADS
                    HTTP_NETTY_EVENT_EXECUTOR_THREADS HTTP_NETTY_EVENT_EXECUTOR_QUEUE_SIZE
                    HTTP_REQUEST_EXECUTOR_CORE_SIZE HTTP_REQUEST_EXECUTOR_MAX_SIZE
                    HTTP_REQUEST_EXECUTOR_QUEUE_SIZE ENGINE_SYNC_EXECUTOR_CORE_SIZE
                    ENGINE_SYNC_EXECUTOR_MAX_SIZE STATUS_CHECK_EXECUTOR_CORE_SIZE
                    STATUS_CHECK_EXECUTOR_MAX_SIZE SERVICE_DISCOVERY_MAX_SIZE TRAFFIC_POLICY
                    TRAFFIC_POLICY_CONFIG TRAFFIC_POLICY_CONFIG_FILE STRATEGY_CONFIGS
                    PREFILL_TIME_FORMULA DEFAULT_SCHEDULE_MODE FLEXLB_BATCH_ENABLED ENABLE_QUEUEING
                    FLEXLB_BATCH_SIZE_MAX FLEXLB_BATCH_WINDOW_MS FLEXLB_BATCH_MIN_SIZE
                    FLEXLB_BATCH_EMERGENCY_BUDGET_MS FLEXLB_BATCH_DISPATCH_GUARD_MS
                    FLEXLB_BATCH_ARRIVAL_EMA_ALPHA FLEXLB_BATCH_ARRIVAL_WAIT_GUARD_MS
                    FLEXLB_BATCH_SLO_MAX_INFLIGHT_BATCHES FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES
                    FLEXLB_BATCH_ENQUEUE_DEADLINE_MS FLEXLB_INFLIGHT_TTL_MS
                    FLEXLB_BATCH_DISPATCH_POOL_SIZE FLEXLB_BATCH_DISPATCH_QUEUE_SIZE
                    FLEXLB_BATCH_MAX_CAPACITY FLEXLB_BATCH_SCAN_AHEAD FLEXLB_BATCH_QUEUE_MAX_SIZE
                    FLEXLB_BATCH_MAX_INFLIGHT FLEXLB_BATCH_ALGORITHM FLEXLB_BATCH_FIXED_WAIT_MS
                    FLEXLB_BATCH_PREDICT_THRESHOLD_MS COST_SLO_FILTER_ENABLED COST_SLO_MS
                    COST_SLO_RISK_MARGIN_MS COST_SLO_BUCKETS COST_HOTSPOT_MULTIPLIER
                    COST_IMBALANCE_MULTIPLIER COST_FORMULA SCORE_TIE_RANDOM_ENABLED
                    SCORE_TIE_THRESHOLD_PCT SCORE_TIE_THRESHOLD_MS
                    SHORTEST_TTFT_CANDIDATE_POOL_MODE SHORTEST_TTFT_CANDIDATE_POOL_RATIO
                    SHORTEST_TTFT_CANDIDATE_POOL_MIN_SIZE SHORTEST_TTFT_CANDIDATE_POOL_SIZE
                    CACHE_AFFINITY_ENABLED CACHE_AFFINITY_MAX_EXTRA_TTFT_MS
                    CACHE_AFFINITY_MIN_HIT_RATE PREFILL_PREDICTOR_TYPE PREFILL_LB_TIMEOUT_MS
                    DECODE_HOTSPOT_MULTIPLIER DECODE_IMBALANCE_MULTIPLIER MAX_NEW_TOKENS_CAP
                    AUTO_TPM_ENABLED AUTO_TPM_DEFAULT_PRIORITY AUTO_TPM_SLO_LENGTH_BUCKETS
                    AUTO_TPM_PRIORITY_SLO_MULTIPLIERS AUTO_TPM_PREFILL_QUEUE_EVICT_ENABLED
                    AUTO_TPM_DECODE_RESERVED_EVICT_ENABLED AUTO_TPM_PLAN_CACHE_HIT_BENEFIT_CAP
                    AUTO_TPM_POST_SUCCESS_SOFT_TIMEOUT_MS AUTO_TPM_POST_SUCCESS_BACKPRESSURE_LIMIT
                    AUTO_TPM_DECODE_ACCEPTED_EVICT_ENABLED AUTO_TPM_CANCEL_ACK_TIMEOUT_MS
                    AUTO_TPM_CANCEL_COMPLETION_TIMEOUT_MS AUTO_TPM_COMMIT_STRATEGY
                    AUTO_TPM_VICTIM_GUARD_MODE WORKER_TIMEOUT_MS WORKER_TIMEOUT_US TASK_TIMEOUT_US
                    CACHE_STATUS_DIFF_SIZE CACHE_STATUS_MIN_INTERVAL_MS CACHE_STATUS_MAX_INTERVAL_MS
                    SYNC_STATUS_INTERVAL SYNC_REQUEST_TIMEOUT_MS WHALE_CACHE_DEBUG_MODE
                    VIT_SYNC_REQUEST_TIMEOUT_MS VIT_WORKER_TIMEOUT_US VIT_RETAIN_ALIVE_ON_TIMEOUT
                    FLEXLB_MONITOR_MODE ENGINE_TYPE FLEXLB_ENGINE_TYPE
                    BATCH_SCHEDULE_MAX_COUNT FLEXLB_BATCH_SCHEDULE_MAX_COUNT
                    BATCH_LOAD_BALANCE_STRATEGY FLEXLB_BATCH_LOAD_BALANCE_STRATEGY
                    """.trim().split("\\s+")));

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
        rejectRemovedLegacyEnvironment(environment);
        String document = environment.get(FLEXLB_CONFIG_ENV);
        this.flexlbConfig = document == null ? new FlexlbConfig() : parse(document);
        FlexlbConfigValidator.validate(flexlbConfig);
        logEffectiveConfig(flexlbConfig);
    }

    private static void rejectRemovedLegacyEnvironment(Map<String, String> environment) {
        var removed = environment.keySet().stream()
                .filter(REMOVED_LEGACY_ENV_VARS::contains)
                .sorted()
                .toList();
        if (removed.isEmpty()) {
            return;
        }
        String monitorMigration = removed.contains("FLEXLB_MONITOR_MODE")
                ? " Replace FLEXLB_MONITOR_MODE with FLEXLB_MONITOR_METRIC_WHITELIST"
                        + " (the bare flexlb_ prefix exposes all FlexLB metrics)."
                : "";
        throw new ConfigValidationException(
                "environment",
                "Removed legacy FlexLB environment variables are no longer read: "
                        + String.join(", ", removed)
                        + ". Migrate scheduling behavior into FLEXLB_CONFIG with schemaVersion 2."
                        + monitorMigration);
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

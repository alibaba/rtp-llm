package org.flexlb.config.loading;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * Rejects null values and behavior settings in model topology documents before binding configuration.
 */
public final class ConfigJsonValidation {
    private static final Set<String> MODEL_BEHAVIOR_FIELDS =
            Set.of(
                    "connect_timeout_ms",
                    "connectTimeoutMs",
                    "read_timeout_ms",
                    "readTimeoutMs",
                    "poll_interval_ms",
                    "pollIntervalMs",
                    "max_idle_connections",
                    "maxIdleConnections",
                    "keep_alive_duration_ms",
                    "keepAliveDurationMs",
                    "request_timeout_ms",
                    "requestTimeoutMs",
                    "leader_refresh_interval_ms",
                    "leaderRefreshIntervalMs",
                    "heartbeat_failure_threshold",
                    "heartbeatFailureThreshold",
                    "query_failure_threshold",
                    "queryFailureThreshold",
                    "max_query_retry_count",
                    "maxQueryRetryCount",
                    "recovery_success_threshold",
                    "recoverySuccessThreshold",
                    "global_kvs_host_count",
                    "globalKvsHostCount",
                    "enable_p2p",
                    "enableP2p",
                    "local_standby",
                    "localStandby");

    private ConfigJsonValidation() {}

    public static void rejectJsonNull(JsonNode node, String path, String sourceName) {
        if (node == null || node.isNull()) {
            throw new ConfigValidationException(sourceName, "JSON null is not allowed at " + path);
        }
        if (node.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                rejectJsonNull(field.getValue(), path + "." + field.getKey(), sourceName);
            }
        } else if (node.isArray()) {
            for (int index = 0; index < node.size(); index++) {
                rejectJsonNull(node.get(index), path + "[" + index + "]", sourceName);
            }
        }
    }

    public static void rejectModelBehaviorFields(JsonNode node, String sourceName) {
        org.flexlb.util.JsonUtils.forEachField(
                node,
                "$",
                null,
                (fieldPath, parentField, fieldName) -> {
                    boolean rootBehavior = "$.load_balance".equals(fieldPath);
                    boolean enableBehavior =
                            "enabled".equals(fieldName)
                                    && ("kvcm".equals(parentField)
                                            || "optimizer".equals(parentField));
                    if (rootBehavior
                            || enableBehavior
                            || MODEL_BEHAVIOR_FIELDS.contains(fieldName)) {
                        throw new ConfigValidationException(
                                sourceName,
                                fieldPath
                                        + " is a FlexLB behavior field; configure it in FLEXLB_CONFIG");
                    }
                });
    }
}

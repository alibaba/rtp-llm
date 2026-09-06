package org.flexlb.util;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigValidationException;
import org.flexlb.enums.StatusEnum;
import org.flexlb.exception.FlexLBException;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
public class JsonUtils {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final ObjectMapper STRICT_MAPPER = JsonMapper.builder()
            .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
            .enable(DeserializationFeature.FAIL_ON_NUMBERS_FOR_ENUMS)
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
            .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS)
            .serializationInclusion(JsonInclude.Include.NON_NULL)
            .build();

    private static final Set<String> MODEL_BEHAVIOR_FIELDS = Set.of(
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
            "p2p_host_count",
            "p2pHostCount",
            "local_standby",
            "localStandby");

    private static final ObjectWriter WRITER;

    private static final ObjectMapper MAPPER_WITH_INDENT = new ObjectMapper();

    private static final ObjectWriter WRITER_WITH_INDENT;

    static {
        MAPPER.registerModule(new JavaTimeModule());
        MAPPER.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        MAPPER.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        MAPPER.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
        MAPPER.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        WRITER = MAPPER.writer();

        MAPPER_WITH_INDENT.registerModule(new JavaTimeModule());
        MAPPER_WITH_INDENT.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        MAPPER_WITH_INDENT.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
        MAPPER_WITH_INDENT.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        MAPPER_WITH_INDENT.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        MAPPER_WITH_INDENT.enable(SerializationFeature.INDENT_OUTPUT);
        MAPPER_WITH_INDENT.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
        WRITER_WITH_INDENT = MAPPER_WITH_INDENT.writer();
    }

    /**
     * Convert a json string to a java object.
     *
     * @param input  The input.
     * @param clazz The expected java object type.
     * @return The java object.
     */
    public static <I, T> T toObject(I input, Class<T> clazz) throws FlexLBException {
        ObjectReader reader = MAPPER.readerFor(clazz);
        try {
            T res;
            if (input instanceof String string) {
                res = reader.readValue(string);
            } else if (input instanceof byte[] bytes) {
                res = reader.readValue(bytes);
            } else {
                throw new IllegalArgumentException("Unsupported input type: " + input.getClass());
            }
            if (res == null) {
                throw new RuntimeException("The result of json mapper is null.");
            }
            return res;
        } catch (Throwable e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException("msg=" + e.getMessage() + ", text=" + input, e);
        }
    }

    /**
     * Convert a json string to a java object.
     *
     * @param text  The json string.
     * @param clazz The expected java object type.
     * @return The java object.
     */
    public static <T> T toObject(String text, Class<T> clazz) throws FlexLBException {
        ObjectReader reader = MAPPER.readerFor(clazz);
        try {
            T res = reader.readValue(text);
            if (res == null) {
                throw new RuntimeException("The result of json mapper is null.");
            }
            return res;
        } catch (Throwable e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException("msg=" + e.getMessage() + ", text=" + text, e);
        }
    }

    public static <T> T toObjectOrNull(String text, Class<T> clazz) {
        ObjectReader reader = MAPPER.readerFor(clazz);
        try {
            return reader.readValue(text);
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Convert byte array to a java object.
     *
     * @param content The byte array.
     * @param clazz   The expected java object type.
     * @return The java object.
     */
    public static <T> T toObject(byte[] content, Class<T> clazz) throws FlexLBException {
        ObjectReader reader = MAPPER.readerFor(clazz);
        try {
            return reader.readValue(content);
        } catch (IOException e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException(e);
        }
    }

    /**
     * Convert a json string to a java object.
     *
     * @param text The json string.
     * @param type The type reference.
     * @return The java object.
     */
    public static <T> T toObject(String text, TypeReference<T> type) throws FlexLBException {
        ObjectReader reader = MAPPER.readerFor(type);
        try {
            return reader.readValue(text);
        } catch (IOException e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException(text, e);
        }
    }

    public static <Result> Result toObject(byte[] bodyBytes, TypeReference<Result> type) {
        ObjectReader reader = MAPPER.readerFor(type);
        try {
            return reader.readValue(bodyBytes);
        } catch (IOException e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException(e);
        }
    }

    /**
     * Convert a json tree to a java object.
     *
     * @param tree  The json tree.
     * @param clazz The java type.
     * @return The Java object.
     */
    public static <T> T toObject(JsonNode tree, Class<T> clazz) throws FlexLBException {
        ObjectReader reader = MAPPER.readerFor(clazz);
        try {
            return reader.readValue(tree);
        } catch (IOException e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException(e);
        }
    }

    public static <T> List<T> toList(String text, Class<T> valueType) throws FlexLBException {
        CollectionType type = MAPPER.getTypeFactory().constructCollectionType(List.class, valueType);
        ObjectReader reader = MAPPER.readerFor(type);
        try {
            return reader.readValue(text);
        } catch (IOException e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException(text, e);
        }
    }

    /**
     * Convert an object to json string.
     *
     * @param object The object.
     * @return The json string.
     * @throws FlexLBException Failed to convert.
     */
    public static String toString(Object object) throws FlexLBException {
        try {
            return WRITER.writeValueAsString(object);
        } catch (JsonProcessingException error) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException("Failed to convert object to json string!", error);
        }
    }

    /**
     * Convert a Java object to a json string.
     *
     * @param object The Java object.
     * @return The json string.
     */
    public static String toStringOrEmpty(Object object) {
        try {
            return WRITER.writeValueAsString(object);
        } catch (JsonProcessingException e) {
            if (log.isTraceEnabled()) {
                log.trace("Failed to convert json to string:", e);
            }
            return "";
        }
    }

    /**
     * Convert an object to a formatted string.
     */
    public static String toFormattedString(Object object) {
        try {
            return WRITER_WITH_INDENT.writeValueAsString(object);
        } catch (JsonProcessingException e) {
            if (log.isTraceEnabled()) {
                log.trace("Format json failed:", e);
            }
            return "";
        }
    }

    public static JsonNode readStrictTree(String document) throws JsonProcessingException {
        return STRICT_MAPPER.readTree(document);
    }

    public static <T> T strictTreeToValue(JsonNode tree, Class<T> valueType) throws JsonProcessingException {
        return STRICT_MAPPER.treeToValue(tree, valueType);
    }

    public static <T extends JsonNode> T strictValueToTree(Object value) {
        return STRICT_MAPPER.valueToTree(value);
    }

    public static String toStrictString(Object value) throws JsonProcessingException {
        return STRICT_MAPPER.writeValueAsString(value);
    }

    public static ObjectNode createObjectNode() {
        return STRICT_MAPPER.createObjectNode();
    }

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

    public static void forEachField(JsonNode node, String path, String parentField, JsonFieldConsumer consumer) {
        if (node == null || !node.isContainerNode()) {
            return;
        }
        if (node.isArray()) {
            for (int index = 0; index < node.size(); index++) {
                forEachField(node.get(index), path + "[" + index + "]", parentField, consumer);
            }
            return;
        }
        Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            String fieldPath = path + "." + field.getKey();
            consumer.accept(fieldPath, parentField, field.getKey());
            forEachField(field.getValue(), fieldPath, field.getKey(), consumer);
        }
    }

    public static void rejectModelBehaviorFields(JsonNode node, String sourceName) {
        forEachField(node, "$", null, (fieldPath, parentField, fieldName) -> {
            boolean rootBehavior = "$.load_balance".equals(fieldPath);
            boolean enableBehavior = "enabled".equals(fieldName) && ("kvcm".equals(parentField) || "optimizer".equals(parentField));
            if (rootBehavior || enableBehavior || MODEL_BEHAVIOR_FIELDS.contains(fieldName)) {
                throw new ConfigValidationException(sourceName,
                        fieldPath + " is a FlexLB behavior field; configure it in FLEXLB_CONFIG");
            }
        });
    }

    @FunctionalInterface
    public interface JsonFieldConsumer {

        void accept(String fieldPath, String parentField, String fieldName);
    }
}

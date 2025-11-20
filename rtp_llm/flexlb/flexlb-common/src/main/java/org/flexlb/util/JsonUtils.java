package org.flexlb.util;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.enums.StatusEnum;
import org.flexlb.exception.FlexLBException;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Slf4j
public class JsonUtils {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final ObjectWriter WRITER;

    private static final ObjectReader READER;

    private static final ObjectMapper MAPPER_WITH_INDENT = new ObjectMapper();

    private static final ObjectWriter WRITER_WITH_INDENT;

    static {
        MAPPER.registerModule(new JavaTimeModule());
        MAPPER.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        MAPPER.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        MAPPER.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
        MAPPER.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        WRITER = MAPPER.writer();
        READER = MAPPER.reader();

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

    public static <T> Map<String, T> toMap(String text, Class<T> valueType) throws FlexLBException {
        MapType type = MAPPER.getTypeFactory().constructMapType(Map.class, String.class, valueType);
        ObjectReader reader = MAPPER.readerFor(type);
        try {
            return reader.readValue(text);
        } catch (IOException e) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException(text, e);
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
     * Convert a Java object to bytes
     *
     * @param object The Java object.
     * @return The bytes array
     */
    public static byte[] toBytes(Object object) {
        try {
            return WRITER.writeValueAsBytes(object);
        } catch (JsonProcessingException e) {
            if (log.isTraceEnabled()) {
                log.trace("Failed to convert json to string:", e);
            }
            return "".getBytes();
        }
    }

    /**
     * Convert a java object to a json node.
     *
     * @param object The java object.
     * @return The json node.
     */
    public static JsonNode toTreeNode(Object object) {
        return MAPPER.valueToTree(object);
    }

    /**
     * Convert a string to a json node.
     *
     * @param text The json string.
     * @return The json node.
     */
    public static JsonNode toTreeNode(String text) throws FlexLBException {
        try {
            return READER.readTree(text);
        } catch (JsonProcessingException error) {
            throw StatusEnum.JSON_MAPPER_ERROR.toException("Failed to parse text to json tree!, text=" + text, error);
        }
    }

    public static JsonNode toTreeNodeOrNull(String text) {
        if (StringUtils.isBlank(text)) {
            return null;
        }
        try {
            return READER.readTree(text);
        } catch (JsonProcessingException e) {
            return null;
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

    /**
     * Create an empty tree object.
     *
     * @return The tree object.
     */
    public static ObjectNode createTreeObject() {
        return MAPPER.createObjectNode();
    }

    public static ArrayNode createArrayNode() {
        return MAPPER.createArrayNode();
    }
}

package org.flexlb.util;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.StatusEnum;
import org.flexlb.exception.FlexLBException;

import java.io.IOException;

@Slf4j
public class JsonUtils {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final ObjectWriter WRITER;

    private static final ObjectWriter WRITER_WITH_INDENT;

    static {
        MAPPER.registerModule(new JavaTimeModule());
        MAPPER.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        MAPPER.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        MAPPER.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
        MAPPER.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        WRITER = MAPPER.writer();

        WRITER_WITH_INDENT = MAPPER.writerWithDefaultPrettyPrinter();
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
}

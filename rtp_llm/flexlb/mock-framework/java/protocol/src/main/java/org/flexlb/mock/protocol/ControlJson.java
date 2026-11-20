package org.flexlb.mock.protocol;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * Defines and validates the JSON shapes accepted by the Mock control API.
 */
final class ControlJson {
    static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private ControlJson() {}

    static Map<String, Object> schema(Type type) {
        if (type instanceof ParameterizedType p) {
            Type[] args = p.getActualTypeArguments();
            if (p.getRawType() == List.class) return Map.of("type", "array", "items", schema(args[0]));
            if (p.getRawType() == Map.class && args[0] == String.class)
                return Map.of("type", "object", "values", schema(args[1]));
            throw new IllegalArgumentException("Unsupported generic type: " + type);
        }
        if (!(type instanceof Class<?> c)) throw new IllegalArgumentException("Unsupported type: " + type);
        if (c == String.class) return Map.of("type", "string");
        if (c == int.class || c == Integer.class || c == long.class || c == Long.class)
            return Map.of("type", "integer", "bits", c == int.class || c == Integer.class ? 32 : 64);
        if (c == double.class || c == Double.class) return Map.of("type", "number");
        if (c == boolean.class || c == Boolean.class) return Map.of("type", "boolean");
        if (c.isEnum()) return Map.of("type", "string", "enum",
                Arrays.stream(c.getEnumConstants()).map(Object::toString).toList());
        if (!c.isRecord()) throw new IllegalArgumentException("Unsupported API type: " + type);
        Map<String, Object> fields = new TreeMap<>();
        List<String> required = new ArrayList<>();
        for (RecordComponent field : c.getRecordComponents()) {
            if (!field.isAnnotationPresent(ControlHttpServer.OptionalArgument.class)) required.add(field.getName());
            fields.put(field.getName(), schema(field.getGenericType()));
        }
        return Map.of("type", "object", "properties", fields, "required", required);
    }

    @SuppressWarnings("unchecked")
    static void validate(JsonNode node, Map<String, Object> spec) {
        String type = (String) spec.get("type");
        if (node == null || node.isNull()) throw new IllegalArgumentException("Missing/null value");
        switch (type) {
            case "object" -> {
                require(node.isObject(), "Expected object");
                if (spec.containsKey("values")) {
                    var schema = (Map<String, Object>) spec.get("values");
                    node.forEach(value -> validate(value, schema));
                    return;
                }
                var properties = (Map<String, Map<String, Object>>) spec.get("properties");
                var required = (List<String>) spec.get("required");
                for (String key : required) require(node.has(key), "Missing field: " + key);
                node.fieldNames().forEachRemaining(key -> {
                    require(properties.containsKey(key), "Unknown field: " + key);
                    validate(node.get(key), properties.get(key));
                });
            }
            case "array" -> {
                require(node.isArray(), "Expected array");
                for (var child : node) validate(child, (Map<String, Object>) spec.get("items"));
            }
            case "string" -> {
                require(node.isTextual() && !node.textValue().isBlank(), "Expected nonempty string");
                if (spec.containsKey("enum")) require(((List<?>) spec.get("enum")).contains(node.textValue()), "Invalid enum value");
            }
            case "integer", "number" -> {
                require(node.isNumber() && Double.isFinite(node.doubleValue()), "Expected finite number");
                if (type.equals("integer")) {
                    int bits = ((Number) spec.get("bits")).intValue();
                    require(node.isIntegralNumber() && node.bigIntegerValue().bitLength() < bits, "Integer overflow/type mismatch");
                }
            }
            case "boolean" -> require(node.isBoolean(), "Expected boolean");
            default -> throw new IllegalArgumentException("Unknown schema type: " + type);
        }
    }

    static void require(boolean valid, String message) {
        if (!valid) throw new IllegalArgumentException(message);
    }
}

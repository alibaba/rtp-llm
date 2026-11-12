package org.flexlb.mock.protocol;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import java.io.IOException;
import java.lang.reflect.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;

/**
 * Derives schemas for Java control records, validates and decodes messages, and publishes JSON files atomically.
 */
public final class JsonWire {
    static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private JsonWire() {}

    public static void writeAtomically(Path target, Object value) throws IOException {
        writeAtomically(target, value, JSON.writer(), false);
    }

    public static void writeAtomically(Path target, Object value, ObjectWriter writer, boolean replaceExisting) throws IOException {
        Path temporary = Files.createTempFile(target.toAbsolutePath().getParent(), ".json-", ".tmp");
        try {
            writer.writeValue(temporary.toFile(), value);
            if (replaceExisting) {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

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
            if (!field.isAnnotationPresent(ControlServer.OptionalArgument.class)) required.add(field.getName());
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

    @SuppressWarnings({"unchecked", "rawtypes"})
    static Object decode(JsonNode node, Type type) throws ReflectiveOperationException {
        if (node == null) return null; // Optional record fields only; validation rejects missing required ones.
        if (type instanceof ParameterizedType parameterized) {
            var args = parameterized.getActualTypeArguments();
            if (parameterized.getRawType() == List.class) {
                var values = new ArrayList<>();
                for (var child : node) values.add(decode(child, args[0]));
                return List.copyOf(values);
            }
            var values = new LinkedHashMap<>();
            var fields = node.fields();
            while (fields.hasNext()) { var field = fields.next(); values.put(field.getKey(), decode(field.getValue(), args[1])); }
            return Collections.unmodifiableMap(values);
        }
        var clazz = (Class<?>) type;
        if (clazz.isEnum()) return Enum.valueOf((Class<Enum>) clazz, node.textValue());
        if (!clazz.isRecord()) return JSON.convertValue(node, clazz);
        var fields = clazz.getRecordComponents();
        Object[] args = new Object[fields.length];
        for (int i = 0; i < fields.length; i++) args[i] = decode(node.get(fields[i].getName()), fields[i].getGenericType());
        return clazz.getDeclaredConstructor(Arrays.stream(fields).map(RecordComponent::getType).toArray(Class<?>[]::new)).newInstance(args);
    }
}

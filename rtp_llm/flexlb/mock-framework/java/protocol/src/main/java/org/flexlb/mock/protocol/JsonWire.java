package org.flexlb.mock.protocol;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import java.lang.reflect.*;
import java.math.BigDecimal;
import java.util.*;

/** The generated schema validates both network input and component output. */
final class JsonWire {
    static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
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
                    int bits = ((Number) spec.getOrDefault("bits", 64)).intValue();
                    require(node.isIntegralNumber() && node.bigIntegerValue().bitLength() < bits, "Integer overflow/type mismatch");
                }
                if (spec.containsKey("minimum"))
                    require(node.decimalValue().compareTo(new BigDecimal(spec.get("minimum").toString())) >= 0, "Below minimum");
                if (spec.containsKey("maximum"))
                    require(node.decimalValue().compareTo(new BigDecimal(spec.get("maximum").toString())) <= 0, "Above maximum");
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

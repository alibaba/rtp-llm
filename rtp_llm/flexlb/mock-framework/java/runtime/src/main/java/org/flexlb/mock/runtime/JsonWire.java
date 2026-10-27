package org.flexlb.mock.runtime;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import java.lang.reflect.*;
import java.math.BigDecimal;
import java.util.*;
import java.util.regex.Pattern;
import org.flexlb.mock.api.Variant;
import org.flexlb.mock.api.model.Types;

/** The generated schema validates both network input and component output. */
final class JsonWire {
    static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
    private static final Pattern DURATION = Pattern.compile("(-?(?:\\d+(?:\\.\\d+)?|\\.\\d+))(ns|us|ms|s|m|h)");

    static BigDecimal seconds(String input) {
        var match = DURATION.matcher(input);
        if (!match.matches()) throw new IllegalArgumentException("Expected duration with unit");
        var factor = switch (match.group(2)) {
            case "ns" -> "0.000000001"; case "us" -> "0.000001"; case "ms" -> "0.001";
            case "m" -> "60"; case "h" -> "3600"; default -> "1";
        };
        return new BigDecimal(match.group(1)).multiply(new BigDecimal(factor));
    }

    @SuppressWarnings("unchecked")
    static void validate(JsonNode node, Map<String, Object> spec) {
        String type = (String) spec.get("type");
        if (node == null || node.isNull()) throw new IllegalArgumentException("Missing/null value");
        switch (type) {
            case "union" -> {
                for (var variant : (List<Map<String, Object>>) spec.get("variants")) {
                    try { validate(node, variant); return; } catch (IllegalArgumentException ignored) { }
                }
                throw new IllegalArgumentException("No matching parameter variant");
            }
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
                for (var check : (List<Map<String, Object>>) spec.getOrDefault("checks", List.of())) {
                    var fields = (List<String>) check.get("fields");
                    if (fields.stream().anyMatch(key -> !node.has(key))) continue;
                    var first = node.get(fields.getFirst());
                    boolean valid = switch ((String) check.get("rule")) {
                        case "ordered" -> first.decimalValue().compareTo(node.get(fields.get(1)).decimalValue()) <= 0;
                        case "same_length" -> first.size() == node.get(fields.get(1)).size();
                        case "positive_sum" -> { boolean any = false; for (var item : first) any |= item.doubleValue() > 0; yield any; }
                        default -> throw new IllegalArgumentException("Unknown constraint");
                    };
                    require(valid, "Constraint failed: " + check);
                }
            }
            case "array" -> {
                require(node.isArray(), "Expected array");
                require(node.size() >= ((Number) spec.getOrDefault("minItems", 0)).intValue(), "Too few items");
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
            }
            case "boolean" -> require(node.isBoolean(), "Expected boolean");
            case "duration" -> {
                require(node.isTextual(), "Expected duration string");
                require(seconds(node.textValue()).signum() >= 0, "Negative duration");
            }
            case "range" -> {
                var allowed = (List<String>) spec.getOrDefault("ranges", List.of());
                if (node.isTextual()) require(allowed.contains(node.textValue()), "Unsupported range");
                else {
                    require(allowed.contains("interval") && node.isArray() && node.size() == 2
                        && node.get(0).isTextual() && node.get(1).isTextual(), "Invalid range");
                    var start = seconds(node.get(0).textValue()); var end = seconds(node.get(1).textValue());
                    require(start.compareTo(end) < 0 && end.signum() <= 0, "Invalid range bounds");
                }
            }
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
        if (clazz == Types.Duration.class) return new Types.Duration(node.textValue());
        if (clazz == Types.TimeRange.class) {
            if (node.isTextual()) return node.textValue().equals("all") ? new Types.All() : new Types.Latest();
            return new Types.Interval(new Types.Duration(node.get(0).textValue()), new Types.Duration(node.get(1).textValue()));
        }
        if (clazz.isSealed()) {
            for (var variant : clazz.getPermittedSubclasses()) {
                var tag = variant.getAnnotation(Variant.class);
                if (tag.scalar() && node.isIntegralNumber()) {
                    var field = variant.getRecordComponents()[0];
                    return variant.getDeclaredConstructor(field.getType()).newInstance(decode(node, field.getGenericType()));
                }
                if (!tag.scalar() && node.path("distribution").asText().equals(tag.value())) return decode(node, variant);
            }
            throw new IllegalArgumentException("Unknown sealed variant");
        }
        if (clazz.isEnum()) return Enum.valueOf((Class<Enum>) clazz, node.textValue());
        if (!clazz.isRecord()) return JSON.convertValue(node, clazz);
        var fields = clazz.getRecordComponents();
        Object[] args = new Object[fields.length];
        for (int i = 0; i < fields.length; i++) args[i] = decode(node.get(fields[i].getName()), fields[i].getGenericType());
        return clazz.getDeclaredConstructor(Arrays.stream(fields).map(RecordComponent::getType).toArray(Class<?>[]::new)).newInstance(args);
    }
}

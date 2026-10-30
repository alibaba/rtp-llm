package org.flexlb.mock.api;

import java.lang.reflect.*;
import java.util.*;
import org.flexlb.mock.api.model.Types;

/** Metadata comes from method signatures, not a parallel handwritten catalog. */
public final class InterfaceRegistry {
    private InterfaceRegistry() {}

    public static Map<String, Object> describe(Class<?>... interfaces) {
        Map<String, Object> result = new TreeMap<>();
        for (Class<?> api : interfaces) {
            if (!api.isInterface()) throw new IllegalArgumentException("Not an interface: " + api);
            for (Method method : api.getMethods()) {
                Endpoint endpoint = method.getAnnotation(Endpoint.class);
                if (endpoint == null || method.getParameterCount() != 1
                        || !method.getParameterTypes()[0].isRecord())
                    throw new IllegalArgumentException("API method needs an endpoint and one record argument: " + method);
                if (!endpoint.value().matches("[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)+"))
                    throw new IllegalArgumentException("Invalid interface name: " + endpoint.value());
                if (endpoint.background() && endpoint.kind() != Endpoint.Kind.action)
                    throw new IllegalArgumentException("Only actions can run in background");
                Map<String, Object> spec = new LinkedHashMap<>();
                spec.put("kind", endpoint.kind().name());
                spec.put("background", endpoint.background());
                spec.put("unit", endpoint.unit());
                spec.put("args", schema(method.getGenericParameterTypes()[0]));
                spec.put("returns", schema(method.getGenericReturnType()));
                if (result.putIfAbsent(endpoint.value(), spec) != null)
                    throw new IllegalArgumentException("Duplicate interface: " + endpoint.value());
            }
        }
        return Collections.unmodifiableMap(result);
    }

    private static Map<String, Object> schema(Type type) {
        if (type instanceof ParameterizedType p) {
            Type[] args = p.getActualTypeArguments();
            if (p.getRawType() == List.class) return Map.of("type", "array", "items", schema(args[0]));
            if (p.getRawType() == Map.class && args[0] == String.class)
                return Map.of("type", "object", "values", schema(args[1]));
            throw new IllegalArgumentException("Unsupported generic type: " + type);
        }
        if (!(type instanceof Class<?> c)) throw new IllegalArgumentException("Unsupported type: " + type);
        if (c == String.class) return Map.of("type", "string", "minLength", 1);
        if (c == int.class || c == Integer.class || c == long.class || c == Long.class)
            return Map.of("type", "integer", "bits", c == int.class || c == Integer.class ? 32 : 64);
        if (c == double.class || c == Double.class) return Map.of("type", "number");
        if (c == boolean.class || c == Boolean.class) return Map.of("type", "boolean");
        if (c == Types.Duration.class) return Map.of("type", "duration");
        if (c == Types.TimeRange.class) return Map.of("type", "range");
        if (c.isSealed()) {
            List<Map<String, Object>> variants = new ArrayList<>();
            for (Class<?> subtype : c.getPermittedSubclasses()) {
                Variant variant = subtype.getAnnotation(Variant.class);
                if (variant == null || !subtype.isRecord())
                    throw new IllegalArgumentException("Sealed variant needs a record and @Variant: " + subtype);
                Map<String, Object> spec = new LinkedHashMap<>(schema(subtype));
                if (variant.scalar()) {
                    if (subtype.getRecordComponents().length != 1)
                        throw new IllegalArgumentException("Scalar variant must have one field");
                    @SuppressWarnings("unchecked")
                    var properties = (Map<String, Map<String, Object>>) spec.get("properties");
                    variants.add(properties.values().iterator().next());
                } else {
                    @SuppressWarnings("unchecked")
                    var properties = new TreeMap<>((Map<String, Object>) spec.get("properties"));
                    if (variant.value().isBlank() || properties.containsKey("distribution"))
                        throw new IllegalArgumentException("Invalid distribution variant: " + subtype);
                    properties.put("distribution", Map.of("type", "string", "enum", List.of(variant.value())));
                    spec.put("properties", properties);
                    @SuppressWarnings("unchecked")
                    var required = new ArrayList<>((List<String>) spec.get("required"));
                    required.add("distribution");
                    spec.put("required", required);
                    variants.add(spec);
                }
            }
            return Map.of("type", "union", "variants", variants);
        }
        if (c.isEnum()) return Map.of("type", "string", "enum",
                Arrays.stream(c.getEnumConstants()).map(Object::toString).toList());
        if (!c.isRecord()) throw new IllegalArgumentException("Unsupported API type: " + type);
        Map<String, Object> fields = new TreeMap<>();
        List<String> required = new ArrayList<>();
        for (RecordComponent field : c.getRecordComponents()) {
            Map<String, Object> spec = new LinkedHashMap<>(schema(field.getGenericType()));
            Argument annotation = field.getAnnotation(Argument.class);
            if (annotation == null || !annotation.optional()) required.add(field.getName());
            if (annotation != null) {
                if (!Double.isNaN(annotation.minimum())) spec.put("minimum", annotation.minimum());
                if (!Double.isNaN(annotation.maximum())) spec.put("maximum", annotation.maximum());
                if (annotation.ranges().length > 0) spec.put("ranges", List.of(annotation.ranges()));
            }
            fields.put(field.getName(), spec);
        }
        Map<String, Object> result = new LinkedHashMap<>(Map.of("type", "object", "name", c.getSimpleName(),
                "properties", fields, "required", required, "additionalProperties", false));
        Checks checks = c.getAnnotation(Checks.class);
        if (checks != null) {
            List<Object> rules = new ArrayList<>();
            for (Checks.Check check : checks.value()) {
                if (!fields.keySet().containsAll(List.of(check.fields())))
                    throw new IllegalArgumentException("Constraint names unknown fields in " + c);
                rules.add(Map.of("rule", check.rule().name(), "fields", List.of(check.fields())));
            }
            result.put("checks", rules);
        }
        return result;
    }
}

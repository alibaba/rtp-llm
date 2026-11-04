package org.flexlb.mock.protocol;

import java.lang.reflect.*;
import java.util.*;

/** Derives validation rules from the typed control request and response records. */
final class RecordSchema {
    private RecordSchema() {}

    static Map<String, Object> schema(Type type) {
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
        if (c.isEnum()) return Map.of("type", "string", "enum",
                Arrays.stream(c.getEnumConstants()).map(Object::toString).toList());
        if (!c.isRecord()) throw new IllegalArgumentException("Unsupported API type: " + type);
        Map<String, Object> fields = new TreeMap<>();
        List<String> required = new ArrayList<>();
        for (RecordComponent field : c.getRecordComponents()) {
            Argument annotation = field.getAnnotation(Argument.class);
            if (annotation == null || !annotation.optional()) required.add(field.getName());
            fields.put(field.getName(), schema(field.getGenericType()));
        }
        return Map.of("type", "object", "name", c.getSimpleName(),
                "properties", fields, "required", required, "additionalProperties", false);
    }
}

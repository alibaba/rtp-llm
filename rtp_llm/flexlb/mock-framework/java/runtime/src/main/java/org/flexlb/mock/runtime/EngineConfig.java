package org.flexlb.mock.runtime;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/** Input conversion only. Component parsers own all defaults, units and validation. */
public final class EngineConfig {
    private static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    public static JsonNode read(Path path) throws IOException { return JSON.readTree(path.toFile()); }

    public static ObjectNode normalize(JsonNode input, Set<String> nativeNames, Set<String> mockNames) {
        if (!input.isObject()) throw new IllegalArgumentException("Engine config must be an object");
        var result = JSON.createObjectNode();
        input.fields().forEachRemaining(entry -> {
            String key = entry.getKey();
            if (key.equals("mock") || key.startsWith("mock.")) {
                flatten(result, key, entry.getValue(), mockNames);
            } else {
                String nativeKey = key.toLowerCase(Locale.ROOT);
                if (nativeNames.contains(nativeKey)) {
                    if (!key.equals(nativeKey) && !key.equals(nativeKey.toUpperCase(Locale.ROOT)))
                        throw new IllegalArgumentException("Invalid native setting spelling: " + key);
                    put(result, nativeKey, entry.getValue());
                } else if (nativeKey.startsWith("mock")) {
                    throw new IllegalArgumentException("Mock settings require lowercase dotted names: " + key);
                }
                // Unsupported real options do not affect the simulated model.
            }
        });
        return result;
    }

    private static void flatten(ObjectNode result, String key, JsonNode value, Set<String> names) {
        if (!key.matches("mock(?:\\.[a-z][a-z0-9]*)*"))
            throw new IllegalArgumentException("Invalid Mock setting: " + key);
        if (value.isObject()) {
            if (value.isEmpty()) throw new IllegalArgumentException("Empty Mock setting: " + key);
            value.fields().forEachRemaining(entry -> flatten(result, key + "." + entry.getKey(), entry.getValue(), names));
        } else {
            if (!names.contains(key)) throw new IllegalArgumentException("Unknown Mock setting: " + key);
            put(result, key, value);
        }
    }

    private static void put(ObjectNode result, String key, JsonNode value) {
        if (result.has(key)) throw new IllegalArgumentException("Duplicate setting: " + key);
        if (value.isNull()) throw new IllegalArgumentException("Null setting: " + key);
        result.set(key, value.deepCopy());
    }

    /** Only declared business settings are read; arbitrary process ENV never enters artifacts. */
    public static ObjectNode environment(Map<String, String> env, Set<String> nativeNames, Set<String> mockNames) {
        var result = JSON.createObjectNode();
        for (String key : nativeNames) {
            String value = env.get(key.toUpperCase(Locale.ROOT));
            if (value != null) result.put(key, value);
        }
        var known = new HashSet<String>();
        for (String key : mockNames) {
            String name = key.toUpperCase(Locale.ROOT).replace('.', '_');
            known.add(name);
            if (env.containsKey(name)) result.put(key, env.get(name));
        }
        for (String name : env.keySet()) {
            // Framework startup variables are separate from these reserved business namespaces.
            if ((name.startsWith("MOCK_COMPUTE_") || name.startsWith("MOCK_CACHE_") || name.startsWith("MOCK_VIT_") || name.startsWith("MOCK_MULTIMODAL_")) && !known.contains(name))
                throw new IllegalArgumentException("Unknown Mock environment setting: " + name);
        }
        return result;
    }

    /** Read a signed int64 setting from JSON or a native environment string. */
    public static long signedLong(JsonNode config, String key, long fallback) {
        var value = config.get(key);
        if (value == null) return fallback;
        if (value.isIntegralNumber() || value.isTextual()) {
            try { return Long.parseLong(value.asText()); }
            catch (NumberFormatException ignored) { }
        }
        throw new IllegalArgumentException("Invalid integer setting: " + key);
    }

    private EngineConfig() {}
}

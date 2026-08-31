package org.flexlb.service.config.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ConfigSchemaVersion;

import java.util.HashMap;
import java.util.Map;
import java.util.OptionalInt;

public final class ConfigDocumentParserResolver {

    static final String CONFIG_SCHEMA_VERSION_ENV = "FLEXLB_CONFIG_SCHEMA_VERSION";

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Map<Integer, ConfigDocumentParser> PARSERS = new HashMap<>();

    private ConfigDocumentParserResolver() {}

    public static synchronized void register(ConfigDocumentParser parser) {
        ConfigDocumentParser registered = PARSERS.get(parser.schemaVersion());
        if (registered != null && registered.getClass() != parser.getClass()) {
            throw new IllegalStateException("Multiple ConfigDocumentParsers registered for schemaVersion " + parser.schemaVersion());
        }
        PARSERS.put(parser.schemaVersion(), parser);
    }

    public static ConfigDocumentParser resolve(String rawFlexlbConfig) {
        int schemaVersion = declaredVersion(rawFlexlbConfig).orElseGet(ConfigDocumentParserResolver::environmentVersion);
        ConfigDocumentParser parser = PARSERS.get(schemaVersion);
        if (parser == null) {
            throw new IllegalArgumentException("No ConfigDocumentParser registered for schemaVersion " + schemaVersion);
        }
        return parser;
    }

    private static OptionalInt declaredVersion(String rawFlexlbConfig) {
        if (rawFlexlbConfig == null || rawFlexlbConfig.isBlank()) {
            return OptionalInt.empty();
        }
        try {
            JsonNode document = MAPPER.readTree(rawFlexlbConfig);
            if (document == null || !document.isObject() || !document.has("schemaVersion")) {
                return OptionalInt.empty();
            }
            JsonNode schemaVersion = document.get("schemaVersion");
            if (!schemaVersion.isIntegralNumber() || !schemaVersion.canConvertToInt()) {
                throw new IllegalArgumentException("schemaVersion must be an integer");
            }
            return OptionalInt.of(schemaVersion.intValue());
        } catch (IllegalArgumentException error) {
            throw error;
        } catch (Exception error) {
            return OptionalInt.empty();
        }
    }

    private static int environmentVersion() {
        String configuredVersion = StringUtils.trimToNull(System.getenv(CONFIG_SCHEMA_VERSION_ENV));
        if (configuredVersion == null) {
            return ConfigSchemaVersion.V0_COMPATIBILITY;
        }
        try {
            return Integer.parseInt(configuredVersion);
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(CONFIG_SCHEMA_VERSION_ENV + " must be an integer", error);
        }
    }
}

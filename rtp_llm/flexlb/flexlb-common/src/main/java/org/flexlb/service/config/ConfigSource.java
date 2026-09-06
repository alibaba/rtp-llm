package org.flexlb.service.config;

import org.flexlb.service.config.parser.ConfigDocumentParser;
import org.flexlb.service.config.parser.ConfigDocumentParserResolver;

import java.util.function.Consumer;

public interface ConfigSource extends AutoCloseable {

    String name();

    int priority();

    void setUpdateListener(Consumer<String> listener);

    String load() throws Exception;

    default String loadModelServiceConfig() {
        return null;
    }

    default NormalizedConfig loadConfig() throws Exception {
        return normalize(load());
    }

    default NormalizedConfig normalize(String rawFlexlbConfig) {
        ConfigDocumentParser parser = ConfigDocumentParserResolver.resolve(rawFlexlbConfig);
        return parser.parse(rawFlexlbConfig, loadModelServiceConfig());
    }

    @Override
    default void close() throws Exception {}
}

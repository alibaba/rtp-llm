package org.flexlb.it.configuration;

import org.flexlb.service.config.ConfigSource;

import java.util.function.Consumer;

/**
 * Immutable, high-priority configuration source for a test application context.
 *
 * <p>It lets an initializer insert dynamically allocated fake-server ports through the normal
 * {@code ConfigService} merge path without changing environment or Nacos production behavior.
 */
final class StaticConfigSource implements ConfigSource {

    private final String content;

    StaticConfigSource(String content) {
        this.content = content;
    }

    @Override
    public String name() {
        return "integration-test";
    }

    @Override
    public int priority() {
        return 100;
    }

    @Override
    public void setUpdateListener(Consumer<String> listener) {}

    @Override
    public String load() {
        return content;
    }
}

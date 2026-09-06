package org.flexlb.service.config;

import java.util.function.Consumer;

public interface ConfigSource extends AutoCloseable {

    String name();

    int priority();

    void setUpdateListener(Consumer<String> listener);

    String load() throws Exception;

    @Override
    default void close() throws Exception {}
}

package org.flexlb.config;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class InternalRuntimeSettingsTest {

    @Test
    void should_keep_upstream_defaults_without_env_overrides() {
        InternalRuntimeSettings settings = new InternalRuntimeSettings(Map.of());

        assertEquals(64, settings.getBatchDispatchThreads());
        assertEquals(256, settings.getBatchDispatchQueueCapacity());
    }

    @Test
    void should_apply_positive_env_overrides_for_batch_dispatch_sizing() {
        Map<String, String> env = new HashMap<>();
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_THREADS_ENV, "128");
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_QUEUE_CAPACITY_ENV, "2048");

        InternalRuntimeSettings settings = new InternalRuntimeSettings(env);

        assertEquals(128, settings.getBatchDispatchThreads());
        assertEquals(2048, settings.getBatchDispatchQueueCapacity());
    }

    @Test
    void should_fall_back_to_defaults_for_missing_or_blank_env_values() {
        Map<String, String> env = new HashMap<>();
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_THREADS_ENV, "  ");
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_QUEUE_CAPACITY_ENV, "");

        InternalRuntimeSettings settings = new InternalRuntimeSettings(env);

        assertEquals(InternalRuntimeSettings.DEFAULT_BATCH_DISPATCH_THREADS, settings.getBatchDispatchThreads());
        assertEquals(InternalRuntimeSettings.DEFAULT_BATCH_DISPATCH_QUEUE_CAPACITY,
                settings.getBatchDispatchQueueCapacity());
    }

    @Test
    void should_fall_back_to_defaults_for_invalid_env_values() {
        Map<String, String> env = new HashMap<>();
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_THREADS_ENV, "not-a-number");
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_QUEUE_CAPACITY_ENV, "-5");

        InternalRuntimeSettings settings = new InternalRuntimeSettings(env);

        assertEquals(InternalRuntimeSettings.DEFAULT_BATCH_DISPATCH_THREADS, settings.getBatchDispatchThreads());
        assertEquals(InternalRuntimeSettings.DEFAULT_BATCH_DISPATCH_QUEUE_CAPACITY,
                settings.getBatchDispatchQueueCapacity());
    }

    @Test
    void should_fall_back_to_default_when_env_value_is_zero() {
        Map<String, String> env = new HashMap<>();
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_QUEUE_CAPACITY_ENV, "0");

        InternalRuntimeSettings settings = new InternalRuntimeSettings(env);

        assertEquals(InternalRuntimeSettings.DEFAULT_BATCH_DISPATCH_QUEUE_CAPACITY,
                settings.getBatchDispatchQueueCapacity());
    }

    @Test
    void should_trim_env_values_before_parsing() {
        Map<String, String> env = new HashMap<>();
        env.put(InternalRuntimeSettings.BATCH_DISPATCH_THREADS_ENV, " 96 ");

        InternalRuntimeSettings settings = new InternalRuntimeSettings(env);

        assertEquals(96, settings.getBatchDispatchThreads());
    }
}

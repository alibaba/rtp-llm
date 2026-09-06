package org.flexlb.service.config;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ConfigServiceTest {

    private ConfigService configService;

    @AfterEach
    void closeConfigService() {
        if (configService != null) {
            configService.close();
        }
    }

    @Test
    void loadsEnabledConfigSourcesByPriority() {
        EnvironmentConfigSource environmentSource = environmentSource(Map.of(
                "FLEXLB_CONFIG", """
                        {"router":{"availabilityHysteresisPercent":20}}
                        """));
        FakeConfigSource nacosSource = new FakeConfigSource(
                "Nacos",
                200,
                """
                        {
                          "scheduler":{"type":"DIRECT"},
                          "dispatcher":{"type":"NON_BATCH"},
                          "router":{"availabilityHysteresisPercent":9}
                        }
                        """);

        ConfigService service = createService(List.of(nacosSource, environmentSource));

        assertThat(service.loadBalanceConfig().isDirect()).isTrue();
        assertThat(service.loadBalanceConfig().getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(9);
    }

    @Test
    void doesNotLoadUnregisteredConfigSources() {
        FakeConfigSource unregisteredSource = new FakeConfigSource(
                "unregistered",
                200,
                """
                        {"scheduler":{"type":"DIRECT"},"dispatcher":{"type":"NON_BATCH"}}
                        """);

        ConfigService service = createService(List.of(environmentSource(Map.of())));

        assertThat(unregisteredSource.loaded).isFalse();
        assertThat(service.loadBalanceConfig().isQueue()).isTrue();
    }

    @Test
    void failsFastWhenInitialSourceReadFails() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                new IllegalStateException("Nacos unavailable"));

        assertThatThrownBy(() -> createService(List.of(
                environmentSource(Map.of()),
                source)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Failed to initialize FlexLB configuration from Nacos")
                .hasRootCauseMessage("Nacos unavailable");
        assertThat(source.closed).isTrue();
    }

    @Test
    void failsFastForMissingEmptyOrInvalidInitialContent() {
        assertInvalidInitialContent(null, "must not be blank");
        assertInvalidInitialContent("  ", "must not be blank");
        assertInvalidInitialContent("{}", "at least one FlexlbConfig field");
        assertInvalidInitialContent("[]", "must be a JSON object");
        assertInvalidInitialContent(
                "{\"scheduler\":{\"type\":\"INVALID\"}}",
                "Could not resolve type id 'INVALID'");
    }

    @Test
    void rejectsUnknownSourceFields() {
        assertInvalidInitialContent(
                "{\"unknownField\":1}",
                "Unrecognized field");
    }

    @Test
    void rejectsScalarCoercionInSourceFields() {
        assertInvalidInitialContent(
                "{\"schemaVersion\":\"1\"}",
                "Cannot coerce String value");
    }

    @Test
    void mergesNestedConfigurationWithoutResettingSiblingFields() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"router\":{\"availabilityHysteresisPercent\":9}}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        assertThat(service.loadBalanceConfig().getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(9);
        assertThat(service.loadBalanceConfig().getRouter().getRoles()).isNotNull();
    }

    @Test
    void keepsModelServiceConfigIndependentFromDynamicFlexlbSources() throws Exception {
        ConfigService service = new EnvironmentVariables(Map.of(
                "MODEL_SERVICE_CONFIG",
                "{\"service_id\":\"environment-service\",\"role_endpoints\":[]}",
                "FLEXLB_CONFIG",
                "{\"router\":{\"availabilityHysteresisPercent\":20}}"))
                .execute(() -> {
                    EnvironmentConfigSource environmentSource =
                            new EnvironmentConfigSource();
                    environmentSource.initialize();
                    ConfigService.register(new FakeConfigSource(
                            "Nacos",
                            200,
                            "{\"router\":{\"availabilityHysteresisPercent\":9}}"));
                    return new ConfigService();
                });
        configService = service;

        assertThat(service.loadModelServiceConfig().getServiceId())
                .isEqualTo("environment-service");
        assertThat(service.loadBalanceConfig().getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(9);
    }

    @Test
    void runtimeUpdatesKeepMissingFieldsAndReplaceSnapshot() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                """
                        {
                          "router":{"availabilityHysteresisPercent":9},
                          "observability":{"logging":{"level":"warn"}}
                        }
                        """);
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        FlexlbConfig initialSnapshot = service.loadBalanceConfig();
        source.emit("{\"router\":{\"availabilityHysteresisPercent\":10}}");
        FlexlbConfig updatedSnapshot = service.loadBalanceConfig();

        assertThat(updatedSnapshot).isNotSameAs(initialSnapshot);
        assertThat(initialSnapshot.getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(9);
        assertThat(updatedSnapshot.getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(10);
        assertThat(updatedSnapshot.getObservability().getLogging().getLevel())
                .isEqualTo(LogLevel.WARN);
    }

    @Test
    void notifiesListenerWithCurrentAndRuntimeConfigurations() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"router\":{\"availabilityHysteresisPercent\":9}}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<Long> updates = new ArrayList<>();

        service.addUpdateListener(config -> updates.add(
                config.getRouter().getAvailabilityHysteresisPercent()));
        source.emit("{\"router\":{\"availabilityHysteresisPercent\":10}}");

        assertThat(updates).containsExactly(9L, 10L);
    }

    @Test
    void rejectsInvalidRuntimeUpdatesAndKeepsLastKnownGoodSnapshot() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"router\":{\"availabilityHysteresisPercent\":9}}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        FlexlbConfig lastKnownGood = service.loadBalanceConfig();

        source.emit("");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        source.emit("{}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        source.emit("{\"unknownField\":1}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
    }

    @Test
    void closesRegisteredConfigSources() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"router\":{\"availabilityHysteresisPercent\":9}}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        service.close();

        assertThat(source.closed).isTrue();
    }

    private ConfigService createService(List<ConfigSource> sources) {
        for (ConfigSource source : sources) {
            if (!(source instanceof EnvironmentConfigSource)) {
                ConfigService.register(source);
            }
        }
        configService = new ConfigService();
        return configService;
    }

    private EnvironmentConfigSource environmentSource(Map<String, String> environment) {
        try {
            return new EnvironmentVariables(environment).execute(() -> {
                EnvironmentConfigSource source = new EnvironmentConfigSource();
                source.initialize();
                return source;
            });
        } catch (Exception error) {
            throw new IllegalStateException(error);
        }
    }

    private void assertInvalidInitialContent(String content, String expectedMessage) {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, content);

        assertThatThrownBy(() -> createService(List.of(
                environmentSource(Map.of()),
                source)))
                .isInstanceOf(IllegalStateException.class)
                .hasStackTraceContaining(expectedMessage);
        assertThat(source.closed).isTrue();
    }

    private static final class FakeConfigSource implements ConfigSource {
        private final String name;
        private final int priority;
        private final String initialContent;
        private final Exception loadException;
        private Consumer<String> listener;
        private boolean loaded;
        private boolean closed;

        private FakeConfigSource(String name, int priority, String initialContent) {
            this(name, priority, initialContent, null);
        }

        private FakeConfigSource(String name, int priority, Exception loadException) {
            this(name, priority, null, loadException);
        }

        private FakeConfigSource(
                String name,
                int priority,
                String initialContent,
                Exception loadException) {
            this.name = name;
            this.priority = priority;
            this.initialContent = initialContent;
            this.loadException = loadException;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public int priority() {
            return priority;
        }

        @Override
        public void setUpdateListener(Consumer<String> listener) {
            this.listener = listener;
        }

        @Override
        public String load() throws Exception {
            loaded = true;
            if (loadException != null) {
                throw loadException;
            }
            return initialContent;
        }

        @Override
        public void close() {
            closed = true;
        }

        private void emit(String content) {
            listener.accept(content);
        }
    }
}

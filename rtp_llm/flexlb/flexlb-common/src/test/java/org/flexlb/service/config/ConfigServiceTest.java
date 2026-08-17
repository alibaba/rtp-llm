package org.flexlb.service.config;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.LBConsistencyConfig;
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
                "ENABLE_QUEUEING", "true",
                "MAX_RETRY_COUNT", "4"));
        FakeConfigSource nacosSource = new FakeConfigSource(
                "Nacos",
                200,
                "{\"enableQueueing\":false,\"maxRetryCount\":9}");

        ConfigService service = createService(List.of(nacosSource, environmentSource));

        assertThat(service.loadBalanceConfig().isEnableQueueing()).isFalse();
        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
    }

    @Test
    void doesNotLoadUnregisteredConfigSources() {
        FakeConfigSource unregisteredSource = new FakeConfigSource("unregistered", 200, "{}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of("MAX_RETRY_COUNT", "4"))));

        assertThat(unregisteredSource.loaded).isFalse();
        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(4);
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
        assertInvalidInitialContent("{\"loadBalanceStrategy\":\"INVALID\"}", "not one of the values accepted");
    }

    @Test
    void ignoresUnknownSourceFields() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"unknownField\":1,\"maxRetryCount\":9}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
    }

    @Test
    void letsJacksonConvertSourceFieldValues() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"enableQueueing\":\"true\",\"maxRetryCount\":\"9\"}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        assertThat(service.loadBalanceConfig().isEnableQueueing()).isTrue();
        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
    }

    @Test
    void loadsNestedConsistencyConfigFromNacos() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"flexlbSyncConsistencyConfig\":{\"needConsistency\":true,"
                        + "\"masterElectType\":\"ZOOKEEPER\",\"zookeeperConfig\":{"
                        + "\"zkHost\":\"zk:2181\",\"zkTimeoutMs\":10000}}}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        LBConsistencyConfig consistencyConfig =
                service.loadBalanceConfig().getFlexlbSyncConsistencyConfig();

        assertThat(consistencyConfig.isNeedConsistency()).isTrue();
        assertThat(consistencyConfig.getMasterElectType()).isEqualTo(LBConsistencyConfig.MasterElectType.ZOOKEEPER);
        assertThat(consistencyConfig.getZookeeperConfig().getZkHost()).isEqualTo("zk:2181");
        assertThat(consistencyConfig.getZookeeperConfig().getZkTimeoutMs()).isEqualTo(10000);
    }

    @Test
    void runtimeUpdatesKeepCurrentValuesForMissingFieldsAndReplaceSnapshot() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"enableQueueing\":false,\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of(
                        "ENABLE_QUEUEING", "true",
                        "MAX_RETRY_COUNT", "4")),
                source));

        FlexlbConfig initialSnapshot = service.loadBalanceConfig();
        source.emit("{\"enableQueueing\":false}");
        FlexlbConfig updatedSnapshot = service.loadBalanceConfig();

        assertThat(updatedSnapshot).isNotSameAs(initialSnapshot);
        assertThat(initialSnapshot.getMaxRetryCount()).isEqualTo(9);
        assertThat(updatedSnapshot.getMaxRetryCount()).isEqualTo(9);
        assertThat(updatedSnapshot.isEnableQueueing()).isFalse();
    }

    @Test
    void notifiesListenerWithCurrentAndRuntimeConfigurations() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<FlexlbConfig> updates = new ArrayList<>();

        service.addUpdateListener(updates::add);
        source.emit("{\"maxRetryCount\":10}");

        assertThat(updates).extracting(FlexlbConfig::getMaxRetryCount).containsExactly(9, 10);
    }

    @Test
    void rejectsInvalidRuntimeUpdatesAndKeepsLastKnownGoodSnapshot() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        FlexlbConfig lastKnownGood = service.loadBalanceConfig();

        source.emit("");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        source.emit("{}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        source.emit("{\"maxRetryCount\":\"invalid\"}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
    }

    @Test
    void closesRegisteredConfigSources() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
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
        } catch (Exception e) {
            throw new IllegalStateException(e);
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

        private FakeConfigSource(String name, int priority, String initialContent, Exception loadException) {
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

package org.flexlb.service.config.source;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.LocalSyncCacheMatchingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.enums.LogLevel;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
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
                        {"schemaVersion":1,"router":{"availabilityHysteresisPercent":20}}
                        """));
        FakeConfigSource nacosSource = new FakeConfigSource(
                "Nacos",
                200,
                """
                        {
                          "schemaVersion":1,
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
                        {"schemaVersion":1,"scheduler":{"type":"DIRECT"},"dispatcher":{"type":"NON_BATCH"}}
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
    void rejectsMissingOrBlankInitialContent() {
        assertInvalidInitialContent(null, "must not be null or blank");
        assertInvalidInitialContent("  ", "must not be null or blank");
    }

    @Test
    void treatsEmptyV2ObjectAsCompatibilityDefaults() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{}");

        ConfigService service = createService(List.of(environmentSource(Map.of()), source));

        assertThat(service.loadBalanceConfig().isDirect()).isTrue();
        assertThat(service.loadBalanceConfig().isBatchDispatch()).isFalse();
    }

    @Test
    void failsFastForInvalidInitialContent() {
        assertInvalidInitialContent("[]", "must be a JSON object");
        assertInvalidInitialContent(
                "{\"schemaVersion\":1,\"scheduler\":{\"type\":\"INVALID\"}}",
                "Could not resolve type id 'INVALID'");
    }

    @Test
    void rejectsUnknownSourceFields() {
        assertInvalidInitialContent(
                "{\"schemaVersion\":1,\"unknownField\":1}",
                "Unrecognized field");
    }

    @Test
    void rejectsScalarCoercionInSourceFields() {
        assertInvalidInitialContent(
                "{\"schemaVersion\":\"1\"}",
                "schemaVersion must be an integer");
    }

    @Test
    void mergesNestedConfigurationWithoutResettingSiblingFields() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}");

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
                "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":20}}"))
                .execute(() -> {
                    EnvironmentConfigSource environmentSource =
                            new EnvironmentConfigSource();
                    environmentSource.initialize();
                    ConfigService.register(new FakeConfigSource("Nacos", 200,
                            "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}"));
                    return new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
                });
        configService = service;

        assertThat(service.modelServiceConfig().getServiceId())
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
                          "schemaVersion":1,
                          "router":{"availabilityHysteresisPercent":9},
                          "observability":{"logging":{"level":"warn"}}
                        }
                        """);
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        FlexlbConfig initialSnapshot = service.loadBalanceConfig();
        source.emit("{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":10}}");
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
    void runtimeUpdateLogsTheNewSourceSchemaVersion() {
        ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(ConfigService.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"schemaVersion\":0,\"enableQueueing\":true}");
            createService(List.of(environmentSource(Map.of()), source));
            assertThat(appender.list)
                    .extracting(ILoggingEvent::getFormattedMessage)
                    .anyMatch(message -> message.startsWith("FlexLB config loaded: schemaVersion=0,"));

            appender.list.clear();
            source.emit("{\"schemaVersion\":1,\"scheduler\":{\"type\":\"QUEUE\"},\"dispatcher\":{\"type\":\"NON_BATCH\"}}");

            assertThat(appender.list)
                    .extracting(ILoggingEvent::getFormattedMessage)
                    .anyMatch(message -> message.startsWith("FlexLB config loaded: schemaVersion=1,"));
        } finally {
            logger.detachAppender(appender);
            appender.stop();
        }
    }

    @Test
    void taggedUnionTypeChangesReplaceTheWholeBranch() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                """
                        {
                          "schemaVersion":1,
                          "cacheMatching": {
                            "type": "KVCM",
                            "requestTimeoutMs": 900,
                            "leaderRefreshIntervalMs": 20000
                          }
                        }
                        """);
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        source.emit("{\"schemaVersion\":1,\"cacheMatching\":{\"requestTimeoutMs\":750}}");
        KvcmCacheMatchingConfig patched = (KvcmCacheMatchingConfig)
                service.loadBalanceConfig().getCacheMatching();
        assertThat(patched.getRequestTimeoutMs()).isEqualTo(750);
        assertThat(patched.getLeaderRefreshIntervalMs()).isEqualTo(20_000);

        source.emit("{\"schemaVersion\":1,\"cacheMatching\":{\"type\":\"LOCAL_SYNC\"}}");
        assertThat(service.loadBalanceConfig().getCacheMatching())
                .isInstanceOf(LocalSyncCacheMatchingConfig.class);

        source.emit("{\"schemaVersion\":1,\"cacheMatching\":{\"type\":\"KVCM\"}}");
        KvcmCacheMatchingConfig replaced = (KvcmCacheMatchingConfig)
                service.loadBalanceConfig().getCacheMatching();
        assertThat(replaced.getRequestTimeoutMs())
                .isEqualTo(KvcmCacheMatchingConfig.DEFAULT_REQUEST_TIMEOUT_MS);
        assertThat(replaced.getLeaderRefreshIntervalMs())
                .isEqualTo(KvcmCacheMatchingConfig.DEFAULT_LEADER_REFRESH_INTERVAL_MS);
    }

    @Test
    void arraysReplaceAsAWholeAndJsonNullIsRejected() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                """
                        {
                          "schemaVersion":1,
                          "scheduler": {
                            "type": "QUEUE",
                            "ordering": {
                              "type": "PRIORITY",
                              "preemption": {
                                "allowedVictimStages": [
                                  "PREFILL_QUEUED",
                                  "DECODE_RESERVED"
                                ]
                              }
                            }
                          },
                          "dispatcher": {"type": "NON_BATCH"}
                        }
                        """);
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        source.emit("""
                {
                  "schemaVersion":1,
                  "scheduler": {
                    "ordering": {
                      "preemption": {
                        "allowedVictimStages": ["PREFILL_QUEUED"]
                      }
                    }
                  }
                }
                """);
        assertThat(service.loadBalanceConfig().priorityOrdering()
                .getPreemption().getAllowedVictimStages())
                .containsExactly(VictimStage.PREFILL_QUEUED);

        FlexlbConfig lastKnownGood = service.loadBalanceConfig();
        source.emit("{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":null}}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
    }

    @Test
    void notifiesListenerWithCurrentAndRuntimeConfigurations() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<Long> updates = new ArrayList<>();

        service.addUpdateListener(config -> updates.add(
                config.getRouter().getAvailabilityHysteresisPercent()));
        source.emit("{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":10}}");

        assertThat(updates).containsExactly(9L, 10L);
    }

    @Test
    void rejectsBlankAndInvalidRuntimeUpdates() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        FlexlbConfig lastKnownGood = service.loadBalanceConfig();

        source.emit("");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        source.emit("{\"schemaVersion\":1,\"unknownField\":1}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
    }

    @Test
    void doesNotNotifyListenersForBlankRuntimeUpdates() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<Long> updates = new ArrayList<>();
        service.addUpdateListener(config -> updates.add(
                config.getRouter().getAvailabilityHysteresisPercent()));

        source.emit("");

        assertThat(updates).containsExactly(9L);
    }

    @Test
    void closesRegisteredConfigSources() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}");
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
        configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
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

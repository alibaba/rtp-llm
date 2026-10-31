package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.CacheMatchMode;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheMatchHotSwitchTest {

    @Test
    void switchesBothWaysWithoutResettingParametersOrOldSnapshots() throws Exception {
        try (Fixture fixture = new Fixture("KVCM", true)) {
            FlexlbConfig initial = fixture.configService.loadBalanceConfig();
            fixture.switchTo("LOCAL_STANDBY");
            assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.manager.activeSource());
            assertEquals(CacheMatchMode.LOCAL_STANDBY, fixture.orchestrator.status().configuredMode());
            assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.orchestrator.status().effectiveSource());
            assertEquals(CacheMatchMode.KVCM, initial.getKvcmAndStandbyConfig().getType());
            assertEquals(731, fixture.configuration.getKvcmRuntimeConfig().getRequestTimeoutMs());
            assertEquals(7, fixture.configuration.getKvcmRuntimeConfig().getTopKHostCount());
            assertEquals(1234, fixture.configuration.getLocalStandbyConfig().getMaximumEntries());
            assertTrue(fixture.configuration.isKvcmEnabled());
            assertTrue(fixture.configuration.isLocalStandbyEnabled());
            fixture.heartbeat(true);
            assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.manager.activeSource());
            fixture.switchTo("KVCM");
            assertEquals(CacheMatchSource.KVCM, fixture.manager.activeSource());
            assertEquals(CacheMatchMode.KVCM, fixture.orchestrator.status().configuredMode());
            assertEquals(731, fixture.configuration.getKvcmRuntimeConfig().getRequestTimeoutMs());
            assertEquals(1234,
                    fixture.configService.loadBalanceConfig()
                            .getKvcmAndStandbyConfig()
                            .getLocalStandby()
                            .getMaximumEntries());
            assertEquals("KVCM",
                    JsonUtils.strictValueToTree(fixture.configService.loadBalanceConfig())
                            .path("cacheMatching")
                            .path("type")
                            .asText());
        }
    }

    @Test
    void startsOnStandbyAndRestoresKvcmHealthPolicy() throws Exception {
        for (boolean autoSwitch : new boolean[]{true, false}) {
            try (Fixture fixture = new Fixture("LOCAL_STANDBY", autoSwitch)) {
                assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.manager.activeSource());
                fixture.heartbeat(false);
                fixture.switchTo("KVCM");
                assertEquals(autoSwitch ? CacheMatchSource.LOCAL_STANDBY : CacheMatchSource.KVCM,
                        fixture.manager.activeSource());
                fixture.heartbeat(true);
                assertEquals(CacheMatchSource.KVCM, fixture.manager.activeSource());
            }
        }
    }

    @Test
    void otherParameterUpdatesPreserveConfiguredSource() throws Exception {
        try (Fixture fixture = new Fixture("LOCAL_STANDBY", true)) {
            fixture.emit("{\"schemaVersion\":3,\"cacheMatching\":{\"requestTimeoutMs\":900}}");
            fixture.heartbeat(true);
            assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.manager.activeSource());
            fixture.switchTo("KVCM");
            fixture.heartbeat(true);
            assertEquals(CacheMatchSource.KVCM, fixture.manager.activeSource());
            assertEquals(900, fixture.configuration.getKvcmRuntimeConfig().getRequestTimeoutMs());
        }
    }

    @Test
    void invalidUpdatesKeepConfigurationAndEffectiveSource() throws Exception {
        try (Fixture fixture = new Fixture("LOCAL_STANDBY", true)) {
            FlexlbConfig initial = fixture.configService.loadBalanceConfig();
            for (String patch : List.of("{\"type\":\"UNKNOWN\"}", "{\"type\":null}",
                    "{\"type\":\"KVCM\",\"requestTimeoutMs\":-1}")) {
                fixture.emit("{\"schemaVersion\":3,\"cacheMatching\":" + patch + "}");
                assertSame(initial, fixture.configService.loadBalanceConfig());
                assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.manager.activeSource());
            }
            assertEquals("LOCAL_STANDBY",
                    JsonUtils.strictValueToTree(initial).path("cacheMatching").path("type").asText());
        }
    }

    @Test
    void concurrentHealthNotificationsCannotOverrideConfiguredStandby() throws Exception {
        try (Fixture fixture = new Fixture("KVCM", true); var executor = Executors.newSingleThreadExecutor()) {
            CountDownLatch start = new CountDownLatch(1);
            var heartbeats = executor.submit(() -> {
                start.await();
                for (int i = 0; i < 200; i++) {
                    fixture.healthUpdate.accept(fixture.health(true));
                }
                return null;
            });
            start.countDown();
            for (int i = 0; i < 20; i++) {
                fixture.switchTo("LOCAL_STANDBY");
                assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.manager.activeSource());
                fixture.switchTo("KVCM");
                assertEquals(CacheMatchSource.KVCM, fixture.manager.activeSource());
            }
            fixture.switchTo("LOCAL_STANDBY");
            heartbeats.get(10, TimeUnit.SECONDS);
            assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.manager.activeSource());
        }
    }

    @Test
    void routesThroughSelectedProviderAndKeepsPerRequestFallback() throws Exception {
        try (Fixture fixture = new Fixture("KVCM", true)) {
            CacheMatchQuery query = new CacheMatchQuery("request", List.of(1L), 128, RoleType.PREFILL, "group");
            assertEquals(CacheMatchSource.KVCM, fixture.orchestrator.findMatchingEngines(query).source());
            fixture.switchTo("LOCAL_STANDBY");
            assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.orchestrator.findMatchingEngines(query).source());
            verify(fixture.kvcmProvider, times(1)).findMatchingEngines("request", List.of(1L), 128, RoleType.PREFILL,
                    "group");
            fixture.switchTo("KVCM");
            when(fixture.kvcmProvider.findMatchingEngines("request", List.of(1L), 128, RoleType.PREFILL, "group"))
                    .thenThrow(new IllegalStateException("query failed"));
            assertEquals(CacheMatchSource.LOCAL_STANDBY, fixture.orchestrator.findMatchingEngines(query).source());
            assertEquals(CacheMatchSource.KVCM, fixture.manager.activeSource());
            verify(fixture.standbyProvider, times(2)).findMatchingEngines("request", List.of(1L), 128, RoleType.PREFILL,
                    "group");
        }
    }

    private static final class Fixture implements AutoCloseable {
        private final ConfigService configService;
        private final CacheMatchConfiguration configuration;
        private final CacheMatchFailoverManager manager;
        private final KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        private final KvcmCacheMatchProvider kvcmProvider = mock(KvcmCacheMatchProvider.class);
        private final LocalStandbyCacheMatchProvider standbyProvider = mock(LocalStandbyCacheMatchProvider.class);
        private final CacheMatchQueryOrchestrator orchestrator;
        private final Consumer<String> update;
        private final Consumer<KvcmHealthSnapshot> healthUpdate;

        @SuppressWarnings("unchecked")
        private Fixture(String mode, boolean autoSwitch) throws Exception {
            ConfigSource source = mock(ConfigSource.class, CALLS_REAL_METHODS);
            when(source.name()).thenReturn("test");
            when(source.load()).thenReturn("""
                    {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}},
                     "cacheMatching":{"type":"%s","requestTimeoutMs":731,"topKHostCount":7,
                        "localStandby":{"autoSwitch":%s,"maximumEntries":1234}}}
                    """.formatted(mode, autoSwitch));
            ConfigService.register(source);
            configService = new ConfigService(List.of(new StandardConfigDocumentParser()));
            ArgumentCaptor<Consumer<String>> updateCaptor = ArgumentCaptor.forClass(Consumer.class);
            verify(source).setUpdateListener(updateCaptor.capture());
            update = updateCaptor.getValue();
            ServiceRoute route = new ServiceRoute();
            route.setKvcm(new KvcmConfig());
            ModelMetaConfig model = new ModelMetaConfig(route);
            configuration = new CacheMatchConfiguration(model, configService);
            when(client.healthSnapshot()).thenReturn(health(true));
            CacheMetricsReporter reporter = mock(CacheMetricsReporter.class);
            manager = new CacheMatchFailoverManager(configuration, client, reporter, configService);
            ArgumentCaptor<Consumer<KvcmHealthSnapshot>> healthCaptor = ArgumentCaptor.forClass(Consumer.class);
            verify(client).setHealthSnapshotListener(healthCaptor.capture());
            healthUpdate = healthCaptor.getValue();
            orchestrator = new CacheMatchQueryOrchestrator(mock(LocalSyncCacheMatchProvider.class), kvcmProvider,
                    standbyProvider, mock(LocalStandbyCacheManager.class), manager,
                    mock(LocalStandbyComparisonService.class), reporter, configuration);
        }

        private void switchTo(String mode) {
            emit("{\"schemaVersion\":3,\"cacheMatching\":{\"type\":\"" + mode + "\"}}");
        }

        private void emit(String document) {
            update.accept(document);
        }

        private void heartbeat(boolean healthy) {
            KvcmHealthSnapshot snapshot = health(healthy);
            when(client.healthSnapshot()).thenReturn(snapshot);
            healthUpdate.accept(snapshot);
        }

        private KvcmHealthSnapshot health(boolean healthy) {
            return new KvcmHealthSnapshot(healthy ? KvcmHealthState.HEALTHY : KvcmHealthState.UNHEALTHY, 0, 0, 0, 100,
                    0, "test health");
        }

        @Override
        public void close() {
            configService.close();
        }
    }
}

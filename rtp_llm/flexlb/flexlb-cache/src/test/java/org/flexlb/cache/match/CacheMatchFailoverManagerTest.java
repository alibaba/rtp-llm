package org.flexlb.cache.match;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.flexlb.cache.CacheMatchTestConfigurations.kvcm;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheMatchFailoverManagerTest {

    @Test
    void manualFallbackRecordsOperatorActionWhenAlreadyOnStandby() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        when(client.healthSnapshot()).thenReturn(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure"));
        CacheMatchFailoverManager manager = new CacheMatchFailoverManager(
                configuration(true), client, metricsReporter);
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());
        long automaticFailoverTimeMs = manager.lastFailoverTimeMs();

        manager.activateFallbackManually();

        assertEquals("manual failover activated", manager.lastFailoverReason());
        assertTrue(manager.lastFailoverTimeMs() >= automaticFailoverTimeMs);
        verify(metricsReporter, times(1)).reportCacheMatchSourceChange(
                CacheMatchSource.KVCM, CacheMatchSource.LOCAL_STANDBY);
    }

    @Test
    void manualRecoveryUsesValidatedHealthBeforeLaterNotifications() {
        for (boolean autoSwitch : new boolean[]{true, false}) {
            KvcmGrpcClient client = mock(KvcmGrpcClient.class);
            KvcmHealthSnapshot unhealthy = health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure");
            when(client.healthSnapshot()).thenReturn(
                    health(KvcmHealthState.HEALTHY, 0, 0, 0, "initial"),
                    health(KvcmHealthState.HEALTHY, 0, 3, 0, "recovered"),
                    unhealthy);
            CacheMetricsReporter reporter = mock(CacheMetricsReporter.class);
            CacheMatchFailoverManager manager = new CacheMatchFailoverManager(
                    configuration(autoSwitch), client, reporter);
            manager.activateFallbackManually();

            manager.recoverPrimaryManually();

            assertEquals(CacheMatchSource.KVCM, manager.activeSource());
            verify(client, times(2)).healthSnapshot();
            verify(reporter).reportCacheMatchSourceChange(
                    CacheMatchSource.LOCAL_STANDBY, CacheMatchSource.KVCM);

            healthSnapshotListener(client).accept(unhealthy);
            assertEquals(autoSwitch ? CacheMatchSource.LOCAL_STANDBY : CacheMatchSource.KVCM,
                    manager.activeSource());
        }
    }

    @Test
    void lateHealthCallbackDoesNotOverrideNewerClientState() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot()).thenReturn(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "recovered"));
        CacheMatchFailoverManager manager = new CacheMatchFailoverManager(
                configuration(true), client, mock(CacheMetricsReporter.class));

        healthSnapshotListener(client).accept(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "outdated heartbeat"));

        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @Test
    void automaticallyFollowsKvcmClientHealth() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        AtomicReference<KvcmHealthSnapshot> currentHealth = new AtomicReference<>(
                health(KvcmHealthState.HEALTHY, 0, 0, 0, "initial"));
        when(client.healthSnapshot()).thenAnswer(ignored -> currentHealth.get());
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(true), client, metricsReporter);
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        currentHealth.set(health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure"));
        healthSnapshotListener.accept(currentHealth.get());
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());
        assertEquals("heartbeat failure", manager.lastFailoverReason());

        currentHealth.set(health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        healthSnapshotListener.accept(currentHealth.get());
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
        assertEquals("KVCM health recovered", manager.lastFailoverReason());
        verify(metricsReporter).reportCacheMatchSourceChange(
                CacheMatchSource.KVCM, CacheMatchSource.LOCAL_STANDBY);
        verify(metricsReporter).reportCacheMatchSourceChange(
                CacheMatchSource.LOCAL_STANDBY, CacheMatchSource.KVCM);
    }

    @Test
    void keepsKvcmActiveUntilManualFailoverWhenAutoSwitchIsDisabled() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        AtomicReference<KvcmHealthSnapshot> currentHealth = new AtomicReference<>(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 10, "query failure"));
        when(client.healthSnapshot()).thenAnswer(ignored -> currentHealth.get());
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(false), client, mock(CacheMetricsReporter.class));
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        assertEquals(CacheMatchSource.KVCM, manager.activeSource());

        manager.activateFallbackManually();
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        assertThrows(IllegalStateException.class, manager::recoverPrimaryManually);
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        currentHealth.set(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        healthSnapshotListener.accept(currentHealth.get());
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        manager.recoverPrimaryManually();
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @Test
    void warnsOncePerUnhealthyPeriodWhenManualFailoverIsRequired() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        AtomicReference<KvcmHealthSnapshot> currentHealth = new AtomicReference<>(
                health(KvcmHealthState.HEALTHY, 0, 0, 0, "initial"));
        when(client.healthSnapshot()).thenAnswer(ignored -> currentHealth.get());
        var logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(CacheMatchFailoverManager.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            CacheMatchFailoverManager manager = new CacheMatchFailoverManager(
                    configuration(false), client, mock(CacheMetricsReporter.class));
            Consumer<KvcmHealthSnapshot> listener = healthSnapshotListener(client);

            currentHealth.set(health(KvcmHealthState.UNHEALTHY, 3, 0, 1, "first failure"));
            listener.accept(currentHealth.get());
            currentHealth.set(health(KvcmHealthState.UNHEALTHY, 4, 0, 2, "continued failure"));
            listener.accept(currentHealth.get());
            assertEquals(1, manualFailoverWarnings(appender));
            assertEquals(CacheMatchSource.KVCM, manager.activeSource());

            currentHealth.set(health(KvcmHealthState.HEALTHY, 0, 3, 0, "recovered"));
            listener.accept(currentHealth.get());
            currentHealth.set(health(KvcmHealthState.UNHEALTHY, 3, 0, 1, "new failure"));
            listener.accept(currentHealth.get());
            assertEquals(2, manualFailoverWarnings(appender));
            assertEquals(CacheMatchSource.KVCM, manager.activeSource());
        } finally {
            logger.detachAppender(appender);
            appender.stop();
        }
    }

    private long manualFailoverWarnings(ListAppender<ILoggingEvent> appender) {
        return appender.list.stream()
                .filter(event -> event.getFormattedMessage().contains("manual failover is required"))
                .count();
    }

    @Test
    void manualFallbackRemainsActiveAfterKvcmRecovers() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot())
                .thenReturn(health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(true), client, mock(CacheMetricsReporter.class));

        manager.activateFallbackManually();

        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        manager.recoverPrimaryManually();
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @Test
    void rejectsManualRecoveryForUnhealthyKvcmWhenAutoSwitchIsEnabled() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        AtomicReference<KvcmHealthSnapshot> currentHealth = new AtomicReference<>(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure"));
        when(client.healthSnapshot()).thenAnswer(ignored -> currentHealth.get());
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(true), client, mock(CacheMetricsReporter.class));
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        manager.activateFallbackManually();
        assertThrows(IllegalStateException.class, manager::recoverPrimaryManually);
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        currentHealth.set(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        healthSnapshotListener.accept(currentHealth.get());
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        manager.recoverPrimaryManually();
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @SuppressWarnings("unchecked")
    private Consumer<KvcmHealthSnapshot> healthSnapshotListener(KvcmGrpcClient client) {
        ArgumentCaptor<Consumer<KvcmHealthSnapshot>> captor =
                ArgumentCaptor.forClass(Consumer.class);
        verify(client).setHealthSnapshotListener(captor.capture());
        return captor.getValue();
    }

    private KvcmHealthSnapshot health(KvcmHealthState state, int heartbeatFailures, int heartbeatSuccesses, int queryFailures, String reason) {
        return new KvcmHealthSnapshot(
                state,
                heartbeatFailures,
                heartbeatSuccesses,
                queryFailures,
                100,
                0,
                reason);
    }

    private CacheMatchConfiguration configuration(boolean autoSwitch) {
        KvcmConfig kvcmTopology = new KvcmConfig();

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcmTopology);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return kvcm(config,
                runtime -> runtime.getLocalStandby().setAutoSwitch(autoSwitch));
    }
}

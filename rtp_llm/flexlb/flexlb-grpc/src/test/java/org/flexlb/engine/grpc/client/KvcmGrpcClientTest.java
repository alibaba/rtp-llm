package org.flexlb.engine.grpc.client;

import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.monitor.KvcmMetricsReporter;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.kvcm.grpc.Status;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.metric.NoOpFlexMonitor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class KvcmGrpcClientTest {

    private KvcmGrpcClient client;

    @AfterEach
    void tearDown() {
        if (client != null) {
            client.shutdown();
        }
    }

    @Test
    void reportsLogicalWorkerMatchesAndMillisecondCallMetrics() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmConfig config = new KvcmConfig();
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
        runtimeConfig.setMedium(List.of("hbm", "kvs"));
        runtimeConfig.setGlobalKvsHostCount(5);
        runtimeConfig.setEnableP2p(true);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(config);
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metaServiceClient.getHostCacheState(any(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    TimeUnit.MILLISECONDS.sleep(5L);
                    return GetHostCacheStateResponse.newBuilder()
                        .setHeader(okHeader())
                        .addHosts(HostCacheMatch.newBuilder()
                                .setHostIpPort("10.0.0.1:8601@1")
                                .setLocal(2)
                                .setGlobal(10))
                        .build();
                });

        GrpcReporter reporter = mock(GrpcReporter.class);
        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                reporter);

        long startedAt = System.nanoTime();
        Map<String, org.flexlb.dao.cache.HostCacheMatch> result =
                client.findMatchingEngines(
                        "request-1", List.of(11L, 22L), 2192L,
                        RoleType.PREFILL, "default");

        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startedAt) + 1L;
        ArgumentCaptor<Long> duration = ArgumentCaptor.forClass(Long.class);
        verify(reporter).reportCallMetrics(eq("KVCM_GET_HOST_CACHE_STATE"), duration.capture(), anyInt(), eq(false));
        assertTrue(duration.getValue() >= 5L);
        assertTrue(duration.getValue() <= elapsedMs, "Call duration must use milliseconds");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        verify(metaServiceClient).getHostCacheState(any(), sentRequest.capture(), anyLong());
        assertEquals(List.of("hbm", "kvs"), sentRequest.getValue().getMediumList());
        assertEquals(5, sentRequest.getValue().getGlobalKvsHostCount());
        assertTrue(sentRequest.getValue().getEnableP2P());

        assertEquals(2, result.get("10.0.0.1:8601@1").localMatchBlocks());
        assertEquals(10, result.get("10.0.0.1:8601@1").globalMatchBlocks());
    }

    @Test
    void queriesThreeGlobalHostsWithoutP2pByDefault() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metaServiceClient.getHostCacheState(any(), any(), anyLong()))
                .thenReturn(GetHostCacheStateResponse.newBuilder()
                        .setHeader(okHeader())
                        .build());

        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        client.findMatchingEngines(
                "request-3", List.of(11L), 2192L, RoleType.PREFILL, "default");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        verify(metaServiceClient).getHostCacheState(any(), sentRequest.capture(), anyLong());
        assertEquals(3, sentRequest.getValue().getGlobalKvsHostCount());
        assertFalse(sentRequest.getValue().getEnableP2P());
        assertTrue(sentRequest.getValue().getMediumList().isEmpty());
    }

    @Test
    void skipsQueriesWhenDisabled() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        when(configuration.isKvcmEnabled()).thenReturn(false);
        client = createClient(
                configuration,
                mock(KvcmMetaServiceClient.class),
                mock(KvcmLeaderResolver.class),
                mock(KvcmWorkerMetadataResolver.class),
                mock(GrpcReporter.class));

        assertTrue(client.findMatchingEngines(
                "request-2", List.of(11L), 2192L,
                RoleType.PREFILL, "default").isEmpty());
    }

    @Test
    void usesTheLatestRuntimeConfigForEveryQuery() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setMaxQueryRetryCount(0);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metaServiceClient.getHostCacheState(any(), any(), anyLong()))
                .thenReturn(GetHostCacheStateResponse.newBuilder()
                        .setHeader(okHeader())
                        .build());

        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        client.findMatchingEngines(
                "request-4", List.of(11L), 2192L, RoleType.PREFILL, "default");

        KvcmCacheMatchingConfig updatedConfig = new KvcmCacheMatchingConfig();
        updatedConfig.setLeaderRefreshIntervalMs(60_000);
        updatedConfig.setMaxQueryRetryCount(0);
        updatedConfig.setGlobalKvsHostCount(9);
        updatedConfig.setEnableP2p(true);
        updatedConfig.setMedium(List.of("kvs"));
        updatedConfig.setRequestTimeoutMs(900L);
        runtimeConfig.set(updatedConfig);
        client.findMatchingEngines(
                "request-5", List.of(11L), 2192L, RoleType.PREFILL, "default");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        ArgumentCaptor<Long> timeout = ArgumentCaptor.forClass(Long.class);
        verify(metaServiceClient, times(2))
                .getHostCacheState(any(), sentRequest.capture(), timeout.capture());
        assertEquals(3, sentRequest.getAllValues().get(0).getGlobalKvsHostCount());
        assertEquals(9, sentRequest.getAllValues().get(1).getGlobalKvsHostCount());
        assertTrue(sentRequest.getAllValues().get(1).getEnableP2P());
        assertEquals(List.of("kvs"), sentRequest.getAllValues().get(1).getMediumList());
        assertEquals(900L, timeout.getAllValues().get(1));
    }

    @Test
    void reusesRuntimeConfigSnapshotAcrossQueryRetries() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setMaxQueryRetryCount(1);
        initialConfig.setGlobalKvsHostCount(4);
        initialConfig.setEnableP2p(true);
        initialConfig.setMedium(List.of("hbm"));
        initialConfig.setRequestTimeoutMs(700L);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        AtomicInteger attempts = new AtomicInteger();
        when(metaServiceClient.getHostCacheState(any(), any(), anyLong()))
                .thenAnswer(ignored -> {
                    if (attempts.getAndIncrement() == 0) {
                        KvcmCacheMatchingConfig updatedConfig = new KvcmCacheMatchingConfig();
                        updatedConfig.setLeaderRefreshIntervalMs(60_000);
                        updatedConfig.setMaxQueryRetryCount(0);
                        updatedConfig.setGlobalKvsHostCount(9);
                        updatedConfig.setEnableP2p(false);
                        updatedConfig.setMedium(List.of("kvs"));
                        updatedConfig.setRequestTimeoutMs(900L);
                        runtimeConfig.set(updatedConfig);
                        throw io.grpc.Status.UNAVAILABLE.asRuntimeException();
                    }
                    return GetHostCacheStateResponse.newBuilder()
                            .setHeader(okHeader())
                            .build();
                });

        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        client.findMatchingEngines(
                "request-6", List.of(11L), 2192L, RoleType.PREFILL, "default");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        ArgumentCaptor<Long> timeout = ArgumentCaptor.forClass(Long.class);
        verify(metaServiceClient, times(2))
                .getHostCacheState(any(), sentRequest.capture(), timeout.capture());
        for (GetHostCacheStateRequest request : sentRequest.getAllValues()) {
            assertEquals(4, request.getGlobalKvsHostCount());
            assertTrue(request.getEnableP2P());
            assertEquals(List.of("hbm"), request.getMediumList());
        }
        assertEquals(List.of(700L, 700L), timeout.getAllValues());
    }

    @Test
    void appliesLatestHeartbeatThresholdsWithoutRestart() throws Exception {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setHeartbeatFailureThreshold(5);
        initialConfig.setRecoverySuccessThreshold(5);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        AtomicReference<Boolean> refreshResult = new AtomicReference<>(true);
        CountDownLatch initialRefresh = new CountDownLatch(1);
        when(leaderResolver.refresh()).thenAnswer(ignored -> refreshResult.get());
        doAnswer(ignored -> {
            initialRefresh.countDown();
            return null;
        }).when(metadataResolver).refreshNamespacesAndQueryTypes();
        client = createClient(
                configuration,
                mock(KvcmMetaServiceClient.class),
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        assertTrue(initialRefresh.await(2, TimeUnit.SECONDS));

        KvcmCacheMatchingConfig failureConfig = new KvcmCacheMatchingConfig();
        failureConfig.setHeartbeatFailureThreshold(1);
        failureConfig.setRecoverySuccessThreshold(5);
        runtimeConfig.set(failureConfig);
        refreshResult.set(false);
        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());

        KvcmCacheMatchingConfig recoveryConfig = new KvcmCacheMatchingConfig();
        recoveryConfig.setHeartbeatFailureThreshold(1);
        recoveryConfig.setRecoverySuccessThreshold(1);
        runtimeConfig.set(recoveryConfig);
        refreshResult.set(true);
        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
    }

    @Test
    void appliesLatestQueryFailureThresholdWithoutRestart() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setMaxQueryRetryCount(0);
        initialConfig.setQueryFailureThreshold(10);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(leaderResolver.refresh()).thenReturn(true);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(metaServiceClient.getHostCacheState(any(), any(), anyLong()))
                .thenThrow(io.grpc.Status.UNAVAILABLE.asRuntimeException());
        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));

        KvcmCacheMatchingConfig updatedConfig = new KvcmCacheMatchingConfig();
        updatedConfig.setMaxQueryRetryCount(0);
        updatedConfig.setQueryFailureThreshold(1);
        runtimeConfig.set(updatedConfig);
        assertThrows(KvcmQueryException.class, () -> client.findMatchingEngines(
                "request-7", List.of(11L), 2192L, RoleType.PREFILL, "default"));

        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
    }

    private static KvcmGrpcClient createClient(CacheMatchConfiguration configuration,
                                               KvcmMetaServiceClient metaServiceClient,
                                               KvcmLeaderResolver leaderResolver,
                                               KvcmWorkerMetadataResolver workerMetadataResolver,
                                               GrpcReporter grpcReporter) {
        ApplicationWarmupState warmupState = new ApplicationWarmupState();
        warmupState.setWarmupFinished(true);
        return new KvcmGrpcClient(configuration, metaServiceClient, leaderResolver, workerMetadataResolver,
                warmupState, grpcReporter, new KvcmMetricsReporter(NoOpFlexMonitor.getInstance()));
    }

    private static CommonResponseHeader okHeader() {
        return CommonResponseHeader.newBuilder()
                .setStatus(Status.newBuilder().setCode(ErrorCode.OK))
                .build();
    }
}

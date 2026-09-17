package org.flexlb.engine.grpc.client;

import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.kvcm.grpc.Status;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
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
        client = new KvcmGrpcClient(
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

        client = new KvcmGrpcClient(
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
        client = new KvcmGrpcClient(
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

        client = new KvcmGrpcClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        client.findMatchingEngines(
                "request-4", List.of(11L), 2192L, RoleType.PREFILL, "default");

        runtimeConfig.setGlobalKvsHostCount(9);
        runtimeConfig.setEnableP2p(true);
        runtimeConfig.setMedium(List.of("kvs"));
        runtimeConfig.setRequestTimeoutMs(900L);
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

    private static CommonResponseHeader okHeader() {
        return CommonResponseHeader.newBuilder()
                .setStatus(Status.newBuilder().setCode(ErrorCode.OK))
                .build();
    }
}

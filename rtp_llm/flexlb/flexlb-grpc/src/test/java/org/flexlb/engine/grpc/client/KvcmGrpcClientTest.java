package org.flexlb.engine.grpc.client;

import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.kvcm.grpc.Status;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
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
    void returnsP2pAwareMatches() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmConfig config = new KvcmConfig();
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
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
                .thenReturn(GetHostCacheStateResponse.newBuilder()
                        .setHeader(okHeader())
                        .addHosts(HostCacheMatch.newBuilder()
                                .setHostIpPort("10.0.0.1:8601")
                                .setLocal(2)
                                .setP2P1Fetch(8)
                                .setP2P1TotalMatch(10))
                        .build());

        client = new KvcmGrpcClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));

        Map<String, org.flexlb.dao.cache.HostCacheMatch> result =
                client.findMatchingEngines(
                        "request-1", List.of(11L, 22L), 2192L,
                        RoleType.PREFILL, "default");

        assertEquals(2, result.get("10.0.0.1:8601").localMatchBlocks());
        assertEquals(8, result.get("10.0.0.1:8601").p2pFetchBlocks());
        assertEquals(10, result.get("10.0.0.1:8601").p2pTotalMatchBlocks());
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

    private static CommonResponseHeader okHeader() {
        return CommonResponseHeader.newBuilder()
                .setStatus(Status.newBuilder().setCode(ErrorCode.OK))
                .build();
    }
}

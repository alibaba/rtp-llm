package org.flexlb.engine.grpc.client;

import io.grpc.Status;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.kvcm.grpc.QueryType;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class KvcmQueryTypeResolverTest {

    @Test
    void usesPrefixMatchForFullAttention() {
        TestContext context = context(metadata("full_attention"));

        context.resolver().refresh();

        assertEquals(QueryType.QT_PREFIX_MATCH,
                context.resolver().resolve(RoleType.PDFUSION, "default"));
        assertEquals(QueryType.QT_PREFIX_MATCH,
                context.resolver().resolve(RoleType.DECODE, "default"));
        verifyMetadataRequest(context.engineGrpcClient());
    }

    @Test
    void usesMambaPrefixMatchForMixedAttention() {
        TestContext context = context(metadata("mamba", "full_attention"));

        context.resolver().refresh();

        assertEquals(QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                context.resolver().resolve(RoleType.PDFUSION, "default"));
    }

    @Test
    void defaultsToPrefixMatchWhenMetadataIsUnavailable() {
        EngineRpcService.KvCacheGroupListPB unavailable = EngineRpcService.KvCacheGroupListPB.newBuilder()
                .setErrCode(EngineRpcService.KvCacheGroupMetadataErrorCode.KV_CACHE_GROUP_METADATA_UNAVAILABLE)
                .setErrMsg("metadata unavailable")
                .build();
        TestContext context = context(unavailable);

        context.resolver().refresh();

        assertEquals(QueryType.QT_PREFIX_MATCH,
                context.resolver().resolve(RoleType.PDFUSION, "default"));
        verifyMetadataRequest(context.engineGrpcClient());
    }

    @Test
    void queriesReplicasUntilOneReturnsTrustedMetadata() {
        WorkerHost unavailableHost = worker("10.0.0.1");
        WorkerHost availableHost = worker("10.0.0.2");
        TestContext context = context(
                metadata("mamba", "full_attention"),
                List.of(unavailableHost, availableHost));
        when(context.engineGrpcClient().getKvCacheGroupsMetadata(
                eq(unavailableHost.getIp()),
                eq(unavailableHost.getWorkerStatusPort()),
                any(EngineRpcService.KvCacheGroupsRequestPB.class),
                eq(123L)))
                .thenThrow(Status.DEADLINE_EXCEEDED.asRuntimeException());

        context.resolver().refresh();

        assertEquals(QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                context.resolver().resolve(RoleType.PDFUSION, "default"));
        verify(context.engineGrpcClient()).getKvCacheGroupsMetadata(
                eq(unavailableHost.getIp()),
                eq(unavailableHost.getWorkerStatusPort()),
                any(EngineRpcService.KvCacheGroupsRequestPB.class),
                eq(123L));
        verify(context.engineGrpcClient()).getKvCacheGroupsMetadata(
                eq(availableHost.getIp()),
                eq(availableHost.getWorkerStatusPort()),
                any(EngineRpcService.KvCacheGroupsRequestPB.class),
                eq(123L));
    }

    private TestContext context(EngineRpcService.KvCacheGroupListPB metadata) {
        return context(metadata, List.of(worker("10.0.0.1")));
    }

    private TestContext context(
            EngineRpcService.KvCacheGroupListPB metadata,
            List<WorkerHost> hosts) {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(ServiceDiscoveryType.DASHSCOPE);

        Endpoint workerEndpoint = new Endpoint();
        workerEndpoint.setAddress("v-workers");
        workerEndpoint.setProtocol("grpc");
        workerEndpoint.setDiscovery(discovery);

        GroupRoleEndPoint group = new GroupRoleEndPoint();
        group.setGroup("default");
        group.setPdFusionEndpoint(workerEndpoint);
        group.setDecodeEndpoint(workerEndpoint);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setAddress("v-kvcm");
        kvcm.setDiscovery(discovery);
        kvcm.setRequestTimeoutMs(123L);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);
        route.setRoleEndpoints(List.of(group));

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(route.getServiceId(), route);

        RoutingServiceDiscovery serviceDiscovery = mock(RoutingServiceDiscovery.class);
        when(serviceDiscovery.getHosts(workerEndpoint)).thenReturn(hosts);

        EngineGrpcClient engineGrpcClient = mock(EngineGrpcClient.class);
        when(engineGrpcClient.getKvCacheGroupsMetadata(
                anyString(), anyInt(), any(EngineRpcService.KvCacheGroupsRequestPB.class), eq(123L)))
                .thenReturn(metadata);

        return new TestContext(
                new KvcmQueryTypeResolver(modelMetaConfig, serviceDiscovery, engineGrpcClient),
                engineGrpcClient);
    }

    private WorkerHost worker(String ip) {
        return new WorkerHost(ip, 8080, 8081, 8085, 18002, "", "default", "deployment");
    }

    private EngineRpcService.KvCacheGroupListPB metadata(String... kinds) {
        EngineRpcService.KvCacheGroupListPB.Builder builder = EngineRpcService.KvCacheGroupListPB.newBuilder()
                .setErrCode(EngineRpcService.KvCacheGroupMetadataErrorCode.KV_CACHE_GROUP_METADATA_OK);
        for (int index = 0; index < kinds.length; index++) {
            builder.addItems(EngineRpcService.KvCacheGroupMetadataPB.newBuilder()
                    .setGroupIdx(index)
                    .setKind(kinds[index])
                    .setBlockSize(1072)
                    .setSlidingWindow(-1));
        }
        return builder.build();
    }

    private void verifyMetadataRequest(EngineGrpcClient engineGrpcClient) {
        ArgumentCaptor<EngineRpcService.KvCacheGroupsRequestPB> requestCaptor =
                ArgumentCaptor.forClass(EngineRpcService.KvCacheGroupsRequestPB.class);
        verify(engineGrpcClient).getKvCacheGroupsMetadata(
                eq("10.0.0.1"), eq(18002), requestCaptor.capture(), eq(123L));
        assertEquals(EngineRpcService.KvCacheGroupsRequestPB.getDefaultInstance(), requestCaptor.getValue());
    }

    private record TestContext(
            KvcmQueryTypeResolver resolver,
            EngineGrpcClient engineGrpcClient) {
    }
}

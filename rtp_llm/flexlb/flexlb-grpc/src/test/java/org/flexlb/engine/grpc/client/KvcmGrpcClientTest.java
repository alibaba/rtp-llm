package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.MetaNodeEndpoint;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.kvcm.grpc.Status;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.when;

class KvcmGrpcClientTest {

    private final AtomicReference<GetHostCacheStateRequest> lastCacheRequest = new AtomicReference<>();
    private Server seedServer;
    private Server leaderServer;
    private KvcmGrpcClient client;

    @BeforeEach
    void setUp() throws IOException {
        leaderServer = ServerBuilder.forPort(0)
                .addService(new LeaderMetaService(lastCacheRequest))
                .build()
                .start();
        seedServer = ServerBuilder.forPort(0)
                .addService(new SeedMetaService(leaderServer.getPort()))
                .build()
                .start();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (client != null) {
            client.shutdown();
        }
        seedServer.shutdownNow();
        seedServer.awaitTermination();
        leaderServer.shutdownNow();
        leaderServer.awaitTermination();
    }

    @Test
    void usesBootstrapPortThenLeaderRpcPortAndQueriesFirstDeploymentNamespace() throws Exception {
        RoutingServiceDiscovery serviceDiscovery = serviceDiscovery();
        client = newClient(
                modelMetaConfig(seedServer.getPort()),
                serviceDiscovery,
                KvCacheGroupMode.FULL_ATTENTION_ONLY);

        Map<String, Integer> matches = waitForMatches(RoleType.PDFUSION);
        Map<String, Integer> decodeMatches = waitForMatches(RoleType.DECODE);

        assertEquals(2, matches.get("10.0.0.1:8601"));
        assertEquals(2, decodeMatches.get("10.0.0.1:8601"));
        GetHostCacheStateRequest request = lastCacheRequest.get();
        assertEquals("deployment-first_2192", request.getInstanceId());
        assertEquals(QueryType.QT_PREFIX_MATCH, request.getQueryType());
        assertEquals(List.of(11L, 22L, 33L), request.getBlockCacheKeysList());
        assertFalse(request.getUseEaglePop());
        assertEquals(0, request.getMediumCount());
        assertTrue(client.findMatchingEngines(
                "request-null-group", List.of(11L, 22L, 33L), 2192L, RoleType.PDFUSION, null).isEmpty());
        assertTrue(client.findMatchingEngines(
                "request-empty-group", List.of(11L, 22L, 33L), 2192L, RoleType.PDFUSION, "").isEmpty());
    }

    @Test
    void configuredNamespaceTakesPriorityForCacheQuery() throws Exception {
        RoutingServiceDiscovery serviceDiscovery = serviceDiscovery();
        client = newClient(
                modelMetaConfig(seedServer.getPort(), "vllm-test-0"),
                serviceDiscovery,
                KvCacheGroupMode.WITH_MAMBA,
                1);

        Map<String, Integer> matches = waitForMatches(RoleType.PDFUSION, null);

        assertEquals(2, matches.get("10.0.0.1:8601"));
        assertEquals("vllm-test-0_2192", lastCacheRequest.get().getInstanceId());
        assertEquals(QueryType.QT_PREFIX_MATCH_WITH_MAMBA, lastCacheRequest.get().getQueryType());
        assertTrue(lastCacheRequest.get().getUseEaglePop());
    }

    @Test
    void marksKvcmUnhealthyAndRecoversOnlyAfterHeartbeatThresholds() {
        KvcmMetaServiceClient metaServiceClient = Mockito.mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = Mockito.mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver workerMetadataResolver =
                Mockito.mock(KvcmWorkerMetadataResolver.class);
        when(leaderResolver.refresh())
                .thenReturn(false, false, false, true, true, true);
        client = new KvcmGrpcClient(
                new CacheMatchConfiguration(new ModelMetaConfig()),
                metaServiceClient,
                leaderResolver,
                workerMetadataResolver,
                () -> true);
        List<KvcmHealthState> healthSnapshots = new ArrayList<>();
        client.setHealthSnapshotListener(snapshot -> healthSnapshots.add(snapshot.state()));

        KvcmHealthSnapshot initial = client.healthSnapshot();
        assertEquals(KvcmHealthState.HEALTHY, initial.state());

        client.refreshKvcmServiceStateSafely();
        KvcmHealthSnapshot firstFailure = client.healthSnapshot();
        assertEquals(KvcmHealthState.HEALTHY, firstFailure.state());
        assertEquals(1, firstFailure.consecutiveHeartbeatFailures());
        assertEquals(0, firstFailure.consecutiveHeartbeatSuccesses());

        client.refreshKvcmServiceStateSafely();
        client.refreshKvcmServiceStateSafely();
        KvcmHealthSnapshot unhealthy = client.healthSnapshot();
        assertEquals(KvcmHealthState.UNHEALTHY, unhealthy.state());
        assertEquals(3, unhealthy.consecutiveHeartbeatFailures());

        client.refreshKvcmServiceStateSafely();
        KvcmHealthSnapshot firstSuccess = client.healthSnapshot();
        assertEquals(KvcmHealthState.UNHEALTHY, firstSuccess.state());
        assertEquals(0, firstSuccess.consecutiveHeartbeatFailures());
        assertEquals(1, firstSuccess.consecutiveHeartbeatSuccesses());
        assertEquals(unhealthy.lastHeartbeatFailureTimeMs(),
                firstSuccess.lastHeartbeatFailureTimeMs());

        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());

        client.refreshKvcmServiceStateSafely();
        KvcmHealthSnapshot recovered = client.healthSnapshot();
        assertEquals(KvcmHealthState.HEALTHY, recovered.state());
        assertEquals(3, recovered.consecutiveHeartbeatSuccesses());
        assertEquals(List.of(
                KvcmHealthState.HEALTHY,
                KvcmHealthState.HEALTHY,
                KvcmHealthState.UNHEALTHY,
                KvcmHealthState.UNHEALTHY,
                KvcmHealthState.UNHEALTHY,
                KvcmHealthState.HEALTHY), healthSnapshots);
    }

    @Test
    void ignoresHeartbeatFailuresUntilApplicationWarmupFinishes() {
        KvcmMetaServiceClient metaServiceClient = Mockito.mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = Mockito.mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver workerMetadataResolver =
                Mockito.mock(KvcmWorkerMetadataResolver.class);
        AtomicBoolean warmupFinished = new AtomicBoolean();
        when(leaderResolver.refresh()).thenReturn(false);
        client = new KvcmGrpcClient(
                new CacheMatchConfiguration(new ModelMetaConfig()),
                metaServiceClient,
                leaderResolver,
                workerMetadataResolver,
                warmupFinished::get);

        client.refreshKvcmServiceStateSafely();
        client.refreshKvcmServiceStateSafely();
        client.refreshKvcmServiceStateSafely();

        KvcmHealthSnapshot duringWarmup = client.healthSnapshot();
        assertEquals(KvcmHealthState.HEALTHY, duringWarmup.state());
        assertEquals(0, duringWarmup.consecutiveHeartbeatFailures());
        assertTrue(duringWarmup.lastHeartbeatFailureTimeMs() > 0);

        warmupFinished.set(true);
        client.refreshKvcmServiceStateSafely();
        client.refreshKvcmServiceStateSafely();
        client.refreshKvcmServiceStateSafely();

        KvcmHealthSnapshot afterWarmup = client.healthSnapshot();
        assertEquals(KvcmHealthState.UNHEALTHY, afterWarmup.state());
        assertEquals(3, afterWarmup.consecutiveHeartbeatFailures());
    }

    @Test
    void ignoresQueryFailuresUntilApplicationWarmupFinishes() {
        KvcmMetaServiceClient metaServiceClient = Mockito.mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = Mockito.mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver workerMetadataResolver =
                Mockito.mock(KvcmWorkerMetadataResolver.class);
        AtomicBoolean warmupFinished = new AtomicBoolean();
        GrpcTarget leader = new GrpcTarget("127.0.0.1", 6381);

        when(leaderResolver.resolve()).thenReturn(leader);
        when(workerMetadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L))
                .thenReturn("test-namespace");
        when(workerMetadataResolver.resolveQueryType(
                RoleType.PREFILL, "default"))
                .thenReturn(QueryType.QT_PREFIX_MATCH);
        when(metaServiceClient.getHostCacheState(
                any(GrpcTarget.class),
                any(GetHostCacheStateRequest.class),
                anyLong()))
                .thenThrow(io.grpc.Status.UNAVAILABLE.asRuntimeException());

        ModelMetaConfig modelMetaConfig =
                modelMetaConfig(seedServer.getPort(), "test-namespace");
        KvcmConfig kvcm = modelMetaConfig
                .getServiceRoute("test-service")
                .getKvcm();
        kvcm.setQueryFailureThreshold(1);
        kvcm.setMaxQueryRetryCount(0);
        client = new KvcmGrpcClient(
                new CacheMatchConfiguration(modelMetaConfig),
                metaServiceClient,
                leaderResolver,
                workerMetadataResolver,
                warmupFinished::get);

        assertThrows(KvcmQueryException.class, this::queryMockClient);
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
        assertEquals(0, client.healthSnapshot().consecutiveQueryFailures());

        warmupFinished.set(true);
        assertThrows(KvcmQueryException.class, this::queryMockClient);
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        assertEquals(1, client.healthSnapshot().consecutiveQueryFailures());
    }

    @Test
    void countsOneFailurePerRetriedQueryAndUsesOnlyHeartbeatForRecovery() {
        KvcmMetaServiceClient metaServiceClient = Mockito.mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = Mockito.mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver workerMetadataResolver =
                Mockito.mock(KvcmWorkerMetadataResolver.class);
        AtomicBoolean querySucceeds = new AtomicBoolean();
        GrpcTarget leader = new GrpcTarget("127.0.0.1", 6381);

        when(leaderResolver.resolve()).thenReturn(leader);
        when(leaderResolver.refresh()).thenReturn(true);
        when(workerMetadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L))
                .thenReturn("test-namespace");
        when(workerMetadataResolver.resolveQueryType(
                RoleType.PREFILL, "default"))
                .thenReturn(QueryType.QT_PREFIX_MATCH);
        when(metaServiceClient.getHostCacheState(
                any(GrpcTarget.class),
                any(GetHostCacheStateRequest.class),
                anyLong()))
                .thenAnswer(invocation -> {
                    if (!querySucceeds.get()) {
                        throw io.grpc.Status.UNAVAILABLE.asRuntimeException();
                    }
                    return GetHostCacheStateResponse.newBuilder()
                            .setHeader(okHeader())
                            .build();
                });

        ModelMetaConfig modelMetaConfig =
                modelMetaConfig(seedServer.getPort(), "test-namespace");
        KvcmConfig kvcm = modelMetaConfig
                .getServiceRoute("test-service")
                .getKvcm();
        kvcm.setQueryFailureThreshold(2);
        kvcm.setMaxQueryRetryCount(2);
        kvcm.setRecoverySuccessThreshold(2);
        client = new KvcmGrpcClient(
                new CacheMatchConfiguration(modelMetaConfig),
                metaServiceClient,
                leaderResolver,
                workerMetadataResolver,
                () -> true);
        Mockito.verify(leaderResolver, Mockito.timeout(1_000)).refresh();
        int heartbeatSuccessesBeforeQueries =
                client.healthSnapshot().consecutiveHeartbeatSuccesses();

        assertThrows(KvcmQueryException.class, this::queryMockClient);
        assertEquals(1, client.healthSnapshot().consecutiveQueryFailures());
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
        assertEquals(heartbeatSuccessesBeforeQueries,
                client.healthSnapshot().consecutiveHeartbeatSuccesses());
        Mockito.verify(metaServiceClient, Mockito.times(3)).getHostCacheState(
                any(GrpcTarget.class), any(GetHostCacheStateRequest.class), anyLong());

        assertThrows(KvcmQueryException.class, this::queryMockClient);
        assertEquals(2, client.healthSnapshot().consecutiveQueryFailures());
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        Mockito.verify(metaServiceClient, Mockito.times(6)).getHostCacheState(
                any(GrpcTarget.class), any(GetHostCacheStateRequest.class), anyLong());

        querySucceeds.set(true);
        assertTrue(queryMockClient().isEmpty());
        assertEquals(0, client.healthSnapshot().consecutiveQueryFailures());
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        Mockito.verify(metaServiceClient, Mockito.times(7)).getHostCacheState(
                any(GrpcTarget.class), any(GetHostCacheStateRequest.class), anyLong());
        Mockito.verify(leaderResolver, Mockito.times(1)).refresh();

        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
    }

    private Map<String, Integer> queryMockClient() {
        return client.findMatchingEngines(
                "request-health",
                List.of(11L),
                2192L,
                RoleType.PREFILL,
                "default");
    }

    private Map<String, Integer> waitForMatches(RoleType roleType) throws InterruptedException {
        return waitForMatches(roleType, "default");
    }

    private Map<String, Integer> waitForMatches(RoleType roleType, String group) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 3000L;
        while (System.currentTimeMillis() < deadline) {
            try {
                Map<String, Integer> result = client.findMatchingEngines(
                        "request-1", List.of(11L, 22L, 33L), 2192L, roleType, group);
                if (!result.isEmpty()) {
                    return result;
                }
            } catch (KvcmQueryException ignored) {
                // Background discovery may still be completing.
            }
            Thread.sleep(20L);
        }
        fail("KVCM client did not become ready before the test deadline");
        return Map.of();
    }

    private KvcmGrpcClient newClient(ModelMetaConfig modelMetaConfig, RoutingServiceDiscovery serviceDiscovery, KvCacheGroupMode mode) {
        return newClient(modelMetaConfig, serviceDiscovery, mode, 0);
    }

    private KvcmGrpcClient newClient(ModelMetaConfig modelMetaConfig, RoutingServiceDiscovery serviceDiscovery,
                                     KvCacheGroupMode mode, int rollbackBlocks) {
        KvcmMetaServiceClient metaServiceClient = new KvcmMetaServiceClient(channelFactory());
        CacheMatchConfiguration configuration = new CacheMatchConfiguration(modelMetaConfig);
        return new KvcmGrpcClient(
                configuration,
                metaServiceClient,
                new KvcmLeaderResolver(configuration, serviceDiscovery, metaServiceClient),
                new KvcmWorkerMetadataResolver(
                        configuration, workerStatusProvider(mode, rollbackBlocks)),
                () -> true);
    }

    private WorkerStatusProvider workerStatusProvider(KvCacheGroupMode mode, int rollbackBlocks) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setDeploymentName("deployment-first");
        workerStatus.setKvCacheGroupMode(mode);
        workerStatus.setCacheMatchRollbackBlocks(rollbackBlocks);
        return new WorkerStatusProvider() {
            @Override
            public Collection<WorkerStatus> getWorkerStatuses(RoleType roleType, String group) {
                return List.of(workerStatus);
            }
        };
    }

    private ModelMetaConfig modelMetaConfig(int bootstrapPort) {
        return modelMetaConfig(bootstrapPort, null);
    }

    private ModelMetaConfig modelMetaConfig(int bootstrapPort, String namespace) {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(ServiceDiscoveryType.DASHSCOPE);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setAddress("v-kvcm");
        kvcm.setPort(bootstrapPort);
        kvcm.setNamespace(namespace);
        kvcm.setDiscovery(discovery);
        kvcm.setRequestTimeoutMs(1000L);
        kvcm.setLeaderRefreshIntervalMs(60000L);

        Endpoint workerEndpoint = new Endpoint();
        workerEndpoint.setAddress("v-workers");
        workerEndpoint.setProtocol("grpc");
        workerEndpoint.setDiscovery(discovery);

        GroupRoleEndPoint group = new GroupRoleEndPoint();
        group.setGroup("default");
        group.setPdFusionEndpoint(workerEndpoint);
        group.setDecodeEndpoint(workerEndpoint);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);
        route.setRoleEndpoints(List.of(group));

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(route.getServiceId(), route);
        return modelMetaConfig;
    }

    private RoutingServiceDiscovery serviceDiscovery() {
        RoutingServiceDiscovery serviceDiscovery = Mockito.mock(RoutingServiceDiscovery.class);
        when(serviceDiscovery.getHosts(any(Endpoint.class))).thenAnswer(invocation -> {
            Endpoint endpoint = invocation.getArgument(0);
            if ("v-kvcm".equals(endpoint.getAddress())) {
                // The discovery port is not the MetaService gRPC port.
                return List.of(WorkerHost.of("127.0.0.1", 8080));
            }
            if ("v-workers".equals(endpoint.getAddress())) {
                return List.of(
                        WorkerHost.of("10.0.0.1", 8601, "", "deployment-first"),
                        WorkerHost.of("10.0.0.2", 8601, "", "deployment-second"));
            }
            return List.of();
        });
        return serviceDiscovery;
    }

    private GrpcChannelFactory channelFactory() {
        GrpcChannelFactory channelFactory = Mockito.mock(GrpcChannelFactory.class);
        when(channelFactory.create(any(GrpcTarget.class))).thenAnswer(invocation -> {
            GrpcTarget target = invocation.getArgument(0);
            return ManagedChannelBuilder.forAddress(target.host(), target.port())
                    .usePlaintext()
                    .build();
        });
        return channelFactory;
    }

    private static CommonResponseHeader okHeader() {
        return CommonResponseHeader.newBuilder()
                .setStatus(Status.newBuilder().setCode(ErrorCode.OK))
                .build();
    }

    private static final class SeedMetaService extends MetaServiceGrpc.MetaServiceImplBase {

        private final int leaderPort;

        private SeedMetaService(int leaderPort) {
            this.leaderPort = leaderPort;
        }

        @Override
        public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> responseObserver) {
            responseObserver.onNext(GetClusterInfoResponse.newBuilder()
                    .setHeader(okHeader())
                    .setLeaderNodeId("leader")
                    .setLeaderEndpoint(MetaNodeEndpoint.newBuilder()
                            .setNodeId("leader")
                            .setHost("127.0.0.1")
                            .setMetaRpcPort(leaderPort))
                    .build());
            responseObserver.onCompleted();
        }
    }

    private static final class LeaderMetaService extends MetaServiceGrpc.MetaServiceImplBase {

        private final AtomicReference<GetHostCacheStateRequest> lastCacheRequest;

        private LeaderMetaService(AtomicReference<GetHostCacheStateRequest> lastCacheRequest) {
            this.lastCacheRequest = lastCacheRequest;
        }

        @Override
        public void getHostCacheState(GetHostCacheStateRequest request, StreamObserver<GetHostCacheStateResponse> responseObserver) {
            lastCacheRequest.set(request);
            responseObserver.onNext(GetHostCacheStateResponse.newBuilder()
                    .setHeader(okHeader())
                    .addHosts(HostCacheMatch.newBuilder()
                            .setHostIpPort("10.0.0.1:8601")
                            .setPrefixMatchBlocks(2))
                    .build());
            responseObserver.onCompleted();
        }
    }
}

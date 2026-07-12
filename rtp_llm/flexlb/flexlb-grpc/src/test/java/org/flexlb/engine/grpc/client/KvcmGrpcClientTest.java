package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
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
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.MetaNodeEndpoint;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;
import org.flexlb.kvcm.grpc.Status;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
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
        client = new KvcmGrpcClient(
                modelMetaConfig(seedServer.getPort()), serviceDiscovery, channelFactory());

        Map<String, Integer> matches = waitForMatches(RoleType.PDFUSION);
        Map<String, Integer> decodeMatches = waitForMatches(RoleType.DECODE);

        assertEquals(2, matches.get("10.0.0.1:8601"));
        assertEquals(2, decodeMatches.get("10.0.0.1:8601"));
        GetHostCacheStateRequest request = lastCacheRequest.get();
        assertEquals("deployment-first", request.getInstanceId());
        assertEquals(List.of(11L, 22L, 33L), request.getBlockCacheKeysList());
        assertEquals(0, request.getMediumCount());
        assertTrue(client.findMatchingEngines(
                List.of(11L, 22L, 33L), RoleType.PDFUSION, null).isEmpty());
        assertTrue(client.findMatchingEngines(
                List.of(11L, 22L, 33L), RoleType.PDFUSION, "").isEmpty());
    }

    @Test
    void configuredNamespaceTakesPriorityAndSkipsWorkerMetadataDiscovery() throws Exception {
        RoutingServiceDiscovery serviceDiscovery = serviceDiscovery();
        client = new KvcmGrpcClient(
                modelMetaConfig(seedServer.getPort(), "vllm-test-0"),
                serviceDiscovery,
                channelFactory());

        Map<String, Integer> matches = waitForMatches(RoleType.PDFUSION, null);

        assertEquals(2, matches.get("10.0.0.1:8601"));
        assertEquals("vllm-test-0", lastCacheRequest.get().getInstanceId());
        Mockito.verify(serviceDiscovery, Mockito.never()).getHosts(Mockito.argThat(
                endpoint -> "v-workers".equals(endpoint.getAddress())));
    }

    private Map<String, Integer> waitForMatches(RoleType roleType) throws InterruptedException {
        return waitForMatches(roleType, "default");
    }

    private Map<String, Integer> waitForMatches(RoleType roleType, String group) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 3000L;
        while (System.currentTimeMillis() < deadline) {
            Map<String, Integer> result = client.findMatchingEngines(
                    List.of(11L, 22L, 33L), roleType, group);
            if (!result.isEmpty()) {
                return result;
            }
            Thread.sleep(20L);
        }
        fail("KVCM client did not become ready before the test deadline");
        return Map.of();
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
        public void getClusterInfo(
                GetClusterInfoRequest request,
                StreamObserver<GetClusterInfoResponse> responseObserver) {
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
        public void getHostCacheState(
                GetHostCacheStateRequest request,
                StreamObserver<GetHostCacheStateResponse> responseObserver) {
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

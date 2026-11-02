package org.flexlb.engine.grpc;

import io.grpc.*;
import io.grpc.stub.StreamObserver;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.discovery.LocalServiceDiscovery;
import org.flexlb.kvcm.grpc.*;
import org.junit.jupiter.api.Test;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.*;
import static org.junit.jupiter.api.Assertions.*;

class KvcmGrpcClientTest {
    private static final CommonResponseHeader OK = CommonResponseHeader.newBuilder()
            .setStatus(org.flexlb.kvcm.grpc.Status.newBuilder().setCode(ErrorCode.OK)).build();

    @Test void nativeLeaderDiscoveryPreservesKeysNamespaceAndLocalBlockCounts() throws Exception {
        var observed = new AtomicReference<GetHostCacheStateRequest>();
        var port = new AtomicInteger();
        var discoveries = new AtomicInteger();
        var failure = new AtomicBoolean();
        Server server = ServerBuilder.forPort(0).addService(new MetaServiceGrpc.MetaServiceImplBase() {
            @Override public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> out) {
                assertEquals("deployment_16", request.getInstanceId()); discoveries.incrementAndGet();
                out.onNext(GetClusterInfoResponse.newBuilder().setHeader(OK).setLeaderEndpoint(
                        MetaNodeEndpoint.newBuilder().setHost("127.0.0.1").setMetaRpcPort(port.get())).build()); out.onCompleted();
            }
            @Override public void getHostCacheState(GetHostCacheStateRequest request, StreamObserver<GetHostCacheStateResponse> out) {
                observed.set(request);
                if (failure.get()) { out.onError(io.grpc.Status.UNAVAILABLE.asRuntimeException()); return; }
                out.onNext(GetHostCacheStateResponse.newBuilder().setHeader(OK)
                        .addHosts(HostCacheMatch.newBuilder().setHostIpPort("127.0.0.1:18080@0").setLocal(2).setGlobal(100))
                        .addHosts(HostCacheMatch.newBuilder().setHostIpPort("old:18080").setLocal(3)).build()); out.onCompleted();
            }
        }).build().start();
        port.set(server.getPort());
        var topology = new KvcmConfig(); topology.setAddress("meta-service"); topology.setPort(port.get()); topology.setNamespace("deployment");
        var options = new KvcmCacheMatchingConfig(); options.setRequestTimeoutMs(3000);
        try (var client = new KvcmGrpcClient(options, topology,
                new LocalServiceDiscovery(Map.of("meta-service", List.of("127.0.0.1:" + port.get()))))) {
            var keys = List.of(Long.MIN_VALUE, -42L, Long.MAX_VALUE);
            var candidates = List.of("127.0.0.1:18080");
            assertEquals(Map.of(candidates.getFirst(), 2), client.findMatchingEngines(keys, 16, candidates));
            assertEquals(keys, observed.get().getBlockCacheKeysList());
            assertEquals("deployment_16", observed.get().getInstanceId());
            assertEquals(QueryType.QT_PREFIX_MATCH, observed.get().getQueryType());
            assertEquals(List.of("hbm"), observed.get().getMediumList());
            assertEquals(0, observed.get().getGlobalKvsHostCount()); assertFalse(observed.get().getEnableP2P());
            failure.set(true);
            assertThrows(StatusRuntimeException.class, () -> client.findMatchingEngines(keys, 16, candidates));
            failure.set(false); client.findMatchingEngines(keys, 16, candidates);
            assertEquals(2, discoveries.get());
            assertThrows(IllegalArgumentException.class, () -> client.findMatchingEngines(keys, 0, candidates));
        } finally { server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

    @Test void disabledModeNeedsNoTopologyAndMakesNoConnection() {
        try (var client = new KvcmGrpcClient((KvcmCacheMatchingConfig) null, null, null)) { assertFalse(client.enabled()); }
        assertThrows(IllegalArgumentException.class, () -> new KvcmGrpcClient(new KvcmCacheMatchingConfig(), null, null));
    }
}

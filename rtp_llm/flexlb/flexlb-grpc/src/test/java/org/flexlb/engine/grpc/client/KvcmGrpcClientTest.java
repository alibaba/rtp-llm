package org.flexlb.engine.grpc.client;

import io.grpc.Server;
import io.grpc.ManagedChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import java.net.InetSocketAddress;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.config.*;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.LocalServiceDiscovery;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.monitor.KvcmMetricsReporter;
import org.flexlb.kvcm.grpc.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.any;

class KvcmGrpcClientTest {
    private static final CommonResponseHeader OK = CommonResponseHeader.newBuilder()
            .setStatus(Status.newBuilder().setCode(ErrorCode.OK)).build();

    @Test void nativeQueryPreservesSignedKeysNamespaceAndQueryBudget() throws Exception {
        var observed = new AtomicReference<GetHostCacheStateRequest>();
        var port = new AtomicInteger();
        Server server = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0))
                .addService(new MetaServiceGrpc.MetaServiceImplBase() {
            @Override public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> out) {
                out.onNext(GetClusterInfoResponse.newBuilder().setHeader(OK).setLeaderEndpoint(
                        MetaNodeEndpoint.newBuilder().setHost("127.0.0.1").setMetaRpcPort(port.get())).build());
                out.onCompleted();
            }
            @Override public void getHostCacheState(GetHostCacheStateRequest request, StreamObserver<GetHostCacheStateResponse> out) {
                observed.set(request);
                out.onNext(GetHostCacheStateResponse.newBuilder().setHeader(OK)
                        .addHosts(HostCacheMatch.newBuilder().setHostIpPort("127.0.0.1:18080@0").setLocal(2).setGlobal(3)).build());
                out.onCompleted();
            }
        }).build().start();
        port.set(server.getPort());
        var route = new ServiceRoute(); var topology = new KvcmConfig();
        topology.setAddress("meta"); topology.setPort(port.get()); topology.setNamespace("deployment"); route.setKvcm(topology);
        var options = new KvcmCacheMatchingConfig(); options.setRequestTimeoutMs(3000); options.setGlobalKvsHostCount(0); options.setMedium(List.of("hbm"));
        var config = new FlexlbConfig(); config.setCacheMatching(options);
        var configuration = new CacheMatchConfiguration(route, config);
        var factory = mock(GrpcChannelFactory.class);
        when(factory.create(any())).thenAnswer(call -> {
            org.flexlb.engine.grpc.core.GrpcTarget target = call.getArgument(0);
            return ManagedChannelBuilder.forAddress(target.host(), target.port()).usePlaintext().build();
        });
        var meta = new KvcmMetaServiceClient(factory);
        var resolver = new KvcmLeaderResolver(configuration, new LocalServiceDiscovery(Map.of("meta", List.of("127.0.0.1:" + port.get()))), meta);
        var client = new KvcmGrpcClient(configuration, meta, resolver, mock(GrpcReporter.class), mock(KvcmMetricsReporter.class));
        try {
            client.refreshKvcmServiceStateSafely();
            var keys = List.of(Long.MIN_VALUE, -42L, Long.MAX_VALUE);
            var matches = client.findMatchingEngines("request", keys, 16, RoleType.PREFILL, null);
            assertEquals(2, matches.get("127.0.0.1:18080@0").localMatchBlocks());
            assertEquals(3, matches.get("127.0.0.1:18080@0").globalMatchBlocks());
            assertEquals(keys, observed.get().getBlockCacheKeysList());
            assertEquals("deployment_16", observed.get().getInstanceId());
            assertEquals("request", observed.get().getTraceId());
            assertEquals(QueryType.QT_PREFIX_MATCH, observed.get().getQueryType());
            assertEquals(List.of("hbm"), observed.get().getMediumList());
            assertEquals(0, observed.get().getGlobalKvsHostCount());
        } finally { client.shutdown(); meta.shutdown(); server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

    @Test void disabledModeCreatesNoChannelsOrRefresher() {
        var configuration = new CacheMatchConfiguration(new ServiceRoute(), new FlexlbConfig());
        var meta = mock(KvcmMetaServiceClient.class); var resolver = mock(KvcmLeaderResolver.class);
        var client = new KvcmGrpcClient(configuration, meta, resolver, mock(GrpcReporter.class), mock(KvcmMetricsReporter.class));
        assertTrue(client.findMatchingEngines("request", List.of(1L), 16, RoleType.PREFILL, null).isEmpty());
        client.shutdown(); verifyNoInteractions(meta, resolver);
    }
}

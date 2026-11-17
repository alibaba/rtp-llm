package org.flexlb.engine.grpc;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.discovery.LocalServiceDiscovery;
import org.flexlb.engine.grpc.monitor.KvcmMetricsReporter;
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
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

/**
 * Exercises leader discovery and prefix matching through a local KVCM RPC service.
 */
class KvcmGrpcClientTest {
    private static final CommonResponseHeader OK = CommonResponseHeader.newBuilder()
            .setStatus(org.flexlb.kvcm.grpc.Status.newBuilder().setCode(ErrorCode.OK)).build();

    @Test void nativeLeaderDiscoveryPreservesKeysNamespaceAndLocalBlockCounts() throws Exception {
        var observed = new AtomicReference<GetHostCacheStateRequest>();
        var port = new AtomicInteger();
        var discoveries = new AtomicInteger();
        var failuresRemaining = new AtomicInteger();
        var omitResponse = new AtomicBoolean();
        var emptyMatch = new AtomicBoolean();
        var attempts = new AtomicInteger();
        var reporter = mock(KvcmMetricsReporter.class);
        Server server = ServerBuilder.forPort(0).addService(new MetaServiceGrpc.MetaServiceImplBase() {
            @Override public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> out) {
                assertFalse(request.getTraceId().isBlank()); discoveries.incrementAndGet();
                out.onNext(GetClusterInfoResponse.newBuilder().setHeader(OK).setLeaderEndpoint(
                        MetaNodeEndpoint.newBuilder().setHost("127.0.0.1").setMetaRpcPort(port.get())).build()); out.onCompleted();
            }
            @Override public void getHostCacheState(GetHostCacheStateRequest request, StreamObserver<GetHostCacheStateResponse> out) {
                observed.set(request);
                attempts.incrementAndGet();
                if (failuresRemaining.getAndUpdate(n -> Math.max(0, n - 1)) > 0) {
                    out.onError(io.grpc.Status.UNAVAILABLE.asRuntimeException());
                    return;
                }
                if (omitResponse.get()) {
                    return;
                }
                if (emptyMatch.get()) {
                    out.onNext(GetHostCacheStateResponse.newBuilder().setHeader(OK).build());
                    out.onCompleted();
                    return;
                }
                out.onNext(GetHostCacheStateResponse.newBuilder().setHeader(OK)
                        .addHosts(HostCacheMatch.newBuilder().setHostIpPort("127.0.0.1:18080@0").setLocal(2).setGlobal(100))
                        .addHosts(HostCacheMatch.newBuilder().setHostIpPort("old:18080").setLocal(3)).build()); out.onCompleted();
            }
        }).build().start();
        port.set(server.getPort());
        var topology = new KvcmConfig(); topology.setAddress("meta-service"); topology.setPort(port.get()); topology.setNamespace("deployment");
        var options = new KvcmCacheMatchingConfig(); options.setRequestTimeoutMs(3000);
        try (var client = new KvcmGrpcClient(options, topology,
                new LocalServiceDiscovery(Map.of("meta-service", List.of("127.0.0.1:" + port.get()))),
                reporter)) {
            var keys = List.of(Long.MIN_VALUE, -42L, Long.MAX_VALUE);
            var candidates = List.of("127.0.0.1:18080");
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while (true) {
                try {
                    assertEquals(Map.of(candidates.getFirst(), 2), client.findMatchingEngines(keys, 16, candidates));
                    break;
                } catch (IllegalStateException unavailable) {
                    if (System.nanoTime() >= deadline) throw unavailable;
                    Thread.sleep(10);
                }
            }
            assertEquals(keys, observed.get().getBlockCacheKeysList());
            assertEquals("deployment_16", observed.get().getInstanceId());
            assertEquals(QueryType.QT_PREFIX_MATCH, observed.get().getQueryType());
            assertEquals(List.of("hbm"), observed.get().getMediumList());
            assertEquals(0, observed.get().getGlobalKvsHostCount()); assertFalse(observed.get().getEnableP2P());
            clearInvocations(reporter);
            attempts.set(0);
            failuresRemaining.set(1);
            assertEquals(Map.of(candidates.getFirst(), 2), client.findMatchingEngines(keys, 16, candidates));
            assertEquals(2, attempts.get());
            verify(reporter).reportQueryRetry(1);
            verifyNoMoreInteractions(reporter);

            clearInvocations(reporter);
            attempts.set(0);
            failuresRemaining.set(2);
            assertThrows(StatusRuntimeException.class, () -> client.findMatchingEngines(keys, 16, candidates));
            assertEquals(2, attempts.get());
            verify(reporter).reportQueryRetry(1);
            verify(reporter).reportQueryFailure();
            verifyNoMoreInteractions(reporter);

            clearInvocations(reporter);
            emptyMatch.set(true);
            assertEquals(Map.of(), client.findMatchingEngines(keys, 16, candidates));
            verifyNoMoreInteractions(reporter);

            options.setRequestTimeoutMs(200);
            omitResponse.set(true);
            var timeout = assertThrows(StatusRuntimeException.class,
                    () -> client.findMatchingEngines(keys, 16, candidates));
            assertEquals(io.grpc.Status.Code.DEADLINE_EXCEEDED, timeout.getStatus().getCode());
            verify(reporter).reportQueryFailure();
            verifyNoMoreInteractions(reporter);
            assertTrue(discoveries.get() >= 1);
            assertThrows(IllegalArgumentException.class, () -> client.findMatchingEngines(keys, 0, candidates));
        } finally { server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

    @Test void disabledModeNeedsNoTopologyAndMakesNoConnection() {
        var reporter = mock(KvcmMetricsReporter.class);
        try (var client = new KvcmGrpcClient((KvcmCacheMatchingConfig) null, null, null, reporter)) {
            assertFalse(client.enabled());
        }
        assertThrows(IllegalArgumentException.class, () -> new KvcmGrpcClient(new KvcmCacheMatchingConfig(), null, null, reporter));
    }
}

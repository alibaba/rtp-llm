package org.flexlb.mockengine;

import io.grpc.*;
import io.grpc.netty.NettyServerBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class MockVitMonitorTest {
    private final FlexMonitor sink = mock(FlexMonitor.class);
    private final Map<String, String> tags = Map.of("hippo_app", "app", "hippo_role", "deployment_g03.vit_part0",
            "container_ip", "127.0.0.1", "source", "mock_vit");

    @Test void pollingIsExcludedAndCachedEmbeddingStillCountsAsRpc() throws Exception {
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults());
             var monitor = new MockVitMonitor(sink, tags)) {
            monitor.start(service);
            var server = NettyServerBuilder.forPort(0).addService(service).intercept(monitor).build().start();
            var channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            try {
                var stub = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5, TimeUnit.SECONDS);
                stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
                verify(sink, never()).report(eq("mock_vit_rpc_qps"), any(), anyDouble());
                var input = MockVitServiceTest.inputs(MockVitServiceTest.image(28, 28, 0xabcdef));
                stub.remoteMultimodalEmbedding(input);
                stub.remoteMultimodalEmbedding(input);
                assertEquals(1, service.computations.get());
                verify(sink, times(1)).report(eq("py_rtp_vit_qps"), any(), eq(1.0));
                verify(sink, times(1)).report(eq("py_rtp_vit_success_qps"), any(), eq(1.0));
                verify(sink, never()).report(eq("py_rtp_vit_error_qps"), any(), anyDouble());
                verify(sink, times(2)).report(eq("mock_vit_rpc_success_qps"), argThat(t ->
                        t.getTags().get("method").equals("RemoteMultimodalEmbedding")
                        && t.getTags().get("code").equals("OK")
                        && t.getTags().get("hippo_role").equals(tags.get("hippo_role"))), eq(1.0));
                stub.getMultimodalHashes(EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys("missing").build());
                verify(sink).report(eq("mock_vit_rpc_qps"), argThat(t ->
                        "GetMultimodalHashes".equals(t.getTags().get("method"))), eq(1.0));
                verify(sink, times(1)).report(eq("py_rtp_vit_qps"), any(), eq(1.0));
                assertThrows(StatusRuntimeException.class, () -> stub.remoteMultimodalEmbedding(
                        EngineRpcService.MultimodalInputsPB.getDefaultInstance()));
                verify(sink).report(eq("mock_vit_rpc_error_qps"), argThat(t ->
                        "INVALID_ARGUMENT".equals(t.getTags().get("code"))), eq(1.0));
            } finally {
                channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
                server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            }
        }
    }

    @Test void logicalPoolAndUrlCacheGaugesAreRegisteredBeforeSampling() {
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults());
             var monitor = new MockVitMonitor(sink, tags)) {
            monitor.sample(service.snapshot(), System.nanoTime());
            monitor.sample(service.snapshot(), System.nanoTime());
            for (String field : java.util.List.of("gpu_pool_used_bytes", "cpu_pool_pending_bytes",
                    "cpu_pool_largest_free_block_bytes", "hash_pool_used_bytes", "url_cache_items")) {
                verify(sink, times(1)).register(eq("mock_vit_" + field), eq(org.flexlb.enums.FlexMetricType.GAUGE));
                verify(sink, times(2)).report(eq("mock_vit_" + field), any(), anyDouble());
            }
        }
    }

    @Test void schedulerMetricsUseRealNamesMillisecondsAndExcludeCacheHits() throws Exception {
        var document = MockEngineConfigFixture.load().snapshot();
        var limits = document.path("roles").path("vit");
        var config = new MockVitService.Config(8, limits.path("vit_concurrency").intValue(),
                limits.path("vit_max_queue_size").intValue(), limits.path("mm_cache_cpu_max_bytes").longValue(),
                limits.path("mm_hash_key_cache_max_bytes").longValue(), 1 << 20, 1 << 20, 20);
        try (var service = new MockVitService(config, null, document);
             var monitor = new MockVitMonitor(sink, tags)) {
            monitor.start(service);
            var image = MockVitServiceTest.image(28, 28, 0xabc123);
            service.acquire(image, 3000).get(3, TimeUnit.SECONDS);
            service.acquire(image, 3000).get(3, TimeUnit.SECONDS);
            verify(sink, times(1)).report(eq("py_rtp_vit_embedding_batch_size"), any(), eq(1.0));
            verify(sink, times(1)).report(eq("py_rtp_vit_embedding_forward_rt"), any(), doubleThat(ms -> ms >= 15 && ms < 2000));
            verify(sink, times(1)).report(eq("py_rtp_vit_embedding_queue_wait_rt"), any(), doubleThat(ms -> ms >= 0));
            verify(sink, timeout(1000).times(1)).report(eq("py_rtp_vit_embedding_rt"), any(), doubleThat(ms -> ms >= 15));
            verify(sink, atLeastOnce()).report(eq("py_rtp_vit_embedding_queue_size"), any(), eq(0.0));
        }
    }

    @Test void cancelledRpcRecordsOnlyOneTerminal() throws Exception {
        var d = MockVitService.Config.defaults();
        try (var service = MockVitTestSupport.service(new MockVitService.Config(8, 1, 4, 1 << 20, 1 << 20,
                d.maxInputBytes(), d.maxOutputBytes(), 500)); var monitor = new MockVitMonitor(sink, tags)) {
            monitor.start(service);
            var server = NettyServerBuilder.forPort(0).addService(service).intercept(monitor).build().start();
            var channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            try {
                MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5, TimeUnit.SECONDS)
                        .getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                var stub = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(100, TimeUnit.MILLISECONDS);
                assertThrows(StatusRuntimeException.class, () -> stub.remoteMultimodalEmbedding(
                        MockVitServiceTest.inputs(MockVitServiceTest.image(28, 28, 0x123456))));
                verify(sink, timeout(2000).times(1)).report(eq("mock_vit_rpc_error_qps"), any(), eq(1.0));
                Thread.sleep(600);
                verify(sink, times(1)).report(eq("mock_vit_rpc_error_qps"), any(), eq(1.0));
                verify(sink, never()).report(eq("mock_vit_rpc_success_qps"), any(), anyDouble());
            } finally {
                channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
                server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            }
        }
    }

    @Test void samplingReportsComputeDeltaAndIdleZero() {
        try (var monitor = new MockVitMonitor(sink, tags)) {
            var snapshot = Map.of("computations", 7L, "active", 0L, "queued", 0L, "pending", 0L,
                    "embedding_bytes", 128L, "hash_bytes", 64L);
            long now = System.nanoTime();
            monitor.sample(snapshot, now);
            clearInvocations(sink);
            monitor.sample(snapshot, now + TimeUnit.SECONDS.toNanos(1));
            verify(sink).report(eq("mock_vit_computation_qps"), any(), eq(0.0));
            verify(sink).report(eq("mock_vit_active"), any(), eq(0.0));
            verify(sink).report(eq("mock_vit_queued"), any(), eq(0.0));
            verify(sink).report(eq("mock_vit_computations_total"), any(), eq(7.0));
            monitor.sample(Map.of("computations", 11L), now + TimeUnit.SECONDS.toNanos(2));
            verify(sink).report(eq("mock_vit_computation_qps"), any(), eq(4.0));
        }
        verify(sink).close();
    }

    @Test void brokenSinkDoesNotBreakSamplingAndToggleIsStrict() {
        doThrow(new IllegalStateException("sink unavailable")).when(sink).report(anyString(), any(), anyDouble());
        try (var monitor = new MockVitMonitor(sink, tags)) {
            assertDoesNotThrow(() -> monitor.sample(Map.of("computations", 0L), System.nanoTime()));
        }
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(new String[]{"--mock.kmonitor", "typo"}));
    }
}

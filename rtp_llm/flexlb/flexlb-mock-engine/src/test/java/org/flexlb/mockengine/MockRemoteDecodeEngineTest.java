package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@Timeout(20)
class MockRemoteDecodeEngineTest {
    @TempDir Path directory;
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(4);
    private JavaMockEngineCluster.FastRpcService decode;
    private Server server;
    private ManagedChannel channel;

    private void start() throws Exception {
        Path perf = directory.resolve("performance.json");
        Path master = directory.resolve("master.json");
        new ObjectMapper().writeValue(perf.toFile(), Map.of("block_size", 1024, "sleep_scale", 1,
                "jitter_pct", 0, "prefill", Map.of("scale", 1),
                "decode", Map.of("scale", 1, "tokens_per_step", 1,
                        "step_ms_by_batch", List.of(List.of(1, 10)))));
        MockMasterConfig.writeWithPrefillExpression(master, "10");
        decode = new JavaMockEngineCluster.FastRpcService("decode",
                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE, 0, new ConcurrentHashMap<>(), scheduler,
                MockPerformanceModel.load(perf.toString(), master.toString()), 8,
                new JavaMockEngineCluster.ClusterStats());
        decode.setWhaleRemote(true);
        server = ServerBuilder.forPort(0).addService(decode).build().start();
        channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
    }

    private MockRemoteDecodeStream stream(long id, CompletableFuture<EngineRpcService.GenerateOutputsPB> result) {
        return new MockRemoteDecodeStream(channel, EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(id).addAllTokenIds(List.of(1, 2, 3))
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(4))
                .build(), "p-generation", 10_000, frame -> {
                    if (frame.hasErrorInfo() || frame.getFlattenOutput().getFinishedList().contains(true)) {
                        result.complete(frame);
                    }
                }, result::completeExceptionally, () -> {});
    }

    private void awaitDrain() throws Exception {
        CompletableFuture<Void> drained = new CompletableFuture<>();
        var poll = scheduler.scheduleAtFixedRate(() -> {
            if (decode.getOccupiedKvTokens() == 0 && !decode.hasInflightWork()) drained.complete(null);
        }, 0, 5, TimeUnit.MILLISECONDS);
        try {
            drained.get(3, TimeUnit.SECONDS);
        } finally {
            poll.cancel(false);
        }
    }

    @AfterEach
    void stop() throws Exception {
        if (channel != null) channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
        if (server != null) server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
        if (decode != null) { decode.drainAndShutdown(); decode.shutdown(); }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    @Test
    void allocationUsesRealKvAndCancelReleasesIt() throws Exception {
        start();
        try (var stream = stream(42, new CompletableFuture<>())) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            assertEquals(1024, decode.getOccupiedKvTokens());
            assertEquals(0, decode.getActiveDecodeCount());
            var metrics = decode.whaleMetrics();
            assertEquals(7, metrics.get("rtp_llm_kv_cache_available_blocks").intValue());
            assertEquals(12.5, metrics.get("rtp_llm_kv_cache_used_ratio").doubleValue());
            assertEquals(8, metrics.get("rtp_llm_kv_cache_pool_total_blocks").intValue());
            assertEquals(7, metrics.get("rtp_llm_kv_cache_pool_available_blocks").intValue());
            assertEquals(12.5, metrics.get("rtp_llm_kv_cache_pool_used_ratio").doubleValue());
            assertEquals(7 * 1024L, metrics.get("rtp_llm_kv_cache_left_seq").longValue());
            stream.close();
            awaitDrain();
            assertEquals(8, decode.whaleMetrics().get("rtp_llm_kv_cache_available_blocks").intValue());
            assertEquals(0.0, decode.whaleMetrics().get("rtp_llm_kv_cache_used_ratio").doubleValue());
        }
    }

    @Test
    void remoteGenerateUsesExistingDecodeExecutionAndReleasesLease() throws Exception {
        start();
        CompletableFuture<EngineRpcService.GenerateOutputsPB> result = new CompletableFuture<>();
        try (var stream = stream(42, result)) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            stream.load().get(3, TimeUnit.SECONDS);
            stream.generate(7);
            assertTrue(result.get(3, TimeUnit.SECONDS).getFlattenOutput().getFinishedList().contains(true));
            awaitDrain();
        }
    }

    @Test
    void duplicateStreamCannotReleaseTheOriginalReservation() throws Exception {
        start();
        try (var original = stream(42, new CompletableFuture<>())) {
            original.allocated().get(3, TimeUnit.SECONDS);
            try (var duplicate = stream(42, new CompletableFuture<>())) {
                assertThrows(java.util.concurrent.ExecutionException.class,
                        () -> duplicate.allocated().get(3, TimeUnit.SECONDS));
                assertEquals(1024, decode.getOccupiedKvTokens());
            }
            original.close();
            awaitDrain();
        }
    }

    @Test
    void externalCancelClosesTheUnstartedStreamAndReleasesItsLease() throws Exception {
        start();
        CompletableFuture<EngineRpcService.GenerateOutputsPB> result = new CompletableFuture<>();
        try (var stream = stream(42, result)) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            decode.cancel(42);
            assertThrows(java.util.concurrent.ExecutionException.class, () -> result.get(3, TimeUnit.SECONDS));
            awaitDrain();
        }
    }

    @Test
    void stoppedDecodeClosesReservedStreamsWithoutWaitingForTheirDeadline() throws Exception {
        start();
        CompletableFuture<EngineRpcService.GenerateOutputsPB> result = new CompletableFuture<>();
        try (var stream = stream(42, result)) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            decode.setStopped(true);
            assertThrows(java.util.concurrent.ExecutionException.class, () -> result.get(3, TimeUnit.SECONDS));
            awaitDrain();
        }
    }
}

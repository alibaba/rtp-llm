package org.flexlb.mock.engine.scheduler;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

public class LLMMockEngineTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @TempDir Path directory;

    public static Map<String, Object> configuration(String role) {
        return Map.of("id", role + "-0", "component_role", role,
            "component", Map.of("engine", "rtp", "config", Map.of("seq_size_per_block", 16,
                "max_seq_len", 4096, "test_block_num", 9, "reserve_block_ratio", 0, "mock.compute", Map.of("forward", Map.of("expression", "1 + batchSize")))),
            "environment", Map.of());
    }

    @Test void servesProductionNativeProtocolAndLiveCacheCapacity() throws Exception {
        var root = JSON.<com.fasterxml.jackson.databind.node.ObjectNode>valueToTree(configuration("prefill"));
        ((com.fasterxml.jackson.databind.node.ObjectNode) root.path("component").path("config"))
            .put("dp_size", 3).put("mock.cache.total.blocks", 16);
        try (var node = LLMMockEngine.start(RtpConfig.load(root))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                assertEquals("OK", rpc.checkHealth(EmptyPB.getDefaultInstance()).getHealth());
                var status = rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                assertEquals("RoleType.PREFILL", status.getRole()); assertEquals(RoleTypePB.ROLE_TYPE_PREFILL, status.getRoleType());
                assertEquals(0, status.getAvailableConcurrency()); assertTrue(status.getAlive());
                assertEquals(128, status.getTotalKvCache()); assertEquals(128, status.getAvailableKvCache());
                assertEquals(4096, status.getMaxSeqLen());
                assertEquals(1, status.getDpSize()); assertEquals(0, status.getDpRank()); assertEquals(1, status.getTpSize());
                assertEquals(4096, status.getMaxBatchTokensSize());
                assertTrue(rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getStatusVersion() > status.getStatusVersion());
                var lease = node.cacheManager().acquire(32, List.of()).lease();
                assertEquals(96, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                lease.publish(List.of(1L, -2L), 32); lease.close();
                var snapshot = rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build());
                assertEquals(Set.of(1L, -2L), snapshot.getCacheKeysMap().keySet());
                assertEquals(16, snapshot.getBlockSize()); assertEquals(128, snapshot.getAvailableKvCache());
                try (var extra = node.cacheManager().acquire(112, List.of()).lease()) {
                    assertTrue(extra.publish(List.of(10L, 11L, 12L, 13L, 14L, 15L, 16L), 112));
                }
                assertEquals(9, node.cacheManager().itemCount()); // Logical cache now exceeds eight device blocks.
                assertEquals(128, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getTotalKvCache());
                snapshot = rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build());
                assertEquals(9, snapshot.getCacheKeysCount());
                var pinned = node.cacheManager().acquire(16, List.of(1L)).lease();
                var unchanged = rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(snapshot.getVersion()).build());
                assertEquals(snapshot.getVersion(), unchanged.getVersion()); assertEquals(0, unchanged.getCacheKeysCount());
                assertEquals(112, unchanged.getAvailableKvCache()); pinned.close();
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void decodeReportsItsRoleAndRequestRpcDoesNotInventSuccessfulGeneration() throws Exception {
        try (var node = LLMMockEngine.start(config("decode"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                assertEquals(RoleTypePB.ROLE_TYPE_DECODE, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRoleType());
                var error = assertThrows(StatusRuntimeException.class, () -> rpc.enqueueBatch(EnqueueBatchRequestPB.getDefaultInstance()));
                assertEquals(Status.Code.UNIMPLEMENTED, error.getStatus().getCode());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void defaultFifoModeRejectsGroupRpcsButStillGeneratesIndividualRequests() throws Exception {
        var root = JSON.<com.fasterxml.jackson.databind.node.ObjectNode>valueToTree(configuration("prefill"));
        try (var node = LLMMockEngine.start(RtpConfig.load(root))) {
            assertFalse(((List<?>) node.endpoints().get("supported_rpc")).contains("EnqueueBatch"));
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                var batch = assertThrows(StatusRuntimeException.class, () -> rpc.enqueueBatch(EnqueueBatchRequestPB.getDefaultInstance()));
                assertEquals(Status.Code.UNIMPLEMENTED, batch.getStatus().getCode());
                var group = assertThrows(StatusRuntimeException.class, () -> rpc.enqueueGroup(EnqueueGroupRequestPB.getDefaultInstance()));
                assertEquals(Status.Code.UNIMPLEMENTED, group.getStatus().getCode());
                var request = GenerateInputPB.newBuilder().setRequestId(1).addTokenIds(42)
                    .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(1)).build();
                var outputs = rpc.generateStreamCall(request);
                assertTrue(outputs.next().getFlattenOutput().getFinished(0));
                assertFalse(outputs.hasNext());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void removedRemoteCacheIsRejected() throws Exception {
        var root = JSON.<com.fasterxml.jackson.databind.node.ObjectNode>valueToTree(configuration("prefill"));
        ((com.fasterxml.jackson.databind.node.ObjectNode) root.path("component").path("config"))
            .put("enable_remote_cache", true);
        var error = assertThrows(IllegalArgumentException.class, () -> LLMMockEngine.start(RtpConfig.load(root)));
        assertTrue(error.getMessage().contains("supports only local KV"));
    }

    @Test void exportsClasspathForNativeCrossProcessProbe() throws Exception {
        Files.writeString(Path.of("target/rtp-classpath.txt"), System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
    }

    private RtpConfig config(String role) throws Exception {
        Path file = directory.resolve("config.json"); JSON.writeValue(file.toFile(), configuration(role));
        return RtpConfig.load(file);
    }
}

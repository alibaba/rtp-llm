package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class RtpNodeTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @TempDir Path directory;

    public static Map<String, Object> configuration(String role) {
        return Map.of("id", role + "-0", "component_role", role,
            "component", Map.of("engine", "rtp", "config", Map.of("seq_size_per_block", 16,
                "max_seq_len", 4096, "test_block_num", 9, "reserve_block_ratio", 0, "compute", Map.of("forward", Map.of("expression", "1 + batchSize")))),
            "environment", Map.of());
    }
    private RtpConfig config(String role) throws Exception {
        Path file = directory.resolve("config.json"); JSON.writeValue(file.toFile(), configuration(role));
        return RtpConfig.load(file);
    }

    @Test void servesProductionNativeProtocolAndLiveCacheCapacity() throws Exception {
        try (var node = RtpNode.start(config("prefill"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                assertEquals("OK", rpc.checkHealth(EmptyPB.getDefaultInstance()).getHealth());
                var status = rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                assertEquals("RoleType.PREFILL", status.getRole()); assertEquals(RoleTypePB.ROLE_TYPE_PREFILL, status.getRoleType());
                assertEquals(0, status.getAvailableConcurrency()); assertTrue(status.getAlive());
                assertEquals(128, status.getTotalKvCache()); assertEquals(128, status.getAvailableKvCache());
                assertEquals(4096, status.getMaxSeqLen());
                assertTrue(rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getStatusVersion() > status.getStatusVersion());
                var lease = node.cache().acquire(32, List.of()).lease();
                assertEquals(96, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                lease.publish(List.of(1L, -2L), 32); lease.close();
                var snapshot = rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build());
                assertEquals(Set.of(1L, -2L), snapshot.getCacheKeysMap().keySet());
                assertEquals(16, snapshot.getBlockSize()); assertEquals(128, snapshot.getAvailableKvCache());
                var pinned = node.cache().acquire(16, List.of(1L)).lease();
                var unchanged = rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(snapshot.getVersion()).build());
                assertEquals(snapshot.getVersion(), unchanged.getVersion()); assertEquals(0, unchanged.getCacheKeysCount());
                assertEquals(112, unchanged.getAvailableKvCache()); pinned.close();
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void decodeReportsItsRoleAndRequestRpcDoesNotInventSuccessfulGeneration() throws Exception {
        try (var node = RtpNode.start(config("decode"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                assertEquals(RoleTypePB.ROLE_TYPE_DECODE, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRoleType());
                var error = assertThrows(StatusRuntimeException.class, () -> rpc.enqueueBatch(EnqueueBatchRequestPB.getDefaultInstance()));
                assertEquals(Status.Code.UNIMPLEMENTED, error.getStatus().getCode());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void exportsClasspathForNativeCrossProcessProbe() throws Exception {
        Files.writeString(Path.of("target/rtp-classpath.txt"), System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
    }
}

package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;

/** Test-only external native client. No Engine substitution. */
public final class NativeProbe {
    public static void main(String[] args) throws Exception {
        var channel = ManagedChannelBuilder.forTarget(args[0]).usePlaintext().build();
        try {
            var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
            var worker = rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance());
            var cache = rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build());
            var health = rpc.checkHealth(EmptyPB.getDefaultInstance());
            var result = Map.of("health", health.getHealth(), "role", worker.getRole(), "role_type", worker.getRoleType().name(),
                "alive", worker.getAlive(), "total_tokens", worker.getTotalKvCache(), "available_tokens", cache.getAvailableKvCache(),
                "block_size", cache.getBlockSize(), "cache_keys", cache.getCacheKeysCount());
            new ObjectMapper().writeValue(Path.of(args[1]).toFile(), result);
        } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
}

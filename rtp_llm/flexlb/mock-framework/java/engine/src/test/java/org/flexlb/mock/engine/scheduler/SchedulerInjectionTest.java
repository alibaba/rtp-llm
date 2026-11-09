package org.flexlb.mock.engine.scheduler;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.flexlb.mock.engine.forward.SimulatedCompute;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks that control requests can inject scheduler work through the Engine entry point.
 */
class SchedulerInjectionTest {
    @TempDir Path directory;

    @Test void defaultRtpPolicyRunsALegalSingletonAboveTheBatchTokenBudget() throws Exception {
        try (var node = LLMMockEngine.start(config("prefill"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                var outputs = rpc.generateStreamCall(input(1));
                assertTrue(outputs.hasNext()); assertTrue(outputs.next().getFlattenOutput().getFinished(0));
                assertFalse(outputs.hasNext());
                assertEquals(128, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void factoryFailureOrNullPolicyClosesTheNewInstancesCache() throws Exception {
        for (boolean returnsNull : List.of(false, true)) {
            var resource = new AtomicReference<KvCacheManager>();
            var config = config("prefill");
            var error = assertThrows(RuntimeException.class,
                () -> LLMMockEngine.start(config, (lock, cache, finished) -> {
                    resource.set(cache);
                    if (returnsNull) return null;
                    throw new IllegalStateException("Cannot create scheduler");
                }));
            assertEquals(returnsNull ? NullPointerException.class : IllegalStateException.class, error.getClass());
            assertNotNull(resource.get());
            assertThrows(IllegalStateException.class, () -> resource.get().acquire(1, List.of()));
        }
    }

    private RtpConfig config(String role) throws Exception {
        var path = directory.resolve(role + ".json");
        var json = new ObjectMapper();
        var document = json.valueToTree(LLMMockEngineTest.configuration(role));
        ((com.fasterxml.jackson.databind.node.ObjectNode) document.path("component").path("config"))
            .put("max_batch_tokens_size", 1);
        json.writeValue(path.toFile(), document);
        return RtpConfig.load(path);
    }

    private static GenerateInputPB input(int output) {
        return GenerateInputPB.newBuilder().setRequestId(1).addAllTokenIds(List.of(1, 2, 3))
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(output)).build();
    }
}

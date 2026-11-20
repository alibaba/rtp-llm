package org.flexlb.mock.engine.scheduler;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.worker.rpc.protocol.EngineRpcService.*;
import org.flexlb.worker.rpc.protocol.RpcServiceGrpc;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that the Engine loop uses its supplied scheduler.
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
            assertThrows(IllegalStateException.class, () -> resource.get().prepare(1, List.of()));
        }
    }

    @Test void scrapeReadsQueueStateFromInjectedScheduler() throws Exception {
        var config = config("prefill");
        try (var node = LLMMockEngine.start(config, (lock, cache, finished) -> {
            var recorder = new RtpMetrics("prefill", () -> fail("Only the Engine recorder is scraped"));
            var delegate = new RtpFifoScheduler(lock, config, cache, finished, recorder);
            return (EngineScheduler) Proxy.newProxyInstance(
                EngineScheduler.class.getClassLoader(), new Class<?>[]{EngineScheduler.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("queues")) {
                        return new EngineScheduler.QueueSnapshot(7, 0, 0, 0);
                    }
                    try {
                        return method.invoke(delegate, args);
                    } catch (InvocationTargetException error) {
                        throw error.getCause();
                    }
                });
        }); var http = HttpClient.newHttpClient()) {
            var request = HttpRequest.newBuilder(URI.create((String) node.endpoints().get("metrics")))
                .timeout(Duration.ofSeconds(3)).build();
            var response = http.send(request, HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode());
            assertTrue(response.body().lines().anyMatch(line -> line.startsWith("rtp_llm_scheduler_requests{")
                && line.contains("state=\"waiting\"") && line.endsWith(" 7.0")));
            assertTrue(response.body().contains("rtp_llm_kv_tokens"));
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

package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.*;
import java.time.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.*;

/**
 * Cold/warm calls against the unmodified DSV route path with native KVCM lookup enabled.
 */
public final class KvcmMasterProbe {
    public static void main(String[] args) throws Exception {
        var json = new ObjectMapper();
        int prefillCpSize = args.length > 4 ? Integer.parseInt(args[4]) : 1;
        int inputTokens = Math.max(32, 24 * prefillCpSize);
        try (var client = new NativeClient(() -> List.of(args[0]), event -> {}, 1000, 0L, prefillCpSize)) {
            NativeClient.Result last = null;
            for (long id : List.of(8001L, 8002L)) {
                var input = GenerateInputPB.newBuilder().setRequestId(id)
                    .addAllTokenIds(java.util.stream.IntStream.range(1, inputTokens + 1).boxed().toList()).setTokenIds(8, 1000)
                    .addMultimodalInputs(MultimodalInputPB.newBuilder().setMultimodalType(1).setMultimodalUrl("same-image")
                        .setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(84).setHeight(84)))
                    .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(3).setIgnoreEos(true).setReuseCache(true)
                        .setReturnIncremental(true).setCanUsePdSeparation(true)).build();
                var request = FlexlbScheduleRequestPB.newBuilder().setRequestId(id).setModel("mock").setSeqLen(inputTokens)
                    .setMaxNewTokens(3).setPriority(50).setCacheKeyBlockSize(16).setRequestTimeMs(System.currentTimeMillis())
                    .setGenerateInput(input.toByteString()).build();
                var result = client.send(request, Duration.ofSeconds(8)).completion().toCompletableFuture().get(9, TimeUnit.SECONDS);
                last = result;
                if (result.outcome() != NativeClient.Outcome.success || result.outputTokens() != 3)
                    throw new IllegalStateException("KVCM routed generation failed: " + result);
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                while (id == 8001 && args.length > 3 && json.readTree(URI.create(args[3] + "/stats").toURL()).path("keys").asLong() == 0) {
                    if (System.nanoTime() >= deadline) throw new IllegalStateException("Engine keys were not reported");
                    Thread.sleep(20);
                }
            }
            var stats = args.length > 3 ? json.readTree(URI.create(args[3] + "/stats").toURL()) : json.createObjectNode();
            if (args.length > 3 && (stats.path("queries").asLong() < 1 || stats.path("matching_hosts").asLong() < 1))
                throw new IllegalStateException("KVCM route never matched Engine keys: " + stats);
            json.writeValue(Path.of(args[2]).toFile(), Map.of("enqueued_by_master", false, "outcome", "success", "tokens", 3,
                    "ttft_ns", last.ttftNanos(), "elapsed_ns", last.elapsedNanos(), "kvcm_queries", stats.path("queries").asLong(), "matching_hosts", stats.path("matching_hosts").asLong()));
        }
    }
}

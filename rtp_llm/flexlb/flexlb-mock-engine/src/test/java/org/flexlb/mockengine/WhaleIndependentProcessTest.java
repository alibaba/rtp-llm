package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/** Separate JVMs, including two Decode hosts using the same port. No shared engine objects. */
@Timeout(60)
class WhaleIndependentProcessTest {
    @TempDir Path directory;
    private final List<Process> children = new ArrayList<>();
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build();
    private final int base = Integer.parseInt(System.getenv().getOrDefault("FLEXLB_PORT_BASE", "62400"));
    private ManagedChannel channel;

    private String get(String ip, int port, String path) throws Exception {
        var response = http.send(HttpRequest.newBuilder(URI.create("http://" + ip + ":" + port + path))
                .timeout(Duration.ofSeconds(2)).GET().build(), HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode());
        return response.body();
    }

    private Process start(String ip, int grpcPort, boolean prefill, boolean automatic) throws Exception {
        Path perf = directory.resolve("performance.json");
        Path master = directory.resolve("master.json");
        if (!Files.exists(perf)) {
            new ObjectMapper().writeValue(perf.toFile(), Map.of("block_size", 1024, "sleep_scale", 1,
                    "jitter_pct", 0, "prefill", Map.of("scale", 1), "decode", Map.of("scale", 1,
                            "tokens_per_step", 1, "step_ms_by_batch", List.of(List.of(1, 20)))));
            MockMasterConfig.writeWithPrefillExpression(master, "30");
        }
        Path log = directory.resolve(ip + ".log");
        Process process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp", System.getProperty("java.class.path"), JavaMockEngineCluster.class.getName(),
                "--whale", "true", "--n-prefill", prefill ? "1" : "0", "--n-decode", prefill ? "0" : "1",
                "--host", ip, "--bind-host", ip, "--base-grpc-port", String.valueOf(grpcPort),
                "--performance", perf.toString(), "--master-config", master.toString(),
                "--endpoint-file", directory.resolve(ip + "-unused-endpoint.json").toString(),
                "--auto-fetch", String.valueOf(automatic))
                .redirectErrorStream(true).redirectOutput(log.toFile()).start();
        children.add(process);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
        while (System.nanoTime() < deadline && process.isAlive()) {
            try { get(ip, grpcPort - 1, "/health"); return process; }
            catch (java.io.IOException ignored) { Thread.sleep(30); }
        }
        fail("Engine did not start: " + Files.readString(log));
        return process;
    }

    private void enqueue(long id, String decodeIp) {
        var result = enqueueAck(id, decodeIp);
        assertEquals(0, result.getErrorsCount(), result.toString());
        assertEquals(1, result.getSuccessesCount());
    }

    private EngineRpcService.EnqueueBatchResponsePB enqueueAck(long id, String decodeIp) {
        var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(List.of(1, 2, 3))
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(5)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder().setRoleStr("DECODE")
                                .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                .setIp(decodeIp).setGrpcPort(base + 21)));
        return RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5, TimeUnit.SECONDS)
                .enqueueBatch(EngineRpcService.EnqueueBatchRequestPB.newBuilder().setBatchId(id)
                        .addDpSlots(EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(0)
                                .addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder().setInput(input)))
                        .build());
    }

    @AfterEach
    void stop() throws Exception {
        if (channel != null) channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
        for (Process process : children) process.destroy();
        for (Process process : children) {
            if (!process.waitFor(5, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                assertTrue(process.waitFor(5, TimeUnit.SECONDS));
            }
        }
    }

    @Test
    void samePortDifferentDecodePodsFollowTheMasterSelectedAddress() throws Exception {
        start("127.0.0.3", base + 21, false, false);
        start("127.0.0.4", base + 21, false, false);
        start("127.0.0.2", base + 1, true, false);
        channel = ManagedChannelBuilder.forAddress("127.0.0.2", base + 1).usePlaintext().build();
        for (int i = 0; i < 2; i++) {
            long id = 701 + i;
            enqueue(id, "127.0.0." + (3 + i));
            var frames = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5, TimeUnit.SECONDS)
                    .fetchResponse(EngineRpcService.FetchRequestPB.newBuilder().setRequestId(id).build());
            int count = 0;
            boolean terminal = false;
            while (frames.hasNext()) {
                var frame = frames.next();
                assertFalse(frame.hasErrorInfo(), frame.toString());
                terminal |= frame.getFlattenOutput().getFinishedList().contains(true);
                count++;
            }
            assertEquals(2, count);
            assertTrue(terminal);
        }
        var first = new ObjectMapper().readTree(get("127.0.0.3", base + 20, "/requests")).elements().next();
        var second = new ObjectMapper().readTree(get("127.0.0.4", base + 20, "/requests")).elements().next();
        assertTrue(first.has("701"));
        assertFalse(first.has("702"));
        assertTrue(second.has("702"));
        assertFalse(second.has("701"));
        assertFalse(Files.exists(directory.resolve("discovery.json")));
    }

    private void awaitEndState(String ip, long id, String state) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        String observed = "";
        while (System.nanoTime() < deadline) {
            var lifecycle = new ObjectMapper().readTree(get(ip, base + 20, "/requests")).elements().next();
            observed = lifecycle.path(String.valueOf(id)).path("end_state").asText();
            if (state.equals(observed)) return;
            Thread.sleep(20);
        }
        fail("Expected " + state + ", got " + observed);
    }

    @Test
    void automaticContinuationCompletesWithoutAnyFetch() throws Exception {
        start("127.0.0.3", base + 21, false, true);
        start("127.0.0.2", base + 1, true, true);
        channel = ManagedChannelBuilder.forAddress("127.0.0.2", base + 1).usePlaintext().build();
        enqueue(801, "127.0.0.3");
        awaitEndState("127.0.0.3", 801, "completed");
    }

    @Test
    void prefillProcessDeathReleasesAnUnfetchedDecodeReservation() throws Exception {
        start("127.0.0.3", base + 21, false, false);
        Process prefill = start("127.0.0.2", base + 1, true, false);
        channel = ManagedChannelBuilder.forAddress("127.0.0.2", base + 1).usePlaintext().build();
        enqueue(802, "127.0.0.3");
        prefill.destroyForcibly();
        assertTrue(prefill.waitFor(3, TimeUnit.SECONDS));
        awaitEndState("127.0.0.3", 802, "cancelled");
    }

    @Test
    void decodeProcessDeathReachesTheClientAsFailure() throws Exception {
        Process decode = start("127.0.0.3", base + 21, false, false);
        start("127.0.0.2", base + 1, true, false);
        channel = ManagedChannelBuilder.forAddress("127.0.0.2", base + 1).usePlaintext().build();
        enqueue(803, "127.0.0.3");
        decode.destroyForcibly();
        assertTrue(decode.waitFor(3, TimeUnit.SECONDS));
        var frames = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5, TimeUnit.SECONDS)
                .fetchResponse(EngineRpcService.FetchRequestPB.newBuilder().setRequestId(803).build());
        boolean failure = false;
        while (frames.hasNext()) {
            var frame = frames.next();
            failure |= frame.getErrorInfo().getErrorCodeValue() == 8209;
        }
        assertTrue(failure, "A dead Decode must not produce successful or empty completion");
    }

    @Test
    void unreachableDecodeIsNotMisclassifiedAsKvExhaustion() throws Exception {
        start("127.0.0.2", base + 1, true, false);
        channel = ManagedChannelBuilder.forAddress("127.0.0.2", base + 1).usePlaintext().build();
        var ack = enqueueAck(804, "127.0.0.5");
        assertEquals(0, ack.getSuccessesCount());
        assertEquals(1, ack.getErrorsCount());
        assertEquals(8207, ack.getErrors(0).getErrorInfo().getErrorCode());
    }
}

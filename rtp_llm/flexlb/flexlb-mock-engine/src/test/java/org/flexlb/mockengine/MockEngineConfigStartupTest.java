package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.ManagedChannelBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class MockEngineConfigStartupTest {
    @TempDir Path directory;

    @Test @Timeout(30) void importedConfigurationDrivesActualPdProcessAndReadback() throws Exception {
        int base;
        while (true) {
            base = java.util.concurrent.ThreadLocalRandom.current().nextInt(20000, 28000);
            // Keep the range outside the OS ephemeral client-port range: the
            // HTTP probe must not take a released server port before the child binds.
            try (var p = new ServerSocket(base)) {
                try (var control = new ServerSocket(base - 1); var d = new ServerSocket(base + 1)) { break; }
                catch (java.io.IOException occupied) { /* try another owned range */ }
            }
        }
        Path calibration = directory.resolve("calibration.json"), log = directory.resolve("engine.log");
        Files.writeString(calibration, "{\"mock\":{\"forward\":{\"prefill\":{\"fixed_ms\":1},\"decode\":{\"step_base_ms\":1,\"step_per_running_ms\":0,\"tokens_per_step\":1}}}}");
        var child = new ProcessBuilder(System.getProperty("java.home") + "/bin/java", "-cp",
                System.getProperty("java.class.path"), JavaMockEngineCluster.class.getName(),
                "--mock.engine_config", "../tools/mock_bundle/engine-config.json", "--mock.calibration", calibration.toString(),
                "--mock.endpoint_file", directory.resolve("endpoints.json").toString(),
                "--mock.base_grpc_port", Integer.toString(base), "--mock.n_prefill", "1", "--mock.n_decode", "1",
                "--mock.unique_engine_ips", "false").redirectErrorStream(true).redirectOutput(log.toFile()).start();
        var client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build();
        var url = "http://127.0.0.1:" + (base - 1);
        var channel = ManagedChannelBuilder.forAddress("127.0.0.1", base).usePlaintext().build();
        try {
            String state = null;
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (System.nanoTime() < deadline && child.isAlive()) {
                try {
                    state = client.send(HttpRequest.newBuilder(URI.create(url + "/effective_config"))
                            .timeout(Duration.ofSeconds(1)).GET().build(), HttpResponse.BodyHandlers.ofString()).body();
                    break;
                } catch (java.io.IOException starting) { Thread.sleep(20); }
            }
            assertNotNull(state, () -> {
                try { return Files.readString(log); } catch (Exception error) { return error.toString(); }
            });
            assertEquals(2, new ObjectMapper().readTree(state).path("engines").size());
            var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(123)
                    .addAllTokenIds(Collections.nCopies(64, 7))
                    .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(8).setTimeoutMs(5000)
                            .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder().setRoleStr("DECODE")
                                    .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE).setIp("127.0.0.1").setGrpcPort(base + 1))).build();
            var stub = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(10, TimeUnit.SECONDS);
            var ack = stub.enqueueBatch(EngineRpcService.EnqueueBatchRequestPB.newBuilder().setBatchId(1)
                    .addDpSlots(EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(0)
                            .addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder().setInput(input))).build());
            assertEquals(1, ack.getSuccessesCount(), ack.toString());
            var responses = stub.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder().setRequestId(123).build());
            assertTrue(responses.hasNext());
            while (responses.hasNext()) assertFalse(responses.next().hasErrorInfo());
            var post = HttpRequest.newBuilder(URI.create(url + "/set_perf")).POST(HttpRequest.BodyPublishers.ofString(
                    "{\"engine\":\"prefill-0\",\"mock\":{\"forward\":{\"prefill\":{\"fixed_ms\":2}}}}")).build();
            assertEquals(200, client.send(post, HttpResponse.BodyHandlers.ofString()).statusCode());
            state = client.send(HttpRequest.newBuilder(URI.create(url + "/effective_config")).GET().build(),
                    HttpResponse.BodyHandlers.ofString()).body();
            boolean observed = false;
            for (var engine : new ObjectMapper().readTree(state).path("engines")) {
                if (engine.path("engine").asText().equals("prefill-0")) {
                    assertEquals(1, engine.path("calibration").path("generation").longValue());
                    assertEquals(11042, engine.path("config").path("roles").path("prefill").path("capacity").path("device_blocks").intValue());
                    observed = true;
                }
            }
            assertTrue(observed);
        } finally {
            channel.shutdownNow(); channel.awaitTermination(3, TimeUnit.SECONDS);
            child.destroy();
            if (!child.waitFor(3, TimeUnit.SECONDS)) { child.destroyForcibly(); child.waitFor(3, TimeUnit.SECONDS); }
        }
    }
}

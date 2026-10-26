package org.flexlb.mockengine;

import io.grpc.ManagedChannelBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class MockVitOnlyProcessTest {
    @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory;
    private static int freePort() throws Exception {
        for (;;) {
            int port=java.util.concurrent.ThreadLocalRandom.current().nextInt(20000,27000);
            try (var grpc=new ServerSocket(port); var http=new ServerSocket(port-1)) { return port; }
            catch (java.io.IOException occupied) { }
        }
    }

    @Test void independentVitPublishesHealthAndActualGrpcStatus() throws Exception {
        int grpc = freePort(), http = grpc - 1;
        var calibration = directory.resolve("calibration.json");
        java.nio.file.Files.writeString(calibration, "{\"mock\":{\"vit\":{\"hidden_size\":128,\"max_output_bytes\":16777216},\"forward\":{\"vit\":{\"fixed_ms\":1}}}}");
        var child = new ProcessBuilder(System.getProperty("java.home") + "/bin/java", "-cp",
                System.getProperty("java.class.path"), JavaMockEngineCluster.class.getName(),
                "--mock.engine_config", "../tools/mock_bundle/engine-config.json", "--mock.calibration", calibration.toString(),
                "--mock.n_prefill", "0", "--mock.n_decode", "0", "--mock.n_vit", "1",
                "--mock.endpoint_file", directory.resolve("endpoints.json").toString(),
                "--mock.base_grpc_port", Integer.toString(grpc))
                .redirectErrorStream(true).redirectOutput(directory.resolve("engine.log").toFile()).start();
        var channel = ManagedChannelBuilder.forAddress("127.0.0.1", grpc).usePlaintext().build();
        try {
            var client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build();
            var request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + http + "/health"))
                    .timeout(Duration.ofSeconds(1)).build();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
            boolean ready = false;
            while (child.isAlive() && System.nanoTime() < deadline) {
                try {
                    ready = client.send(request, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
                    if (ready) break;
                } catch (java.io.IOException ignored) { }
                Thread.sleep(50);
            }
            assertTrue(ready, () -> { try { return java.nio.file.Files.readString(directory.resolve("engine.log")); } catch(Exception error) { return error.toString(); } });
            var status = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS)
                    .getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
            assertTrue(status.getAlive());
            assertEquals("VIT", status.getRole());
        } finally {
            java.nio.file.Files.deleteIfExists(calibration);
            channel.shutdownNow();
            child.destroy();
            if (!child.waitFor(5, TimeUnit.SECONDS)) child.destroyForcibly().waitFor();
        }
    }
}

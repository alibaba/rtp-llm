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

class MockVitServerTest {
    private static int freePort() throws Exception {
        try (var socket = new ServerSocket(0)) { return socket.getLocalPort(); }
    }

    @Test void independentVitPublishesHealthAndActualGrpcStatus() throws Exception {
        int grpc = freePort(), http = freePort();
        var child = new ProcessBuilder(System.getProperty("java.home") + "/bin/java", "-cp",
                System.getProperty("java.class.path"), MockVitServer.class.getName(),
                "--port", Integer.toString(grpc), "--http-port", Integer.toString(http))
                .redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.DISCARD).start();
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
            assertTrue(ready, "independent ViT must publish health after gRPC startup");
            var status = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS)
                    .getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
            assertTrue(status.getAlive());
            assertEquals("VIT", status.getRole());
        } finally {
            channel.shutdownNow();
            child.destroy();
            if (!child.waitFor(5, TimeUnit.SECONDS)) child.destroyForcibly().waitFor();
        }
    }
}

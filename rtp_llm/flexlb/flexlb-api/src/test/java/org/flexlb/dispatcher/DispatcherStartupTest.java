package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DispatcherStartupTest {
    @TempDir Path directory;

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    @Timeout(60)
    void applicationMainUsesEnvironmentSettingsWithCommandLineOverrides(boolean override) throws Exception {
        int port;
        try (var socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        Path output = directory.resolve("stdout.log");
        var builder = new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp", System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")),
                "org.flexlb.Application", "--server.port=" + port, "--management.server.port=0",
                "--flexlb.log.path=" + directory);
        if (override) {
            builder.command().add("--dispatch.sub-batch=count:2");
        }
        Path config = directory.resolve("dispatcher.properties");
        Files.writeString(config, "dispatch.sub-batch=size:99\n");
        builder.command().add("--spring.config.additional-location=" + config.toUri());
        builder.environment().put("FLEXLB_CONFIG", """
                {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "grpcServer":{"shutdownQuietPeriodMs":1}}
                """);
        builder.environment().put("MODEL_SERVICE_CONFIG", """
                {"service_id":"aigc.text-generation.generation.startup-test",
                 "role_endpoints":[{"group":"default","pd_fusion_endpoint":
                   {"address":"be","protocol":"http","path":"/"}}],"hosts":{}}
                """);
        builder.environment().put("DISPATCH_ROUTING_TOKEN", "startup-test-secret");
        builder.environment().put("DISPATCH_FE_POOL_SERVICE_ID", "fe");
        builder.environment().put("DISPATCH_SUB_BATCH", "size:1");
        Process process = builder.redirectErrorStream(true).redirectOutput(output.toFile()).start();
        try {
            var request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port
                            + "/dispatcher/_dryrun/batch_infer"))
                    .timeout(Duration.ofSeconds(2)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString("{\"prompt_batch\":[\"a\",\"b\",\"c\"]}")).build();
            var client = HttpClient.newHttpClient();
            HttpResponse<String> response = null;
            long deadline = System.nanoTime() + Duration.ofSeconds(40).toNanos();
            while (response == null && System.nanoTime() < deadline) {
                assertTrue(process.isAlive(), () -> "Application exited; logs: " + directory);
                try {
                    response = client.send(request, HttpResponse.BodyHandlers.ofString());
                } catch (java.io.IOException starting) {
                    Thread.sleep(100);
                }
            }
            assertTrue(response != null, "Application did not become reachable");
            assertEquals(200, response.statusCode(), response.body());
            var preview = new ObjectMapper().readTree(response.body());
            assertEquals(override ? 2 : 3, preview.get("chunk_count").asInt());
            assertEquals(override ? 2 : 1, preview.get("chunks").get(0).get("prompt_batch").size());
            assertFalse(response.body().contains("startup-test-secret"));
        } finally {
            process.destroy();
            if (!process.waitFor(10, TimeUnit.SECONDS)) {
                process.destroyForcibly().waitFor();
            }
        }
        assertFalse(Files.readString(directory.resolve("flexlb.log")).contains("startup-test-secret"));
    }
}

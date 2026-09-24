package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DispatcherStartupTest {
    @TempDir Path directory;
    private final MockWebServer frontend = new MockWebServer();

    @AfterEach
    void closeFrontend() throws Exception {
        frontend.shutdown();
    }

    @ParameterizedTest
    @CsvSource({"true,,false,false,http", "true,,true,false,grpc", "true,,false,true,http", "false,,false,true,http",
            "false,true,false,true,http", "true,false,false,false,http"})
    @Timeout(60)
    void applicationMainUsesEffectiveEnablementAndExistingDiscovery(boolean jsonEnabled, String environmentEnabled,
                                                                 boolean override,
                                                                 boolean separateFe, String protocol) throws Exception {
        boolean enabled = environmentEnabled == null ? jsonEnabled : Boolean.parseBoolean(environmentEnabled);
        int port;
        try (var socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        AtomicInteger delivered = new AtomicInteger();
        frontend.setDispatcher(new okhttp3.mockwebserver.Dispatcher() {
            @Override
            public MockResponse dispatch(RecordedRequest request) {
                if ("/startup_echo".equals(request.getPath())) {
                    delivered.incrementAndGet();
                }
                return new MockResponse().setBody("frontend-ok");
            }
        });
        frontend.start();
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
                 "grpcServer":{"shutdownQuietPeriodMs":1},"httpDispatcher":{"enabled":%s}}
                """.formatted(jsonEnabled));
        if (environmentEnabled == null) {
            builder.environment().remove("DISPATCH_ENABLED");
        } else {
            builder.environment().put("DISPATCH_ENABLED", environmentEnabled);
        }
        builder.environment().put("MODEL_SERVICE_CONFIG", """
                {"service_id":"aigc.text-generation.generation.startup-test",
                 "role_endpoints":[{"group":"default","pd_fusion_endpoint":
                   {"address":"be","protocol":"%s","path":"/"}}],
                 "hosts":{"be":["127.0.0.1:%d"],"fe":["127.0.0.1:%d"]}}
                """.formatted(protocol, frontend.getPort() + (protocol.equals("grpc") ? 1 : 0), frontend.getPort()));
        builder.environment().put("DISPATCH_PRE_ASSIGN_BE", separateFe ? "false" : "true");
        if (separateFe) {
            builder.environment().put("DISPATCH_FE_POOL_SERVICE_ID", "fe");
        } else {
            builder.environment().remove("DISPATCH_FE_POOL_SERVICE_ID");
        }
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
            assertEquals(enabled ? 200 : 404, response.statusCode(), response.body());
            if (enabled) {
                var preview = new ObjectMapper().readTree(response.body());
                assertEquals(override ? 2 : 3, preview.get("chunk_count").asInt());
                assertEquals(override ? 2 : 1, preview.get("chunks").get(0).get("prompt_batch").size());
                var echo = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/dispatcher/startup_echo"))
                        .timeout(Duration.ofSeconds(5)).POST(HttpRequest.BodyPublishers.ofString("hello")).build();
                HttpResponse<String> echoed = client.send(echo, HttpResponse.BodyHandlers.ofString());
                assertEquals(200, echoed.statusCode());
                assertEquals("frontend-ok", echoed.body());
                assertEquals(1, delivered.get());
                if (separateFe) {
                    var allocation = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port
                                    + "/rtp_llm/batch_schedule"))
                            .timeout(Duration.ofSeconds(5)).header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(
                                    "{\"batch_count\":2,\"allocation_type\":\"FE\"}")).build();
                    HttpResponse<String> allocated = client.send(allocation, HttpResponse.BodyHandlers.ofString());
                    assertEquals(200, allocated.statusCode(), allocated.body());
                    var targets = new ObjectMapper().readTree(allocated.body()).get("server_status");
                    assertEquals(2, targets.size());
                    assertEquals("FRONTEND", targets.get(0).get("role").asText());
                    assertEquals(frontend.getPort(), targets.get(0).get("http_port").asInt());
                }
            }
        } finally {
            process.destroy();
            if (!process.waitFor(10, TimeUnit.SECONDS)) {
                process.destroyForcibly().waitFor();
            }
        }
        if (!enabled) {
            assertFalse(Files.readString(directory.resolve("flexlb.log")).contains("dispatcher enabled:"));
            assertEquals(0, delivered.get());
        }
    }
}

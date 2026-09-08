package org.flexlb.constraint;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.httpserver.ConstraintTreeServer;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.transport.HttpNettyConfig;
import org.junit.jupiter.api.Test;
import org.springframework.http.server.reactive.ReactorHttpHandlerAdapter;
import org.springframework.web.reactive.function.server.RouterFunctions;
import reactor.netty.DisposableServer;
import reactor.netty.http.server.HttpServer;

import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.*;

/** Real Java HTTP receiver/publisher -> two C++ HTTP workers. Only discovery/election are stubbed. */
class ConstraintTreeMappedE2ETest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

    @Test
    void bucketInputBuildsCsrAndPublishesToNativeWorkerWhileReadFailureKeepsOldTree() throws Exception {
        String binary = System.getenv("CONSTRAINT_TREE_CPP_WORKER_BINARY");
        assumeTrue(binary != null && Files.isExecutable(Path.of(binary)), "requires compiled C++ test Worker");
        var mapping = ConstraintTreeSidMappingTest.mapping(Map.of("C1", 17, "C2", 19, "C3", 23)).validated();
        Path manifest = Files.createTempFile("csr-bucket-e2e-mapping-", ".json");
        Files.writeString(manifest, JSON.writeValueAsString(mapping));
        int port = freePort();
        Process worker = null;
        WhaleConstraintTreePublisher publisher = null;
        ConstraintTreeBuildService builds = null;
        IgraphConstraintTreePoller poller = null;
        try {
            worker = worker(binary, port, manifest);
            var addresses = mock(WorkerAddressService.class);
            when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of());
            when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of(
                    new WorkerHost("127.0.0.1", port - 5, port - 4, port, "local", "default")));
            var transport = new GeneralHttpNettyService(new HttpNettyConfig().createNettyClientHandler());
            publisher = new WhaleConstraintTreePublisher(addresses, transport, 2, Duration.ofSeconds(5));
            builds = new ConstraintTreeBuildService(new ConstraintTreeBuilder(), Executors.newSingleThreadExecutor(), publisher);
            var failRead = new java.util.concurrent.atomic.AtomicBoolean();
            var sid = new java.util.concurrent.atomic.AtomicReference<>("C1C2");
            org.flexlb.constraint.source.SidBucketClient client = (key, limit, timeout) -> {
                if (failRead.get()) {
                    return java.util.concurrent.CompletableFuture.failedFuture(new IllegalStateException("source unavailable"));
                }
                int bucket = BucketSidReader.bucketForItem("123", 4);
                return java.util.concurrent.CompletableFuture.completedFuture(key.equals("pool_" + bucket)
                        ? List.of(new org.flexlb.constraint.source.SidBucketClient.Row(key, "123", sid.get())) : List.of());
            };
            poller = new IgraphConstraintTreePoller(new BucketSidReader(client, BucketSidReaderTest.settings(4, 2, 100, 0)),
                    builds, () -> true, "gul_item", true, true, 600,
                    java.time.Clock.fixed(java.time.Instant.ofEpochMilli(100), java.time.ZoneOffset.UTC));
            poller.pollOnce();
            assertEquals("SUBMITTED", poller.getStatus().state());
            awaitState(builds, ConstraintTreeModels.BuildState.READY);
            assertEquals(100, get(port, "/constraint_tree_status").path("version").asLong());
            var firstArtifact = builds.getCurrentArtifact().orElseThrow();
            assertEquals(1, ConstraintTreeCsrCodec.decode(firstArtifact.payload()).sidCount());

            failRead.set(true);
            poller.pollOnce();
            assertEquals("FAILED", poller.getStatus().state());
            assertSame(firstArtifact, builds.getCurrentArtifact().orElseThrow());
            assertEquals(100, get(port, "/constraint_tree_status").path("version").asLong());

            failRead.set(false);
            sid.set("C3C2");
            poller.pollOnce();
            awaitState(builds, ConstraintTreeModels.BuildState.READY);
            assertEquals(101, get(port, "/constraint_tree_status").path("version").asLong());
            assertEquals(100, builds.getStatus().backupVersion());
        } finally {
            if (poller != null) { poller.close(); }
            if (builds != null) { builds.destroy(); }
            if (publisher != null) { publisher.destroy(); }
            stop(worker);
            Files.deleteIfExists(manifest);
        }
    }

    @Test
    void fullSidSubmissionMappingRetryAndRestartAcrossTwoWorkers() throws Exception {
        String binary = System.getenv("CONSTRAINT_TREE_CPP_WORKER_BINARY");
        assumeTrue(binary != null && Files.isExecutable(Path.of(binary)), "requires compiled C++ test Worker");
        var mapping = ConstraintTreeSidMappingTest.mapping(Map.of("C1", 17, "C2", 19, "C3", 23)).validated();
        var otherMapping = ConstraintTreeSidMappingTest.mapping(Map.of("C1", 29, "C2", 19, "C3", 23)).validated();
        Path manifest = Files.createTempFile("csr-e2e-mapping-", ".json");
        Path otherManifest = Files.createTempFile("csr-e2e-other-mapping-", ".json");
        Files.writeString(manifest, JSON.writeValueAsString(mapping));
        Files.writeString(otherManifest, JSON.writeValueAsString(otherMapping));
        int firstPort = freePort(), secondPort = freePort();
        Process first = null, second = null;
        WhaleConstraintTreePublisher publisher = null;
        ConstraintTreeBuildService builds = null;
        DisposableServer server = null;
        try {
            first = worker(binary, firstPort, manifest);
            second = worker(binary, secondPort, manifest);
            var addresses = mock(WorkerAddressService.class);
            when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of());
            when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of(
                    new WorkerHost("127.0.0.1", firstPort - 5, firstPort - 4, firstPort, "local", "default"),
                    new WorkerHost("127.0.0.1", secondPort - 5, secondPort - 4, secondPort, "local", "default")));
            var transport = new GeneralHttpNettyService(new HttpNettyConfig().createNettyClientHandler());
            publisher = new WhaleConstraintTreePublisher(addresses, transport, 2, Duration.ofSeconds(5));
            builds = new ConstraintTreeBuildService(new ConstraintTreeBuilder(), Executors.newSingleThreadExecutor(), publisher);
            var leader = mock(LBStatusConsistencyService.class);
            when(leader.isMaster()).thenReturn(true);
            var handler = RouterFunctions.toHttpHandler(new ConstraintTreeServer(builds, leader, transport).constraintTreeRoutes());
            server = HttpServer.create().host("127.0.0.1").port(0).handle(new ReactorHttpHandlerAdapter(handler)).bindNow();
            int masterPort = server.port();
            String body = "{\"version\":100,\"model\":\"gul_item\",\"sids\":[\"C1C2\",\"C3C2\",\"C1C3\"]}";
            assertEquals("ACCEPTED", post(masterPort, "/rtp_llm/constraint_tree/build", body, 200).path("state").asText());
            awaitState(builds, ConstraintTreeModels.BuildState.READY);
            assertEquals(2, builds.getStatus().publishedWorkerCount());
            assertEquals(mapping.fingerprint(), get(firstPort, "/constraint_tree_status").path("mapping_fingerprint").asText());
            assertEquals(100, get(secondPort, "/constraint_tree_status").path("version").asLong());
            var firstArtifact = builds.getCurrentArtifact().orElseThrow();
            var decoded = ConstraintTreeCsrCodec.decode(firstArtifact.payload());
            assertEquals(3, decoded.sidCount());
            assertEquals(firstArtifact.contentSha256(), get(firstPort, "/constraint_tree_status").path("content_sha256").asText());

            String reordered = body.replace("[\"C1C2\",\"C3C2\",\"C1C3\"]", "[\"C3C2\",\"C1C3\",\"C1C2\",\"C3C2\"]");
            assertEquals("ALREADY_ACCEPTED", post(masterPort, "/rtp_llm/constraint_tree/build", reordered, 200).path("state").asText());
            assertEquals("version already exists with different content", post(masterPort, "/rtp_llm/constraint_tree/build", body.replace("C1C2", "C2C2"), 409).path("error").asText());
            assertSame(firstArtifact, builds.getCurrentArtifact().orElseThrow());

            stop(second);
            second = worker(binary, secondPort, manifest);
            assertEquals(0, get(secondPort, "/constraint_tree_status").path("version").asLong());
            post(masterPort, "/rtp_llm/constraint_tree/retry", "{\"version\":100,\"model\":\"gul_item\"}", 200);
            // Reconciliation represents the periodic production loop; wait for actual Worker activation, not just delivery.
            awaitVersion(builds, secondPort, 100);
            assertSame(firstArtifact, builds.getCurrentArtifact().orElseThrow());

            stop(second);
            second = worker(binary, secondPort, otherManifest);
            post(masterPort, "/rtp_llm/constraint_tree/build", body.replace("100", "101"), 200);
            awaitState(builds, ConstraintTreeModels.BuildState.FAILED);
            assertTrue(builds.getStatus().message().contains("disagree"));
            assertEquals(100, builds.getStatus().activeVersion());
            assertSame(firstArtifact, builds.getCurrentArtifact().orElseThrow());
            assertEquals(100, get(firstPort, "/constraint_tree_status").path("version").asLong());

            stop(second);
            second = worker(binary, secondPort, manifest);
            post(masterPort, "/rtp_llm/constraint_tree/retry", "{\"version\":101,\"model\":\"gul_item\"}", 200);
            awaitState(builds, ConstraintTreeModels.BuildState.READY);
            assertEquals(101, builds.getStatus().activeVersion());
            assertEquals(100, builds.getStatus().backupVersion());
            assertEquals(101, get(secondPort, "/constraint_tree_status").path("version").asLong());
            assertEquals(101, get(masterPort, "/rtp_llm/constraint_tree/status").path("active_version").asLong());
        } finally {
            if (server != null) { server.disposeNow(); }
            if (builds != null) { builds.destroy(); }
            if (publisher != null) { publisher.destroy(); }
            stop(first);
            stop(second);
            Files.deleteIfExists(manifest);
            Files.deleteIfExists(otherManifest);
        }
    }

    private JsonNode post(int port, String path, String body, int expected) throws Exception {
        var response = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + path))
                .timeout(Duration.ofSeconds(10)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
        assertEquals(expected, response.statusCode(), response.body());
        return JSON.readTree(response.body());
    }

    private JsonNode get(int port, String path) throws Exception {
        var response = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + path))
                .timeout(Duration.ofSeconds(5)).GET().build(), HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), response.body());
        return JSON.readTree(response.body());
    }

    private Process worker(String binary, int port, Path manifest) throws Exception {
        Process process = new ProcessBuilder(binary, Integer.toString(port), manifest.toString())
                .redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.DISCARD).start();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (System.nanoTime() < deadline) {
            try { get(port, "/constraint_tree_mapping_status"); return process; }
            catch (java.io.IOException e) { Thread.sleep(20); }
            if (!process.isAlive()) { fail("C++ Worker exited: " + process.exitValue()); }
        }
        stop(process);
        return fail("C++ Worker startup timeout");
    }

    private void awaitState(ConstraintTreeBuildService builds, ConstraintTreeModels.BuildState state) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (System.nanoTime() < deadline) {
            if (builds.getStatus().state() == state) { return; }
            builds.reconcileCurrent();
            Thread.sleep(20);
        }
        fail("Master did not reach " + state + ": " + builds.getStatus());
    }

    private void awaitVersion(ConstraintTreeBuildService builds, int port, long version) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (System.nanoTime() < deadline) {
            builds.reconcileCurrent();
            if (get(port, "/constraint_tree_status").path("version").asLong() == version
                    && builds.getStatus().state() == ConstraintTreeModels.BuildState.READY) { return; }
            Thread.sleep(20);
        }
        fail("Worker version not restored");
    }

    private static int freePort() throws Exception {
        try (ServerSocket socket = new ServerSocket(0)) { return socket.getLocalPort(); }
    }

    private static void stop(Process process) throws Exception {
        if (process == null || !process.isAlive()) { return; }
        process.getOutputStream().write('\n');
        process.getOutputStream().flush();
        if (!process.waitFor(5, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            assertTrue(process.waitFor(5, TimeUnit.SECONDS));
        }
    }
}

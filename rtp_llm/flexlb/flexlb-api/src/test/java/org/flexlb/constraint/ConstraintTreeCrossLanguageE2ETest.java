package org.flexlb.constraint;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.constraint.ConstraintTreeModels.Artifact;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.flexlb.constraint.ConstraintTreeModels.BuildState;
import org.flexlb.constraint.ConstraintTreeModels.PublicationResult;
import org.flexlb.constraint.ConstraintTreeModels.SerializedArtifact;
import org.flexlb.constraint.ConstraintTreeModels.WorkerPublication;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Opt-in process-level E2E. The test starts the actual C++ Worker HTTP server,
 * drives it from the Java Master build/reconciliation service, restarts the
 * Worker, and verifies that reconciliation restores the active version.
 */
class ConstraintTreeCrossLanguageE2ETest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    private URI workerUri;

    @Test
    void javaMasterPushesCppWorkerAndRepushesAfterWorkerRestart() throws Exception {
        String binaryValue = System.getenv("CONSTRAINT_TREE_CPP_WORKER_BINARY");
        assumeTrue(binaryValue != null && !binaryValue.isBlank(),
                "set CONSTRAINT_TREE_CPP_WORKER_BINARY to run the Java-to-C++ E2E test");
        Path binary = Path.of(binaryValue).toAbsolutePath();
        assumeTrue(Files.isExecutable(binary), "C++ Worker test binary is not executable: " + binary);

        int port = freePort();
        workerUri = URI.create("http://127.0.0.1:" + port);
        Process worker = startWorker(binary, port);
        ConstraintTreeBuildService master = new ConstraintTreeBuildService(
                new ConstraintTreeBuilder(),
                Executors.newSingleThreadExecutor(),
                this::publishToWorker);
        try {
            long version = Math.max(1, System.currentTimeMillis());
            BuildRequest firstRequest = request(version,
                    "169967_216546", "169967_215835", "42_43_44");

            master.submit(firstRequest);
            JsonNode firstReady = awaitWorkerVersion(version);
            master.reconcileCurrent();
            awaitMasterState(master, BuildState.READY);
            SerializedArtifact first = master.getCurrentArtifact().orElseThrow();
            assertEquals(first.metadata().prefixCount(), firstReady.path("prefix_count").asLong());
            assertEquals(first.metadata().edgeCount(), firstReady.path("edge_count").asLong());
            assertTrue(master.getBackupArtifact().isEmpty());

            try (ConstraintTreeBuilder builder = new ConstraintTreeBuilder()) {
                Artifact stale = builder.build(request(version - 1, "9"));
                JsonNode staleResponse = postArtifact(stale, 409);
                assertEquals("stale_version", staleResponse.path("status").asText());
                assertEquals(version, staleResponse.path("version").asLong());
            }

            stopWorker(worker);
            worker = startWorker(binary, port);
            assertEquals(0, readStatus().path("version").asLong());

            // The Master process and its current artifact survive the Worker restart.
            // The first reconciliation delivers asynchronously; the second observes
            // actual activation and marks the deployment ready.
            master.reconcileCurrent();
            awaitWorkerVersion(version);
            master.reconcileCurrent();
            awaitMasterState(master, BuildState.READY);
            assertEquals(version, master.getCurrentArtifact().orElseThrow().version());

            master.submit(request(version + 1, "7", "7_8", "100_200_300_400"));
            awaitWorkerVersion(version + 1);
            master.reconcileCurrent();
            awaitMasterState(master, BuildState.READY);
            assertEquals(version + 1, master.getCurrentArtifact().orElseThrow().version());
            assertEquals(version, master.getBackupArtifact().orElseThrow().version());
        } finally {
            master.destroy();
            stopWorker(worker);
        }
    }

    private PublicationResult publishToWorker(SerializedArtifact artifact) {
        try {
            JsonNode status = readStatus();
            if (status.path("version").asLong() == artifact.version()
                    && status.path("status").asText().equals("ready")) {
                return publication(artifact.version(), true, "already current");
            }
            HttpRequest request = HttpRequest.newBuilder(workerUri.resolve("/update_constraint_tree"))
                    .timeout(Duration.ofSeconds(10))
                    .header("Content-Type", "application/octet-stream")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(artifact.payload()))
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            JsonNode body = MAPPER.readTree(response.body());
            boolean active = response.statusCode() == 200
                    && body.path("version").asLong() == artifact.version();
            return publication(body.path("version").asLong(), active,
                    active ? "active" : "delivered; activation pending");
        } catch (Exception e) {
            return publication(0, false, e.getMessage());
        }
    }

    private PublicationResult publication(long version, boolean success, String message) {
        return new PublicationResult(1, success ? 1 : 0,
                List.of(new WorkerPublication(workerUri.toString(), success, version, message)));
    }

    private JsonNode postArtifact(Artifact artifact, int expectedStatus) throws Exception {
        HttpRequest request = HttpRequest.newBuilder(workerUri.resolve("/update_constraint_tree"))
                .timeout(Duration.ofSeconds(10))
                .header("Content-Type", "application/octet-stream")
                .POST(HttpRequest.BodyPublishers.ofByteArray(ConstraintTreeCsrCodec.encode(artifact)))
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        assertEquals(expectedStatus, response.statusCode(), response.body());
        return MAPPER.readTree(response.body());
    }

    private JsonNode readStatus() throws Exception {
        HttpRequest request = HttpRequest.newBuilder(workerUri.resolve("/constraint_tree_status"))
                .timeout(Duration.ofSeconds(5))
                .GET()
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), response.body());
        return MAPPER.readTree(response.body());
    }

    private JsonNode awaitWorkerVersion(long expectedVersion) throws Exception {
        long deadlineNanos = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        JsonNode lastStatus = null;
        while (System.nanoTime() < deadlineNanos) {
            try {
                lastStatus = readStatus();
                if (lastStatus.path("version").asLong() == expectedVersion
                        && lastStatus.path("status").asText().equals("ready")) {
                    return lastStatus;
                }
            } catch (Exception ignored) {
                // The process may still be binding its HTTP port.
            }
            Thread.sleep(20);
        }
        return fail("C++ Worker did not activate version " + expectedVersion + "; last status=" + lastStatus);
    }

    private void awaitMasterState(ConstraintTreeBuildService master, BuildState expected) throws Exception {
        long deadlineNanos = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (System.nanoTime() < deadlineNanos) {
            if (master.getStatus().state() == expected) {
                return;
            }
            Thread.sleep(20);
        }
        fail("Java Master did not reach " + expected + "; status=" + master.getStatus());
    }

    private Process startWorker(Path binary, int port) throws Exception {
        Process process = new ProcessBuilder(binary.toString(), Integer.toString(port))
                .redirectErrorStream(true)
                // Surefire uses the forked JVM's stdout as a framed control
                // channel. A native child inheriting that descriptor corrupts
                // the protocol, so keep Worker output away from that channel.
                .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                .start();
        long deadlineNanos = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (System.nanoTime() < deadlineNanos) {
            if (!process.isAlive()) {
                fail("C++ Worker exited during startup with code " + process.exitValue());
            }
            try {
                readStatus();
                return process;
            } catch (Exception ignored) {
                Thread.sleep(20);
            }
        }
        stopWorker(process);
        return fail("timed out waiting for C++ Worker on " + workerUri);
    }

    private void stopWorker(Process process) throws Exception {
        if (process == null || !process.isAlive()) {
            return;
        }
        try {
            process.getOutputStream().write('\n');
            process.getOutputStream().flush();
            process.getOutputStream().close();
        } catch (IOException ignored) {
            process.destroy();
        }
        if (!process.waitFor(5, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            assertTrue(process.waitFor(5, TimeUnit.SECONDS), "C++ Worker did not stop");
        }
    }

    private int freePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private BuildRequest request(long version, String... sids) {
        return new BuildRequest(version, "gul_item", 1699, 151645, "_", null, List.of(sids));
    }
}

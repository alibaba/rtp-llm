package org.flexlb;

import org.flexlb.mock.master.MasterReads;

import com.fasterxml.jackson.databind.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Collectors;
import org.flexlb.Application;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.service.grace.ApplicationLifecycle;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.mock.api.master.MasterStateApi;
import org.flexlb.mock.runtime.ApiServer;
import org.springframework.boot.SpringApplication;
import org.springframework.core.SpringProperties;
import org.springframework.core.env.AbstractEnvironment;

/** Runs the production Spring Master and its shared read adapter in the same JVM. */
public final class MasterNode {
    public static void main(String[] args) throws Exception {
        var json = new ObjectMapper();
        var config = json.readTree(Path.of(required("MOCK_CONFIG_FILE")).toFile());
        if (!config.path("component_role").asText().equals("master") || config.path("environment").path("master").path("deployment").path("replicas").asInt() != 1)
            throw new IllegalArgumentException("Master adapter currently supports one local Master");
        var directory = Path.of(required("MOCK_OUTPUT_DIR"));
        var topology = json.readTree(Path.of(required("MOCK_TOPOLOGY_FILE")).toFile());
        var identities = new HashMap<String, String>();
        for (var node : topology) if (node.has("rpc")) {
            if (identities.putIfAbsent(node.path("rpc").asText(), node.path("id").asText()) != null)
                throw new IllegalArgumentException("Duplicate Engine RPC address");
        }
        if (identities.isEmpty()) throw new IllegalArgumentException("Master requires running Engines");
        int port = port();
        SpringProperties.setFlag(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME);
        org.flexlb.OpenTelemetryBootstrap.configureFromEnvironment();
        var application = new SpringApplication(Application.class);
        application.setRegisterShutdownHook(false);
        var context = application.run("--server.address=127.0.0.1", "--server.port=" + port, "--management.server.port=" + port, "--flexlb.log.path=" + directory.resolve("logs"));
        try {
            var lifecycle = context.getBean(ApplicationLifecycle.class);
            var consistency = context.getBean(LBStatusConsistencyService.class);
            var workers = context.getBean(WorkerDirectory.class);
            lifecycle.online();
            String id = required("MOCK_NODE_ID");
            var reads = new MasterReads(id, consistency::isMaster, role -> workers.statusSnapshot(
                org.flexlb.dao.route.RoleType.valueOf(role.name().toUpperCase(Locale.ROOT))).values().stream()
                .map(s -> s.getIp() + ":" + s.getGrpcPort()).collect(Collectors.toSet()), identities);
            try (var api = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).bind(MasterStateApi.class, reads).start()) {
                String nativeUrl = "http://127.0.0.1:" + port;
                json.writerWithDefaultPrettyPrinter().writeValue(directory.resolve("endpoints.json").toFile(), Map.of(
                    "id", id, "role", "master", "control", api.url(), "http", nativeUrl, "rpc", "127.0.0.1:" + (port + 2),
                    "metrics", nativeUrl + "/prometheus"));
                json.writerWithDefaultPrettyPrinter().writeValue(directory.resolve("effective-config.json").toFile(), Map.of(
                    "FLEXLB_CONFIG", required("FLEXLB_CONFIG"), "MODEL_SERVICE_CONFIG", required("MODEL_SERVICE_CONFIG"),
                    "server_port", port));
                Runtime.getRuntime().addShutdownHook(new Thread(() -> { api.close(); context.close(); }, "mock-master-close"));
                api.publishReady(Path.of(required("MOCK_READY_FILE")));
                new CountDownLatch(1).await();
            }
        } finally { context.close(); }
    }
    private static String required(String key) { return Objects.requireNonNull(System.getenv(key), "Missing " + key); }
    private static int port() throws java.io.IOException {
        for (int attempt = 0; attempt < 32; attempt++) {
            try (var http = new ServerSocket(0, 1, InetAddress.getLoopbackAddress())) {
                int port = http.getLocalPort();
                if (port > 65533) continue;
                try (var grpc = new ServerSocket(port + 2, 1, InetAddress.getLoopbackAddress())) { return port; }
                catch (java.io.IOException occupied) { /* Try another pair; startup still detects a later bind race. */ }
            }
        }
        throw new java.io.IOException("No available Master HTTP/gRPC port pair");
    }
}

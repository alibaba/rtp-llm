package org.flexlb;

import java.util.function.*;
import org.flexlb.mock.api.model.Types.*;

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
import org.flexlb.mock.protocol.ControlServer;
import org.springframework.boot.SpringApplication;
import org.springframework.core.SpringProperties;
import org.springframework.core.env.AbstractEnvironment;

/** Runs the production Spring Master and its shared read adapter in the same JVM. */
public final class MasterNode {
    public static void main(String[] args) {
        try { run(); }
        catch (Throwable error) {
            error.printStackTrace(System.err);
            System.exit(1); // Native libraries may leave non-daemon threads after failed initialization.
        }
    }
    private static void run() throws Exception {
        var json = new ObjectMapper();
        var config = json.readTree(Path.of(required("MOCK_CONFIG_FILE")).toFile());
        if (!config.path("component_role").asText().equals("master"))
            throw new IllegalArgumentException("Expected a Master component");
        var directory = Path.of(required("MOCK_OUTPUT_DIR"));
        int port = Integer.parseInt(required("MOCK_MASTER_PORT"));
        String host = required("FLEXLB_ADVERTISED_IP");
        var election = json.readTree(required("FLEXLB_SYNC_CONSISTENCY_CONFIG"));
        if (!election.path("needConsistency").asBoolean())
            throw new IllegalArgumentException("Master requires ZooKeeper election");
        SpringProperties.setFlag(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME);
        org.flexlb.OpenTelemetryBootstrap.configureFromEnvironment();
        var application = new SpringApplication(Application.class);
        application.setRegisterShutdownHook(false);
        var context = application.run("--server.address=" + host, "--flexlb.grpc.bind-address=" + host,
                "--server.port=" + port, "--management.server.port=" + port,
                "--flexlb.log.path=" + directory.resolve("logs"));
        try {
            var lifecycle = context.getBean(ApplicationLifecycle.class);
            var consistency = context.getBean(LBStatusConsistencyService.class);
            var workers = context.getBean(WorkerDirectory.class);
            lifecycle.online();
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(30);
            while (consistency.getMasterHostIpPort() == null) {
                if (System.nanoTime() >= deadline) throw new IllegalStateException("ZooKeeper leader election timed out");
                Thread.sleep(20);
            }
            String id = required("MOCK_NODE_ID");
            var reads = new MasterReads(id, consistency::isMaster, role -> workers.statusSnapshot(
                org.flexlb.dao.route.RoleType.valueOf(role.name().toUpperCase(Locale.ROOT))).values().stream()
                .map(s -> s.getIp() + ":" + s.getGrpcPort()).collect(Collectors.toSet()));
            try (var api = new ControlServer(new InetSocketAddress(host, 0)).route("master/leadership/get", None.class, MasterSet.class, reads::leadership)
                    .route("master/engine/get", EngineSelection.class, PerMasterEngines.class, reads::engines).start()) {
                String nativeUrl = "http://" + host + ":" + port;
                json.writerWithDefaultPrettyPrinter().writeValue(directory.resolve("endpoints.json").toFile(), Map.of(
                    "id", id, "role", "master", "control", api.url(), "http", nativeUrl, "rpc", host + ":" + (port + 2),
                    "metrics", nativeUrl + "/prometheus"));
                json.writerWithDefaultPrettyPrinter().writeValue(directory.resolve("effective-config.json").toFile(), Map.of(
                    "FLEXLB_CONFIG", required("FLEXLB_CONFIG"), "MODEL_SERVICE_CONFIG", required("MODEL_SERVICE_CONFIG"),
                    "server_port", port, "advertised_ip", host, "FLEXLB_SYNC_CONSISTENCY_CONFIG", election, "HIPPO_ROLE", required("HIPPO_ROLE")));
                Runtime.getRuntime().addShutdownHook(new Thread(() -> { api.close(); context.close(); }, "mock-master-close"));
                api.publishReady(Path.of(required("MOCK_READY_FILE")));
                new CountDownLatch(1).await();
            }
        } finally { context.close(); }
    }
    private static String required(String key) { return Objects.requireNonNull(System.getenv(key), "Missing " + key); }

    public static final class MasterReads {
        private final String id;
        private final BooleanSupplier leader;
        private final Function<Role, Set<String>> observed;
        public MasterReads(String id, BooleanSupplier leader, Function<Role, Set<String>> observed) {
            this.id = Objects.requireNonNull(id); this.leader = Objects.requireNonNull(leader);
            this.observed = Objects.requireNonNull(observed);
        }
        public MasterSet leadership(None args) {
            return new MasterSet(leader.getAsBoolean() ? List.of(new MasterRef(id)) : List.of());
        }
        public PerMasterEngines engines(EngineSelection args) {
            return new PerMasterEngines(Map.of(id, observed.apply(args.role()).stream().sorted().toList()));
        }
    }
}

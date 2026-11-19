package org.flexlb.mock.frontend;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies Master selection as topology and heartbeat responses change.
 */
class MasterRoutesTest {
    @TempDir Path directory;

    final ObjectMapper json = new ObjectMapper();

    @Test void observesLeadershipThenDiscoversReplacementWithoutChangingThePlaybackPlan() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new BackendRpcTransportTest.Harness(); var second = new BackendRpcTransportTest.Harness(); var added = new BackendRpcTransportTest.Harness()) {
            first.leader = second.leader;
            publish(topology, List.of(first.endpoint(), second.endpoint()));
            try (var discovery = new MasterRoutes(topology, null)) {
                assertEquals(List.of(second.endpoint().get("rpc"), first.endpoint().get("rpc")), addresses(discovery));
                first.leader = first.http().substring(7); second.leader = first.leader;
                PlaybackTest.eventually(() -> addresses(discovery).getFirst().equals(first.endpoint().get("rpc")));
                first.heartbeatStatus = 503; second.leader = second.http().substring(7);
                PlaybackTest.eventually(() -> addresses(discovery).equals(List.of(second.endpoint().get("rpc"))));
                publish(topology, List.of(second.endpoint(), added.endpoint()));
                second.leader = added.leader;
                PlaybackTest.eventually(() -> addresses(discovery).equals(List.of(added.endpoint().get("rpc"), second.endpoint().get("rpc"))));
                second.heartbeatStatus = 503; added.heartbeatStatus = 503;
                PlaybackTest.eventually(() -> addresses(discovery).isEmpty());
            }
        }
    }

    void publish(Path path, List<?> nodes) throws Exception {
        Path temp = directory.resolve("next.json");
        json.writeValue(temp.toFile(), nodes);
        Files.move(temp, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    }

    @Test void notReadyMasterIsExcludedOnTheFirstProbe() throws Exception {
        assertRoutesAfterFirstFailure(200, false);
    }

    @Test void ordinaryProbeFailureKeepsMasterUntilTheSecondFailure() throws Exception {
        assertRoutesAfterFirstFailure(503, true);
    }

    @Test void heartbeatDefaultsToSendingGenerateInputWhenCapabilityIsMissing() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var master = new BackendRpcTransportTest.Harness()) {
            publish(topology, List.of(master.endpoint()));
            master.heartbeat.removeContext("/rtp_llm/master/info");
            master.heartbeat.createContext("/rtp_llm/master/info", exchange -> {
                try (exchange) {
                    byte[] bytes = "{\"ready\":true}".getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(200, bytes.length);
                    exchange.getResponseBody().write(bytes);
                }
            });
            try (var discovery = new MasterRoutes(topology, null)) {
                assertEquals(1, discovery.routes().size());
                assertTrue(discovery.routes().getFirst().requiresGenerateInput());
            }
        }
    }

    @Test void heartbeatRejectsMissingReadinessAndNonBooleanFields() throws Exception {
        Path topology = directory.resolve("topology.json");
        for (String response : List.of("{\"requires_generate_input\":false}",
                "{\"ready\":\"true\",\"requires_generate_input\":false}",
                "{\"ready\":true,\"requires_generate_input\":null}")) {
            try (var master = new BackendRpcTransportTest.Harness()) {
                publish(topology, List.of(master.endpoint()));
                master.heartbeat.removeContext("/rtp_llm/master/info");
                master.heartbeat.createContext("/rtp_llm/master/info", exchange -> {
                    try (exchange) {
                        byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
                        exchange.sendResponseHeaders(200, bytes.length);
                        exchange.getResponseBody().write(bytes);
                    }
                });
                try (var discovery = new MasterRoutes(topology, null)) {
                    assertTrue(discovery.routes().isEmpty(), response);
                }
            }
        }
    }

    @Test void explicitSelectionStaysFixedAndMalformedDiscoveryFailsPlayback() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new BackendRpcTransportTest.Harness(); var second = new BackendRpcTransportTest.Harness()) {
            publish(topology, List.of(first.endpoint(), second.endpoint()));
            try (var discovery = new MasterRoutes(topology, List.of(first.endpoint().get("rpc")))) {
                assertEquals(List.of(first.endpoint().get("rpc")), addresses(discovery));
                first.heartbeatStatus = 503;
                PlaybackTest.eventually(() -> addresses(discovery).isEmpty());
                publish(topology, List.of(Map.of("role", "master", "http", "invalid", "rpc", "x:1")));
                PlaybackTest.eventually(() -> {
                    try { discovery.check(); return false; }
                    catch (IllegalStateException expected) { return true; }
                });
                assertThrows(IllegalStateException.class, discovery::routes);
            }
        }
    }

    @Test void deduplicatesRpcAddressesWithoutDerivingThemFromHttpPorts() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var master = new BackendRpcTransportTest.Harness()) {
            String rpc = "127.0.0.1:1";
            var endpoint = Map.of("role", "master", "http", master.http(), "rpc", rpc);
            publish(topology, List.of(endpoint, endpoint));
            try (var discovery = new MasterRoutes(topology, null)) {
                assertEquals(List.of(rpc), addresses(discovery));
                assertTrue(discovery.routes().getFirst().requiresGenerateInput());
            }
        }
    }

    @Test void sendsInputOnlyToTheMasterThatRequiresIt() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var primary = new BackendRpcTransportTest.Harness(); var backup = new BackendRpcTransportTest.Harness()) {
            primary.requiresGenerateInput = false;
            backup.requiresGenerateInput = true;
            backup.leader = primary.leader;
            publish(topology, List.of(primary.endpoint(), backup.endpoint()));
            try (var discovery = new MasterRoutes(topology, null);
                 var frontend = new BackendRpcTransport(event -> {}, 1000, 0L, 1, discovery::routes)) {
                var input = BackendRpcTransportTest.input(1);
                var request = BackendRpcTransportTest.request(1).toBuilder().clearGenerateInput().setCacheKeyBlockSize(1).build();
                var timeout = java.time.Duration.ofSeconds(3);
                assertEquals(BackendRpcTransport.Outcome.success, primary.result(frontend.send(request, input, timeout, List.of())).outcome());
                var nonBatch = primary.scheduleInputs.remove();
                assertTrue(nonBatch.getGenerateInput().isEmpty());
                assertEquals(3, nonBatch.getBlockCacheKeysCount());
                assertEquals(input.getTokenIdsList(), primary.generated.get().getTokenIdsList());

                primary.scheduleFailure = io.grpc.Status.UNAVAILABLE;
                backup.batched = true;
                assertEquals(BackendRpcTransport.Outcome.success, backup.result(frontend.send(request, input, timeout, List.of())).outcome());
                assertTrue(primary.scheduleInputs.remove().getGenerateInput().isEmpty());
                var batch = backup.scheduleInputs.remove();
                assertEquals(nonBatch.getBlockCacheKeysList(), batch.getBlockCacheKeysList());
                assertEquals(input.toByteString(), batch.getGenerateInput());
                assertEquals(1, backup.fetches.get());

            }
        }
    }

    private void assertRoutesAfterFirstFailure(int status, boolean retained) throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new BackendRpcTransportTest.Harness(); var second = new BackendRpcTransportTest.Harness()) {
            first.leader = first.http().substring(7);
            second.leader = first.leader;
            publish(topology, List.of(first.endpoint(), second.endpoint()));
            String primary = first.endpoint().get("rpc");
            String backup = second.endpoint().get("rpc");
            var probes = new AtomicInteger();
            var active = new AtomicReference<MasterRoutes>();
            var routesAfterFirstFailure = new CompletableFuture<List<String>>();
            first.heartbeat.removeContext("/rtp_llm/master/info");
            first.heartbeat.createContext("/rtp_llm/master/info", exchange -> {
                try (exchange) {
                    int probe = probes.incrementAndGet();
                    // The previous refresh has published its snapshot before the next probe starts.
                    if (probe == 3) routesAfterFirstFailure.complete(addresses(active.get()));
                    var body = json.createObjectNode().put("real_master_host", first.leader)
                        .put("requires_generate_input", true).put("ready", probe != 2 || status != 200);
                    byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(probe == 1 ? 200 : probe == 2 ? status : 503, bytes.length);
                    exchange.getResponseBody().write(bytes);
                }
            });
            try (var discovery = new MasterRoutes(topology, null)) {
                active.set(discovery);
                assertEquals(List.of(primary, backup), addresses(discovery));
                assertEquals(retained ? List.of(primary, backup) : List.of(backup),
                    routesAfterFirstFailure.get(5, TimeUnit.SECONDS));
                PlaybackTest.eventually(() -> addresses(discovery).equals(List.of(backup)));
            }
        }
    }

    private static List<String> addresses(MasterRoutes discovery) {
        return discovery.routes().stream().map(MasterRoutes.Target::rpc).toList();
    }
}

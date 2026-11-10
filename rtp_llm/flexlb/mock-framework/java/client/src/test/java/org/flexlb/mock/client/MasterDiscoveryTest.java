package org.flexlb.mock.client;

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
class MasterDiscoveryTest {
    @TempDir Path directory;

    final ObjectMapper json = new ObjectMapper();

    @Test void observesLeadershipThenDiscoversReplacementWithoutChangingThePlaybackPlan() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new BackendRpcClientTest.Harness(); var second = new BackendRpcClientTest.Harness(); var added = new BackendRpcClientTest.Harness()) {
            first.leader = second.leader;
            publish(topology, List.of(first.endpoint(), second.endpoint()));
            try (var discovery = new MasterDiscovery(topology, null)) {
                assertEquals(List.of(second.endpoint().get("rpc"), first.endpoint().get("rpc")), discovery.routes());
                first.leader = first.http().substring(7); second.leader = first.leader;
                PlaybackTest.eventually(() -> discovery.routes().getFirst().equals(first.endpoint().get("rpc")));
                first.heartbeatStatus = 503; second.leader = second.http().substring(7);
                PlaybackTest.eventually(() -> discovery.routes().equals(List.of(second.endpoint().get("rpc"))));
                publish(topology, List.of(second.endpoint(), added.endpoint()));
                second.leader = added.leader;
                PlaybackTest.eventually(() -> discovery.routes().equals(List.of(added.endpoint().get("rpc"), second.endpoint().get("rpc"))));
                second.heartbeatStatus = 503; added.heartbeatStatus = 503;
                PlaybackTest.eventually(() -> discovery.routes().isEmpty());
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

    @Test void explicitSelectionStaysFixedAndMalformedDiscoveryFailsPlayback() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new BackendRpcClientTest.Harness(); var second = new BackendRpcClientTest.Harness()) {
            publish(topology, List.of(first.endpoint(), second.endpoint()));
            try (var discovery = new MasterDiscovery(topology, List.of(first.endpoint().get("rpc")))) {
                assertEquals(List.of(first.endpoint().get("rpc")), discovery.routes());
                first.heartbeatStatus = 503;
                PlaybackTest.eventually(() -> discovery.routes().isEmpty());
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
        try (var master = new BackendRpcClientTest.Harness()) {
            String rpc = "127.0.0.1:1";
            var endpoint = Map.of("role", "master", "http", master.http(), "rpc", rpc);
            publish(topology, List.of(endpoint, endpoint));
            try (var discovery = new MasterDiscovery(topology, null)) {
                assertEquals(List.of(rpc), discovery.routes());
            }
        }
    }

    private void assertRoutesAfterFirstFailure(int status, boolean retained) throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new BackendRpcClientTest.Harness(); var second = new BackendRpcClientTest.Harness()) {
            first.leader = first.http().substring(7);
            second.leader = first.leader;
            publish(topology, List.of(first.endpoint(), second.endpoint()));
            String primary = first.endpoint().get("rpc");
            String backup = second.endpoint().get("rpc");
            var probes = new AtomicInteger();
            var active = new AtomicReference<MasterDiscovery>();
            var routesAfterFirstFailure = new CompletableFuture<List<String>>();
            first.heartbeat.removeContext("/rtp_llm/master/info");
            first.heartbeat.createContext("/rtp_llm/master/info", exchange -> {
                try (exchange) {
                    int probe = probes.incrementAndGet();
                    // The previous refresh has published its snapshot before the next probe starts.
                    if (probe == 3) routesAfterFirstFailure.complete(active.get().routes());
                    var body = json.createObjectNode().put("real_master_host", first.leader);
                    if (probe == 2 && status == 200) body.put("ready", false);
                    byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(probe == 1 ? 200 : probe == 2 ? status : 503, bytes.length);
                    exchange.getResponseBody().write(bytes);
                }
            });
            try (var discovery = new MasterDiscovery(topology, null)) {
                active.set(discovery);
                assertEquals(List.of(primary, backup), discovery.routes());
                assertEquals(retained ? List.of(primary, backup) : List.of(backup),
                    routesAfterFirstFailure.get(5, TimeUnit.SECONDS));
                PlaybackTest.eventually(() -> discovery.routes().equals(List.of(backup)));
            }
        }
    }
}

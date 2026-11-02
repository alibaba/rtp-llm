package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MasterDiscoveryTest {
    @TempDir Path directory;
    final ObjectMapper json = new ObjectMapper();
    void publish(Path path, List<?> nodes) throws Exception {
        Path temp = directory.resolve("next.json");
        json.writeValue(temp.toFile(), nodes);
        Files.move(temp, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    }
    @Test void observesLeadershipThenDiscoversReplacementWithoutChangingThePlaybackPlan() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new NativeClientTest.Harness(); var second = new NativeClientTest.Harness(); var added = new NativeClientTest.Harness()) {
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
    @Test void explicitSelectionStaysFixedAndMalformedDiscoveryFailsPlayback() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var first = new NativeClientTest.Harness(); var second = new NativeClientTest.Harness()) {
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
}

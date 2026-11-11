package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;

/**
 * Subprocess killed by the Python cross-language recovery test after its first durable frame.
 */
public final class EvidenceCrashProbe {
    public static void main(String[] args) throws Exception {
        Path root = Path.of(args[0]), topology = root.resolve("topology.json");
        Files.createDirectories(root);
        try (var master = new BackendRpcClientTest.Harness();
             var client = new ReplayClient("client-0", topology, root.resolve("components/client-0"))) {
            master.output = (id, response) -> response.onNext(BackendRpcClientTest.token(id, 1, false));
            new ObjectMapper().writeValue(topology.toFile(), List.of(master.endpoint(),
                Map.of("id", "client-0", "role", "client", "control", client.url(), "rpc", "client:1")));
            client.start(InputFixture.plan(root, null, 20, 1, 2, 32, 30000, 30000));
            Path path = root.resolve("components/client-0/playback/1/requests.jsonl");
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(8);
            while (!Files.readString(path).contains("\"FIRST_TOKEN\"") && System.nanoTime() < deadline) Thread.sleep(20);
            if (!Files.readString(path).contains("\"FIRST_TOKEN\"")) throw new IllegalStateException("Evidence writer did not flush");
            Files.writeString(root.resolve("written"), "ready");
            new CountDownLatch(1).await();
        }
    }
}

package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;

/** Subprocess killed by the Python cross-language recovery test after its first durable frame. */
public final class EvidenceCrashProbe {
    public static void main(String[] args) throws Exception {
        Path root = Path.of(args[0]), topology = root.resolve("topology.json");
        Files.createDirectories(root);
        try (var master = new NativeClientTest.Harness();
             var client = new ClientNode("client-0", topology, root.resolve("components/client-0"))) {
            master.output = (id, response) -> response.onNext(NativeClientTest.token(id, 1, false));
            new ObjectMapper().writeValue(topology.toFile(), List.of(master.endpoint(),
                Map.of("id", "client-0", "role", "client", "control", client.url(), "rpc", "client:1")));
            client.start(InputFixture.plan(root, null, 20, 1, 2, 32, 30000, 30000));
            Path path = root.resolve("components/client-0/playback/1/requests.binpb.zst");
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(8);
            while (Files.size(path) == 0 && System.nanoTime() < deadline) Thread.sleep(20);
            if (Files.size(path) == 0) throw new IllegalStateException("Evidence writer did not flush");
            Files.writeString(root.resolve("written"), "ready");
            new CountDownLatch(1).await();
        }
    }
}

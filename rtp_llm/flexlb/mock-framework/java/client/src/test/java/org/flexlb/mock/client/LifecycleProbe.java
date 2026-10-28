package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.TimeUnit;

/** Bounded sequential traffic across a topology change; every outcome is retained. */
public final class LifecycleProbe {
    public static void main(String[] args) throws Exception {
        var directory = Path.of(args[1]);
        var json = new ObjectMapper();
        try (var client = new NativeClient(args[0]);
             var writer = Files.newBufferedWriter(directory.resolve("requests.jsonl"), StandardOpenOption.CREATE_NEW)) {
            long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(50);
            for (long id = 1_000_000; !Files.exists(directory.resolve("stop")); id++) {
                if (System.nanoTime() > end) throw new IllegalStateException("Lifecycle test did not stop traffic");
                var result = client.send(MasterProbe.request(id), Duration.ofSeconds(2))
                    .completion().toCompletableFuture().get(3, TimeUnit.SECONDS);
                writer.write(json.writeValueAsString(Map.of("id", id, "sent_at", result.sentAt().toString(),
                    "finished_at", Instant.now().toString(), "outcome", result.outcome().name(), "tokens", result.outputTokens())));
                writer.newLine(); writer.flush();
                Thread.sleep(50);
            }
        }
    }
}

package org.flexlb.mock.compute;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.api.model.Types.Role;

/** Cross-language component test only; not an Engine or production launcher. */
public final class ComputeProbe {
    public static void main(String[] args) throws Exception {
        var root = new ObjectMapper().enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS).readTree(Path.of(args[0]).toFile());
        if (root.path("environment").has("compute")) throw new IllegalArgumentException("Compute belongs in component.config");
        var models = ComputeModel.fromJson(root.path("component").path("config").path("compute"));
        var request = new ForwardBatch.Request("request", 10, 2, 32, 0, 8, 0, List.of());
        var batch = new ForwardBatch("batch", List.of(new ForwardBatch.Partition(0, List.of(request))));
        var sample = models.plan(Role.prefill, "prefill-0", batch);
        try (var timer = new SimulatedCompute()) {
            var outcome = timer.submit(sample).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            var result = Map.of("state", outcome.state().name(), "base_ns", sample.base().toNanos(),
                "planned_ns", sample.planned().toNanos(), "elapsed_ns", outcome.elapsedNanos(),
                "effective", models.effectiveConfiguration());
            Files.writeString(Path.of(args[1]), new ObjectMapper().writeValueAsString(result));
        }
    }
}

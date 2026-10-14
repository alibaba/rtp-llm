package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mockengine.MockEngineTestSupport.*;

class CrashCounterGenerationTest {
    @TempDir Path directory;

    @Test void crashResetsTotalsAndGenerationWhileNetworkStopPreservesThem() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "1", 1), 62220, 1, 1)) {
            var prefill = cluster.prefill(0);
            var decode = cluster.decode(0);
            enqueueAndFetch(prefill, batch(1, slot(0, inputWithDecode(1, 512, decode.getGrpcPort(), 10))));
            cluster.awaitAllInflightZero(2000);
            assertTrue(((Number) prefill.getSnapshot().get("context_tokens_total")).longValue() > 0);
            assertTrue(((Number) decode.getSnapshot().get("generate_tokens_total")).longValue() > 0);
            var oldIdentity = prefill.getSnapshot().get("engine_incarnation");
            var oldTags = prefill.whaleMetricTags();
            prefill.setStopped(true);
            prefill.setStopped(false);
            assertEquals(oldIdentity, prefill.getSnapshot().get("engine_incarnation"));
            assertEquals(oldTags, prefill.whaleMetricTags());
            for (var service : List.of(prefill, decode)) {
                service.crashNow();
                for (String field : List.of("context_compute_tokens_total", "context_tokens_total",
                        "context_requests_total", "generate_tokens_total"))
                    assertEquals(0L, ((Number) service.getSnapshot().get(field)).longValue(), field);
                service.setStopped(false);
            }
            assertNotEquals(oldIdentity, prefill.getSnapshot().get("engine_incarnation"));
            assertNotEquals(oldTags, prefill.whaleMetricTags());
            enqueueAndFetch(prefill, batch(2, slot(0, inputWithDecode(2, 512, decode.getGrpcPort(), 10))));
            cluster.awaitAllInflightZero(2000);
            assertEquals(512L, ((Number) prefill.getSnapshot().get("context_tokens_total")).longValue());
            assertEquals(10L, ((Number) decode.getSnapshot().get("generate_tokens_total")).longValue());
        }
    }
}

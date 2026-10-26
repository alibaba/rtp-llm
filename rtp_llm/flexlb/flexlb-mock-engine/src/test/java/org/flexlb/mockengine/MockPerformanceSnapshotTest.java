package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class MockPerformanceSnapshotTest {
    @TempDir Path directory;

    @Test @Timeout(10)
    void concurrentDecodeUpdatesNeverMixStepAndTokenCoefficients() throws Exception {
        var model = MockEngineTestSupport.performanceModel(directory, "10", 1);
        model.setRuntimeDecode(10, 0, 1);
        var executor = java.util.concurrent.Executors.newSingleThreadExecutor();
        try {
            var writer = executor.submit(() -> {
                for (int i = 0; i < 10000; i++) model.setRuntimeDecode(i % 2 == 0 ? 20 : 10, 0, i % 2 == 0 ? 2 : 1);
            });
            for (int i = 0; i < 10000; i++) assertEquals(120, model.decodeMs(12, 1));
            writer.get();
        } finally { executor.shutdownNow(); }
    }

    @Test void invalidPatchCannotPublishPartialTimingControls() throws Exception {
        var model = MockEngineTestSupport.performanceModel(directory, "10", 1);
        var before = model.decodeModelState();
        var patch = new com.fasterxml.jackson.databind.ObjectMapper().readTree(
                "{\"max_waiting_batches\":-1,\"mock\":{\"forward\":{\"prefill\":{\"fixed_ms\":999},\"decode\":{\"scale\":7}}}}");
        assertThrows(IllegalArgumentException.class, () -> model.applyOverrides(patch));
        assertEquals(before, model.decodeModelState());
    }
}

package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BoundedPerformanceNoiseTest {
    @TempDir Path directory;

    private MockPerformanceModel model(String json) throws Exception {
        Path performance = directory.resolve("performance.json");
        Path master = directory.resolve("master.json");
        Files.writeString(performance, json);
        MockMasterConfig.writeWithPrefillExpression(master, "100 + sum(computeTokens / 1024.) * 100");
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private static MockPerformanceModel.RequestShape request(int tokens) {
        return new MockPerformanceModel.RequestShape(
                EngineRpcService.GenerateInputPB.getDefaultInstance(), tokens, 1,
                List.of(), 0, 0, false);
    }

    @Test void noiseIsOptInAndBoundedForBothRoles() throws Exception {
        var plain = model("{\"decode\":{\"step_base_ms\":100,\"step_per_running_ms\":0}}");
        assertEquals(200, plain.prefillMs(List.of(request(1024))));
        assertEquals(100, plain.decodeStepDelayMs(1));

        var noisy = model("{\"prefill\":{\"noise\":{\"base_std_ms\":2,"
                + "\"variance_per_unit_ms2\":9,\"max_std_ms\":20,\"max_abs_ms\":12}},"
                + "\"decode\":{\"step_base_ms\":100,\"step_per_running_ms\":0,"
                + "\"noise\":{\"base_std_ms\":2,\"variance_per_unit_ms2\":1,"
                + "\"max_std_ms\":20,\"max_abs_ms\":8}}}");
        for (int i = 0; i < 500; i++) {
            assertTrue(Math.abs(noisy.prefillMs(List.of(request(1024))) - 200) <= 12);
            assertTrue(Math.abs(noisy.prefillMs(List.of(request(8192))) - 900) <= 12);
            assertTrue(Math.abs(noisy.decodeStepDelayMs(1) - 100) <= 8);
        }
        int steps = noisy.decodeSteps(50);
        assertTrue(Math.abs(noisy.decodeMs(50, 1) - 100L * steps) <= 8L * steps);
        assertTrue(Math.abs(noisy.forEngine().prefillMs(List.of(request(1024))) - 200) <= 12);
    }

    @Test void standardDeviationGrowsSublinearlyAndInvalidConfigFails() throws Exception {
        var spec = new MockPerformanceModel.NoiseSpec(2, 9, 20, 12);
        assertEquals(2, spec.stdMs(0));
        assertTrue(spec.stdMs(16) > spec.stdMs(1));
        assertTrue(spec.stdMs(16) / 16 < spec.stdMs(1));
        assertEquals(20, spec.stdMs(1000));
        assertThrows(IllegalStateException.class, () -> model("{\"prefill\":{\"noise\":{"
                + "\"base_std_ms\":2,\"max_std_ms\":20}}}"));
        assertThrows(IllegalStateException.class, () -> model("{\"jitter_pct\":0.1,"
                + "\"prefill\":{\"noise\":{\"base_std_ms\":2,"
                + "\"max_std_ms\":20,\"max_abs_ms\":12}}}"));
    }
}

package org.flexlb.mockengine;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class WhalePrefillFormulaRuntimeTest {
    @TempDir Path directory;

    @Test void controlApiValidatesAndScopesUpdates() throws Exception {
        var test = new PythonCompatControlApiTest();
        test.tempDir = directory;
        try { test.prefillFormulaUpdateIsScopedAndInvalidUpdatePreservesState(); }
        finally { test.tearDown(); }
    }

    @Test void updateKeepsCacheAndUsesFormulaWithoutLegacyScale() throws Exception {
        Path perf = directory.resolve("perf.json");
        Files.writeString(perf, "{\"block_size\":2,\"prefill\":{\"scale\":0.5}}");
        Path config = directory.resolve("master.json");
        MockMasterConfig.writeWithPrefillExpression(config, "200");
        var model = MockPerformanceModel.load(perf.toString(), config.toString());
        var cache = new MockLruBlockCache(10);
        cache.admit(List.of(1L, 2L));
        var request = model.shape(EngineRpcService.GenerateInputPB.newBuilder()
                .addAllTokenIds(List.of(1, 2, 3, 4)).build(), cache);
        assertEquals(100, model.prefillMs(List.of(request)));
        assertEquals("200", model.prefillExpressionState().get("expression"));
        model.setPrefillExpression("300 + sum(computeTokens)");
        assertEquals(304, model.prefillMs(List.of(request)));
        assertEquals(1.0, model.prefillExpressionState().get("scale"));
        assertEquals(2, cache.lruKeyBlocks());
        assertEquals(304, model.forEngine().prefillMs(List.of(request)));
        assertThrows(IllegalArgumentException.class, () -> model.setPrefillExpression("bad_function(1)"));
        assertThrows(IllegalArgumentException.class, () -> model.setPrefillExpression("-1"));
        assertEquals(304, model.prefillMs(List.of(request)));
    }
    @Test void runtimeFormulaUsesBatchTotalAndMaximumAcrossDifferentRequests() throws Exception {
        Path perf = directory.resolve("aggregate-perf.json");
        Files.writeString(perf, "{}");
        Path config = directory.resolve("aggregate-master.json");
        MockMasterConfig.writeWithPrefillExpression(config, "200");
        var model = MockPerformanceModel.load(perf.toString(), config.toString());
        var cache = new MockLruBlockCache(10);
        var first = model.shape(EngineRpcService.GenerateInputPB.newBuilder()
                .addAllTokenIds(List.of(1, 2, 3, 4)).build(), cache);
        var second = model.shape(EngineRpcService.GenerateInputPB.newBuilder()
                .addAllTokenIds(List.of(5, 6)).build(), cache);
        model.setPrefillExpression("100 * totalComputeTokens + 10 * maxInputTokens");
        assertEquals(640, model.prefillMs(List.of(first, second)));
        model.setPrefillExpression("totalInputTokens + totalHitCacheTokens + maxComputeTokens");
        assertEquals(10, model.prefillMs(List.of(first, second)));
    }

}

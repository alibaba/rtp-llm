package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class MockCodeDefaultsTest {
    @Test void cacheReuseRequiresAnExplicitEnableEvenForSyntheticKeys() throws Exception {
        var model = MockPerformanceModel.loadCalibration(new ObjectMapper().createObjectNode());
        var cache = new MockLruBlockCache(16);
        assertTrue(cache.admit(List.of(7L)));
        var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(1)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(1)
                        .setUniqueKey("{\"input_len\":2,\"output_len\":1,\"block_cache_keys\":[7]}"))
                .build();
        assertEquals(0, model.shape(input, cache).hitBlocks());
        model.prefillReuseCache = true;
        assertEquals(1, model.shape(input, cache).hitBlocks(), "explicit reuse still enables matching");
    }
}

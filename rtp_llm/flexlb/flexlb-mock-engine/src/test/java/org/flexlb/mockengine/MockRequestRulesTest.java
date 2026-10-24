package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MockRequestRulesTest {
    private EngineRpcService.GenerateInputPB request(int tokens, boolean explicit, boolean scoring) {
        return EngineRpcService.GenerateInputPB.newBuilder().setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                .setMaxNewTokens(tokens).setPrefillOnly(explicit).setReturnPromptLogits(scoring)).build();
    }
    @Test void zeroWithoutExplicitFlagRemainsPrefillOnly() {
        assertEquals(0, MockRequestRules.normalize(request(0, false, false), false).getGenerateConfig().getMaxNewTokens());
        assertThrows(io.grpc.StatusRuntimeException.class, () -> MockRequestRules.normalize(request(0, false, false), true));
    }
    @Test void invalidWireCombinationsAndNegativeTokensCarryDomain605() {
        for (var input : java.util.List.of(request(-1, false, false), request(1, true, false), request(0, true, true))) {
            var error = assertThrows(io.grpc.StatusRuntimeException.class, () -> MockRequestRules.normalize(input, false));
            assertEquals(605, MockRpcErrors.downstream(error).getErrorCode());
            assertEquals(io.grpc.Status.Code.INVALID_ARGUMENT, error.getStatus().getCode());
        }
    }
    @Test void scoringNormalizesToSingleTokenAndDisablesReuseAndPd() {
        var input = request(0, false, true).toBuilder().setGenerateConfig(request(0, false, true).getGenerateConfig()
                .toBuilder().setReuseCache(true).setCanUsePdSeparation(true).setIsStreaming(true)).build();
        var result = MockRequestRules.normalize(input, false).getGenerateConfig();
        assertEquals(1, result.getMaxNewTokens());
        assertFalse(result.getReuseCache()); assertFalse(result.getCanUsePdSeparation()); assertFalse(result.getIsStreaming());
    }
    @Test void reserveValidationPrecedesTotalPoolLimitAndAllowsEquality() {
        assertNull(MockRequestRules.inputError(16, 20, 4, 100));
        assertEquals(511, MockRequestRules.inputError(17, 20, 4, 100).getErrorCode());
        assertEquals(607, MockRequestRules.inputError(101, 1000, 0, 100).getErrorCode());
        assertNull(MockRequestRules.inputError(20, 20, 0, 100), "no synthetic max_seq gate when reserve is zero");
    }
}

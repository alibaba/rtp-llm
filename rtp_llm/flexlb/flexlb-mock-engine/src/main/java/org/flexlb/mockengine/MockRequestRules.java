package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;

/** QueryConverter + GenerateConfig + FIFOSchedulerBase request-side checks. */
final class MockRequestRules {
    private MockRequestRules() {}
    static EngineRpcService.GenerateInputPB normalize(EngineRpcService.GenerateInputPB input, boolean decode) {
        input = normalizeQuery(input);
        var config = input.getGenerateConfig();
        int tokens = config.getMaxNewTokens();
        if (tokens == 0) {
            int probs = config.getReturnAllProbsMode();
            boolean wantsProbs = probs == 2 || probs == 3 || (probs == 0 && config.getReturnAllProbs());
            if (config.getMinNewTokens() > 0 || config.getNumBeams() > 1 || config.getVariableNumBeamsCount() > 0
                    || config.getNumReturnSequences() > 1 || config.getReturnLogits() || config.getCalculateLoss() != 0
                    || config.getReturnSoftmaxProbs() || wantsProbs || config.getReturnCumLogProbs()
                    || config.getReturnHiddenStates() || config.getReturnAllHiddenStates())
                throw invalid("prefill-only request contains generation-only outputs or sampling constraints");
            if (decode) throw invalid("DECODE role does not accept prefill-only requests");
        }
        return input;
    }
    static EngineRpcService.GenerateInputPB normalizeQuery(EngineRpcService.GenerateInputPB input) {
        var config = input.getGenerateConfig();
        int tokens = config.getMaxNewTokens();
        if (tokens < 0 || (config.getPrefillOnly() && (tokens != 0 || config.getReturnPromptLogits())))
            throw invalid("invalid max_new_tokens / prefill_only / return_prompt_logits combination");
        if (config.getReturnPromptLogits()) {
            config = config.toBuilder().setMaxNewTokens(1).setIsStreaming(false)
                    .setReuseCache(false).setCanUsePdSeparation(false).build();
        }
        return input.toBuilder().setGenerateConfig(config).build();
    }
    static EngineRpcService.GenerateInputPB decodeInput(EngineRpcService.GenerateInputPB input) {
        var copy = input.toBuilder().clearGroupSize().clearGroupId();
        copy.getGenerateConfigBuilder().clearGroupTimeout();
        return copy.build();
    }
    static EngineRpcService.ErrorDetailsPB inputError(long input, long maxSequence, long reserve, long totalTokens) {
        if (reserve > 0 && (input > maxSequence || reserve > maxSequence - input))
            return error(511, "input length plus speculative reserve_step exceeds max_seq_len");
        if (input > totalTokens) return error(607, "input length exceeds KV cache maximum token capacity");
        return null;
    }
    private static io.grpc.StatusRuntimeException invalid(String message) {
        return MockRpcErrors.error(io.grpc.Status.INVALID_ARGUMENT, 605, message);
    }
    private static EngineRpcService.ErrorDetailsPB error(long code, String message) {
        return EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(code).setErrorMessage(message).build();
    }
}

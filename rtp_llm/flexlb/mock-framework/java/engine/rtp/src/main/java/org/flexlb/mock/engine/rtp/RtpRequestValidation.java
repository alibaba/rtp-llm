package org.flexlb.mock.engine.rtp;

import io.grpc.Status;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;

/** Common support boundary for local Prefill, P/D Prefill and Decode admission. */
final class RtpRequestValidation {
    private RtpRequestValidation() {}

    static void supported(GenerateInputPB input) {
        var g = input.getGenerateConfig();
        if (input.getMultimodalInputsCount() > 0 || input.hasV41Inputs() || input.hasMultimodalTokenLayout()
                || g.hasAdapterName() || !g.getUniqueKey().isEmpty()
                || g.getNumBeams() > 1 || g.getNumReturnSequences() > 1 || g.getVariableNumBeamsCount() > 0
                || g.getReturnPromptLogits() || g.getReturnHiddenStates() || g.getReturnLogits() || g.getReturnAllHiddenStates()
                || g.getReturnAllProbs() || g.getReturnSoftmaxProbs() || g.getReturnCumLogProbs() || g.getCalculateLoss() != 0
                || g.getStopWordsList().getRowsCount() > 0 || g.getSpEdit() || g.getSpInputLookup())
            throw Status.UNIMPLEMENTED.withDescription("Unsupported multimodal, adapter, sequence shape or output mode").asRuntimeException();
    }
}

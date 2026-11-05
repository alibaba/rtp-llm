package org.flexlb.mock.admission;

import com.google.protobuf.ByteString;
import io.grpc.Status;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.Scheduler;

/**
 * Translation at the native RTP boundary; scheduling state contains no protobuf objects.
 */
final class RpcMessages {
    private RpcMessages() {}

    static Scheduler.Input input(GenerateInputPB input, String identity, long started, long rpcDeadline,
                                 boolean retained, int blockSize, long seed) {
        return new Scheduler.Input(identity, input.getRequestId(), input.hasGroupId() ? input.getGroupId().getValue() : 0,
            input.getTokenIdsCount(), input.getGenerateConfig().getMaxNewTokens(), input.getPriority(), started, rpcDeadline,
            new org.flexlb.mock.cache.CacheSequence(blockSize, input.getTokenIdsList(), java.util.List.of(), seed), input.getGenerateConfig().getReuseCache(), retained, input.getGenerateConfig().getMinNewTokens(), input.getGenerateConfig().getIgnoreEos(), input.getGenerateConfig().getIsStreaming());
    }
    static Scheduler.Task pending(GenerateInputPB input) {
        return new Scheduler.Task(input.getRequestId(), input.hasGroupId() ? input.getGroupId().getValue() : 0,
            input.getTokenIdsCount(), 0, input.getPriority(), Scheduler.ReportedPhase.PENDING, 0, 0, -1, -1, null, false);
    }
    static long deadline(GenerateInputPB input, long started, long rpcDeadline) {
        int timeout = input.getGenerateConfig().getTimeoutMs();
        return Math.min(rpcDeadline, timeout <= 0 ? Long.MAX_VALUE : after(started, TimeUnit.MILLISECONDS.toNanos(timeout)));
    }
    static TaskInfoPB task(Scheduler.Task task) {
        var result = TaskInfoPB.newBuilder().setRequestId(task.requestId()).setBatchId(task.batchId()).setRequestIdStr(Long.toString(task.requestId())).setPrefixLengthValid(task.prefixLengthValid())
            .setInputLength(task.inputLength()).setPrefixLength(task.prefixLength()).setPriority(task.priority())
            .setDpRank(0).setPhase(switch (task.phase()) {
                case PENDING -> TaskPhase.TASK_PHASE_PENDING;
                case RECEIVED -> TaskPhase.TASK_PHASE_RECEIVED;
                case KV_ALLOCATED -> TaskPhase.TASK_PHASE_KV_ALLOCATED;
                case RUNNING -> TaskPhase.TASK_PHASE_RUNNING;
            }).setIsWaiting(task.phase() != Scheduler.ReportedPhase.RUNNING)
            .setIterateCount(task.iterations()).setWaitingTimeMs(task.waitingMillis())
            .setExecutionTimeMs(task.executionMillis()).setEndTimeMs(task.endTimeMillis());
        if (task.failure() != null) {
            long code = RpcErrors.code(task.failure());
            if (code == 0) code = RpcErrors.status(task.failure()).getCode().value();
            result.setErrorInfo(ErrorDetailsPB.newBuilder().setErrorCode(code).setErrorMessage(java.util.Objects.toString(task.failure().getMessage(), "")));
        }
        return result.build();
    }
    static long after(long now, long nanos) {
        try { return Math.addExact(now, nanos); }
        catch (ArithmeticException overflow) { return nanos < 0 ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }

    static int sampledToken(GenerateInputPB input) {
        int last=input.getTokenIds(input.getTokenIdsCount()-1);
        return last>=0?last:42; // Cache-only image hashes are not generated vocabulary tokens.
    }
    static GenerateOutputsPB token(GenerateInputPB input, int tokenId, int generated, int stepTokens, int reused, long elapsedNanos, boolean finished, boolean pd) {
        // Native RPC carries new tokens; Frontend owns cumulative responses.
        int width = stepTokens;
        var tokens = ByteBuffer.allocate(Math.multiplyExact(4, width)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < width; i++) tokens.putInt(tokenId);
        var aux = AuxInfoPB.newBuilder().setInputLen(input.getTokenIdsCount()).setOutputLen(generated).setStepOutputLen(stepTokens)
            .setIterCount(generated).setPrefixLen(reused).setTotalReuseLen(reused).setLocalReuseLen(reused).setPdSep(pd)
            .setCostTimeUs((int) Math.min(Integer.MAX_VALUE, elapsedNanos / 1000))
            .setFirstTokenCostTimeUs((int) Math.min(Integer.MAX_VALUE, elapsedNanos / 1000));
        var output = FlattenOutputPB.newBuilder().addFinished(finished).addAuxInfo(aux).setOutputIds(TensorPB.newBuilder()
            .setDataType(TensorPB.DataType.INT32).addShape(1).addShape(1).addShape(width)
            .setInt32Data(ByteString.copyFrom(tokens.array())));
        return GenerateOutputsPB.newBuilder().setRequestId(input.getRequestId()).setFlattenOutput(output).build();
    }

    static void supported(GenerateInputPB input) {
        var g = input.getGenerateConfig();
        if (g.getPrefillOnly())
            throw Status.UNIMPLEMENTED.withDescription("Mock does not support prefill_only").asRuntimeException();
        if (g.hasAdapterName() || !g.getUniqueKey().isEmpty()
                || g.getNumBeams() > 1 || g.getNumReturnSequences() > 1 || g.getVariableNumBeamsCount() > 0
                || g.getReturnPromptLogits() || g.getReturnHiddenStates() || g.getReturnLogits() || g.getReturnAllHiddenStates()
                || g.getReturnAllProbs() || g.getReturnSoftmaxProbs() || g.getReturnCumLogProbs() || g.getCalculateLoss() != 0
                || g.getStopWordsList().getRowsCount() > 0 || g.getSpEdit() || g.getSpInputLookup())
            throw Status.UNIMPLEMENTED.withDescription("Unsupported adapter, sequence shape or output mode").asRuntimeException();
    }
}

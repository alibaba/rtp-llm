package org.flexlb.mock.engine.rtp;

import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.Scheduler;

/** Translation at the native RTP boundary; scheduling state contains no protobuf objects. */
final class RtpTasks {
    static Scheduler.Input input(GenerateInputPB input, String identity, long started, long rpcDeadline,
                                    boolean retained, int blockSize) {
        long deadline = Math.min(rpcDeadline, input.getGenerateConfig().getTimeoutMs() == 0 ? Long.MAX_VALUE
            : after(started, TimeUnit.MILLISECONDS.toNanos(input.getGenerateConfig().getTimeoutMs())));
        return new Scheduler.Input(identity, input.getRequestId(), input.hasGroupId() ? input.getGroupId().getValue() : 0,
            input.getTokenIdsCount(), input.getGenerateConfig().getMaxNewTokens(), input.getPriority(), started, deadline,
            PrefixKeys.completeBlocks(input, blockSize), input.getGenerateConfig().getReuseCache(), retained);
    }
    static TaskInfoPB task(Scheduler.Task task) {
        var result = TaskInfoPB.newBuilder().setRequestId(task.requestId()).setBatchId(task.batchId())
            .setInputLength(task.inputLength()).setPrefixLength(task.prefixLength()).setPriority(task.priority())
            .setDpRank(0).setPhase(switch (task.phase()) {
                case PENDING -> TaskPhase.TASK_PHASE_PENDING;
                case RECEIVED -> TaskPhase.TASK_PHASE_RECEIVED;
                case KV_ALLOCATED -> TaskPhase.TASK_PHASE_KV_ALLOCATED;
                case RUNNING -> TaskPhase.TASK_PHASE_RUNNING;
            }).setIsWaiting(task.phase() != Scheduler.ReportedPhase.RUNNING)
            .setIterateCount(task.iterations()).setWaitingTimeMs(task.waitingMillis())
            .setExecutionTimeMs(task.executionMillis()).setEndTimeMs(task.endTimeMillis());
        if (task.errorCode() != 0)
            result.setErrorInfo(ErrorDetailsPB.newBuilder().setErrorCode(task.errorCode()).setErrorMessage(task.errorMessage()));
        return result.build();
    }
    static long after(long now, long nanos) {
        try { return Math.addExact(now, nanos); }
        catch (ArithmeticException overflow) { return nanos < 0 ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }
}

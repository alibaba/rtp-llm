package org.flexlb.mock.admission;

import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.Scheduler;

/**
 * Translation at the native RTP boundary; scheduling state contains no protobuf objects.
 */
final class RpcTasks {
    static Scheduler.Input input(GenerateInputPB input, String identity, long started, long rpcDeadline,
                                 boolean retained, int blockSize, long seed) {
        return new Scheduler.Input(identity, input.getRequestId(), input.hasGroupId() ? input.getGroupId().getValue() : 0,
            input.getTokenIdsCount(), input.getGenerateConfig().getMaxNewTokens(), input.getPriority(), started, rpcDeadline,
            new org.flexlb.mock.cache.CacheSequence(blockSize, input.getTokenIdsList(), java.util.List.of(), seed), input.getGenerateConfig().getReuseCache(), retained, input.getGenerateConfig().getMinNewTokens(), input.getGenerateConfig().getIgnoreEos());
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
}

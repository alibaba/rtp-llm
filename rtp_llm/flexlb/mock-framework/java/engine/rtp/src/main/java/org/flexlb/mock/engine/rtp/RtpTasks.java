package org.flexlb.mock.engine.rtp;

import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.RequestFailure;
import org.flexlb.mock.engine.Scheduler;
import org.flexlb.mock.engine.RequestState.Phase;

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
    static TaskInfoPB.Builder task(Scheduler.Request request) {
        var input = request.input(); var progress = request.snapshot();
        boolean running = request.running();
        long now = System.nanoTime();
        long execution = progress.executionNanos() + (progress.phase() == Phase.FORWARD ? Math.max(0, now - request.stepStarted()) : 0);
        return TaskInfoPB.newBuilder().setRequestId(input.requestId()).setBatchId(input.groupId()).setInputLength(input.tokens())
            .setPrefixLength(request.reused()).setPriority(input.priority()).setDpRank(0).setIsWaiting(!running)
            .setPhase(running ? TaskPhase.TASK_PHASE_RUNNING : request.allocated() ? TaskPhase.TASK_PHASE_KV_ALLOCATED
                : progress.phase() == Phase.CAPACITY ? TaskPhase.TASK_PHASE_RECEIVED : TaskPhase.TASK_PHASE_PENDING)
            .setIterateCount(progress.iterations())
            .setWaitingTimeMs(TimeUnit.NANOSECONDS.toMillis(Math.max(0, now - input.started() - execution)))
            .setExecutionTimeMs(TimeUnit.NANOSECONDS.toMillis(execution));
    }
    static TaskInfoPB finished(Scheduler.Request request, RequestFailure failure) {
        var task = task(request).setEndTimeMs(System.currentTimeMillis());
        if (failure != null) {
            var status = RtpErrors.status(failure); long code = RtpErrors.code(failure);
            task.setErrorInfo(ErrorDetailsPB.newBuilder().setErrorCode(code == 0 ? status.getCode().value() : code)
                .setErrorMessage(status.getDescription() == null ? status.getCode().name() : status.getDescription()));
        }
        return task.build();
    }
    static long after(long now, long nanos) {
        try { return Math.addExact(now, nanos); }
        catch (ArithmeticException overflow) { return nanos < 0 ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }
}

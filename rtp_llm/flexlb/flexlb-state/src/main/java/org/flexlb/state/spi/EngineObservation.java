package org.flexlb.state.spi;

import java.util.List;
import java.util.Objects;

/**
 * 一次引擎观察上报的规范化视图（E7）。
 *
 * <p>不可变值对象：内部明细列表经 {@link List#copyOf} 防御拷贝，构造后不可变。
 * 完整性判定见 {@link #isComplete()}——截断上报中的"缺席"不可作为死亡证据
 * （对应 {@code org.flexlb.state.CleanupReason#TRUNCATED_REPORT_EXCLUDED}）。</p>
 *
 * @param endpointRef 上报端点身份（含世代，S8 屏障输入）
 * @param round       上报轮次号
 * @param statusMs    上报时间戳（毫秒）
 * @param detailCount 引擎侧声称的明细总数（E7：detailCount == running.size() 时本次上报完整）
 * @param running     引擎侧仍在执行的请求明细（不可变，防御拷贝）
 * @param finished    引擎侧已完成的请求明细（不可变，防御拷贝）
 */
public record EngineObservation(
        StateEndpointRef endpointRef,
        long round,
        long statusMs,
        int detailCount,
        List<RunningObservation> running,
        List<FinishedObservation> finished) {

    public EngineObservation {
        Objects.requireNonNull(endpointRef, "endpointRef");
        Objects.requireNonNull(running, "running");
        Objects.requireNonNull(finished, "finished");
        running = List.copyOf(running);
        finished = List.copyOf(finished);
        if (detailCount < 0) {
            throw new IllegalArgumentException("detailCount must be >= 0: " + detailCount);
        }
    }

    /**
     * E7 完整性：本次上报是否未截断——引擎声称的明细总数与实际携带的 running 明细数一致。
     */
    public boolean isComplete() {
        return detailCount == running.size();
    }

    /**
     * 引擎侧执行中请求的一次观察明细。
     *
     * @param requestId   请求 ID
     * @param side        该请求在哪个状态侧被跟踪
     * @param enginePhase 引擎报告的执行相位
     * @param batchId     所属批次 ID
     * @param kvTokens    当前 KV token 数
     * @param version     引擎上报序号（单调；裁决矩阵的版本屏障输入，S4/L2）
     */
    public record RunningObservation(
            long requestId,
            StateRole side,
            EnginePhase enginePhase,
            long batchId,
            long kvTokens,
            long version) {
    }

    /**
     * 引擎侧已完成请求的一次观察明细。
     *
     * @param requestId  请求 ID
     * @param side       该请求在哪个状态侧被跟踪
     * @param errorCode  完成错误码（0 = 成功）
     * @param endTimeMs  引擎侧结束时间戳（毫秒）
     * @param version    引擎上报序号（单调；裁决矩阵的版本屏障输入，S4/L2）
     */
    public record FinishedObservation(
            long requestId,
            StateRole side,
            int errorCode,
            long endTimeMs,
            long version) {
    }
}

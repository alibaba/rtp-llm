package org.flexlb.balance.queue.dispatch;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;

/**
 * 一台 Prefill 引擎当前能接下的一批请求的上限：单批最多多少 token、剩余多少 KV cache，
 * 以及引擎上正在运行和排队的请求数。
 * 数值取自引擎最近一次上报的状态，组批时用它判断一批请求放不放得下。
 */
public record BatchCapacityView(long batchTokenCapacity,
                                long batchKvCapacity,
                                long runningQueryLen,
                                long waitingQueryLen) {

    /**
     * 读取引擎最近一次上报的状态。还没有上报时不限制容量。
     */
    public static BatchCapacityView of(PrefillEndpoint prefillEndpoint, FlexlbConfig config) {
        WorkerStatus status = prefillEndpoint != null
                ? prefillEndpoint.getStatus() : null;
        if (status == null) {
            return new BatchCapacityView(Long.MAX_VALUE, Long.MAX_VALUE, -1, -1);
        }
        WorkerStatus.EngineObservation engineStatus =
                status.committedEngineObservation();
        long engineCapacity = engineStatus.maxBatchTokensSize();
        if (engineCapacity <= 0) {
            engineCapacity = engineStatus.maxSeqLen();
        }
        if (engineCapacity <= 0) {
            engineCapacity = config.getFallbackBatchTokenCapacity();
        }
        long batchTokenCapacity = positiveOrUnlimited(engineCapacity);
        long total = engineStatus.totalKvCacheTokens();
        if (total <= 0) {
            return new BatchCapacityView(
                    batchTokenCapacity, Long.MAX_VALUE, engineStatus.runningQueryLen(), engineStatus.waitingQueryLen());
        }
        long available = Math.max(
                0, engineStatus.availableKvCacheTokens());
        return new BatchCapacityView(
                batchTokenCapacity, Math.min(total, available),
                engineStatus.runningQueryLen(), engineStatus.waitingQueryLen());
    }

    /**
     * 这一批可用的 KV cache。引擎完全空闲时，即使这一批超过剩余 KV 也放行，
     * 否则超大请求会一直排不上。
     */
    public long kvCapacityFor(GroupPlanner.Shape shape) {
        if (!shape.fitsKv(batchKvCapacity) && runningQueryLen == 0 && waitingQueryLen == 0) {
            return Long.MAX_VALUE;
        }
        return batchKvCapacity;
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }
}

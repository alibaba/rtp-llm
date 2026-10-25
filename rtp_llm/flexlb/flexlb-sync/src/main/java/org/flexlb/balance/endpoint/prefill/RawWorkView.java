package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.estimate.WorkView;

import java.util.Comparator;
import java.util.List;

/**
 * 还没加工的 WorkView：持锁时把已提交工作复制成原始列表，不排序也不计算。
 *
 * <p>锁外第一次调用 {@link #toWorkView()} 时才排序并生成 WorkView，结果保存下来，之后直接复用。</p>
 */
public final class RawWorkView {
    public final long createdAtMs;
    private final List<WorkView.RequestWork> requests;
    private final List<WorkView.BatchWork> batches;
    private final long unknownRequestCount;
    private volatile WorkView workView;

    public RawWorkView(long createdAtMs,
                       List<WorkView.RequestWork> requests,
                       List<WorkView.BatchWork> batches,
                       long unknownRequestCount) {
        this.createdAtMs = createdAtMs;
        this.requests = List.copyOf(requests);
        this.batches = List.copyOf(batches);
        this.unknownRequestCount = unknownRequestCount;
    }

    /**
     * 排序并生成 WorkView，只做一次。调用方不持有 PrefillState 的锁。
     */
    public WorkView toWorkView() {
        WorkView result = workView;
        if (result != null) {
            return result;
        }
        synchronized (this) {
            if (workView == null) {
                List<WorkView.RequestWork> orderedRequests =
                        requests.stream().sorted(Comparator.comparing(WorkView.RequestWork::requestId)).toList();
                List<WorkView.BatchWork> orderedBatches = batches.stream()
                        .sorted(Comparator.comparingLong(WorkView.BatchWork::batchId))
                        .map(batch -> new WorkView.BatchWork(batch.batchId(),
                                batch.requestIds().stream().sorted().toList(), batch.phase(), batch.remainingWorkMs()))
                        .toList();
                workView = new WorkView(createdAtMs, orderedRequests, orderedBatches, unknownRequestCount);
            }
            return workView;
        }
    }
}

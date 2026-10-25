package org.flexlb.balance.queue;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.QueueView;
import org.flexlb.balance.estimate.QueueView.AdmissionBlock;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.queue.dispatch.BatchCapacityView;
import org.flexlb.balance.queue.dispatch.GroupingLimits;
import org.flexlb.config.FlexlbConfig;

import java.util.Comparator;
import java.util.function.Supplier;

/**
 * 缓存一台 Prefill 的负载视图，供选路时预估 TTFT。
 * <p>
 * 一次选路要对每台 Prefill 都预估一遍。只要队列、调度输入、在途工作和凑组等待时间都没变，
 * 就复用上次的负载视图，不必每次都在队列锁内重新复制排队请求。
 */
public final class LoadViewCache {

    private final QueueState queueState;
    private final PrefillState prefillState;
    private final PrefillEndpoint prefillEndpoint;
    private final FlexlbConfig config;
    private final GroupingLimits groupingLimits;
    private final Comparator<GroupPlanner.Item> estimateOrder;
    /**
     * 队首请求因投递容量不足而阻塞时，返回阻塞信息；否则返回 null。调用方持有队列锁。
     */
    private final Supplier<AdmissionBlock> admissionBlockUnderLock;
    private volatile CapturedLoadView latest;

    public LoadViewCache(QueueState queueState,
                         PrefillState prefillState,
                         PrefillEndpoint prefillEndpoint,
                         FlexlbConfig config,
                         GroupingLimits groupingLimits,
                         Comparator<GroupPlanner.Item> estimateOrder,
                         Supplier<AdmissionBlock> admissionBlockUnderLock) {
        this.queueState = queueState;
        this.prefillState = prefillState;
        this.prefillEndpoint = prefillEndpoint;
        this.config = config;
        this.groupingLimits = groupingLimits;
        this.estimateOrder = estimateOrder;
        this.admissionBlockUnderLock = admissionBlockUnderLock;
    }

    public PrefillTtftEstimate.LoadView capture() {
        CapturedLoadView source = latest;
        if (source == null || !isCurrent(source.version)) {
            queueState.lock();
            try {
                source = latest;
                if (source == null || !isCurrent(source.version)) {
                    source = captureUnderLock();
                    latest = source;
                }
            } finally {
                queueState.unlock();
            }
        }
        // 同一版本的并发调用共用一份。构建得晚的只填自己那份，不会覆盖更新的版本。
        return source.materialize();
    }

    private boolean isCurrent(Version version) {
        return version.queue() == queueState.queueVersion()
                && version.schedulingInputs() == queueState.schedulingInputVersion()
                && version.ownership() == prefillState.mutationVersion()
                && version.collectionWindowMs() == groupingLimits.collectionWindowMs();
    }

    /**
     * 调用方持有队列锁。
     */
    private CapturedLoadView captureUnderLock() {
        long fixedWaitMs = groupingLimits.collectionWindowMs();
        Version version = new Version(queueState.queueVersion(), queueState.schedulingInputVersion(),
                prefillState.mutationVersion(), fixedWaitMs);
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        PrefillState.Snapshot ownership = prefillState.snapshotUnderLock();
        return new CapturedLoadView(
                version, ownership,
                new GroupPlanner.Constraints(
                        groupingLimits.maxRequests(), capacity.batchTokenCapacity(),
                        Long.MAX_VALUE, groupingLimits.predictedExecutionBudgetMs(),
                        fixedWaitMs),
                ownership.activeItems().isEmpty() ? null : admissionBlockUnderLock.get());
    }

    /**
     * 负载视图依赖的各个版本号，任一变化都要重新复制。
     */
    private record Version(long queue,
                           long schedulingInputs,
                           long ownership,
                           long collectionWindowMs) {
    }

    /**
     * 在队列锁内复制下来的排队请求和在途工作。负载视图在锁外构建，同一版本只构建一次。
     */
    private final class CapturedLoadView {
        private final Version version;
        private PrefillState.Snapshot ownership;
        private final GroupPlanner.Constraints constraints;
        private final AdmissionBlock admissionBlock;
        private volatile PrefillTtftEstimate.LoadView materialized;

        private CapturedLoadView(Version version, PrefillState.Snapshot ownership,
                                 GroupPlanner.Constraints constraints, AdmissionBlock admissionBlock) {
            this.version = version;
            this.ownership = ownership;
            this.constraints = constraints;
            this.admissionBlock = admissionBlock;
        }

        private PrefillTtftEstimate.LoadView materialize() {
            PrefillTtftEstimate.LoadView result = materialized;
            if (result != null) {
                return result;
            }
            synchronized (this) {
                if (materialized == null) {
                    var queueSnapshot = new QueueView(ownership.capturedAtMs(),
                            estimateOrder, constraints, ownership.active().plannerItems(), admissionBlock);
                    materialized = new PrefillTtftEstimate.LoadView(
                            queueSnapshot, ownership.work().materialize(), version.ownership());
                    // 缓存的视图不能一直引用已完成请求的上下文。
                    ownership = null;
                }
                return materialized;
            }
        }
    }
}

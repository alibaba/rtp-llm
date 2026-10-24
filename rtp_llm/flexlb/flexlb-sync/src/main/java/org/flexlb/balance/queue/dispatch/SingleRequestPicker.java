package org.flexlb.balance.queue.dispatch;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.queue.ProcessQueueResult;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.OptionalLong;

/**
 * 单请求模式：每一轮只看队首，Prefill KV 容量够就单独交付，不和其他请求凑组。
 */
public final class SingleRequestPicker implements QueuePicker {

    private static final String SINGLE_DECISION_REASON = "single_request";

    private final QueueState queueState;
    private final PrefillEndpoint prefillEndpoint;
    private final FlexlbConfig config;
    private final GroupCommitter groupCommitter;

    public SingleRequestPicker(QueueState queueState,
                        PrefillEndpoint prefillEndpoint,
                        FlexlbConfig config,
                        GroupCommitter groupCommitter) {
        this.queueState = queueState;
        this.prefillEndpoint = prefillEndpoint;
        this.config = config;
        this.groupCommitter = groupCommitter;
    }

    @Override
    public ProcessQueueResult processQueue() {
        ActiveQueueSnapshot snapshot = snapshotActiveQueueHead();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return ProcessQueueResult.NO_ACTION;
        }

        long nowMs = now();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_single_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            groupCommitter.dropHead(head);
            return ProcessQueueResult.CAPACITY_CHANGED;
        }

        GroupPlanner.Shape shape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        if (!shape.fitsKv(capacity.kvCapacityFor(shape))) {
            return ProcessQueueResult.awaitingPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        // Prefill 状态和投递记账是分别更新的。检查投递容量之前，再读一次最新的 KV 容量。
        capacity = BatchCapacityView.of(prefillEndpoint, config);
        if (!shape.fitsKv(capacity.kvCapacityFor(shape))) {
            return ProcessQueueResult.awaitingPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        return groupCommitter.runPredictionBound(head, () -> {
            PrefillTimePredictor predictor =
                    prefillEndpoint.getPredictor();
            PrefillTimePredictor.Evaluator evaluator =
                    predictor.evaluator();
            return groupCommitter.admitAndDeliverCapacityFeasiblePrefix(
                    List.of(head), SINGLE_DECISION_REASON,
                    evaluator, OptionalLong.empty());
        });
    }

    private ActiveQueueSnapshot snapshotActiveQueueHead() {
        queueState.lock();
        try {
            ScheduledRequest head = queueState.activeIndex().peek();
            return new ActiveQueueSnapshot(
                    queueState.queueVersion(), queueState.schedulingInputVersion(),
                    head == null ? List.of() : List.of(head));
        } finally {
            queueState.unlock();
        }
    }

    private static long now() {
        return System.currentTimeMillis();
    }
}

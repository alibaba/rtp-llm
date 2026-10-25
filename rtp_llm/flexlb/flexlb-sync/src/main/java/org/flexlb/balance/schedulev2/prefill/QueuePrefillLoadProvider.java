package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.PrefillLoadProvider;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.queue.dispatch.BatchCapacityView;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.PriorityOrdering;

import java.util.Comparator;

/**
 * 从 NON_BATCH_V2 的 Prefill 队列读取负载视图。
 */
public final class QueuePrefillLoadProvider implements PrefillLoadProvider {

    private static final Comparator<GroupPlanner.Item> PRIORITY_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_ORDER =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq).thenComparing(GroupPlanner.Item::requestId);

    private final PrefillQueueRegistry queues;
    private final ConfigService configService;

    public QueuePrefillLoadProvider(PrefillQueueRegistry queues, ConfigService configService) {
        this.queues = queues;
        this.configService = configService;
    }

    /**
     * 这台 Prefill 还没有队列时返回空视图。NON_BATCH_V2 每次只 dispatch 一个请求，所以每组最多一个请求。
     */
    @Override
    public PrefillTtftEstimate.LoadView loadView(PrefillEndpoint endpoint) {
        FlexlbConfig config = configService.loadBalanceConfig();
        BatchCapacityView capacity = BatchCapacityView.of(endpoint, config);
        GroupPlanner.Constraints constraints =
                new GroupPlanner.Constraints(1, capacity.batchTokenCapacity(), capacity.batchKvCapacity(), 0L, 0L);
        Comparator<GroupPlanner.Item> itemOrder = config.isPriorityOrdering() ? PRIORITY_ORDER : FIFO_ORDER;
        PrefillQueue queue = queues.find(endpoint);
        if (queue == null) {
            queue = new PrefillQueue(endpoint, config.isPriorityOrdering());
        }
        return queue.loadView(System.currentTimeMillis(), itemOrder, constraints);
    }
}

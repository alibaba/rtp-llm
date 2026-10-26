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

    private final ConfigService configService;

    public QueuePrefillLoadProvider(ConfigService configService) {
        this.configService = configService;
    }

    /**
     * 读取节点持有的队列，最终选路通过节点 hold 检查其有效性。
     */
    @Override
    public PrefillTtftEstimate.LoadView loadView(PrefillEndpoint endpoint) {
        FlexlbConfig config = configService.loadBalanceConfig();
        BatchCapacityView capacity = BatchCapacityView.of(endpoint, config);
        GroupPlanner.Constraints constraints =
                new GroupPlanner.Constraints(1, capacity.batchTokenCapacity(), capacity.batchKvCapacity(), 0L, 0L);
        Comparator<GroupPlanner.Item> itemOrder = config.isPriorityOrdering() ? PRIORITY_ORDER : FIFO_ORDER;
        return endpoint.prefillQueue().loadView(System.currentTimeMillis(), itemOrder, constraints);
    }
}

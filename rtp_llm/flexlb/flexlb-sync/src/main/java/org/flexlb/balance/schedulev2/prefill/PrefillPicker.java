package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.PrefillInputs;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;

/**
 * 并行准备缓存匹配和请求自身耗时；调度线程结合最新排队与在途负载作最终选择。
 */
public final class PrefillPicker {

    private final DefaultRouter router;
    private final CostBasedPrefillStrategy prefillStrategy;
    private final PrefillQueueRegistry queues;

    public PrefillPicker(DefaultRouter router, CostBasedPrefillStrategy prefillStrategy, PrefillQueueRegistry queues) {
        this.router = router;
        this.prefillStrategy = prefillStrategy;
        this.queues = queues;
    }

    /**
     * 请求要去的机器 group，按流量策略算出；没配策略时为 null。
     */
    public String groupOf(BalanceContext context) {
        return router.resolvePolicyGroup(context);
    }

    /**
     * worker 准备材料，候选节点全部参与缓存匹配和自身耗时预测，不在这里确定最终节点。
     */
    public PrefillInputs prepare(BalanceContext context, RoleType role, String group) {
        return prefillStrategy.prepare(context, role, group);
    }

    public boolean inputsAreCurrent(PrefillInputs inputs, RoleType role, String group) {
        return prefillStrategy.inputsAreCurrent(inputs, role, group);
    }

    /**
     * 调度线程读取最新队列，选中后紧接着入队，下一次决策才能看到新增负载。
     */
    public PlacementResult<SelectedRole, RoleType> pick(Request request,
                                                        RoleType role,
                                                        String group,
                                                        PrefillInputs inputs) {
        return prefillStrategy.selectPrepared(request.ctx(), role, group, inputs, candidate -> {
            PrefillQueue queue = queues.find(candidate.endpoint());
            if (queue == null) {
                queue = new PrefillQueue(candidate.endpoint(), request.ctx().getConfig().isPriorityOrdering());
                queue.updateFromEngine();
            }
            return queue.estimatedTtftMs(request, candidate.predictedMs());
        });
    }
}

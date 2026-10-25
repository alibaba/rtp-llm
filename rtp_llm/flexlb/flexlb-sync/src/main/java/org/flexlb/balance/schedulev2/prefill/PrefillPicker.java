package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;

/**
 * 给请求选一台 Prefill，只看缓存命中和 TTFT 预估，不看这台机器排了多少、KV 还剩多少。
 */
public final class PrefillPicker {

    private final DefaultRouter router;
    private final CostBasedPrefillStrategy prefillStrategy;

    public PrefillPicker(DefaultRouter router, CostBasedPrefillStrategy prefillStrategy) {
        this.router = router;
        this.prefillStrategy = prefillStrategy;
    }

    /**
     * 请求要去的机器 group，按流量策略算出；没配策略时为 null。
     */
    public String groupOf(BalanceContext context) {
        return router.resolvePolicyGroup(context);
    }

    /**
     * 在 group 里的 role（PREFILL 或 PDFUSION）机器中选一台。成功时调用方负责关闭返回的 SelectedRole。
     */
    public PlacementResult<SelectedRole, RoleType> pick(BalanceContext context, RoleType role, String group) {
        return prefillStrategy.select(context, role, group);
    }
}

package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.domain.balance.BalanceContext;

/**
 * @author zjw
 * description:
 * date: 2025/4/20
 */
public interface Scheduler {

    MasterResponse select(BalanceContext balanceContext);
}

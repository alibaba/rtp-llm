package org.flexlb.balance.strategy;

import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.route.RoleType;

import java.util.List;

/**
 * Optional sub-interface for {@link LoadBalancer} implementations that support
 * batch worker selection.
 *
 * <p>Returns a list of worker targets sized exactly {@code count}, picked in
 * implementation-specific order. The endpoint contract guarantees the returned
 * order corresponds to the caller's positional consumption.
 *
 * <p>Implementations do <strong>not</strong> need to perform per-task
 * bookkeeping; batch path is fire-and-forget and reconciliation is delegated to
 * subsequent worker heartbeats. Return an empty list to signal "no alive
 * worker"; the caller will surface a role-specific error.
 */
public interface BatchLoadBalancer extends LoadBalancer {

    List<BatchScheduleTarget> selectBatch(int count, RoleType roleType, String group);
}

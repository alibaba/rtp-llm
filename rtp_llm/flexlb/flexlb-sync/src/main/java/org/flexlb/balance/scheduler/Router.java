package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

/**
 * 路由接口，负责根据负载均衡上下文选择合适的节点。
 * <p>
 * 该接口定义了负载均衡调度器的核心路由逻辑，
 * 根据当前的负载情况、缓存状态和调度策略，选择最优的节点进行请求处理。
 * </p>
 *
 * @author saichen.sm
 * @since 1.0
 */
public interface Router {

    /**
     * 根据负载均衡进行路由，选择合适的节点。
     *
     * @param balanceContext 负载均衡上下文，包含请求信息、可用节点列表等
     * @return Response 包含选中的节点信息的响应对象
     */
    Response route(BalanceContext balanceContext);

}

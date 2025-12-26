package org.flexlb.service;

import org.flexlb.balance.scheduler.RoutingQueueManager;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.BalanceContext;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicLong;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final RoutingQueueManager queueManager;

    private static final AtomicLong IncNum = new AtomicLong(1);

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        RoutingQueueManager queueManager) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
    }

    /**
     * 路由请求
     * @param balanceContext 负载均衡上下文
     * @return 路由结果
     */
    public MasterResponse route(BalanceContext balanceContext) {
        balanceContext.getSpan().addEvent("start selectEngineWorker");
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(whaleMasterConfig);
        long interRequestId = System.nanoTime() + (IncNum.getAndIncrement()) % 1000;
        balanceContext.setInterRequestId(interRequestId);

        MasterResponse result;
        if (whaleMasterConfig.isEnableQueueing()) {
            result = queueManager.tryRoute(balanceContext);  // 使用排队机制
        } else {
            result = router.route(balanceContext);           // 保持原有逻辑,直接路由
        }

        balanceContext.setMasterResponse(result);
        balanceContext.getSpan().addEvent("finish selectEngineWorker");
        return result;
    }

    /**
     * 取消指定的请求
     * @param balanceContext 负载均衡上下文
     */
    public void cancelRequest(BalanceContext balanceContext) {
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        if (whaleMasterConfig.isEnableQueueing()) {
            queueManager.cancelRequest(balanceContext);
        }
    }
}

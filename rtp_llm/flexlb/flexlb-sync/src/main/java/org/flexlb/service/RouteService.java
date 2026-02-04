package org.flexlb.service;

import java.util.concurrent.CancellationException;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
    }

    /**
     * 路由请求
     * @param balanceContext 负载均衡上下文
     * @return 路由结果
     */
    public Mono<Response> route(BalanceContext balanceContext) {
        balanceContext.getSpan().addEvent("start selectEngineWorker");
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(whaleMasterConfig);

        Mono<Response> resultMono;
        if (whaleMasterConfig.isEnableQueueing()) {
            resultMono = queueManager.tryRouteAsync(balanceContext);  // 使用异步排队机制
        } else {
            resultMono = Mono.fromCallable(() -> router.route(balanceContext));  // 保持原有逻辑,直接路由
        }

        return resultMono.doOnSuccess(result -> {
            balanceContext.setResponse(result);
            balanceContext.getSpan().addEvent("finish selectEngineWorker");
        });
    }

    /**
     * 取消指定的请求
     * @param balanceContext 负载均衡上下文
     */
    public void cancel(BalanceContext balanceContext) {
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        if (whaleMasterConfig.isEnableQueueing()) {
            balanceContext.cancel();
            balanceContext.getFuture().completeExceptionally(new CancellationException("Request cancelled by client"));
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }
}

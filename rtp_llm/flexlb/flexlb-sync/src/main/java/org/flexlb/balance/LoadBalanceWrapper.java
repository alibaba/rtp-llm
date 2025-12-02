package org.flexlb.balance;

import org.flexlb.balance.scheduler.DefaultScheduler;
import org.flexlb.balance.scheduler.Scheduler;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.service.config.ConfigService;
import org.flexlb.util.LoggingUtils;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author zjw
 * description:
 * date: 2025/3/16
 */
@Component
@DependsOn({"defaultScheduler"})
public class LoadBalanceWrapper {

    private final ConfigService configService;
    private final Scheduler roleScheduler;

    private static final AtomicLong IncNum = new AtomicLong(1);

    public LoadBalanceWrapper(ConfigService configService, DefaultScheduler defaultScheduler) {
        LoggingUtils.warn("do LoadBalanceWrapper init.");
        this.configService = configService;
        this.roleScheduler = defaultScheduler;
    }

    public MasterResponse selectEngineWorker(BalanceContext balanceContext) {
        balanceContext.getRequestContext().getSpan().addEvent("start selectEngineWorker");
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(whaleMasterConfig);
        long interRequestId = System.nanoTime() + (IncNum.getAndIncrement()) % 1000;
        balanceContext.setInterRequestId(interRequestId);
        MasterResponse result = roleScheduler.select(balanceContext);
        balanceContext.setMasterResponse(result);
        balanceContext.getRequestContext().getSpan().addEvent("finish selectEngineWorker");
        return result;
    }
}

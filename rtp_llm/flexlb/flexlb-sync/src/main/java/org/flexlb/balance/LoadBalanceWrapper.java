package org.flexlb.balance;

import java.util.concurrent.atomic.AtomicLong;

import org.flexlb.balance.scheduler.Scheduler;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.domain.balance.WhaleMasterConfig;
import org.flexlb.enums.ScheduleType;
import org.flexlb.service.config.ConfigService;
import org.flexlb.utils.LoggingUtils;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

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

    public LoadBalanceWrapper(ConfigService configService) {
        this.configService = configService;
        LoggingUtils.warn("do LoadBalanceWrapper init.");

        this.roleScheduler = SchedulerFactory.getScheduler(ScheduleType.DEFAULT);
        LoggingUtils.warn("LoadBalanceWrapper init success, prefillScheduler:{} lb.", roleScheduler.getClass().getName());
    }

    public MasterResponse selectEngineWorker(BalanceContext balanceContext) {
        MasterRequest masterRequest = balanceContext.getMasterRequest();
        LoggingUtils.info("do selectEngineWorker, masterReq model:{}", masterRequest.getModel());
        LoggingUtils.info("do selectEngineWorker, masterReq seqLen:{}", masterRequest.getSeqLen());
        balanceContext.getRequestContext().getSpan().addEvent("start selectEngineWorker");
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(whaleMasterConfig);
        balanceContext.setWorkerCalcParallel(Runtime.getRuntime().availableProcessors());
        long interRequestId = (System.currentTimeMillis()) * 1000 + (IncNum.getAndIncrement()) % 1000;
        balanceContext.setInterRequestId(interRequestId);
        LoggingUtils.info("go to roleScheduler.select for requestId: {}", interRequestId);
        MasterResponse result = roleScheduler.select(balanceContext);
        balanceContext.setMasterResponse(result);
        balanceContext.getRequestContext().getSpan().addEvent("finish selectEngineWorker");
        return result;
    }
}

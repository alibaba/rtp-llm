package org.flexlb.balance.scheduler;

import lombok.Getter;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.config.ConfigService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.LoggingUtils;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author zjw
 * description:
 * date: 2025/4/20
 */
@Component("defaultScheduler")
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy"})
public class DefaultScheduler implements Scheduler {

    @Getter
    private final LoadBalancer prefillLoadBalancer;
    private final LoadBalancer decodeLoadBalancer;
    private final LoadBalancer vitLoadBalancer;
    private final LoadBalancer fusionLoadBalancer;

    public DefaultScheduler(ConfigService configService) {
        prefillLoadBalancer = LoadBalanceStrategyFactory.getLoadBalanceStrategy(
                configService.loadBalanceConfig().getLoadBalanceStrategy());
        decodeLoadBalancer = LoadBalanceStrategyFactory.getLoadBalanceStrategy(LoadBalanceStrategyEnum.WEIGHTED_CACHE);
        vitLoadBalancer = LoadBalanceStrategyFactory.getLoadBalanceStrategy(
                configService.loadBalanceConfig().getLoadBalanceStrategy());
        fusionLoadBalancer = LoadBalanceStrategyFactory.getLoadBalanceStrategy(
                configService.loadBalanceConfig().getLoadBalanceStrategy());
    }

    public MasterResponse select(BalanceContext balanceContext) {
        if (balanceContext.getMasterRequest() == null) {
            LoggingUtils.error("masterRequest is null");
            return MasterResponse.code(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
        }
        MasterRequest masterRequest = balanceContext.getMasterRequest();
        String modelName = masterRequest.getModel();

        Map<String/*modelName*/, ModelWorkerStatus> targetModelRoleWorkerStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP;
        if (MapUtils.isEmpty(targetModelRoleWorkerStatusMap)) {
            LoggingUtils.error("targetModelRoleWorkerStatusMap is empty");
            return MasterResponse.code(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
        }
        if (targetModelRoleWorkerStatusMap.get(modelName) == null) {
            LoggingUtils.error("targetModelRoleWorkerStatusMap has no key named {}", modelName);
            return MasterResponse.code(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
        }
        ModelWorkerStatus workerStatus = targetModelRoleWorkerStatusMap.get(modelName);
        List<RoleType> roleTypeList = workerStatus.getRoleTypeList();
        List<ServerStatus> serverStatusList = new ArrayList<>();
        String group = null;
        MasterResponse masterResponse = new MasterResponse();
        long interRequestId = balanceContext.getInterRequestId();
        masterResponse.setInterRequestId(interRequestId);

        long startTimeInMs = System.nanoTime() / 1000;

        // INFO  暂时不支持既有prefill又有pd fusion的服务
        if (roleTypeList.contains(RoleType.PREFILL)) {
            ServerStatus prefillServerStatus = prefillLoadBalancer.select(balanceContext, RoleType.PREFILL, null);
            if (!prefillServerStatus.isSuccess()) {
                masterResponse.setSuccess(false);
                masterResponse.setCode(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode());
                masterResponse.setErrorCode(StrategyErrorType.NO_PREFILL_WORKER.getErrorMsg() + " : " + prefillServerStatus.getMessage());
                LoggingUtils.warn("cannot find prefill worker {}", prefillServerStatus.getMessage());
                return masterResponse;
            }
            String ip = prefillServerStatus.getServerIp();
            String port = String.valueOf(prefillServerStatus.getHttpPort());
            balanceContext.getRequestContext().getSpan().setAttribute("prefill_server_ip", prefillServerStatus.getServerIp());
            balanceContext.getRequestContext().getSpan().setAttribute("prefill_time", String.valueOf(prefillServerStatus.getPrefillTime()));
            serverStatusList.add(prefillServerStatus);
            group = prefillServerStatus.getGroup();
            ServerStatus decodeServerStatus = decodeLoadBalancer.select(balanceContext, RoleType.DECODE, group);
            if (!decodeServerStatus.isSuccess()) {
                prefillLoadBalancer.releaseLocalCache(modelName, ip + ":" + port, interRequestId);
                masterResponse.setSuccess(false);
                masterResponse.setCode(StrategyErrorType.NO_DECODE_WORKER.getErrorCode());
                masterResponse.setErrorCode(StrategyErrorType.NO_DECODE_WORKER.getErrorMsg() + " : " + decodeServerStatus.getMessage());
                LoggingUtils.warn("cannot find decode worker {}", decodeServerStatus.getMessage());
                return masterResponse;
            }
            balanceContext.getRequestContext().getSpan().setAttribute("decode_server_ip", decodeServerStatus.getServerIp());
            balanceContext.getRequestContext().getSpan().setAttribute("prefill_time", String.valueOf(decodeServerStatus.getPrefillTime()));
            serverStatusList.add(decodeServerStatus);

        } else if (roleTypeList.contains(RoleType.PDFUSION)) {
            ServerStatus fusionServerStatus = fusionLoadBalancer.select(balanceContext, RoleType.PDFUSION, null);
            group = fusionServerStatus.getGroup();
            masterResponse.setInterRequestId(fusionServerStatus.getInterRequestId());
            if (!fusionServerStatus.isSuccess()) {
                masterResponse.setSuccess(false);
                masterResponse.setCode(StrategyErrorType.NO_PDFUSION_WORKER.getErrorCode());
                masterResponse.setErrorCode(StrategyErrorType.NO_PDFUSION_WORKER.getErrorMsg() + " : " + fusionServerStatus.getMessage());
                LoggingUtils.warn("cannot find fusion worker {}", fusionServerStatus.getMessage());
                return masterResponse;
            }
            balanceContext.getRequestContext().getSpan().setAttribute("fusion_server_ip", fusionServerStatus.getServerIp());
            balanceContext.getRequestContext().getSpan().setAttribute("prefill_time", String.valueOf(fusionServerStatus.getPrefillTime()));
            serverStatusList.add(fusionServerStatus);
        }

        if (roleTypeList.contains(RoleType.VIT)) {
            LoggingUtils.info("vit roleTypeList contains");
            ServerStatus vitServerStatus = vitLoadBalancer.select(balanceContext, RoleType.VIT, group);
            if (!vitServerStatus.isSuccess()) {
                masterResponse.setSuccess(false);
                masterResponse.setCode(StrategyErrorType.NO_VIT_WORKER.getErrorCode());
                masterResponse.setErrorCode(StrategyErrorType.NO_VIT_WORKER.getErrorMsg() + " : " + vitServerStatus.getMessage());
                LoggingUtils.warn("cannot find vit worker {}", vitServerStatus.getMessage());
                return masterResponse;
            }
            balanceContext.getRequestContext().getSpan().setAttribute("vit_server_ip", vitServerStatus.getServerIp());
            balanceContext.getRequestContext().getSpan().setAttribute("prefill_time", String.valueOf(vitServerStatus.getPrefillTime()));
            serverStatusList.add(vitServerStatus);
        }

        long costTime = System.nanoTime() / 1000 - startTimeInMs;
        balanceContext.getRequestContext().getSpan().addEvent("load balance finish(" + costTime + "ms)");

        masterResponse.setServerStatus(serverStatusList);
        masterResponse.setSuccess(true);
        return masterResponse;
    }
}

package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

@Component("randomStrategy")
public class RandomStrategy implements LoadBalancer {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger(RandomStrategy.class);

    private final EngineWorkerStatus engineWorkerStatus;
    private final ConfigService configService;
    private final ResourceMeasureFactory resourceMeasureFactory;

    public RandomStrategy(EngineWorkerStatus engineWorkerStatus,
                          ConfigService configService,
                          ResourceMeasureFactory resourceMeasureFactory) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.configService = configService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, this);
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null);
        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(requestId);
        }
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        logger.debug("Selecting worker for , role: {}, group: {}", roleType, group);

        Map<String/*ip*/, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            logger.warn("No worker status map found");
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        List<WorkerStatus> workerStatuses = new ArrayList<>(workerStatusMap.values());
        if (CollectionUtils.isEmpty(workerStatuses)) {
            logger.warn("No available workers");
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Random select with wrap-around to skip dead workers, no extra allocation
        int size = workerStatuses.size();
        int startIndex = ThreadLocalRandom.current().nextInt(size);
        WorkerStatus selectedWorker = null;
        for (int i = 0; i < size; i++) {
            WorkerStatus ws = workerStatuses.get((startIndex + i) % size);
            if (isWorkerAvailable(balanceContext, roleType, ws)) {
                selectedWorker = ws;
                break;
            }
        }
        if (selectedWorker == null) {
            logger.warn("No serviceable workers available out of {} total workers", size);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        logger.debug("Selected worker ip: {}, httpPort: {}", selectedWorker.getIp(), selectedWorker.getPort());
        return buildServerStatus(selectedWorker, roleType, balanceContext.getRequestId(), request);
    }

    private boolean isWorkerAvailable(BalanceContext balanceContext, RoleType roleType, WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        FlexlbConfig config = balanceContext.getConfig() != null
                ? balanceContext.getConfig()
                : configService.loadBalanceConfig();
        ResourceMeasureIndicatorEnum indicator = config.getResourceMeasureIndicator(roleType);
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        return resourceMeasure == null || resourceMeasure.isResourceAvailable(workerStatus);
    }

    private ServerStatus buildServerStatus(WorkerStatus worker, RoleType roleType, long requestId, Request request) {
        ServerStatus result = new ServerStatus();
        try {
            if (RoleType.DECODE == roleType) {
                TaskInfo taskInfo = new TaskInfo();
                taskInfo.setRequestId(requestId);
                taskInfo.setInputLength(request == null ? 0 : request.getSeqLen());
                taskInfo.setPrefixLength(0);
                worker.putLocalTask(requestId, taskInfo);
            }
            result.setSuccess(true);
            result.setServerIp(worker.getIp());
            result.setHttpPort(worker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(worker.getPort()));
            result.setRole(roleType);
            result.setGroup(worker.getGroup());
            result.setRequestId(requestId);
        } catch (Exception e) {
            Logger.error("buildServerStatus error", e);
            result.setSuccess(false);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
        }
        return result;
    }
}

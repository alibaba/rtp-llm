package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
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

    public RandomStrategy(EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, this);
    }

    @Override
    public void rollBack(String modelName, String ipPort, String interRequestId) {
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        String modelName = request.getModel();
        logger.debug("Selecting worker for model: {}, role: {}, group: {}", modelName, roleType, group);

        Map<String/*ip*/, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, roleType, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            logger.warn("No worker status map found for model: {}", modelName);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        List<WorkerStatus> workerStatuses = new ArrayList<>(workerStatusMap.values());
        if (CollectionUtils.isEmpty(workerStatuses)) {
            logger.warn("No available workers for model: {}", modelName);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        int randomIndex = ThreadLocalRandom.current().nextInt(workerStatuses.size());
        WorkerStatus selectedWorker = workerStatuses.get(randomIndex);

        if (selectedWorker == null) {
            logger.error("Selected worker is null for model: {}", modelName);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        if (!selectedWorker.isAlive()) {
            logger.warn("Selected worker is not alive, ip: {}, model: {}", selectedWorker.getIp(), modelName);
        }

        logger.debug("Selected worker ip: {}, httpPort: {}, model: {}", selectedWorker.getIp(), selectedWorker.getPort(), modelName);
        return buildServerStatus(selectedWorker, roleType, balanceContext.getRequestId());
    }

    private ServerStatus buildServerStatus(WorkerStatus worker, RoleType roleType, String interRequestId) {
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setServerIp(worker.getIp());
            result.setHttpPort(worker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(worker.getPort()));
            result.setRole(roleType);
            result.setGroup(worker.getGroup());
            result.setInterRequestId(interRequestId);
        } catch (Exception e) {
            Logger.error("buildServerStatus error", e);
            result.setSuccess(false);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
        }
        return result;
    }
}

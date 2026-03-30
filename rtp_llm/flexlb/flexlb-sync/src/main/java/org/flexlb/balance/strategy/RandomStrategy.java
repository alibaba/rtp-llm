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
    public void rollBack(String ipPort, long requestId) {
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
            if (ws != null && ws.isAlive()) {
                selectedWorker = ws;
                break;
            }
        }
        if (selectedWorker == null) {
            logger.warn("No alive workers available out of {} total workers", size);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        logger.debug("Selected worker ip: {}, httpPort: {}", selectedWorker.getIp(), selectedWorker.getPort());
        return buildServerStatus(selectedWorker, roleType, balanceContext.getRequestId());
    }

    private ServerStatus buildServerStatus(WorkerStatus worker, RoleType roleType, long requestId) {
        ServerStatus result = new ServerStatus();
        try {
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

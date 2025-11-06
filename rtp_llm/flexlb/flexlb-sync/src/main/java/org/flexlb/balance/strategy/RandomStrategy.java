package org.flexlb.balance.strategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.LoadBalanceStrategyFactory;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * @author zjw
 * description:
 * date: 2025/3/20
 */
@Component("randomStrategy")

public class RandomStrategy implements LoadBalancer {
    private static final Logger logger = LoggerFactory.getLogger(RandomStrategy.class);

    private final EngineWorkerStatus engineWorkerStatus;

    public RandomStrategy(EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, this);
    }
    @Override
    public boolean releaseLocalCache(String modelName, String ip, Long interRequestId) {
        return true;
    }
    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        String modelName = balanceContext.getMasterRequest().getModel();
        logger.debug("Selecting worker for model: {}, role: {}, group: {}", modelName, roleType, group);

        Map<String/*ip*/, WorkerStatus> workerStatusMap =
                Optional.ofNullable(engineWorkerStatus.getModelRoleWorkerStatusMap().get(modelName))
                    .map(entry -> entry.getRoleStatusMap(roleType, group))
                        .orElse(null);

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
        
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setBatchId(UUID.randomUUID().toString());
        result.setServerIp(selectedWorker.getIp());
        result.setHttpPort(selectedWorker.getPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(selectedWorker.getPort()));
        result.setRole(roleType);
        result.setGroup(group);
        
        logger.debug("Selected worker ip: {}, httpPort: {}, model: {}", selectedWorker.getIp(), selectedWorker.getPort(), modelName);
        return result;
    }

}

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
import org.springframework.stereotype.Component;

/**
 * @author zjw
 * description:
 * date: 2025/3/20
 */
@Component("randomStrategy")

public class RandomStrategy implements LoadBalancer {

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

        Map<String/*ip*/, WorkerStatus> workerStatusMap =
                Optional.ofNullable(engineWorkerStatus.getModelRoleWorkerStatusMap().get(balanceContext.getMasterRequest().getModel()))
                        .map(ModelWorkerStatus::getPrefillStatusMap)
                        .orElse(null);

        if (MapUtils.isEmpty(workerStatusMap)) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        List<WorkerStatus> workerStatuses = new ArrayList<>(workerStatusMap.values());
        if (CollectionUtils.isEmpty(workerStatuses)) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        ThreadLocalRandom.current().nextInt(workerStatuses.size());
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setBatchId(UUID.randomUUID().toString());
        return result;
    }

}

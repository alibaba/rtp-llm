package org.flexlb.balance.strategy;

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
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;

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
    public void releaseLocalCache(String modelName, String ip, Long interRequestId) {
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
        int randomIndex = ThreadLocalRandom.current().nextInt(workerStatuses.size());
        WorkerStatus selectedWorker = workerStatuses.get(randomIndex);
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setServerIp(selectedWorker.getIp());
        result.setHttpPort(selectedWorker.getPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(selectedWorker.getPort()));
        result.setGroup(selectedWorker.getGroup());
        result.setInterRequestId(balanceContext.getInterRequestId());

        return result;
    }

}

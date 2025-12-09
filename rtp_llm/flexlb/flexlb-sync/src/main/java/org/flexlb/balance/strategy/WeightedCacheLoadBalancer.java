package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.LoadBalanceStrategyFactory;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.config.ConfigService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.LoggingUtils;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author zjw
 * description: 基于归一化缓存使用的加权随机负载均衡策略
 * 通过计算所有worker缓存使用的平均值，进行归一化处理后加权随机选择
 * date: 2025/3/21
 */
@Component("weightedCacheStrategy")
public class WeightedCacheLoadBalancer implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final double decayFactor;

    public WeightedCacheLoadBalancer(ConfigService configService, EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.decayFactor = configService.loadBalanceConfig().getWeightedCacheDecayFactor();
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, this);
    }

    private record WeightedWorker(WorkerStatus worker, long normalizedCacheUsed, double weight) {
    }

    @Override
    public void releaseLocalCache(String modelName, String ip, Long interRequestId) {
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        MasterRequest masterRequest = balanceContext.getMasterRequest();
        long seqLen = masterRequest.getSeqLen();
        String modelName = masterRequest.getModel();
        Map<String/*ip*/, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            LoggingUtils.warn("select ROLE: {} failed, workerStatusMap is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        List<WorkerStatus> workerStatusList = new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)
                .toList();
        if (CollectionUtils.isEmpty(workerStatusList)) {
            LoggingUtils.warn("select ROLE: {} failed, workerStatusList is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // 实现新的加权随机选择算法
        WorkerStatus selectedWorker = weightedRandomSelection(workerStatusList);

        if (selectedWorker != null) {
            long prefixLength = calcPrefixMatchLength(selectedWorker.getCacheStatus(), balanceContext.getMasterRequest().getBlockCacheKeys());
            // 更新本地任务状态
            return buildServerStatus(selectedWorker, seqLen, prefixLength, roleType, balanceContext.getInterRequestId());
        }

        // 如果没有找到合适的Worker，返回失败
        LoggingUtils.warn("选择Worker失败，没有找到合适的Worker");
        return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
    }

    private long calcPrefixMatchLength(CacheStatus cacheStatus, List<Long> promptCacheKeys) {

        if (cacheStatus == null || promptCacheKeys == null) {
            return 0;
        }
        long blockSize = cacheStatus.getBlockSize();
        Set<Long> cachePrefixHash = cacheStatus.getCachedKeys();
        if (cachePrefixHash == null) {
            return 0;
        }
        
        // 从前往后遍历，找到第一个不匹配的位置
        for (int index = 0; index < promptCacheKeys.size(); index++) {
            long hash = promptCacheKeys.get(index);
            if (!cachePrefixHash.contains(hash)) {
                // 返回匹配的前缀长度（匹配的block数量 * block大小）
                return blockSize * index;
            }
        }
        
        // 如果全部匹配，返回总长度
        return blockSize * promptCacheKeys.size();
    }

    /**
     * 加权随机选择算法：基于归一化cacheUsed进行加权随机选择
     *
     * @param candidateWorkers 候选Worker列表
     * @return 选择的WorkerStatus，如果没有合适的返回null
     */
    private WorkerStatus weightedRandomSelection(List<WorkerStatus> candidateWorkers) {
        int workerCount = candidateWorkers.size();
        if (workerCount == 0) {
            return null;
        }

        // 1. 计算cacheUsed的总和和平均值
        long totalCacheUsed = 0;
        for (WorkerStatus worker : candidateWorkers) {
            totalCacheUsed += worker.getUsedKvCacheTokens().get();
        }
        double avgCacheUsed = (double) totalCacheUsed / workerCount;

        // 2. 归一化cacheUsed并计算权重
        List<WeightedWorker> weightedWorkers = new ArrayList<>();
        boolean allSameUsage = true;
        double totalWeight = 0;

        // 检查所有worker cacheUsed是否相同
        for (WorkerStatus worker : candidateWorkers) {
            long cacheUsed = worker.getUsedKvCacheTokens().get();
            double normalizedValue = cacheUsed - avgCacheUsed;

            // 检查所有worker cacheUsed是否相同
            if (allSameUsage && !weightedWorkers.isEmpty()) {
                long firstCacheUsed = weightedWorkers.getFirst().worker.getUsedKvCacheTokens().get();
                if (cacheUsed != firstCacheUsed) {
                    allSameUsage = false;
                }
            }

            // 权重计算：使用指数衰减法，归一化值越小权重越大
            // 通过DECAY_FACTOR控制权重差异程度，避免极端权重比例
            double weight = Math.exp(-decayFactor * normalizedValue);

            weightedWorkers.add(new WeightedWorker(worker, (long) normalizedValue, weight));
            totalWeight += weight;
        }

        // 检查总权重是否有效
        if (totalWeight <= 0) {
            LoggingUtils.warn("总权重为0或负数: {}, 采用均匀随机选择", totalWeight);
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateWorkers.get(randomIndex);
        }

        // 如果所有worker的cacheUsed都相同，采用均匀随机
        if (allSameUsage) {
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateWorkers.get(randomIndex);
        }

        // 3. 轮盘赌算法进行加权随机选择
        double randomValue = ThreadLocalRandom.current().nextDouble() * totalWeight;
        double cumulativeWeight = 0;

        for (WeightedWorker weightedWorker : weightedWorkers) {
            cumulativeWeight += weightedWorker.weight;
            if (Double.compare(randomValue, cumulativeWeight) <= 0) {
                return weightedWorker.worker;
            }
        }

        // 作为兜底方案：选择cacheUsed最小的worker
        return weightedWorkers.stream()
                .min(Comparator.comparingLong(w -> w.worker.getUsedKvCacheTokens().get()))
                .map(w -> w.worker)
                .orElse(null);
    }

    private ServerStatus buildServerStatus(WorkerStatus optimalWorker, long seqLen, long prefixLength, RoleType roleType, long interRequestId) {
        ServerStatus result = new ServerStatus();
        try {
            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setPrefillTime(0);
            taskInfo.setWaitingTime(0);
            taskInfo.setInputLength(seqLen);
            taskInfo.setPrefixLength(prefixLength);
            taskInfo.setInterRequestId(interRequestId);

            // 本地增量更新KcCache Tokens
            long needNewKvCacheLen = seqLen - prefixLength;
            optimalWorker.decKvCacheFree(needNewKvCacheLen);
            optimalWorker.addKvCacheUsed(needNewKvCacheLen);

            optimalWorker.putLocalTask(interRequestId, taskInfo);

            result.setSuccess(true);
            result.setRole(roleType);
            result.setServerIp(optimalWorker.getIp());
            result.setHttpPort(optimalWorker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(optimalWorker.getPort()));
            result.setGroup(optimalWorker.getGroup());
            result.setInterRequestId(interRequestId);
        } catch (Exception e) {
            LoggingUtils.error("buildServerStatus error", e);
            result.setSuccess(false);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
        }
        return result;
    }
}

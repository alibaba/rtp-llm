package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author saichen.sm
 * description: 基于归一化缓存使用的加权随机负载均衡策略
 * 通过计算所有worker缓存使用的平均值，进行归一化处理后加权随机选择
 * date: 2025/3/21
 */
@Component("weightedCacheStrategy")
public class WeightedCacheLoadBalancer implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final double decayFactor;
    private final ResourceMeasureFactory resourceMeasureFactory;

    public WeightedCacheLoadBalancer(ConfigService configService,
                                     EngineWorkerStatus engineWorkerStatus,
                                     ResourceMeasureFactory resourceMeasureFactory) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.decayFactor = configService.loadBalanceConfig().getWeightedCacheDecayFactor();
        this.resourceMeasureFactory = resourceMeasureFactory;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, this);
    }

    private record WeightedWorker(WorkerStatus worker, long normalizedCacheUsed, double weight) {
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        Map<String/*ip*/, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            Logger.warn("select ROLE: {} failed, workerStatusMap is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(roleType.getResourceMeasureIndicator());
        List<WorkerStatus> workerStatusList = new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)                   // 校验资源是否可用
                .filter(resourceMeasure::isResourceAvailable)    // 校验worker是否有可用资源
                .toList();
        if (CollectionUtils.isEmpty(workerStatusList)) {
            Logger.warn("select ROLE: {} failed, workerStatusList is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // 实现新的加权随机选择算法
        WorkerStatus selectedWorker = weightedRandomSelection(workerStatusList);

        if (selectedWorker != null) {
            long prefixLength = calcPrefixMatchLength(selectedWorker.getCacheStatus(), balanceContext.getRequest().getBlockCacheKeys());
            // 更新本地任务状态
            return buildServerStatus(selectedWorker, seqLen, prefixLength, roleType, balanceContext.getRequestId());
        }

        // 如果没有找到合适的Worker，返回失败
        Logger.warn("选择Worker失败，没有找到合适的Worker");
        return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
    }

    /**
     * 释放指定Worker上的本地缓存任务
     *
     * @param ipPort Worker IP地址
     * @param interRequestId 内部请求ID
     */
    @Override
    public void rollBack(String ipPort, String interRequestId) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null);
        Logger.debug("Decode rollBack - ip: {}, interRequestId: {}",
                ipPort, interRequestId);

        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(interRequestId);
        }
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
        Long firstCacheUsed = null;

        for (WorkerStatus worker : candidateWorkers) {
            long cacheUsed = worker.getUsedKvCacheTokens().get();
            double normalizedValue = cacheUsed - avgCacheUsed;

            if (firstCacheUsed == null) {
                firstCacheUsed = cacheUsed;
            } else if (cacheUsed != firstCacheUsed) {
                allSameUsage = false;
            }

            double weight = Math.exp(-decayFactor * normalizedValue);

            weightedWorkers.add(new WeightedWorker(worker, (long) normalizedValue, weight));
            totalWeight += weight;
        }

        // 检查总权重是否有效
        if (totalWeight <= 0) {
            Logger.warn("总权重为0或负数: {}, 采用均匀随机选择", totalWeight);
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

    private ServerStatus buildServerStatus(WorkerStatus optimalWorker, long seqLen, long prefixLength, RoleType roleType, String interRequestId) {
        ServerStatus result = new ServerStatus();
        try {
            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setPrefillTime(0);
            taskInfo.setWaitingTime(0);
            taskInfo.setInputLength(seqLen);
            taskInfo.setPrefixLength(prefixLength);
            taskInfo.setInterRequestId(interRequestId);

            // 更新本地任务状态
            optimalWorker.putLocalTask(interRequestId, taskInfo);

            result.setSuccess(true);
            result.setRole(roleType);
            result.setServerIp(optimalWorker.getIp());
            result.setHttpPort(optimalWorker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(optimalWorker.getPort()));
            result.setGroup(optimalWorker.getGroup());
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

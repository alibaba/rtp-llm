package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 基于最短首Token时间(TTFT)的负载均衡策略
 *
 * <p>该策略通过综合考虑以下因素选择最优Worker：
 * 1. KV-Cache命中率：优先选择缓存命中率高的Worker
 * 2. 排队时间：考虑Worker当前的任务队列情况
 * 3. 调度公平性：在性能相近的Worker间实现负载均衡
 *
 * @author saichen.sm
 * @since 2025/3/10
 */
@Component("shortestTTFTStrategy")
public class ShortestTTFTStrategy implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;

    private static final int MIN_CANDIDATE_COUNT = 1;
    private static final double CANDIDATE_PERCENTAGE = 0.3;
    private static final double TTFT_THRESHOLD_PERCENTAGE = 0.1;
    private static final double STDDEV_THRESHOLD_FACTOR = 0.5;

    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                EngineHealthReporter engineHealthReporter,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, this);
    }

    /**
     * 选择最优Worker执行任务
     *
     * @param balanceContext 负载均衡上下文
     * @param roleType Worker角色类型
     * @param group Worker分组
     * @return 选中的服务器状态
     */
    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("Failed to select worker", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    /**
     * 释放指定Worker上的本地缓存任务
     *
     * @param modelName 模型名称
     * @param ipPort Worker IP地址
     * @param interRequestId 内部请求ID
     */
    @Override
    public void rollBack(String modelName, String ipPort, String interRequestId) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, RoleType.PREFILL, null);
        Logger.debug("Prefill rollBack - ipPort: {}, interRequestId: {}", modelName, ipPort, interRequestId);

        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(interRequestId);
        }
    }

    /**
     * 执行Worker选择的核心逻辑
     *
     * @param balanceContext 负载均衡上下文
     * @param roleType Worker角色类型
     * @param group Worker分组
     * @return 选中的服务器状态
     */
    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        String interRequestId = balanceContext.getRequestId();
        String modelName = balanceContext.getRequest().getModel();
        long seqLen = balanceContext.getRequest().getSeqLen();

        Logger.debug("Starting shortest TTFT selection for model: {}, role: {}", modelName, roleType);

        // 获取可用的Worker列表
        List<WorkerStatus> availableWorkers = getAvailableWorkers(modelName, roleType, group);
        if (CollectionUtils.isEmpty(availableWorkers)) {
            Logger.warn("No available workers for role: {}", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // 计算每个引擎的缓存匹配结果
        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, modelName, roleType, group);

        List<ScoredWorker> scoredWorkers = scoreWorkers(availableWorkers, cacheMatchResults, seqLen);

        ScoredWorker bestWorker = selectBestWorker(scoredWorkers);
        if (bestWorker == null) {
            Logger.warn("Failed to find best worker for role: {}", roleType);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        return finalizeWorkerSelection(bestWorker, balanceContext, roleType, interRequestId, seqLen);
    }

    /**
     * 获取可用的Worker列表
     *
     * @param modelName 模型名称
     * @param roleType Worker角色类型
     * @param group Worker分组
     * @return 可用的Worker列表
     */
    private List<WorkerStatus> getAvailableWorkers(String modelName, RoleType roleType, String group) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            return new ArrayList<>();
        }

        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(roleType.getResourceMeasureIndicator());

        return new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)                   // 校验资源是否可用
                .filter(resourceMeasure::isResourceAvailable)    // 校验worker是否有可用资源
                .toList();
    }

    /**
     * 获取缓存匹配结果
     *
     * @param balanceContext 负载均衡上下文
     * @param modelName 模型名称
     * @param roleType Worker角色类型
     * @param group Worker分组
     * @return 缓存匹配结果: key: engineIpPort，value: prefixMatchLength
     */
    private Map<String /*engineIpPort*/, Integer /*prefixMatchLength*/> getCacheMatchResults(BalanceContext balanceContext,
                                                                                             String modelName,
                                                                                             RoleType roleType,
                                                                                             String group) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(blockCacheKeys, modelName, roleType, group);
    }

    /**
     * 为所有存活的Worker计算TTFT评分
     *
     * @param workers Worker列表
     * @param cacheMatchResults 缓存匹配结果
     * @param seqLen 序列长度
     * @return 已评分的Worker列表
     */
    private List<ScoredWorker> scoreWorkers(List<WorkerStatus> workers, Map<String, Integer> cacheMatchResults, long seqLen) {
        return workers.stream()
                .filter(WorkerStatus::isAlive)
                .map(workerStatus -> {
                    long hitCacheTokens = calculatePrefixMatchLength(workerStatus, cacheMatchResults);
                    long prefillTime = TaskInfo.estimatePrefillTimeMs(seqLen, hitCacheTokens);
                    long queueTime = workerStatus.getRunningQueueTime().get();
                    long newTTFT = prefillTime + queueTime;
                    long lastSelectedTime = workerStatus.getLastSelectedTime().get();
                    Logger.debug("Calculate TTFT for worker - ip: {}, port: {}, hitCacheTokens: {}, prefillTime: {}, queueTime: {}, newTTFT: {}",
                            workerStatus.getIp(),
                            workerStatus.getPort(),
                            hitCacheTokens,
                            prefillTime,
                            queueTime,
                            newTTFT);
                    return new ScoredWorker(workerStatus, newTTFT, hitCacheTokens, lastSelectedTime);
                })
                .collect(Collectors.toList());
    }

    /**
     * 完成Worker选择并更新状态
     *
     * @param selectedWorker 已选Worker
     * @param balanceContext 负载均衡上下文
     * @param roleType Worker角色类型
     * @param interRequestId 内部请求ID
     * @param seqLen 序列长度
     * @return 服务器状态
     */
    private ServerStatus finalizeWorkerSelection(ScoredWorker selectedWorker,
                                                 BalanceContext balanceContext,
                                                 RoleType roleType,
                                                 String interRequestId,
                                                 long seqLen) {
        WorkerStatus workerStatus = selectedWorker.worker();

        logWorkerSelection(selectedWorker, roleType);
        reportCacheHitMetrics(balanceContext.getRequest().getModel(), roleType, workerStatus.getIp(), selectedWorker.hitCacheTokens(), seqLen);

        TaskInfo task = createTaskInfo(interRequestId, balanceContext.getRequest().getSeqLen(), selectedWorker.hitCacheTokens());
        workerStatus.putLocalTask(interRequestId, task);

        return buildServerStatus(selectedWorker, roleType, interRequestId);
    }

    /**
     * 记录Worker选择日志
     *
     * @param selectedWorker 已选Worker
     * @param roleType Worker角色类型
     */
    private void logWorkerSelection(ScoredWorker selectedWorker, RoleType roleType) {
        WorkerStatus workerStatus = selectedWorker.worker();
        Logger.debug("Selected {} worker - ip: {}, port: {}, hitCacheTokens: {}, ttft: {}",
                roleType,
                workerStatus.getIp(),
                workerStatus.getPort(),
                selectedWorker.hitCacheTokens(),
                selectedWorker.ttft());
    }

    /**
     * 上报缓存命中指标
     *
     * @param modelName 模型名称
     * @param roleType Worker角色类型
     * @param ip Worker IP地址
     * @param hitCacheTokens 命中的缓存Token数量
     * @param seqLen 序列长度
     */
    private void reportCacheHitMetrics(String modelName, RoleType roleType, String ip, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(modelName, roleType, ip, hitCacheTokens, hitRate);
    }

    /**
     * 创建任务信息
     *
     * @param interRequestId 内部请求ID
     * @param inputLength 输入长度
     * @param prefixLength 前缀长度
     * @return 任务信息
     */
    private TaskInfo createTaskInfo(String interRequestId, long inputLength, long prefixLength) {
        TaskInfo task = new TaskInfo();
        task.setInterRequestId(interRequestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        return task;
    }

    /**
     * 选择最佳Worker，综合考虑TTFT和调度公平性
     *
     * <p>算法流程： 1. 按TTFT对所有Worker排序 2. 选择前30%的Worker作为候选者（至少1个） 3. 在TTFT相近的候选者中，优先选择最近未被调度的Worker
     *
     * @param scoredWorkers 已评分的Worker列表
     * @return 最佳Worker
     */
    private ScoredWorker selectBestWorker(List<ScoredWorker> scoredWorkers) {
        if (scoredWorkers.isEmpty()) {
            return null;
        }

        List<ScoredWorker> sortedWorkers = sortByTTFT(scoredWorkers);
        List<ScoredWorker> candidates = selectTopCandidates(sortedWorkers);
        Logger.debug("Select best worker, sortedWorkers size: {}, candidates size: {}", sortedWorkers.size(), candidates.size());

        if (candidates.isEmpty()) {
            return null;
        }

        long minTTFT = candidates.getFirst().ttft();
        double threshold = calculateTTFTThreshold(candidates);

        List<ScoredWorker> similarWorkers = filterSimilarWorkers(candidates, minTTFT, threshold);

        return selectWorkerByScheduleFairness(similarWorkers, candidates);
    }

    /**
     * 按TTFT对Worker排序
     *
     * @param workers Worker列表
     * @return 排序后的Worker列表 从小到大
     */
    private List<ScoredWorker> sortByTTFT(List<ScoredWorker> workers) {
        // 二级排序
        // 1. 第一级排序：按 ttft（首Token时间）从小到大排序
        // 2. 第二级排序：当 ttft 相等时，按 lastSelectedTime（最后选择时间）从小到大排序
        return workers.stream()
                .sorted(Comparator.comparingLong(ScoredWorker::ttft)
                        .thenComparingLong(ScoredWorker::lastSelectedTime))
                .toList();
    }

    /**
     * 选择前N个候选Worker
     *
     * @param sortedWorkers 已排序的Worker列表
     * @return 候选Worker列表
     */
    private List<ScoredWorker> selectTopCandidates(List<ScoredWorker> sortedWorkers) {
        int candidateCount = Math.max(MIN_CANDIDATE_COUNT, (int) (sortedWorkers.size() * CANDIDATE_PERCENTAGE));
        return sortedWorkers.stream().limit(candidateCount).toList();
    }

    /**
     * 计算TTFT相似度阈值
     *
     * @param candidates 候选Worker列表
     * @return TTFT阈值
     */
    private double calculateTTFTThreshold(List<ScoredWorker> candidates) {
        double avgTTFT = candidates.stream().mapToLong(ScoredWorker::ttft).average().orElse(0.0);

        double stdDev = Math.sqrt(
                candidates.stream()
                        .mapToLong(ScoredWorker::ttft)
                        .mapToDouble(v -> Math.pow(v - avgTTFT, 2))
                        .average()
                        .orElse(0.0));
        double percentageAvgTTFT = avgTTFT * TTFT_THRESHOLD_PERCENTAGE;
        double factoredStdDev = stdDev * STDDEV_THRESHOLD_FACTOR;
        Logger.debug("Calculate TTFT threshold, avgTTFT: {}, stdDev: {}, percentageAvgTTFT: {}, factoredStdDev: {}", avgTTFT, stdDev, percentageAvgTTFT, factoredStdDev);
        return Math.max(percentageAvgTTFT, factoredStdDev);
    }

    /**
     * 筛选TTFT相近的Worker
     *
     * @param candidates 候选Worker列表
     * @param minTTFT 最小TTFT值
     * @param threshold 阈值
     * @return TTFT相近的Worker列表
     */
    private List<ScoredWorker> filterSimilarWorkers(List<ScoredWorker> candidates, long minTTFT, double threshold) {
        List<ScoredWorker> scoredWorkers = candidates.stream()
                .filter(worker -> Math.abs(worker.ttft() - minTTFT) <= threshold)
                .toList();
        Logger.debug("Filter similar workers, minTTFT: {}, threshold: {}, candidates size: {}", minTTFT, threshold, scoredWorkers.size());
        return scoredWorkers;
    }

    /**
     * 根据调度公平性选择Worker
     * 在TTFT相近的Worker中，优先选择最近未被调度的Worker
     *
     * @param similarWorkers TTFT相近的Worker列表
     * @param fallbackCandidates 候补Worker列表
     * @return 最终选择的Worker
     */
    private ScoredWorker selectWorkerByScheduleFairness(List<ScoredWorker> similarWorkers, List<ScoredWorker> fallbackCandidates) {
        if (similarWorkers.isEmpty()) {
            return fallbackCandidates.getFirst();
        }

        return similarWorkers.stream()
                // 优先选择最近未被调度的Worker
                .min(Comparator.comparingLong(ScoredWorker::lastSelectedTime))
                .orElse(fallbackCandidates.getFirst());
    }

    /**
     * 构建服务器状态响应
     *
     * @param selectedWorker 已选Worker
     * @param roleType Worker角色类型
     * @param interRequestId 内部请求ID
     * @return 服务器状态
     */
    private ServerStatus buildServerStatus(ScoredWorker selectedWorker, RoleType roleType, String interRequestId) {
        WorkerStatus workerStatus = selectedWorker.worker();
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setRole(roleType);
            result.setInterRequestId(interRequestId);
            result.setPrefillTime(selectedWorker.ttft());
            result.setGroup(workerStatus.getGroup());
            result.setServerIp(workerStatus.getIp());
            result.setHttpPort(workerStatus.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(workerStatus.getPort()));
        } catch (Exception e) {
            Logger.error("Failed to build server status for requestId: {}", interRequestId, e);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
            result.setSuccess(false);
        }
        return result;
    }

    /**
     * 计算前缀匹配长度（缓存命中的Token数量）
     *
     * @param workerStatus Worker状态
     * @param cacheMatchResults 缓存匹配结果
     * @return 命中的Token数量
     */
    private long calculatePrefixMatchLength(WorkerStatus workerStatus, Map<String, Integer> cacheMatchResults) {
        if (workerStatus.getCacheStatus() == null || cacheMatchResults == null) {
            return 0L;
        }

        Integer prefixMatchLength = cacheMatchResults.get(workerStatus.getIpPort());
        if (prefixMatchLength == null) {
            return 0L;
        }

        long blockSize = workerStatus.getCacheStatus().getBlockSize();
        return blockSize * prefixMatchLength;
    }
}

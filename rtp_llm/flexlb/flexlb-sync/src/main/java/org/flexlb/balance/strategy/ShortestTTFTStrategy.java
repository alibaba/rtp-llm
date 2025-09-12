package org.flexlb.balance.strategy;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.AccessLevel;
import lombok.Getter;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.LoadBalanceStrategyFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.domain.batch.RequestBatchMeta;
import org.flexlb.domain.monitor.SelectMonitorContext;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.domain.worker.WorkerTTFT;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.utils.LoggingUtils;
import org.springframework.stereotype.Component;
import reactor.core.scheduler.Schedulers;

/**
 * @author zjw
 * description:
 * date: 2025/3/10
 */
@Component("shortestTTFTStrategy")
public class ShortestTTFTStrategy implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;

    private final EngineHealthReporter engineHealthReporter;
    
    private final CacheAwareService cacheAwareService;

    // trigger by expireEviction or qps checking
    @Getter(AccessLevel.PUBLIC)
    private Cache<String, RequestBatchMeta> batchMetaCache;

    // TODO optimize this workerStatusUpdateTs
    @Getter(AccessLevel.PUBLIC)
    private Cache<String, Long> workerStatusUpdateTs;

    @Getter(AccessLevel.PUBLIC)
    private Cache<String, List<Long>> IdListCache;

    @Getter(AccessLevel.PUBLIC)
    private Cache<Long, RequestBatchMeta> RequestCache;

    private static final int TOP_K = 1;

    public ShortestTTFTStrategy(
                                EngineWorkerStatus engineWorkerStatus,
                                EngineHealthReporter engineHealthReporter,
                                CacheAwareService cacheAwareService) {

        this.engineWorkerStatus = engineWorkerStatus;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheAwareService = cacheAwareService;
        init();
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, this);
    }

    private void init() {
        ExecutorService executorService = new ThreadPoolExecutor(200, 500, 60L, TimeUnit.SECONDS,
            new SynchronousQueue<>(), new NamedThreadFactory("worker-match-thread"),
            new ThreadPoolExecutor.CallerRunsPolicy());
        Schedulers.fromExecutorService(executorService);
        batchMetaCache = Caffeine.newBuilder()
                .maximumSize(10 * 1000)
                .expireAfterWrite(120, TimeUnit.SECONDS)
                .build();
        workerStatusUpdateTs = Caffeine.newBuilder()
                .maximumSize(2000)
                .expireAfterWrite(5, TimeUnit.SECONDS)
                .build();
        IdListCache = Caffeine.newBuilder()
                .maximumSize(10 * 1000)
                .expireAfterWrite(120, TimeUnit.SECONDS)
                .build();
    }

    public boolean releaseLocalCache(String modelName, String ip, Long interRequestId) {
        Map<String/*ip*/, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, RoleType.PREFILL, null);
        LoggingUtils.debug("releaseLocalCache, ip: {}, interRequestId: {}", modelName, ip, interRequestId);
        LoggingUtils.debug(workerStatusMap.keySet().stream().collect(Collectors.joining(",")));
        WorkerStatus workerStatus = workerStatusMap.get(ip);
        workerStatus.removeLocalTask(interRequestId);
        return true;
    }

    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        SelectMonitorContext monitorContext = new SelectMonitorContext();

        monitorContext.setStartTime(System.currentTimeMillis());
        Exception exception = null;
        try {
            ServerStatus result = select0(balanceContext, monitorContext, roleType, group);
            if (!result.isSuccess()) {
                monitorContext.markError(result.getMessage());
            }
            return result;
        } catch (Exception e) {
            exception = e;
            LoggingUtils.warn("select worker error", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        } finally {
            if (exception != null && monitorContext.getErrorCode() == null) {
                monitorContext.setErrorCode("LB_UNKNOWN_ERROR");
            }
            monitorContext.setTotalCost(System.currentTimeMillis() - monitorContext.getStartTime());
            engineHealthReporter.reportPrefillBalanceSelectMetric(balanceContext.getMasterRequest().getModel(),
                    monitorContext.getErrorCode() == null,
                    monitorContext.getErrorCode(),
                    monitorContext.getTotalCost(),
                    monitorContext.getTokenizeEndTime() - monitorContext.getTokenizeStartTime(),
                    monitorContext.getCalcPrefixEndTime() - monitorContext.getCalcPrefixStartTime(),
                    monitorContext.getCalcTTFTEndTime() - monitorContext.getCalcTTFTStartTime());
        }
    }

    private ServerStatus select0(BalanceContext balanceContext, SelectMonitorContext monitorContext, RoleType roleType, String group) {

        long interRequestId = balanceContext.getInterRequestId();
        String modelName = balanceContext.getMasterRequest().getModel();
        long seqLen = balanceContext.getMasterRequest().getSeqLen();

        LoggingUtils.debug("do shortest ttft select");

        monitorContext.setStartTime(System.currentTimeMillis());

        Map<String/*ip*/, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, roleType, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            LoggingUtils.warn("select ROLE: {} failed, workerStatusMap is empty");
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        List<WorkerStatus> workerStatusList = new ArrayList<>(workerStatusMap.values());
        if (CollectionUtils.isEmpty(workerStatusList)) {
            LoggingUtils.warn("select ROLE: {} failed, workerStatusList is empty");
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        monitorContext.setCalcTTFTStartTime(System.currentTimeMillis());

        // 调用 Local KV-Cache 匹配引擎
        List<Long> blockCacheKeys = balanceContext.getMasterRequest().getBlockCacheKeys();
        Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> matchResultsMap =
            cacheAwareService.findMatchingEngines(blockCacheKeys, modelName, roleType, group);

        // 使用全局锁方式选择最优Worker，避免复杂的并发控制
        synchronized (ShortestTTFTStrategy.class) {
            // 1. 计算所有存活Worker的TTFT
            List<ScoredWorker> scoredWorkers = workerStatusList.stream()
                    .filter(WorkerStatus::isAlive) // 只考虑存活的Worker
                    .map(workerStatus -> {
                        WorkerTTFT ttft = calcWorkerTTFT(workerStatus, matchResultsMap, seqLen);
                        return new ScoredWorker(workerStatus, ttft);
                    })
                    .collect(Collectors.toList());

            // 2. 按TTFT排序，选择最优的Worker
            Optional<ScoredWorker> bestWorker = findBestWorkerWithFreshness(scoredWorkers);

            if (!bestWorker.isPresent()) {
                LoggingUtils.warn("select ROLE: {} failed, bestWorker is empty");
                return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
            }
            LocalDateTime now = LocalDateTime.now();
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String formattedDateTime = now.format(formatter);
            LoggingUtils.debug("");
            LoggingUtils.debug("select time: {}, inter req id : {}", formattedDateTime, interRequestId);
            for(WorkerStatus worker : workerStatusList) {
                LoggingUtils.debug("worker: {}", worker.getIp());
                for (TaskInfo task : worker.getRunningTaskList()) {
                    LoggingUtils.debug("remote task inter req id : {}, input len: {}, prefix len: {}", task.getInterRequestId(), task.getInputLength(), task.getPrefixLength());
                }
                for (TaskInfo task : worker.getLocalTaskMap().values()) {
                    LoggingUtils.debug("local task inter req id : {}, input len: {}, prefix len: {}", task.getInterRequestId(), task.getInputLength(), task.getPrefixLength());
                }
            }
            // 3. 选择最优Worker并更新状态
            ScoredWorker selectedWorker = bestWorker.get();
            WorkerStatus workerStatus = selectedWorker.worker();
            LoggingUtils.debug("selected worker ip: {}", workerStatus.getIp());
            engineHealthReporter.reportCacheHitMetrics(modelName, roleType, workerStatus.getIp(),
                selectedWorker.getWorkerTTFT().getHitCacheTokens(),
                selectedWorker.getWorkerTTFT().getHitCacheTokens() / (double) seqLen);

            TaskInfo task = new TaskInfo();
            task.setEnqueueTimeMs(System.currentTimeMillis());
            task.setInterRequestId(interRequestId);
            task.setInputLength(balanceContext.getMasterRequest().getSeqLen());
            task.setPrefixLength(selectedWorker.getWorkerTTFT().getHitCacheTokens());
            LoggingUtils.debug("成功选择{} Worker, ip:{}, port: {}, prefill_time: {}, ttft:{}",
                    roleType.toString(), workerStatus.getIp(), workerStatus.getPort(),
                    selectedWorker.getWorkerTTFT().getPrefillTime(),
                    selectedWorker.ttft()
                    );
            workerStatus.putLocalTask(interRequestId, task);
            
            return buildServerStatus(workerStatus, bestWorker.get().getWorkerTTFT(), roleType, interRequestId);
        }
    }

    private Optional<ScoredWorker> findBestWorkerWithFreshness(List<ScoredWorker> scoredWorkers) {
        if (scoredWorkers.isEmpty()) {
            return Optional.empty();
        }

        // 先按TTFT排序
        List<ScoredWorker> sortedWorkers = scoredWorkers.stream()
                .sorted(Comparator.comparingLong(ScoredWorker::ttft))
                .collect(Collectors.toList());

        // 选择前30%或至少5个Worker作为候选者
        int candidateCount = Math.max(TOP_K, (int) (sortedWorkers.size() * 0.3));
        List<ScoredWorker> candidates = sortedWorkers.stream()
                .limit(candidateCount)
                .collect(Collectors.toList());

        // 找到候选者中的最小TTFT
        long minTTFT = candidates.get(0).ttft();

        // 计算候选者的TTFT平均值和标准差
        double avgTTFT = candidates.stream()
                .mapToLong(ScoredWorker::ttft)
                .average()
                .orElse(0.0);

        double stdDev = Math.sqrt(
                candidates.stream()
                        .mapToLong(ScoredWorker::ttft)
                        .mapToDouble(v -> Math.pow(v - avgTTFT, 2))
                        .average()
                        .orElse(0.0)
        );

        // 定义"相近"阈值为平均值的10%或标准差，取较大者
        double threshold = Math.max(avgTTFT * 0.1, stdDev * 0.5);

        // 筛选出TTFT相近的Worker
        List<ScoredWorker> similarWorkers = candidates.stream()
                .filter(worker -> Math.abs(worker.ttft() - minTTFT) <= threshold)
                .collect(Collectors.toList());

        if (similarWorkers.isEmpty()) {
            // 如果没有相近的Worker，直接选择TTFT最小的
            return Optional.of(candidates.get(0));
        } else {
            // 在相近的Worker中，优先选择最近没有被调度的（lastScheduleTime较小的）
            // 使用正值使最近没有被调度的Worker优先级更高
            // 如果找不到最近没有被调度的Worker，则直接返回第一个
            return similarWorkers.stream()
                    .min(Comparator.comparingLong(worker -> {
                        // 使用正值使最近没有被调度的Worker优先级更高
                        return worker.worker().getLastScheduleTime().get();
                    }));
        }
    }

    private ServerStatus buildServerStatus(WorkerStatus workerStatus, WorkerTTFT workerTTFT, RoleType roleType, long interRequestId) {
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setRole(roleType);
            result.setInterRequestId(interRequestId);
            result.setPrefillTime(workerTTFT.getTtft());
            result.setGroup(workerStatus.getGroup());
            result.setServerIp(workerStatus.getIp());
            result.setHttpPort(workerStatus.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(workerStatus.getPort()));
        } catch (Exception e) {
            LoggingUtils.error("buildServerStatus error, requestId:{} ", interRequestId, e);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
            result.setSuccess(false);
        }
        return result;
    }

    private WorkerTTFT calcWorkerTTFT(WorkerStatus workerStatus, Map<String,Integer> matchResultsMap, long seqLength) {

        // 获取最长前缀匹配长度
        long hitCacheTokens = calcPrefixMatchLength(workerStatus, matchResultsMap);

        // 预估执行Prefill的时间
        long prefillTime = TaskInfo.estimatePrefillTimeMs(seqLength, hitCacheTokens);

        // 预估排队的时间
        long queueTime = workerStatus.getRunningQueueTime().get();

        return new WorkerTTFT(workerStatus, prefillTime + queueTime, prefillTime, hitCacheTokens);
    }

    private long calcPrefixMatchLength(WorkerStatus workerStatus, Map<String, Integer> matchResultsMap) {
        if (workerStatus.getCacheStatus() == null || matchResultsMap == null) {
            return 0;
        }
        Integer prefixMatchLength = matchResultsMap.get(workerStatus.getIpPort());
        if (prefixMatchLength == null) {
            return 0;
        }
        long blockSize = workerStatus.getCacheStatus().getBlockSize();
        return blockSize * prefixMatchLength;
    }
}

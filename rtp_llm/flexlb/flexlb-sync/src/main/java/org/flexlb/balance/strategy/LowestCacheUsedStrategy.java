package org.flexlb.balance.strategy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
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
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.utils.LoggingUtils;
import org.springframework.stereotype.Component;

/**
 * @author zjw
 * description:
 * date: 2025/3/21
 */
@Component("lowestCacheUsedStrategy")
public class LowestCacheUsedStrategy implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;

    public LowestCacheUsedStrategy(EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.LOWEST_CACHE_USED, this);
    }

    public static final class ScoredWorker {
        public final WorkerStatus w;
        public final long needKvCacheLen;

        public ScoredWorker(WorkerStatus w, long needKvCacheLen) {
            this.w = w;
            this.needKvCacheLen = needKvCacheLen;
        }
    }
    @Override
    public boolean releaseLocalCache(String modelName, String ip, Long interRequestId) {
        return true;
    }
    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        MasterRequest masterRequest = balanceContext.getMasterRequest();
        long seqLen = masterRequest.getSeqLen();
        Map<String/*ipPort*/, WorkerStatus> workerStatusMap;
        if (StringUtils.isNotEmpty(group)) {
            workerStatusMap =  Optional.ofNullable(engineWorkerStatus.getModelRoleWorkerStatusMap().get(balanceContext.getMasterRequest().getModel()))
                    .map(entry -> entry.getRoleStatusMap(roleType, group))
                    .orElse(null);
        }  else {
            workerStatusMap = Optional.ofNullable(engineWorkerStatus.getModelRoleWorkerStatusMap().get(balanceContext.getMasterRequest().getModel()))
                    .map(entry -> entry.getRoleStatusMap(roleType))
                    .orElse(null);
        }
        if (MapUtils.isEmpty(workerStatusMap)) {
            LoggingUtils.warn("no available worker, workerStatusMap is empty, roleType:{}, group:{}, seqLen:{}, requestId:{}",
                    roleType, group, seqLen, balanceContext.getInterRequestId());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        // 使用类对象作为全局锁，确保同一时间只有一个线程可以执行Worker选择逻辑
        // 这样可以避免多个线程都选择同一个Worker的问题
        synchronized (LowestCacheUsedStrategy.class) {
            List<WorkerStatus> candidateWorkerStatus = new ArrayList<>();
            for(String ipPort : workerStatusMap.keySet()) {
                WorkerStatus workerStatus = workerStatusMap.get(ipPort);
                if (workerStatus.isAlive()) {
                    candidateWorkerStatus.add(workerStatus);
                }
            }
            LoggingUtils.info("candidateWorkerStatus size: {}", candidateWorkerStatus.size());
            if(candidateWorkerStatus.isEmpty()){
                return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
            }
            // 直接选择缓存使用量最少且满足条件的Worker
            List<ScoredWorker> candidates = candidateWorkerStatus.stream()
                    .filter(WorkerStatus::isAlive)
                    .map(ws -> new ScoredWorker(ws,
                            seqLen - calcPrefixMatchLength(ws.getCacheStatus(),
                                    balanceContext.getMasterRequest().getBlockCacheKeys())))
                    .collect(Collectors.toList());
            
            Optional<ScoredWorker> bestWorker = candidates.stream()
                    .min(Comparator.comparingLong(candidate -> candidate.w.getKvCacheUsed().get()))
                    .map(minCacheUsed -> {
                        // 在所有具有相同最小缓存使用的Worker中随机选择一个
                        List<ScoredWorker> bestCandidates = candidates.stream()
                                .filter(candidate -> candidate.w.getKvCacheUsed().get() == minCacheUsed.w.getKvCacheUsed().get())
                                .collect(Collectors.toList());
                        int randomIndex = new java.util.Random().nextInt(bestCandidates.size());
                        return bestCandidates.get(randomIndex);
                    });
            
            if (bestWorker.isPresent()) {
                ScoredWorker candidate = bestWorker.get();
                WorkerStatus worker = candidate.w;
                long needKvCacheLen = candidate.needKvCacheLen;
                
                // 更新其他状态
                LoggingUtils.info("成功选择{} Worker, ip:{}, port: {} need mem:{}, kvCache Used:{}, free:{}",
                        roleType.toString(), worker.getIp(), worker.getPort(), seqLen - needKvCacheLen, worker.getKvCacheUsed().get(), worker.getKvCacheFree().get());
                return buildServerStatus(worker, seqLen, needKvCacheLen, roleType, balanceContext.getInterRequestId());
            }
            
            // 如果没有找到合适的Worker，返回失败
            LoggingUtils.warn("选择Worker失败，没有找到合适的Worker");
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    private long calcPrefixMatchLength(CacheStatus cacheStatus, List<Long> blockCacheKeys) {
        // 计算 prefix match length
        if(cacheStatus == null){
            return 0;
        }
        long blockSize = cacheStatus.getBlockSize();
        Set<Long> cachePrefixHash = cacheStatus.getCachedKeys();
        if (blockCacheKeys == null || cachePrefixHash == null) {
            return 0;
        }
        for(int index = blockCacheKeys.size()-1; index >= 0; index-- ){
            long hash = blockCacheKeys.get(index);
            // 将Long转换为BigInteger进行比较
            if(cachePrefixHash.contains(hash)){
                return blockSize * (index + 1);
            }
        }
        return 0;
    }

    
    private ServerStatus buildServerStatus(WorkerStatus optimalWorker, long seqLen, long prefixLength, RoleType roleType, long interRequestId) {
        ServerStatus result = new ServerStatus();
        try {
            String batchId = UUID.randomUUID().toString();
            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setPrefillTime(0);
            taskInfo.setWaitingTime(0);
            taskInfo.setInputLength(seqLen);
            taskInfo.setPrefixLength(prefixLength);
            taskInfo.setInterRequestId(interRequestId);
            optimalWorker.decKvCacheFree(seqLen - prefixLength);
            optimalWorker.addKvCacheUsed(seqLen - prefixLength);
            optimalWorker.getLocalTaskMap().put(interRequestId, taskInfo);

            LoggingUtils.info("decode local add ip:{} , need mem:{},  kvcache Used:{}, free:{}", optimalWorker.getIp(),
                    seqLen - prefixLength,
                    optimalWorker.getKvCacheUsed().get(), optimalWorker.getKvCacheFree().get());

            result.setSuccess(true);
            result.setRole(roleType);
            result.setBatchId(batchId);
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

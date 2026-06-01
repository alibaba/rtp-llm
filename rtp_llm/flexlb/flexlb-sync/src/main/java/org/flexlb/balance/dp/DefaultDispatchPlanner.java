package org.flexlb.balance.dp;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

@Component
@DependsOn({"weightedCacheStrategy", "shortestTTFTStrategy", "randomStrategy"})
public class DefaultDispatchPlanner implements DispatchPlanner {

    private final EngineWorkerStatus engineWorkerStatus;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final CacheAwareService cacheAwareService;
    private final PrefillTimePredictor prefillTimePredictor;

    public DefaultDispatchPlanner(EngineWorkerStatus engineWorkerStatus,
                                  ResourceMeasureFactory resourceMeasureFactory,
                                  CacheAwareService cacheAwareService,
                                  PrefillTimePredictor prefillTimePredictor) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.cacheAwareService = cacheAwareService;
        this.prefillTimePredictor = prefillTimePredictor;
    }

    @Override
    public WorkerStatus selectPrefillWorker(String model, FlexlbConfig cfg,
                                            BalanceContext ctx,
                                            ToIntFunction<String> queueSizeByWorker) {
        List<WorkerStatus> candidates = dpEnabledPrefillCandidates(cfg);
        if (candidates.isEmpty()) {
            return null;
        }

        Map<String, WorkerStatus> decodeWorkers =
                engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null);

        int maxQueueSize = cfg.getMaxGroupQueueSize();
        List<WorkerStatus> eligible = new ArrayList<>(candidates.size());
        for (WorkerStatus w : candidates) {
            if (!hasAliveDecodeInGroup(decodeWorkers, w.getGroup())) {
                continue;
            }
            if (maxQueueSize > 0 && queueSizeByWorker.applyAsInt(w.getIpPort()) > maxQueueSize) {
                continue;
            }
            eligible.add(w);
        }

        if (eligible.isEmpty()) {
            eligible = candidates.stream()
                    .filter(w -> hasAliveDecodeInGroup(decodeWorkers, w.getGroup()))
                    .collect(Collectors.toCollection(ArrayList::new));
            if (eligible.isEmpty()) {
                return null;
            }
        }

        if (eligible.size() == 1) {
            return eligible.get(0);
        }

        List<Long> cacheKeys = ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getBlockCacheKeys() : null;
        boolean hasCacheKeys = cacheKeys != null && !cacheKeys.isEmpty();
        long seqLen = ctx != null && ctx.getRequest() != null ? ctx.getRequest().getSeqLen() : 0;
        double alpha = cfg.getGroupQueueWeight();

        WorkerStatus best = null;
        long bestScore = Long.MAX_VALUE;
        long bestCacheMatchedTokens = 0;

        for (WorkerStatus w : eligible) {
            long cacheMatchedTokens = 0;
            if (hasCacheKeys && cfg.isCacheAwareSchedulingEnabled()) {
                long blockSize = getBlockSize(w);
                int prefixBlocks = cacheAwareService.findMatchingPrefixLength(w.getIpPort(), cacheKeys);
                cacheMatchedTokens = prefixBlocks * blockSize;
            }

            long prefillCost = prefillTimePredictor.estimateMs(seqLen, cacheMatchedTokens);
            long queueWait = w.getRunningQueueTime().get();
            long score = prefillCost + (long) (alpha * queueWait);

            if (score < bestScore) {
                bestScore = score;
                best = w;
                bestCacheMatchedTokens = cacheMatchedTokens;
            }
        }

        if (best != null && ctx != null) {
            ctx.setCacheMatchedTokens(bestCacheMatchedTokens);
        }

        return best;
    }

    @Override
    public ServerStatus selectDecodeWorker(BalanceContext ctx, String group) {
        FlexlbConfig cfg = ctx.getConfig();
        if (cfg == null) {
            cfg = new FlexlbConfig();
        }
        LoadBalancer decodeSelector = LoadBalanceStrategyFactory.getLoadBalancer(
                cfg.getStrategyForRoleType(RoleType.DECODE));
        return decodeSelector.select(ctx, RoleType.DECODE, group);
    }

    private static boolean hasAliveDecodeInGroup(Map<String, WorkerStatus> allDecode, String group) {
        if (group == null) {
            return false;
        }
        if (allDecode == null || allDecode.isEmpty()) {
            return false;
        }
        return allDecode.values().stream()
                .anyMatch(w -> w.isAlive() && group.equals(w.getGroup()));
    }

    static long getBlockSize(WorkerStatus w) {
        if (w.getCacheStatus() != null && w.getCacheStatus().getBlockSize() > 0) {
            return w.getCacheStatus().getBlockSize();
        }
        return 1;
    }

    private List<WorkerStatus> dpEnabledPrefillCandidates(FlexlbConfig cfg) {
        Map<String, WorkerStatus> all = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (all == null || all.isEmpty()) {
            Logger.warn("dpEnabledPrefillCandidates: workers={}", all == null ? "null" : "empty");
            return List.of();
        }
        ResourceMeasureIndicatorEnum indicator = cfg.getResourceMeasureIndicator(RoleType.PREFILL);
        ResourceMeasure measure = resourceMeasureFactory.getMeasure(indicator);
        List<WorkerStatus> result = all.values().stream()
                .filter(WorkerStatus::isAlive)
                .filter(w -> measure == null || measure.isResourceAvailable(w))
                .toList();
        if (result.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, WorkerStatus> e : all.entrySet()) {
                WorkerStatus w = e.getValue();
                boolean alive = w != null && w.isAlive();
                boolean resOk = measure == null || (w != null && measure.isResourceAvailable(w));
                sb.append("[").append(e.getKey())
                  .append(":alive=").append(alive)
                  .append(",resOk=").append(resOk)
                  .append("]");
            }
            Logger.warn("dpEnabledPrefillCandidates: 0 candidates, indicator={}, measure={}, workers={}",
                    indicator, measure == null ? "null" : measure.getClass().getSimpleName(), sb);
        }
        return result;
    }

    public static ServerStatus toPrefillServerStatus(WorkerStatus w) {
        ServerStatus s = new ServerStatus();
        s.setSuccess(true);
        s.setRole(RoleType.PREFILL);
        s.setGroup(w.getGroup());
        s.setServerIp(w.getIp());
        s.setHttpPort(w.getPort());
        s.setGrpcPort(org.flexlb.util.CommonUtils.toGrpcPort(w.getPort()));
        return s;
    }
}

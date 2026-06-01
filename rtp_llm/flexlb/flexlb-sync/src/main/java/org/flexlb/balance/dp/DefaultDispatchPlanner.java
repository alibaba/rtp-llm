package org.flexlb.balance.dp;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
@DependsOn({"weightedCacheStrategy", "shortestTTFTStrategy", "randomStrategy"})
public class DefaultDispatchPlanner implements DispatchPlanner {

    private final EngineWorkerStatus engineWorkerStatus;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final Map<String, GroupSelector> groupSelectors;

    public DefaultDispatchPlanner(EngineWorkerStatus engineWorkerStatus,
                                  ResourceMeasureFactory resourceMeasureFactory,
                                  List<GroupSelector> allSelectors) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.groupSelectors = allSelectors.stream()
                .collect(Collectors.toMap(GroupSelector::name, Function.identity()));
    }

    @Override
    public WorkerStatus selectPrefillWorker(String model, FlexlbConfig cfg, int dpSize) {
        List<WorkerStatus> candidates = dpEnabledPrefillCandidates(cfg, dpSize);
        if (candidates.isEmpty()) {
            return null;
        }
        Map<String, WorkerStatus> decodeWorkers =
                engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null);
        if (decodeWorkers != null && !decodeWorkers.isEmpty()) {
            List<WorkerStatus> withDecode = candidates.stream()
                    .filter(w -> hasAliveDecodeInGroup(decodeWorkers, w.getGroup()))
                    .toList();
            if (!withDecode.isEmpty()) {
                candidates = withDecode;
            }
        }
        if (candidates.size() == 1) {
            return candidates.get(0);
        }
        return candidates.stream()
                .min(Comparator.comparingLong(w -> w.getRunningQueueTime().get()))
                .orElse(candidates.get(0));
    }

    @Override
    public ServerStatus selectDecodeWorker(String model, FlexlbConfig cfg, String group) {
        LoadBalancer decodeSelector = LoadBalanceStrategyFactory.getLoadBalancer(
                cfg.getStrategyForRoleType(RoleType.DECODE));
        if (decodeSelector == null) {
            return null;
        }
        org.flexlb.dao.BalanceContext probeCtx = new org.flexlb.dao.BalanceContext();
        probeCtx.setConfig(cfg);
        ServerStatus result = decodeSelector.select(probeCtx, RoleType.DECODE, group);
        if (result == null || !result.isSuccess()) {
            return null;
        }
        return result;
    }

    private static boolean hasAliveDecodeInGroup(Map<String, WorkerStatus> allDecode, String group) {
        if (group == null) {
            return false;
        }
        return allDecode.values().stream()
                .anyMatch(w -> w.isAlive() && group.equals(w.getGroup()));
    }



    @Override
    public DispatchPlan plan(List<QueuedRequest> drained, DispatchContext context) {
        if (drained == null || drained.isEmpty()) {
            return DispatchPlan.empty();
        }

        FlexlbConfig cfg = context.config();
        WorkerStatus group;

        if (context.preSelectedPrefill() != null) {
            group = context.preSelectedPrefill();
        } else {
            List<WorkerStatus> candidates = dpEnabledPrefillCandidates(cfg, context.dpSize());
            if (candidates.isEmpty()) {
                return DispatchPlan.allFailed(drained, StrategyErrorType.NO_PREFILL_WORKER,
                        "no DP-enabled prefill worker available");
            }

            String selectorName = cfg.isCacheAwareSchedulingEnabled()
                    ? cfg.getDpGroupSelector()
                    : FirstAliveGroupSelector.NAME;
            GroupSelector groupSelector = groupSelectors.getOrDefault(selectorName,
                    groupSelectors.get(CacheAwareGroupSelector.NAME));
            if (groupSelector == null) {
                groupSelector = groupSelectors.values().iterator().next();
            }

            group = groupSelector.select(candidates, context);
            if (group == null) {
                return DispatchPlan.allFailed(drained, StrategyErrorType.NO_PREFILL_WORKER,
                        "GroupSelector(" + groupSelector.name() + ") returned no candidate");
            }
        }

        ServerStatus prefill = toPrefillServerStatus(group);

        List<PendingRequest> placed = new ArrayList<>(drained.size());
        List<FailedRequest> failures = new ArrayList<>();

        if (context.preSelectedDecode() != null) {
            ServerStatus decode = context.preSelectedDecode();
            for (QueuedRequest qr : drained) {
                qr.ctx().setConfig(cfg);
                placed.add(new PendingRequest(qr.ctx(), prefill, decode, qr.future(), qr.enqueuedAtMicros()));
            }
        } else {
            LoadBalancer decodeSelector = LoadBalanceStrategyFactory.getLoadBalancer(
                    cfg.getStrategyForRoleType(RoleType.DECODE));
            for (QueuedRequest qr : drained) {
                qr.ctx().setConfig(cfg);
                ServerStatus decode = decodeSelector.select(qr.ctx(), RoleType.DECODE, prefill.getGroup());
                if (decode == null || !decode.isSuccess()) {
                    String msg = decode == null ? "decode selector returned null" : decode.getMessage();
                    failures.add(new FailedRequest(qr, StrategyErrorType.NO_DECODE_WORKER, msg));
                    continue;
                }
                placed.add(new PendingRequest(qr.ctx(), prefill, decode, qr.future(), qr.enqueuedAtMicros()));
            }
        }

        if (placed.isEmpty()) {
            return new DispatchPlan(List.of(), failures);
        }

        long blockSize = group.getCacheStatus() != null ? group.getCacheStatus().getBlockSize() : 1;
        PrefillBatch batch = new PrefillBatch(prefill, placed, context.dpSize(), blockSize);
        Logger.debug("DefaultDispatchPlanner placed {}/{} requests on group {}:{} (drain={}, dpSize={})",
                placed.size(), drained.size(), prefill.getServerIp(), prefill.getGrpcPort(),
                drained.size(), context.dpSize());
        return new DispatchPlan(List.of(batch), failures);
    }

    private List<WorkerStatus> dpEnabledPrefillCandidates(FlexlbConfig cfg, int targetDpSize) {
        Map<String, WorkerStatus> all = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (all == null || all.isEmpty()) {
            Logger.warn("dpEnabledPrefillCandidates: workers={}", all == null ? "null" : "empty");
            return List.of();
        }
        ResourceMeasureIndicatorEnum indicator = cfg.getResourceMeasureIndicator(RoleType.PREFILL);
        ResourceMeasure measure = resourceMeasureFactory.getMeasure(indicator);
        List<WorkerStatus> result = all.values().stream()
                .filter(WorkerStatus::isAlive)
                .filter(w -> targetDpSize <= 0 || w.getDpSize() == targetDpSize)
                .filter(w -> measure == null || measure.isResourceAvailable(w))
                .toList();
        if (result.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, WorkerStatus> e : all.entrySet()) {
                WorkerStatus w = e.getValue();
                boolean alive = w != null && w.isAlive();
                long dpSize = w == null ? -1L : w.getDpSize();
                boolean dpOk = targetDpSize <= 0 || dpSize == targetDpSize;
                boolean resOk = measure == null || (w != null && measure.isResourceAvailable(w));
                sb.append("[").append(e.getKey())
                  .append(":alive=").append(alive)
                  .append(",dpSize=").append(dpSize)
                  .append(",dpOk=").append(dpOk)
                  .append(",resOk=").append(resOk)
                  .append("]");
            }
            Logger.warn("dpEnabledPrefillCandidates: 0 candidates, targetDpSize={}, indicator={}, measure={}, workers={}",
                    targetDpSize, indicator, measure == null ? "null" : measure.getClass().getSimpleName(), sb);
        }
        return result;
    }

    private static ServerStatus toPrefillServerStatus(WorkerStatus w) {
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

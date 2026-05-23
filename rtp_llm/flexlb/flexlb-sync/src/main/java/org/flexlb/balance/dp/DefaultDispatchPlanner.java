package org.flexlb.balance.dp;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
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
    public DispatchPlan plan(List<QueuedRequest> drained, DispatchContext context) {
        if (drained == null || drained.isEmpty()) {
            return DispatchPlan.empty();
        }

        FlexlbConfig cfg = context.config();
        List<WorkerStatus> candidates = dpEnabledPrefillCandidates(cfg);
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

        WorkerStatus group = groupSelector.select(candidates, context);
        if (group == null) {
            return DispatchPlan.allFailed(drained, StrategyErrorType.NO_PREFILL_WORKER,
                    "GroupSelector(" + groupSelector.name() + ") returned no candidate");
        }

        ServerStatus prefill = toPrefillServerStatus(group);
        LoadBalancer decodeSelector = LoadBalanceStrategyFactory.getLoadBalancer(
                cfg.getStrategyForRoleType(RoleType.DECODE));

        List<PendingRequest> placed = new ArrayList<>(drained.size());
        List<FailedRequest> failures = new ArrayList<>();
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

    private List<WorkerStatus> dpEnabledPrefillCandidates(FlexlbConfig cfg) {
        Map<String, WorkerStatus> all = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (all == null || all.isEmpty()) {
            return List.of();
        }
        ResourceMeasureIndicatorEnum indicator = cfg.getResourceMeasureIndicator(RoleType.PREFILL);
        ResourceMeasure measure = resourceMeasureFactory.getMeasure(indicator);
        return all.values().stream()
                .filter(WorkerStatus::isAlive)
                .filter(w -> w.getDpSize() > 1)
                .filter(w -> measure == null || measure.isResourceAvailable(w))
                .toList();
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

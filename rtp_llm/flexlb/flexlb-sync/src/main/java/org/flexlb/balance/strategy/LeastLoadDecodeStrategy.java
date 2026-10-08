package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Decode placement ordered by projected request count, then KV usage ratio. */
@Component("leastLoadDecodeStrategy")
public class LeastLoadDecodeStrategy implements LoadBalancer {
    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final Map<String, String> lastSelectedByGroup = new HashMap<>();

    public LeastLoadDecodeStrategy(ConfigService configService,
                                   EngineWorkerStatus engineWorkerStatus,
                                   ResourceMeasureFactory resourceMeasureFactory) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
        this.resourceMeasureFactory = resourceMeasureFactory;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.LEAST_LOAD_DECODE, this);
    }

    // Serialize selection with local reservation: concurrent schedulers must see each
    // other's new requests even when the engine snapshot has not refreshed yet.
    @Override
    public ServerStatus select(BalanceContext context, RoleType roleType, String group) {
        return select(context, roleType, group, Set.of());
    }

    /** Exclusions apply only to this routing attempt, after a downstream group failure. */
    public synchronized ServerStatus select(BalanceContext context, RoleType roleType, String group,
                                            Set<String> excludedGroups) {
        if (roleType != RoleType.DECODE) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        FlexlbConfig config = context.getConfig() != null
                ? context.getConfig() : configService.loadBalanceConfig();
        ResourceMeasure measure = resourceMeasureFactory.getMeasure(config.getResourceMeasureIndicator(roleType));
        if (measure == null) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        long minimumLoad = Long.MAX_VALUE;
        double minimumKvRatio = Double.POSITIVE_INFINITY;
        List<WorkerStatus> candidates = new ArrayList<>();
        for (WorkerStatus worker : engineWorkerStatus.selectModelWorkerStatus(roleType, group).values()) {
            if (worker == null || !worker.isAlive() || (!excludedGroups.isEmpty() && excludedGroups.contains(worker.getGroup()))) {
                continue;
            }
            long load = worker.getDecodeConcurrency();
            boolean availableResource = measure instanceof DecodeResourceMeasure decodeMeasure
                    ? decodeMeasure.isResourceAvailable(worker, load) : measure.isResourceAvailable(worker);
            if (!availableResource) {
                continue;
            }
            long used = Math.max(0L, worker.getUsedKvCacheTokens().get());
            long available = Math.max(0L, worker.getAvailableKvCacheTokens().get());
            double total = (double) used + available;
            double kvRatio = total == 0 ? 0.0 : used / total;
            if (load < minimumLoad || (load == minimumLoad && kvRatio < minimumKvRatio)) {
                minimumLoad = load;
                minimumKvRatio = kvRatio;
                candidates.clear();
                candidates.add(worker);
            } else if (load == minimumLoad && kvRatio == minimumKvRatio) {
                candidates.add(worker);
            }
        }
        if (candidates.isEmpty()) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Address ordering keeps rotation stable across map iteration order and
        // candidate additions/removals. The cursor is independent for each group.
        candidates.sort(Comparator.comparing(WorkerStatus::getIpPort));
        String previous = lastSelectedByGroup.get(group);
        WorkerStatus selected = candidates.getFirst();
        if (previous != null) {
            for (WorkerStatus candidate : candidates) {
                if (candidate.getIpPort().compareTo(previous) > 0) {
                    selected = candidate;
                    break;
                }
            }
        }

        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setServerIp(selected.getIp());
        result.setHttpPort(selected.getPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(selected.getPort()));
        result.setGroup(selected.getGroup());
        result.setRequestId(context.getRequestId());

        TaskInfo task = new TaskInfo();
        task.setRequestId(context.getRequestId());
        task.setInputLength(context.getRequest().getSeqLen());
        task.setPrefixLength(prefixTokens(selected.getCacheStatus(), context.getRequest().getBlockCacheKeys()));
        selected.putLocalTask(context.getRequestId(), task);
        lastSelectedByGroup.put(group, selected.getIpPort());
        return result;
    }

    private long prefixTokens(CacheStatus cache, List<Long> keys) {
        if (cache == null || cache.getCachedKeys() == null || keys == null) {
            return 0;
        }
        int matched = 0;
        for (Long key : keys) {
            if (!cache.getCachedKeys().contains(key)) {
                break;
            }
            matched++;
        }
        return cache.getBlockSize() * matched;
    }

    @Override
    public synchronized void rollBack(String ipPort, long requestId) {
        WorkerStatus worker = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null).get(ipPort);
        if (worker != null) {
            worker.removeLocalTask(requestId);
        }
    }
}

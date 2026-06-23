package org.flexlb.balance.strategy;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.IntUnaryOperator;
import java.util.function.LongSupplier;

@Component("forceChatStickyStrategy")
public class ForceChatStickyStrategy implements LoadBalancer {
    private static final int MIN_CANDIDATE_COUNT = 1;
    private static final double CANDIDATE_PERCENTAGE = 0.3;
    private static final String NULL_GROUP = "";

    private final EngineWorkerStatus engineWorkerStatus;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final LongSupplier nowMsSupplier;
    private final IntUnaryOperator randomIndexProvider;
    private final Map<StickyKey, StickyEntry> prefillStickyMap = new ConcurrentHashMap<>();

    public ForceChatStickyStrategy(EngineWorkerStatus engineWorkerStatus,
                                   ResourceMeasureFactory resourceMeasureFactory) {
        this(
                engineWorkerStatus,
                resourceMeasureFactory,
                System::currentTimeMillis,
                bound -> ThreadLocalRandom.current().nextInt(bound));
    }

    ForceChatStickyStrategy(EngineWorkerStatus engineWorkerStatus,
                            ResourceMeasureFactory resourceMeasureFactory,
                            LongSupplier nowMsSupplier,
                            IntUnaryOperator randomIndexProvider) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.nowMsSupplier = nowMsSupplier;
        this.randomIndexProvider = randomIndexProvider;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.FORCE_CHAT_STICKY, this);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("Failed to select worker with force chat sticky strategy", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        for (RoleType roleType : RoleType.values()) {
            WorkerStatus workerStatus = engineWorkerStatus.selectModelWorkerStatus(roleType, null).get(ipPort);
            if (workerStatus != null) {
                workerStatus.removeLocalTask(requestId);
            }
        }
    }

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        if (request == null) {
            return ServerStatus.code(StrategyErrorType.INVALID_REQUEST);
        }

        FlexlbConfig config = balanceContext.getConfig() != null ? balanceContext.getConfig() : new FlexlbConfig();
        ResourceMeasure resourceMeasure = getResourceMeasure(config, roleType);
        if (resourceMeasure == null) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            Logger.warn("No worker status map found for role: {}, group: {}", roleType, group);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        WorkerStatus selectedWorker = selectStickyWorker(request, roleType, group, workerStatusMap, resourceMeasure);
        if (selectedWorker == null) {
            selectedWorker = selectFallbackWorker(workerStatusMap, resourceMeasure);
        }
        if (selectedWorker == null) {
            Logger.warn("No available workers found for role: {}, group: {}", roleType, group);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        ServerStatus serverStatus = buildServerStatus(selectedWorker, request, roleType);
        if (serverStatus.isSuccess()) {
            storeStickySelection(request, roleType, group, selectedWorker, config.getChatStickyTtlMs());
        }
        return serverStatus;
    }

    private ResourceMeasure getResourceMeasure(FlexlbConfig config, RoleType roleType) {
        ResourceMeasureIndicatorEnum indicator = config.getResourceMeasureIndicator(roleType);
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            Logger.warn("No ResourceMeasure registered for indicator: {}, roleType: {}", indicator, roleType);
        }
        return resourceMeasure;
    }

    private WorkerStatus selectStickyWorker(Request request,
                                            RoleType roleType,
                                            String group,
                                            Map<String, WorkerStatus> workerStatusMap,
                                            ResourceMeasure resourceMeasure) {
        String chatId = normalizeChatId(request.getChatId());
        if (roleType != RoleType.PREFILL || chatId == null) {
            return null;
        }

        StickyKey key = new StickyKey(roleType, normalizeGroup(group), chatId);
        StickyEntry entry = prefillStickyMap.get(key);
        if (entry == null) {
            return null;
        }

        long nowMs = nowMsSupplier.getAsLong();
        if (entry.expiresAtMs() <= nowMs) {
            prefillStickyMap.remove(key, entry);
            return null;
        }

        WorkerStatus workerStatus = workerStatusMap.get(entry.ipPort());
        if (isSelectable(workerStatus, resourceMeasure)) {
            return workerStatus;
        }

        prefillStickyMap.remove(key, entry);
        return null;
    }

    private WorkerStatus selectFallbackWorker(Map<String, WorkerStatus> workerStatusMap, ResourceMeasure resourceMeasure) {
        List<WorkerStatus> sortedWorkers = workerStatusMap.values().stream()
                .filter(workerStatus -> isSelectable(workerStatus, resourceMeasure))
                .sorted(
                        Comparator.comparingLong((WorkerStatus workerStatus) -> workerStatus.getRunningQueueTime().get())
                                .thenComparingLong(workerStatus -> workerStatus.getLastSelectedTime().get())
                                .thenComparing(WorkerStatus::getIpPort))
                .toList();
        if (sortedWorkers.isEmpty()) {
            return null;
        }

        int candidateCount = Math.max(MIN_CANDIDATE_COUNT, (int) (sortedWorkers.size() * CANDIDATE_PERCENTAGE));
        candidateCount = Math.min(candidateCount, sortedWorkers.size());
        int selectedIndex = Math.floorMod(randomIndexProvider.applyAsInt(candidateCount), candidateCount);
        return sortedWorkers.get(selectedIndex);
    }

    private boolean isSelectable(WorkerStatus workerStatus, ResourceMeasure resourceMeasure) {
        return workerStatus != null && workerStatus.isAlive() && resourceMeasure.isResourceAvailable(workerStatus);
    }

    private ServerStatus buildServerStatus(WorkerStatus workerStatus, Request request, RoleType roleType) {
        ServerStatus result = new ServerStatus();
        try {
            long prefixLength = 0L;
            long queueTime = workerStatus.getRunningQueueTime().get();
            long prefillTime = TaskInfo.estimatePrefillTimeMs(request.getSeqLen(), prefixLength);

            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setRequestId(request.getRequestId());
            taskInfo.setInputLength(request.getSeqLen());
            taskInfo.setPrefixLength(prefixLength);
            workerStatus.putLocalTask(request.getRequestId(), taskInfo);

            result.setSuccess(true);
            result.setRole(roleType);
            result.setRequestId(request.getRequestId());
            result.setPrefillTime(queueTime + prefillTime);
            result.setGroup(workerStatus.getGroup());
            result.setServerIp(workerStatus.getIp());
            result.setHttpPort(workerStatus.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(workerStatus.getPort()));
        } catch (Exception e) {
            Logger.error("Failed to build server status for requestId: {}", request.getRequestId(), e);
            result.setSuccess(false);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
        }
        return result;
    }

    private void storeStickySelection(Request request,
                                      RoleType roleType,
                                      String group,
                                      WorkerStatus selectedWorker,
                                      long ttlMs) {
        String chatId = normalizeChatId(request.getChatId());
        if (roleType != RoleType.PREFILL || chatId == null || ttlMs <= 0) {
            return;
        }

        StickyKey key = new StickyKey(roleType, normalizeGroup(group), chatId);
        StickyEntry entry = new StickyEntry(selectedWorker.getIpPort(), nowMsSupplier.getAsLong() + ttlMs);
        prefillStickyMap.put(key, entry);
    }

    private String normalizeChatId(String chatId) {
        return chatId == null || chatId.isBlank() ? null : chatId;
    }

    private String normalizeGroup(String group) {
        return group == null ? NULL_GROUP : group;
    }

    private record StickyKey(RoleType roleType, String group, String chatId) {
    }

    private record StickyEntry(String ipPort, long expiresAtMs) {
    }
}

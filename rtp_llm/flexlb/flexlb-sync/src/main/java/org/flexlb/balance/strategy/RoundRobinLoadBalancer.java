package org.flexlb.balance.strategy;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Cursor-based round-robin load balancer.
 *
 * <p>Intentionally load-unaware: each call advances a per-role atomic cursor and returns
 * the next alive worker. No queue time, cache hit, or resource-availability gating.
 *
 * <p>Trade-offs vs {@link ShortestTTFTStrategy}:
 * <ul>
 *   <li>50-200x cheaper schedule cost (cursor + modulo only)</li>
 *   <li>Will hit hot workers proportionally under load skew (no avoidance)</li>
 *   <li>Does not consult {@link org.flexlb.balance.resource.ResourceMeasure};
 *       resource-availability gating is intentionally skipped.</li>
 * </ul>
 *
 * <p>Batch behavior ({@link BatchLoadBalancer}):
 * <ul>
 *   <li>{@link #selectBatch(int, RoleType, String)} advances the per-role cursor {@code count} times.</li>
 *   <li>Cursor wraps naturally; when {@code count > alive.size()} one worker receives multiple
 *       assignments (e.g. 20 over 10 workers -&gt; each worker gets 2).</li>
 *   <li>The role cursor is shared with {@link #select}, so single-call and batch-call interleave.</li>
 *   <li>If no workers are alive, returns an empty list.</li>
 *   <li><strong>No bookkeeping</strong>: batch path skips {@code localTaskMap} entirely. Reconciliation
 *       and lost-task detection are not provided for batch dispatches.</li>
 * </ul>
 */
@Component("roundRobinStrategy")
public class RoundRobinLoadBalancer implements BatchLoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final ConfigService configService;
    private final Map<RoleType, AtomicInteger> cursors = new EnumMap<>(RoleType.class);

    public RoundRobinLoadBalancer(EngineWorkerStatus engineWorkerStatus, ConfigService configService) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.configService = configService;
        for (RoleType role : RoleType.values()) {
            cursors.put(role, new AtomicInteger(0));
        }
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.ROUND_ROBIN, this);
    }

    @Override
    public ServerStatus select(BalanceContext context, RoleType roleType, String group) {
        List<WorkerStatus> alive = aliveWorkers(roleType, group);
        if (alive.isEmpty()) {
            return ServerStatus.code(roleType.getErrorType());
        }
        WorkerStatus selected = alive.get(nextIndex(roleType, alive.size()));
        long requestId = context.getRequestId();
        recordTask(selected, requestId, context.getRequest().getSeqLen());
        return buildServerStatus(selected, roleType, requestId);
    }

    @Override
    public List<BatchScheduleTarget> selectBatch(int count, RoleType roleType, String group) {
        List<WorkerStatus> alive = aliveWorkers(roleType, group);
        if (alive.isEmpty()) {
            return new ArrayList<>();
        }
        int aliveSize = alive.size();
        int start = cursors.get(roleType).getAndAdd(count);
        List<BatchScheduleTarget> targets = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int idx = Math.floorMod(start + i, aliveSize);
            targets.add(buildTarget(alive.get(idx), roleType));
        }
        return targets;
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        for (RoleType role : RoleType.values()) {
            Map<String, WorkerStatus> map = engineWorkerStatus.selectModelWorkerStatus(role, null);
            if (map == null) {
                continue;
            }
            WorkerStatus worker = map.get(ipPort);
            if (worker != null) {
                worker.removeLocalTask(requestId);
                return;
            }
        }
    }

    private List<WorkerStatus> aliveWorkers(RoleType roleType, String group) {
        Map<String, WorkerStatus> map = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        List<WorkerStatus> alive = new ArrayList<>();
        if (map == null) {
            return alive;
        }
        for (WorkerStatus w : map.values()) {
            if (w != null && w.isAlive()) {
                alive.add(w);
            }
        }
        return alive;
    }

    private int nextIndex(RoleType roleType, int size) {
        AtomicInteger cursor = cursors.get(roleType);
        return Math.floorMod(cursor.getAndIncrement(), size);
    }

    private void recordTask(WorkerStatus worker, long requestId, long seqLen) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(seqLen);
        task.setPrefixLength(0);
        worker.putLocalTask(requestId, task);
    }

    private ServerStatus buildServerStatus(WorkerStatus worker, RoleType roleType, long requestId) {
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setServerIp(worker.getIp());
            result.setHttpPort(worker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(worker.getPort()));
            result.setRole(roleType);
            result.setGroup(worker.getGroup());
            result.setRequestId(requestId);
        } catch (Exception e) {
            Logger.error("buildServerStatus error", e);
            result.setSuccess(false);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
        }
        return result;
    }

    private BatchScheduleTarget buildTarget(WorkerStatus worker, RoleType roleType) {
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setServerIp(worker.getIp());
        target.setHttpPort(worker.getPort());
        if (configService.loadBalanceConfig().getEngineType() == EngineType.EMBEDDING) {
            target.setArpcPort(CommonUtils.toArpcPort(worker.getPort()));
        } else {
            target.setGrpcPort(CommonUtils.toGrpcPort(worker.getPort()));
        }
        target.setRole(roleType);
        return target;
    }
}

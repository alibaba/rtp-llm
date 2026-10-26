package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.schedulev2.prefill.PrefillQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * 新链路测试共用的请求、Prefill 机器和选择结果。
 */
public final class V2TestSupport {

    private V2TestSupport() {
    }

    public static Request request(FlexlbConfig config, String requestId, int priority, long seqLen) {
        org.flexlb.dao.loadbalance.Request body = new org.flexlb.dao.loadbalance.Request();
        body.setRequestId(requestId);
        body.setSeqLen(seqLen);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(body);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, System.currentTimeMillis() + 60_000L));
        return new Request(context, new CompletableFuture<>(), System.currentTimeMillis());
    }

    /**
     * 一台 PREFILL 机器，WorkerStatus 是 mock，可以用 {@link #report} 设置引擎上报。
     */
    public static PrefillEndpoint prefillEndpoint(String ipPort) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(endpoint.ipPort()).thenReturn(ipPort);
        when(endpoint.getStatus()).thenReturn(status);
        when(status.getRole()).thenReturn(RoleType.PREFILL);
        when(endpoint.prefillQueue()).thenReturn(new PrefillQueue(endpoint, false));
        return endpoint;
    }

    /**
     * 设置引擎上报：版本号、剩余和总 KV、任务列表（requestId → 阶段）。
     */
    public static void report(PrefillEndpoint endpoint,
                              long statusVersion,
                              long availableKv,
                              long totalKv,
                              Map<String, TaskPhase> tasks) {
        Map<String, WorkerStatus.TaskObservation> running = tasks.entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> task(entry.getKey(), entry.getValue())));
        WorkerStatus.EngineObservation engine = new WorkerStatus.EngineObservation(RoleType.PREFILL, null, availableKv,
                totalKv, running, 0.0, 0L, 1L, 1L, 0L, 0L, 0L, 0L, 0L);
        WorkerStatus.CommittedWorkerStatus committed =
                new WorkerStatus.CommittedWorkerStatus(engine, new WorkerStatus.AppliedStatusCursor(statusVersion, 0L));
        WorkerStatus status = endpoint.getStatus();
        when(status.committedWorkerStatus()).thenReturn(committed);
    }

    /**
     * 模拟节点完成首次状态初始化并发布到选路目录。
     */
    public static void register(PrefillQueueRegistry queues, PrefillEndpoint endpoint) {
        endpoint.prefillQueue().updateFromEngine();
        queues.start(endpoint);
    }

    /**
     * 选中 endpoint 的选择结果，返回的 hold 可用来检查是否被关闭。
     */
    public static SelectedRole selected(PrefillEndpoint endpoint, long hitCache, long predictedMs) {
        SelectedRole selected = mock(SelectedRole.class);
        WorkerEndpoint.EndpointHold hold = mock(WorkerEndpoint.EndpointHold.class);
        when(hold.endpoint()).thenReturn(endpoint);
        ServerStatus serverStatus = new ServerStatus();
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCache);
        serverStatus.setDebugInfo(debugInfo);
        when(selected.takeEndpointHold()).thenReturn(hold);
        when(selected.serverStatus()).thenReturn(serverStatus);
        when(selected.prefillWorkMs()).thenReturn(predictedMs);
        when(selected.takeCommitTelemetry()).thenReturn(() -> {});
        return selected;
    }

    /**
     * 直接给请求设置选好的 Prefill，跳过主队列。
     */
    public static PrefillChoice choose(Request request, PrefillEndpoint endpoint, long hitCache, long predictedMs) {
        PrefillChoice choice = new PrefillChoice(endpoint, new ServerStatus(), hitCache, predictedMs, () -> {});
        request.setPrefillChoice(choice);
        return choice;
    }

    private static WorkerStatus.TaskObservation task(String requestId, TaskPhase phase) {
        return new WorkerStatus.TaskObservation(requestId, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, "", 0L, phase, 0L,
                PriorityPreemptionProgress.NONE, mock(WorkerStatus.TaskTelemetry.class));
    }
}

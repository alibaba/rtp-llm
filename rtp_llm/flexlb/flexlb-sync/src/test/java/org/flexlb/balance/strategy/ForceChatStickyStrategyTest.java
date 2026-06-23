package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntUnaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ForceChatStickyStrategyTest {
    private final EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
    private final ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
    private final ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
    private final AtomicLong nowMs = new AtomicLong(1_000L);
    private IntUnaryOperator randomIndexProvider = bound -> 0;

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
    }

    @Test
    void shouldRouteSamePrefillChatToPreviousAliveWorker() {
        WorkerStatus stickyWorker = addWorker(RoleType.PREFILL, "10.0.0.1", 8080, "group-a", 0);
        addWorker(RoleType.PREFILL, "10.0.0.2", 8080, "group-a", 1);
        addWorker(RoleType.PREFILL, "10.0.0.3", 8080, "group-a", 2);

        ForceChatStickyStrategy strategy = createStrategy();

        ServerStatus first = strategy.select(context(1L, "chat-a", 256, 600_000L), RoleType.PREFILL, null);
        ServerStatus second = strategy.select(context(2L, "chat-a", 256, 600_000L), RoleType.PREFILL, null);

        assertTrue(first.isSuccess());
        assertTrue(second.isSuccess());
        assertEquals(stickyWorker.getIp(), first.getServerIp());
        assertEquals(stickyWorker.getIp(), second.getServerIp());
    }

    @Test
    void shouldFallbackAndRefreshStickyWhenPreviousWorkerIsDead() {
        WorkerStatus previousWorker = addWorker(RoleType.PREFILL, "10.0.0.1", 8080, "group-a", 0);
        WorkerStatus fallbackWorker = addWorker(RoleType.PREFILL, "10.0.0.2", 8080, "group-a", 1);
        addWorker(RoleType.PREFILL, "10.0.0.3", 8080, "group-a", 2);

        ForceChatStickyStrategy strategy = createStrategy();

        ServerStatus first = strategy.select(context(1L, "chat-a", 256, 600_000L), RoleType.PREFILL, null);
        previousWorker.setAlive(false);
        ServerStatus second = strategy.select(context(2L, "chat-a", 256, 600_000L), RoleType.PREFILL, null);
        previousWorker.setAlive(true);
        ServerStatus third = strategy.select(context(3L, "chat-a", 256, 600_000L), RoleType.PREFILL, null);

        assertEquals(previousWorker.getIp(), first.getServerIp());
        assertEquals(fallbackWorker.getIp(), second.getServerIp());
        assertEquals(fallbackWorker.getIp(), third.getServerIp());
    }

    @Test
    void shouldExpireStickyMappingAfterConfiguredTtl() {
        WorkerStatus firstWorker = addWorker(RoleType.PREFILL, "10.0.0.1", 8080, "group-a", 0);
        WorkerStatus secondWorker = addWorker(RoleType.PREFILL, "10.0.0.2", 8080, "group-a", 1);
        addWorker(RoleType.PREFILL, "10.0.0.3", 8080, "group-a", 2);

        ForceChatStickyStrategy strategy = createStrategy();

        ServerStatus first = strategy.select(context(1L, "chat-a", 256, 100L), RoleType.PREFILL, null);
        nowMs.set(1_101L);
        ServerStatus second = strategy.select(context(2L, "chat-a", 256, 100L), RoleType.PREFILL, null);

        assertEquals(firstWorker.getIp(), first.getServerIp());
        assertEquals(secondWorker.getIp(), second.getServerIp());
    }

    @Test
    void shouldRandomSelectAmongTopThirtyPercentLowestQueueWorkersForNewChat() {
        for (int i = 0; i < 10; i++) {
            addWorker(RoleType.PREFILL, "10.0.0." + i, 8080, "group-a", i);
        }
        randomIndexProvider = bound -> 2;

        ForceChatStickyStrategy strategy = createStrategy();

        ServerStatus selected = strategy.select(context(1L, "chat-a", 256, 600_000L), RoleType.PREFILL, null);

        assertTrue(selected.isSuccess());
        assertEquals("10.0.0.2", selected.getServerIp());
    }

    @Test
    void shouldNotPersistStickyMappingForDecodeRole() {
        addWorker(RoleType.DECODE, "10.0.0.1", 8080, "group-a", 0);
        addWorker(RoleType.DECODE, "10.0.0.2", 8080, "group-a", 1);

        ForceChatStickyStrategy strategy = createStrategy();

        ServerStatus first = strategy.select(context(1L, "chat-a", 256, 600_000L), RoleType.DECODE, null);
        ServerStatus second = strategy.select(context(2L, "chat-a", 256, 600_000L), RoleType.DECODE, null);

        assertTrue(first.isSuccess());
        assertTrue(second.isSuccess());
        assertNotEquals(first.getServerIp(), second.getServerIp());
    }

    private ForceChatStickyStrategy createStrategy() {
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        return new ForceChatStickyStrategy(
                engineWorkerStatus,
                resourceMeasureFactory,
                nowMs::get,
                randomIndexProvider);
    }

    private BalanceContext context(long requestId, String chatId, long seqLen, long ttlMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setChatId(chatId);
        request.setBlockCacheKeys(List.of(1L, 2L));

        FlexlbConfig config = new FlexlbConfig();
        config.setChatStickyTtlMs(ttlMs);

        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        return context;
    }

    private WorkerStatus addWorker(RoleType roleType, String ip, int port, String group, long runningQueueTime) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        workerStatus.setPort(port);
        workerStatus.setSite("na61");
        workerStatus.setAlive(true);
        workerStatus.setRole(roleType.getCode());
        workerStatus.setGroup(group);
        workerStatus.setWaitingTaskList(new HashMap<>());
        workerStatus.setRunningTaskList(new HashMap<>());
        workerStatus.getRunningQueueTime().set(runningQueueTime);

        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(256);
        cacheStatus.setAvailableKvCache(100_000);
        cacheStatus.setTotalKvCache(100_000);
        workerStatus.setCacheStatus(cacheStatus);

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(roleType)
                .put(workerStatus.getIpPort(), workerStatus);
        return workerStatus;
    }
}

package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ForceChatStickyRoutingIntegrationTest {
    private final EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
    private final ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
    private final ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
    private final AtomicLong nowMs = new AtomicLong(1_000L);

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    @Test
    void shouldRouteRepeatedPrefillChatThroughForceChatStickyStrategy() {
        WorkerStatus stickyWorker = addWorker("10.0.0.1", 8080, "group-a", 0);
        addWorker("10.0.0.2", 8080, "group-a", 1);

        FlexlbConfig config = new FlexlbConfig();
        config.setLoadBalanceStrategy(LoadBalanceStrategyEnum.FORCE_CHAT_STICKY);
        config.setChatStickyTtlMs(600_000L);

        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);

        registerUnusedStrategies();
        new ForceChatStickyStrategy(engineWorkerStatus, resourceMeasureFactory, nowMs::get, bound -> 0);
        DefaultRouter router = new DefaultRouter(configService);

        Response first = router.route(context(config, 1L, "chat-a"));
        Response second = router.route(context(config, 2L, "chat-a"));

        assertTrue(first.isSuccess());
        assertTrue(second.isSuccess());
        assertEquals(stickyWorker.getIp(), first.getServerStatus().getFirst().getServerIp());
        assertEquals(stickyWorker.getIp(), second.getServerStatus().getFirst().getServerIp());
    }

    private void registerUnusedStrategies() {
        LoadBalancer unavailable = new LoadBalancer() {
            @Override
            public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
                return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
            }

            @Override
            public void rollBack(String ipPort, long requestId) {
            }
        };
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, unavailable);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, unavailable);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, unavailable);
    }

    private BalanceContext context(FlexlbConfig config, long requestId, String chatId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(256);
        request.setChatId(chatId);
        request.setBlockCacheKeys(List.of(1L, 2L));

        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        return context;
    }

    private WorkerStatus addWorker(String ip, int port, String group, long runningQueueTime) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        workerStatus.setPort(port);
        workerStatus.setSite("na61");
        workerStatus.setAlive(true);
        workerStatus.setRole(RoleType.PREFILL.getCode());
        workerStatus.setGroup(group);
        workerStatus.setWaitingTaskList(new HashMap<>());
        workerStatus.setRunningTaskList(new HashMap<>());
        workerStatus.getRunningQueueTime().set(runningQueueTime);

        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(256);
        cacheStatus.setAvailableKvCache(100_000);
        cacheStatus.setTotalKvCache(100_000);
        workerStatus.setCacheStatus(cacheStatus);

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(workerStatus.getIpPort(), workerStatus);
        return workerStatus;
    }
}

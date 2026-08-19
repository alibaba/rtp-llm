package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;

/**
 * Task34 类别一：策略层复杂多节点选择正确性 —— 3~8 个 endpoint 组成的
 * 快照矩阵（不同存活状态 / 资源可用性 / endpoint 等待 / batcher 队列构成），
 * 验证：正常 placement 评分选对（含 auto-tpm 开启时实测 queue-age 的
 * batcherEstimatedWaitMs 参与 P 评分且是优先级无关的实测拥堵度）、不可行
 * endpoint 绝不入选、全不可行 → 明确失败、仅一可行 → 必选。
 * 另含 Round-2 拥挤过滤（CONGESTED_QUEUE_FILTERED，编译期 RATIO=0.8）：
 * 超阈引擎即使评分最优也被 bench、未超阈正常参选、全超回退 least-loaded。
 */
class CostBasedPrefillMultiNodeSelectionTest {

    private FlexlbConfig config;
    private ConfigService configService;
    private CacheAwareService cacheAwareService;
    private PrefillResourceMeasure prefillResourceMeasure;
    private EndpointRegistry endpointRegistry;
    private CostBasedPrefillStrategy strategy;
    private final List<PrefillEndpoint> createdEndpoints = new ArrayList<>();

    @BeforeEach
    void setUp() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();

        config = new FlexlbConfig();
        config.setCostSloMs(500_000L);
        config.setCostSloRiskMarginMs(50L);
        config.setScoreTieRandomEnabled(false);
        configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config);

        cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        FlexlbBatchScheduler batchScheduler = Mockito.mock(FlexlbBatchScheduler.class);

        endpointRegistry = new EndpointRegistry(configService, () -> batchScheduler,
                Mockito.mock(BatchSchedulerReporter.class));
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(endpointRegistry);

        prefillResourceMeasure = Mockito.mock(PrefillResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(any())).thenReturn(prefillResourceMeasure);
        Mockito.when(prefillResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(new HashMap<>());

        strategy = new CostBasedPrefillStrategy(
                engineWorkerStatus, cacheAwareService, resourceMeasureFactory,
                engineHealthReporter);
    }

    @AfterEach
    void tearDown() {
        // 显式关闭 batcher 线程（本测试会向队列驻留大量 item 并 park）
        for (PrefillEndpoint ep : createdEndpoints) {
            ep.close();
        }
        createdEndpoints.clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
    }

    // ============ auto-tpm 开启：实测 queue-age（batcherEstimatedWaitMs）参与 P 评分 ============

    @Test
    void autoTpmScoringPrefersEndpointWhoseQueueHeadIsYounger() {
        // w1 队头已积压 5s（慢引擎实测拥堵），w2 队头刚入队（age≈0）。
        // 除队头年龄外两 endpoint 完全对称 → 必选 w2，证明评分吃到实测
        // queue-age 项（队头等得久的引擎看起来更贵）。
        setUpAutoTpmBatcherConfig();
        PrefillEndpoint w1 = parkedEndpoint("10.0.0.1");
        PrefillEndpoint w2 = parkedEndpoint("10.0.0.2");
        fillQueue(w1, 50, 10, 1000, -5_000);
        fillQueue(w2, 50, 10, 2000, 0);

        ServerStatus result = strategy.select(
                priorityContext(9001L, 50), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void autoTpmScoringFlipsWhenQueueHeadAgeIsReversed() {
        // 与上一用例镜像对称：w1 队头年轻、w2 队头老 → 必选 w1。两个方向
        // 都成立，排除“恰好偏向某个 endpointId”的假阳性。
        setUpAutoTpmBatcherConfig();
        PrefillEndpoint w1 = parkedEndpoint("10.0.0.1");
        PrefillEndpoint w2 = parkedEndpoint("10.0.0.2");
        fillQueue(w1, 50, 10, 1000, 0);
        fillQueue(w2, 50, 10, 2000, -5_000);

        ServerStatus result = strategy.select(
                priorityContext(9002L, 50), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    // ============ Round-2 拥挤过滤（CONGESTED_QUEUE_FILTERED，RATIO=0.8） ============

    @Test
    void congestedQueueEndpointIsBenchedEvenWhenItsScoreIsBest() {
        // w1 队列 90 ≥ ceil(0.8×100)=80 → congested；其队头年龄≈0 使
        // score 严格更优（w2 队头老 5s，score 差 5000ms）。拥挤过滤必须
        // 压倒评分：必选 w2（8/17 慢引擎吸引子的直接反制）。
        setUpAutoTpmBatcherConfig();
        PrefillEndpoint w1 = parkedEndpoint("10.0.0.1");
        PrefillEndpoint w2 = parkedEndpoint("10.0.0.2");
        fillQueue(w1, 50, 90, 1000, 0);
        fillQueue(w2, 50, 10, 2000, -5_000);

        ServerStatus result = strategy.select(
                priorityContext(9101L, 50), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void belowThresholdQueueStillCompetesAndWins() {
        // w1 队列 79 < 80：未达阈值，正常参选；w2 队头老 5s（score 更
        // 差）→ 必选 w1。阈值边界不误伤。
        setUpAutoTpmBatcherConfig();
        PrefillEndpoint w1 = parkedEndpoint("10.0.0.1");
        PrefillEndpoint w2 = parkedEndpoint("10.0.0.2");
        fillQueue(w1, 50, 79, 1000, 0);
        fillQueue(w2, 50, 10, 2000, -5_000);

        ServerStatus result = strategy.select(
                priorityContext(9201L, 50), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void allCongestedFallsBackToLeastLoadedEndpoint() {
        // 两队列都 90（全 congested）→ survivor 全滤空 → least-loaded 回退。
        // w2 追加第二个 inflight batch（totalPredict 120_000 > w1 的 60_000，
        // 两端的 elapsed 衰减差只有几 ms）→ w1 恒为 least-loaded → 回退必
        // 选 w1，且路由不 fail-closed。
        setUpAutoTpmBatcherConfig();
        PrefillEndpoint w1 = parkedEndpoint("10.0.0.1");
        PrefillEndpoint w2 = parkedEndpoint("10.0.0.2");
        w2.commitBatch(810_002L, 60_000, List.of());
        fillQueue(w1, 50, 90, 1000, 0);
        fillQueue(w2, 50, 90, 2000, 0);

        ServerStatus result = strategy.select(
                priorityContext(9301L, 50), RoleType.PREFILL, null);

        assertTrue(result.isSuccess(), "all-congested must fall back, never fail closed");
        assertEquals("10.0.0.1", result.getServerIp(),
                "the strictly lower-wait endpoint must win the least-loaded fallback");
    }

    // ============ 多节点矩阵：不可行 endpoint 绝不入选 ============

    @Test
    void sixNodeMatrixNeverSelectsInfeasibleEndpointsAndPicksCheapestFeasible() {
        // 6 节点矩阵：dead / resource-unavailable 两个"评分最优"的陷阱节点 +
        // SLO 超限节点 + 三个可行节点（wait 300/100/200）→ 必选 wait=100 的节点。
        createWorker("10.0.1.1", 0, false);          // dead：分数最优但绝不可入选
        createWorker("10.0.1.2", 0, true);           // resource-unavailable（下方 stub）
        createWorker("10.0.1.3", 600_000, true);     // SLO 超限：wait >> sloMs
        createWorker("10.0.1.4", 300, true);
        createWorker("10.0.1.5", 100, true);
        createWorker("10.0.1.6", 200, true);
        Mockito.when(prefillResourceMeasure.isResourceAvailable(any()))
                .thenAnswer(inv -> !"10.0.1.2".equals(
                        ((PrefillEndpoint) inv.getArgument(0)).getIp()));

        ServerStatus result = strategy.select(buildContext(9101L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.1.5", result.getServerIp());
    }

    @Test
    void allEndpointsDeadFailsExplicitly() {
        createWorker("10.0.2.1", 0, false);
        createWorker("10.0.2.2", 0, false);
        createWorker("10.0.2.3", 0, false);

        ServerStatus result = strategy.select(buildContext(9201L), RoleType.PREFILL, null);

        assertFalse(result.isSuccess());
    }

    @Test
    void onlyAliveEndpointIsAlwaysSelectedEvenWithWorstScore() {
        createWorker("10.0.3.1", 0, false);
        createWorker("10.0.3.2", 0, false);
        // 唯一存活者哪怕负载最重也必须被选中
        createWorker("10.0.3.3", 5_000, true);

        ServerStatus result = strategy.select(buildContext(9301L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.3.3", result.getServerIp());
    }

    // ==================== helpers ====================

    /** auto-tpm 开启 + 背压 park 配置：batcher 队列确定性驻留不被 drain。 */
    private void setUpAutoTpmBatcherConfig() {
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchFixedWaitMs(1_000);
        config.setFlexlbBatchSizeMax(10);
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        // Small hard cap so the congestion tests can cross the 0.8 threshold
        // (ceil(0.8×100)=80) with ~90 quick offers: filling 900 items against
        // the default cap of 1024 is slow enough under a loaded CI JVM that
        // the fixed window (1000ms) expires on the early head and the
        // batcher starts draining the queue mid-test (observed as a flaky
        // all-congested fallback on the full-module run).
        config.setFlexlbBatchQueueMaxSize(100);
    }

    /**
     * 注册 endpoint 并 commit 一个 dummy inflight batch，使 batcher 线程因
     * 背压（inflightBatchCount >= max）确定性 park，队列内容不会被 dispatch。
     */
    private PrefillEndpoint parkedEndpoint(String ip) {
        WorkerStatus w = createUnregisteredWorker(ip);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(ip + ":8080", w);
        PrefillEndpoint ep = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ip + ":8080", w);
        createdEndpoints.add(ep);
        // 两 endpoint 使用同样的 predictMs，endpointWaitMs 对称抵消（几 ms 漂移
        // 远小于 batcherEstimatedWaitMs 的队头年龄差 5000ms）
        ep.commitBatch(800_000L + ip.hashCode(), 60_000, List.of());
        return ep;
    }

    /**
     * 向 parked batcher 队列灌入 count 个指定优先级的 item；
     * {@code ageOffsetMs} 直接偏移 enqueuedAtMs，用于控制队头年龄
     * （负值 = 队头已积压 |offset| ms）。
     */
    private void fillQueue(PrefillEndpoint ep, int priority, int count, long idBase, long ageOffsetMs) {
        long now = System.currentTimeMillis();
        for (int i = 0; i < count; i++) {
            Request req = new Request();
            req.setRequestId(idBase + i);
            req.setSeqLen(100);
            req.setPriority(priority);
            BalanceContext ctx = new BalanceContext();
            ctx.setRequest(req);
            ctx.setBudget(ScheduleBudget.forDeadline(priority, now, now + 30_000));
            BatchItem item = new BatchItem(ctx, null, null, null, null, ep, null, now + ageOffsetMs);
            ep.getBatcher().offer(item);
        }
        assertEquals(count, ep.getBatcher().queueSize(),
                "parked batcher queue must retain all items for " + ep.getIp());
    }

    private BalanceContext priorityContext(long requestId, int priority) {
        BalanceContext ctx = buildContext(requestId);
        ctx.getRequest().setPriority(priority);
        long now = System.currentTimeMillis();
        ctx.setBudget(ScheduleBudget.forDeadline(priority, now, now + 60_000));
        return ctx;
    }

    private void createWorker(String ip, long estimatedWaitMs, boolean alive) {
        WorkerStatus w = createUnregisteredWorker(ip);
        w.setAlive(alive);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(ip + ":8080", w);
        PrefillEndpoint ep = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ip + ":8080", w);
        createdEndpoints.add(ep);
        if (estimatedWaitMs > 0) {
            ep.commitBatch(900_000L + ip.hashCode(), estimatedWaitMs, List.of());
        }
    }

    private WorkerStatus createUnregisteredWorker(String ip) {
        WorkerStatus w = new WorkerStatus();
        w.setIp(ip);
        w.setPort(8080);
        w.setGrpcPort(8081);
        w.setAlive(true);
        w.setRole(RoleType.PREFILL);
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10_000);
        cacheStatus.setBlockSize(256);
        w.setCacheStatus(cacheStatus);
        w.setRunningTaskList(new HashMap<>());
        return w;
    }

    private BalanceContext buildContext(long requestId) {
        Request req = new Request();
        req.setSeqLen(500);
        req.setRequestId(requestId);
        req.setBlockCacheKeys(new ArrayList<>(List.of(1L, 2L)));
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(req);
        ctx.setConfig(config);
        return ctx;
    }
}

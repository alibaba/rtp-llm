package org.flexlb.balance.scheduler;

import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.policy.GroupRoutingPolicy;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.LeastLoadDecodeStrategy;
import org.flexlb.balance.strategy.ShortestTTFTStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.StrategyConfigs;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.VitCacheDirectory;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/** Exercises router retries with actual Decode scoring, reservation and rollback. */
class LeastLoadGroupRoutingTest {
    private FlexlbConfig config;
    private ConfigService configService;
    private EngineWorkerStatus engineWorkerStatus;
    private ResourceMeasureFactory measures;
    private GroupRoutingPolicy policy;
    private VitCacheDirectory vitDirectory;
    private DefaultRouter router;
    private LeastLoadDecodeStrategy decode;
    private ReservingBalancer prefill;
    private ReservingBalancer vit;
    private Map<RoleType, LoadBalancer> balancers;
    private Map<LoadBalanceStrategyEnum, LoadBalancer> factory;
    private Map<LoadBalanceStrategyEnum, LoadBalancer> savedFactory;
    private int address;

    @BeforeEach
    @SuppressWarnings("unchecked")
    void setUp() throws Exception {
        clearWorkers();
        address = 0;
        Field factoryField = LoadBalanceStrategyFactory.class.getDeclaredField("loadBalancerFactory");
        factoryField.setAccessible(true);
        factory = (Map<LoadBalanceStrategyEnum, LoadBalancer>) factoryField.get(null);
        savedFactory = new HashMap<>(factory);
        configService = mock(ConfigService.class);
        config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
        for (RoleType role : RoleType.values()) {
            LoadBalanceStrategyFactory.register(config.getStrategyForRoleType(role), mock(LoadBalancer.class));
        }
        measures = mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeMeasure = new DecodeResourceMeasure(configService);
        when(measures.getMeasure(any())).thenReturn(decodeMeasure);
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        decode = new LeastLoadDecodeStrategy(configService, engineWorkerStatus, measures);
        policy = mock(GroupRoutingPolicy.class);
        when(policy.route(any())).thenReturn(GroupRoutingDecision.none());
        vitDirectory = mock(VitCacheDirectory.class);
        router = new DefaultRouter(configService, policy, vitDirectory);
        Field balancersField = DefaultRouter.class.getDeclaredField("loadBalancerMap");
        balancersField.setAccessible(true);
        balancers = (Map<RoleType, LoadBalancer>) balancersField.get(router);
        prefill = new ReservingBalancer(RoleType.PREFILL);
        vit = new ReservingBalancer(RoleType.VIT);
        balancers.put(RoleType.DECODE, decode);
        balancers.put(RoleType.PREFILL, prefill);
        balancers.put(RoleType.VIT, vit);
        balancers.put(RoleType.PDFUSION, new ReservingBalancer(RoleType.PDFUSION));
    }

    @AfterEach
    void tearDown() {
        clearWorkers();
        if (factory != null && savedFactory != null) {
            factory.clear();
            factory.putAll(savedFactory);
        }
    }

    @Test
    void unavailable_prefill_in_lowest_load_group_should_retry_and_restore_decode_reservation() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        WorkerStatus p = worker(RoleType.PREFILL, "b", false);

        Response response = router.route(context(1000));

        assertTrue(response.isSuccess());
        assertEquals(List.of("a", "b"), prefill.groups);
        assertEquals("b", response.getServerStatus().getFirst().getGroup());
        assertRestored(a);
        assertReserved(b, 1000);
        assertReserved(p, 1000);
    }

    @Test
    void real_prefill_admission_should_reject_busy_a_and_allow_healthy_b() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        WorkerStatus pa = worker(RoleType.PREFILL, "a", false);
        WorkerStatus pb = worker(RoleType.PREFILL, "b", false);
        Map<String, TaskInfo> waiting = new HashMap<>();
        for (long id = 1; id <= config.getPrefillQueueSizeThreshold() + 1; id++) {
            waiting.put(String.valueOf(id), task(id, 0));
        }
        pa.setWaitingTaskList(waiting);
        DecodeResourceMeasure decodeMeasure = new DecodeResourceMeasure(configService);
        PrefillResourceMeasure prefillMeasure = new PrefillResourceMeasure(configService);
        when(measures.getMeasure(any())).thenAnswer(invocation -> {
            Object indicator = invocation.getArgument(0);
            if (indicator == decodeMeasure.getResourceMeasureIndicator()) {
                return decodeMeasure;
            }
            if (indicator == prefillMeasure.getResourceMeasureIndicator()) {
                return prefillMeasure;
            }
            return null;
        });
        when(configService.getStrategyConfigs()).thenReturn(new StrategyConfigs());
        CacheAwareService cacheAwareService = mock(CacheAwareService.class);
        when(cacheAwareService.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        ShortestTTFTStrategy realPrefill = new ShortestTTFTStrategy(engineWorkerStatus,
                mock(EngineHealthReporter.class), cacheAwareService, measures, configService);
        balancers.put(RoleType.PREFILL, realPrefill);

        Response response = router.route(context(1000));

        assertTrue(response.isSuccess());
        assertEquals(List.of("b", "b"), response.getServerStatus().stream().map(ServerStatus::getGroup).toList());
        assertEquals(List.of(RoleType.DECODE, RoleType.PREFILL),
                response.getServerStatus().stream().map(ServerStatus::getRole).toList());
        assertRestored(a);
        assertRestored(pa);
        assertEquals(waiting.size(), pa.getWaitingTaskList().size());
        assertReserved(b, 1000);
        assertReserved(pb, 1000);
    }

    @Test
    void exhausted_groups_should_keep_last_prefill_error_and_release_every_reservation() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus anotherA = worker(RoleType.DECODE, "a", true);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        WorkerStatus unavailable = worker(RoleType.PREFILL, "other", false);
        unavailable.setAlive(false);

        Response response = assertTimeoutPreemptively(Duration.ofSeconds(2), () -> router.route(context(1000)));

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("PREFILL unavailable in b"));
        assertEquals(List.of("a", "b"), prefill.groups);
        assertRestored(a);
        assertRestored(anotherA);
        assertRestored(b);
    }

    @Test
    void vit_failure_after_decode_and_prefill_should_rollback_both_before_retry() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        WorkerStatus pa = worker(RoleType.PREFILL, "a", false);
        WorkerStatus pb = worker(RoleType.PREFILL, "b", false);
        WorkerStatus vb = worker(RoleType.VIT, "b", false);

        Response response = router.route(context(1000));

        assertTrue(response.isSuccess());
        assertEquals(List.of("a", "b"), prefill.groups);
        assertEquals(List.of("a", "b"), vit.groups);
        assertEquals(1, prefill.rollbacks);
        assertRestored(a);
        assertRestored(pa);
        assertReserved(b, 1000);
        assertReserved(pb, 1000);
        assertReserved(vb, 1000);
    }

    @Test
    void explicit_group_should_fail_without_spilling_into_available_group() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        worker(RoleType.PREFILL, "b", false);
        when(policy.route(any())).thenReturn(GroupRoutingDecision.of("a", "explicit"));

        Response response = router.route(context(1000));

        assertFalse(response.isSuccess());
        assertEquals(List.of("a"), prefill.groups);
        assertRestored(a);
        assertRestored(b);
    }

    @Test
    void selected_vit_should_pin_group_and_never_spill_or_release_its_reservation() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        worker(RoleType.PREFILL, "b", false);
        WorkerStatus pinnedWorker = worker(RoleType.VIT, "a", false);
        BalanceContext context = context(1000);
        TaskInfo pinnedTask = task(1000, 10);
        pinnedWorker.putLocalTask(1000L, pinnedTask);
        ServerStatus pinned = status(pinnedWorker, RoleType.VIT, 1000);
        context.getRequest().setSelectedVit(pinned);
        when(vitDirectory.validate(any(), any())).thenReturn(pinned);

        Response response = router.route(context);

        assertFalse(response.isSuccess());
        assertEquals(List.of("a"), prefill.groups);
        assertRestored(a);
        assertRestored(b);
        assertReserved(pinnedWorker, 1000);
        assertEquals(0, vit.rollbacks);
    }

    @Test
    void legacy_decode_strategy_should_keep_single_attempt_behavior() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        worker(RoleType.PREFILL, "b", false);
        // Delegation keeps real reservation accounting but deliberately lacks the new strategy type.
        balancers.put(RoleType.DECODE, new LoadBalancer() {
            public ServerStatus select(BalanceContext c, RoleType r, String g) { return decode.select(c, r, g); }
            public void rollBack(String endpoint, long id) { decode.rollBack(endpoint, id); }
        });

        Response response = router.route(context(1000));

        assertFalse(response.isSuccess());
        assertEquals(List.of("a"), prefill.groups);
        assertRestored(a);
        assertRestored(b);
    }

    @Test
    void pdfusion_first_should_keep_single_attempt_and_rollback_fusion_and_decode() {
        WorkerStatus fusion = worker(RoleType.PDFUSION, "a", false);
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        worker(RoleType.PREFILL, "b", false);

        Response response = router.route(context(1000));

        assertFalse(response.isSuccess());
        assertEquals(List.of("a"), prefill.groups);
        assertRestored(fusion);
        assertRestored(a);
        assertRestored(b);
    }

    @Test
    void recovered_group_should_be_eligible_again_on_next_request() {
        WorkerStatus a = worker(RoleType.DECODE, "a", false);
        worker(RoleType.DECODE, "b", true);
        worker(RoleType.PREFILL, "b", false);
        assertTrue(router.route(context(1000)).isSuccess());
        assertRestored(a);
        worker(RoleType.PREFILL, "a", false);

        Response next = router.route(context(1001));

        assertTrue(next.isSuccess());
        assertEquals("a", next.getServerStatus().getFirst().getGroup());
        assertEquals(List.of("a", "b", "a"), prefill.groups);
        assertReserved(a, 1001);
    }

    @Test
    void null_group_should_be_excludable_and_retry_a_named_group() {
        WorkerStatus ungrouped = worker(RoleType.DECODE, null, false);
        WorkerStatus b = worker(RoleType.DECODE, "b", true);
        worker(RoleType.PREFILL, "b", false);
        prefill.rejectNullGroup = true;

        Response response = router.route(context(1000));

        assertTrue(response.isSuccess());
        assertEquals(2, prefill.groups.size());
        assertNull(prefill.groups.getFirst());
        assertEquals("b", prefill.groups.get(1));
        assertRestored(ungrouped);
        assertReserved(b, 1000);
    }

    @Test
    void multiple_endpoints_in_only_null_group_should_fail_once_without_leaks() {
        WorkerStatus first = worker(RoleType.DECODE, null, false);
        WorkerStatus second = worker(RoleType.DECODE, null, true);
        worker(RoleType.PREFILL, "other", false);
        prefill.rejectNullGroup = true;

        Response response = assertTimeoutPreemptively(Duration.ofSeconds(2), () -> router.route(context(1000)));

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("PREFILL unavailable in null"));
        assertEquals(1, prefill.groups.size());
        assertRestored(first);
        assertRestored(second);
    }

    private void clearWorkers() {
        for (RoleType role : RoleType.values()) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role).clear();
        }
    }

    private WorkerStatus worker(RoleType role, String group, boolean loaded) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp("127.0.0." + (++address));
        worker.setPort(8080);
        worker.setAlive(true);
        worker.setRole(role.getCode());
        worker.setGroup(group);
        worker.getUsedKvCacheTokens().set(100);
        worker.getAvailableKvCacheTokens().set(9900);
        if (loaded) {
            worker.setRunningTaskList(Map.of("1", task(1, 0)));
        }
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role).put(worker.getIpPort(), worker);
        return worker;
    }

    private BalanceContext context(long id) {
        Request request = new Request();
        request.setRequestId(id);
        request.setSeqLen(10L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        return context;
    }

    private TaskInfo task(long id, long inputLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(id);
        task.setInputLength(inputLength);
        return task;
    }

    private ServerStatus status(WorkerStatus worker, RoleType role, long id) {
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(role);
        result.setGroup(worker.getGroup());
        result.setServerIp(worker.getIp());
        result.setHttpPort(worker.getPort());
        result.setRequestId(id);
        return result;
    }

    private void assertRestored(WorkerStatus worker) {
        assertTrue(worker.getLocalTaskMap().isEmpty(), worker.getIpPort());
        assertEquals(100, worker.getUsedKvCacheTokens().get(), worker.getIpPort());
        assertEquals(9900, worker.getAvailableKvCacheTokens().get(), worker.getIpPort());
        assertEquals(0, worker.getRunningQueueTime().get(), worker.getIpPort());
    }

    private void assertReserved(WorkerStatus worker, long id) {
        assertEquals(1, worker.getLocalTaskMap().size(), worker.getIpPort());
        assertTrue(worker.getLocalTaskMap().containsKey(id));
        assertEquals(110, worker.getUsedKvCacheTokens().get(), worker.getIpPort());
        assertEquals(9890, worker.getAvailableKvCacheTokens().get(), worker.getIpPort());
    }

    /** Group eligibility double; reservations and rollback use production WorkerStatus accounting. */
    private class ReservingBalancer implements LoadBalancer {
        private final RoleType role;
        private final List<String> groups = new ArrayList<>();
        private int rollbacks;
        private boolean rejectNullGroup;

        private ReservingBalancer(RoleType role) { this.role = role; }

        @Override
        public ServerStatus select(BalanceContext context, RoleType ignored, String group) {
            groups.add(group);
            if (!(rejectNullGroup && group == null)) {
                for (WorkerStatus worker : EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role).values()) {
                    if (worker.isAlive() && (Objects.equals(group, worker.getGroup())
                            || (role == RoleType.PDFUSION && group == null))) {
                        worker.putLocalTask(context.getRequestId(), task(context.getRequestId(), context.getRequest().getSeqLen()));
                        return status(worker, role, context.getRequestId());
                    }
                }
            }
            ServerStatus failure = new ServerStatus();
            failure.setSuccess(false);
            failure.setMessage(role + " unavailable in " + group);
            return failure;
        }

        @Override
        public void rollBack(String endpoint, long id) {
            rollbacks++;
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role).get(endpoint).removeLocalTask(id);
        }
    }
}

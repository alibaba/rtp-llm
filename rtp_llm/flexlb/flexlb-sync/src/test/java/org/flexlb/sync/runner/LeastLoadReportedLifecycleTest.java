package org.flexlb.sync.runner;

import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.LeastLoadDecodeStrategy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.schedule.ExpirationCleaner;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Actual status-runner/converter and expiry lifecycle; no global admission or live-KV guarantee. */
class LeastLoadReportedLifecycleTest {
    private FlexlbConfig config;
    private LeastLoadDecodeStrategy strategy;
    private ExpirationCleaner cleaner;
    private EngineGrpcService grpc;
    private EngineHealthReporter health;
    private Map<String, WorkerStatus> workers;
    private Map<LoadBalanceStrategyEnum, LoadBalancer> factory;
    private Map<LoadBalanceStrategyEnum, LoadBalancer> savedFactory;

    @BeforeEach
    @SuppressWarnings("unchecked")
    void setUp() throws Exception {
        workers = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();
        workers.clear();
        Field field = LoadBalanceStrategyFactory.class.getDeclaredField("loadBalancerFactory");
        field.setAccessible(true);
        factory = (Map<LoadBalanceStrategyEnum, LoadBalancer>) field.get(null);
        savedFactory = new HashMap<>(factory);
        config = new FlexlbConfig();
        config.setDecodeConcurrencyLimit(10);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        ResourceMeasureFactory measures = mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(service);
        when(measures.getMeasure(any())).thenReturn(measure);
        strategy = new LeastLoadDecodeStrategy(service,
                new EngineWorkerStatus(new ModelMetaConfig()), measures);
        cleaner = new ExpirationCleaner(mock(FlexMonitor.class));
        grpc = mock(EngineGrpcService.class);
        health = mock(EngineHealthReporter.class);
    }

    @AfterEach
    void tearDown() {
        workers.clear();
        if (factory != null && savedFactory != null) {
            factory.clear();
            factory.putAll(savedFactory);
        }
    }

    @Test
    void reports_and_same_version_heartbeat_should_deduplicate_then_finish_local_requests() {
        WorkerStatus worker = worker("127.0.0.1");
        assertSelected(worker, 1001);
        assertEquals(TaskStateEnum.IN_TRANSIT, worker.getLocalTaskMap().get(1001L).getTaskState());
        assertEquals(1, worker.getDecodeConcurrency());

        report(worker, status(1).addRunningTaskInfo(task(1001, true)).build());
        assertTrue(worker.getWaitingTaskList().containsKey("1001"));
        assertTrue(worker.getRunningTaskList().isEmpty());
        assertEquals(TaskStateEnum.CONFIRMED, worker.getLocalTaskMap().get(1001L).getTaskState());
        assertEquals(1, worker.getDecodeConcurrency());

        report(worker, status(2).addRunningTaskInfo(task(1001, false)).build());
        assertTrue(worker.getWaitingTaskList().isEmpty());
        assertTrue(worker.getRunningTaskList().containsKey("1001"));
        assertEquals(TaskStateEnum.RUNNING, worker.getLocalTaskMap().get(1001L).getTaskState());
        assertEquals(1, worker.getDecodeConcurrency());
        assertSelected(worker, 1002);

        // The heartbeat branch refreshes basic health, while preserving reported-map snapshots.
        report(worker, status(2).setAlive(false).setDpSize(4).setTpSize(2)
                .addRunningTaskInfo(task(1001, false)).build());
        assertFalse(worker.isAlive());
        assertEquals(4, worker.getDpSize());
        assertEquals(2, worker.getTpSize());
        assertEquals(2, worker.getStatusVersion().get());
        assertEquals(2, worker.getLocalTaskMap().size());
        assertEquals(2, worker.getDecodeConcurrency());
        assertFalse(strategy.select(context(1003), RoleType.DECODE, null).isSuccess());
        report(worker, status(2).addRunningTaskInfo(task(1001, false)).build());
        assertTrue(worker.isAlive());
        assertEquals(2, worker.getDecodeConcurrency());

        report(worker, status(3).setLatestFinishedVersion(3)
                .addFinishedTaskList(task(1001, false))
                .addRunningTaskInfo(task(1002, false)).build());
        assertFalse(worker.getLocalTaskMap().containsKey(1001L));
        assertEquals(1, worker.getDecodeConcurrency());
        assertEquals(3, worker.getLatestFinishedTaskVersion().get());
        report(worker, status(4).setLatestFinishedVersion(4)
                .addFinishedTaskList(task(1002, false)).build());
        assertTrue(worker.getLocalTaskMap().isEmpty());
        assertEquals(0, worker.getDecodeConcurrency());
        assertSelected(worker, 1003);
        strategy.rollBack(worker.getIpPort(), 1003);
        assertEquals(0, worker.getDecodeConcurrency());
    }

    @Test
    void missing_previously_reported_request_should_be_lost_then_cleaned() {
        WorkerStatus worker = worker("127.0.0.1");
        assertSelected(worker, 1001);
        report(worker, status(1).addRunningTaskInfo(task(1001, false)).build());
        report(worker, status(2).build());
        assertEquals(TaskStateEnum.LOST, worker.getLocalTaskMap().get(1001L).getTaskState());
        assertEquals(1, worker.getDecodeConcurrency());

        cleaner.doClean(workers, RoleType.DECODE);

        assertTrue(workers.containsKey(worker.getIpPort()));
        assertTrue(worker.getLocalTaskMap().isEmpty());
        assertEquals(0, worker.getDecodeConcurrency());
        assertSelected(worker, 1002);
    }

    @Test
    void unreported_in_transit_request_should_expire_without_expiring_healthy_endpoint() {
        WorkerStatus worker = worker("127.0.0.1");
        report(worker, status(1).build());
        assertSelected(worker, 1001);
        worker.getLocalTaskMap().get(1001L).setLastActiveTimeUs(0);

        cleaner.doClean(workers, RoleType.DECODE);

        assertTrue(workers.containsKey(worker.getIpPort()));
        assertTrue(worker.getLocalTaskMap().isEmpty());
        assertEquals(0, worker.getDecodeConcurrency());
        assertSelected(worker, 1002);
    }

    @Test
    void expired_endpoint_should_be_removed_and_next_selection_should_use_healthy_endpoint() {
        WorkerStatus expired = worker("127.0.0.1");
        assertSelected(expired, 1001);
        expired.getStatusLastUpdateTime().set(0);
        WorkerStatus healthy = worker("127.0.0.2");
        report(healthy, status(1).build());

        cleaner.doClean(workers, RoleType.DECODE);

        assertFalse(workers.containsKey(expired.getIpPort()));
        assertTrue(workers.containsKey(healthy.getIpPort()));
        strategy.rollBack(expired.getIpPort(), 1001); // removed endpoints are safe to roll back
        assertSelected(healthy, 1002);
        assertEquals(1, healthy.getDecodeConcurrency());
    }

    private void report(WorkerStatus worker, EngineRpcService.WorkerStatusPB status) {
        when(grpc.getWorkerStatus(eq(worker.getIp()), eq(worker.getPort() + 1), anyLong(),
                anyLong(), eq(RoleType.DECODE))).thenReturn(status);
        worker.getStatusCheckInProgress().set(true);
        new GrpcWorkerStatusRunner("test-model", worker.getIpPort(), "test-site", RoleType.DECODE,
                "test-group", worker, health, grpc, 20).run();
        assertFalse(worker.getStatusCheckInProgress().get());
    }

    private EngineRpcService.WorkerStatusPB.Builder status(long version) {
        return EngineRpcService.WorkerStatusPB.newBuilder().setRole(RoleType.DECODE.getCode())
                .setStatusVersion(version).setAlive(true).setDpSize(1).setTpSize(1);
    }

    private EngineRpcService.TaskInfoPB task(long id, boolean waiting) {
        return EngineRpcService.TaskInfoPB.newBuilder().setRequestId(id).setInputLength(10)
                .setIsWaiting(waiting).build();
    }

    private WorkerStatus worker(String ip) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8080);
        worker.setRole(RoleType.DECODE.getCode());
        worker.setAlive(true);
        worker.getUsedKvCacheTokens().set(100);
        worker.getAvailableKvCacheTokens().set(999900);
        worker.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        workers.put(worker.getIpPort(), worker);
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

    private void assertSelected(WorkerStatus worker, long id) {
        ServerStatus selected = strategy.select(context(id), RoleType.DECODE, null);
        assertTrue(selected.isSuccess());
        assertEquals(worker.getIp(), selected.getServerIp());
        assertTrue(worker.getLocalTaskMap().containsKey(id));
    }
}

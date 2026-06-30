package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author claude
 * description: RandomStrategy unit tests
 * date: 2025/10/20
 */
@Slf4j
class RandomStrategyTest {

    private RandomStrategy randomStrategy;
    private ResourceMeasure resourceMeasure;
    private EndpointRegistry endpointRegistry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        endpointRegistry = new EndpointRegistry(configService, null, Mockito.mock(BatchSchedulerReporter.class));
        resourceMeasure = Mockito.mock(ResourceMeasure.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any(WorkerEndpoint.class))).thenReturn(true);
        randomStrategy = new RandomStrategy(
                new EngineWorkerStatus(new ModelMetaConfig(), endpointRegistry),
                configService,
                resourceMeasureFactory,
                endpointRegistry);
    }

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    /** Register a mock PrefillEndpoint for the given ipPort and WorkerStatus. */
    private void registerPrefill(String ipPort, WorkerStatus ws) {
        PrefillEndpoint ep = Mockito.mock(PrefillEndpoint.class);
        Mockito.when(ep.getIp()).thenReturn(ws.getIp());
        Mockito.when(ep.getHttpPort()).thenReturn(ws.getPort());
        Mockito.when(ep.getGrpcPort()).thenReturn(ws.getGrpcPort());
        Mockito.when(ep.getStatus()).thenReturn(ws);
        Mockito.when(ep.ipPort()).thenReturn(ipPort);
        endpointRegistry.putPrefill(ipPort, ep);
    }

    /** Register a mock DecodeEndpoint for the given ipPort and WorkerStatus. */
    private void registerDecode(String ipPort, WorkerStatus ws) {
        DecodeEndpoint ep = Mockito.mock(DecodeEndpoint.class);
        Mockito.when(ep.getIp()).thenReturn(ws.getIp());
        Mockito.when(ep.getHttpPort()).thenReturn(ws.getPort());
        Mockito.when(ep.getGrpcPort()).thenReturn(ws.getGrpcPort());
        Mockito.when(ep.getStatus()).thenReturn(ws);
        Mockito.when(ep.ipPort()).thenReturn(ipPort);
        endpointRegistry.putDecode(ipPort, ep);
    }

    @Test
    void should_return_error_when_no_workers_available() {
        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        assertFalse(result.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg(), result.getMessage());
    }

    @Test
    void should_return_error_when_worker_map_is_empty() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        assertFalse(result.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg(), result.getMessage());
    }

    @Test
    void should_return_success_when_workers_available() {
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1");
        prefillStatusMap.put("127.0.0.1:8080", workerStatus);
        registerPrefill("127.0.0.1:8080", workerStatus);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
    }

    @Test
    void should_select_randomly_from_available_workers() {
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");

        prefillStatusMap.put("127.0.0.1:8080", worker1);
        prefillStatusMap.put("127.0.0.2:8080", worker2);
        prefillStatusMap.put("127.0.0.3:8080", worker3);
        registerPrefill("127.0.0.1:8080", worker1);
        registerPrefill("127.0.0.2:8080", worker2);
        registerPrefill("127.0.0.3:8080", worker3);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result1 = randomStrategy.select(balanceContext, RoleType.PREFILL, null);
        ServerStatus result2 = randomStrategy.select(balanceContext, RoleType.PREFILL, null);
        ServerStatus result3 = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        assertTrue(result1.isSuccess());
        assertTrue(result2.isSuccess());
        assertTrue(result3.isSuccess());
    }

    @Test
    void should_work_with_different_role_types() {
        WorkerStatus prefillWorker = createWorkerStatus("127.0.0.1");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.1:8080", prefillWorker);
        registerPrefill("127.0.0.1:8080", prefillWorker);

        WorkerStatus decodeWorker = createWorkerStatus("127.0.0.2");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.2:8080", decodeWorker);
        registerDecode("127.0.0.2:8080", decodeWorker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus prefillResult = randomStrategy.select(balanceContext, RoleType.PREFILL, null);
        ServerStatus decodeResult = randomStrategy.select(balanceContext, RoleType.DECODE, null);

        assertTrue(prefillResult.isSuccess());
        assertTrue(decodeResult.isSuccess());
    }

    @Test
    void should_work_with_group_parameter() {
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-a");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, "group-a");

        assertTrue(result.isSuccess());
    }

    @Test
    void should_return_error_when_no_workers_in_specified_group() {
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-a");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, "group-b");

        assertFalse(result.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg(), result.getMessage());
    }

    @Test
    void should_register_strategy_in_factory() {
        RandomStrategy strategyFromFactory = (RandomStrategy) LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.RANDOM);
        assertNotNull(strategyFromFactory);
        assertSame(randomStrategy, strategyFromFactory);
    }

    @Test
    void should_distribute_requests_uniformly_across_workers() {
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");

        prefillStatusMap.put("127.0.0.1:8080", worker1);
        prefillStatusMap.put("127.0.0.2:8080", worker2);
        prefillStatusMap.put("127.0.0.3:8080", worker3);
        registerPrefill("127.0.0.1:8080", worker1);
        registerPrefill("127.0.0.2:8080", worker2);
        registerPrefill("127.0.0.3:8080", worker3);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        int totalRuns = 10000;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.getRequest().setRequestId(1000L + i);
            ServerStatus status = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        int expectedCountPerWorker = totalRuns / 3;
        double tolerance = 0.10;

        for (String ip : selectionCount.keySet()) {
            int count = selectionCount.get(ip);
            double ratio = (double) count / totalRuns;
            assertTrue(ratio >= 0.33 - tolerance && ratio <= 0.33 + tolerance,
                    "Worker " + ip + " selection ratio " + ratio + " is outside expected range");
        }

        log.info("Uniform distribution test: worker1={}, worker2={}, worker3={}",
                selectionCount.get("127.0.0.1"), selectionCount.get("127.0.0.2"), selectionCount.get("127.0.0.3"));
    }

    @Test
    void should_skip_dead_workers() {
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus deadWorker = createWorkerStatus("127.0.0.1");
        deadWorker.setAlive(false);
        prefillStatusMap.put("127.0.0.1:8080", deadWorker);
        registerPrefill("127.0.0.1:8080", deadWorker);

        WorkerStatus aliveWorker = createWorkerStatus("127.0.0.2");
        prefillStatusMap.put("127.0.0.2:8080", aliveWorker);
        registerPrefill("127.0.0.2:8080", aliveWorker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        int totalRuns = 100;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.getRequest().setRequestId(1000L + i);
            ServerStatus status = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        assertFalse(selectionCount.containsKey("127.0.0.1"));
        assertEquals(totalRuns, selectionCount.getOrDefault("127.0.0.2", 0));
    }

    @Test
    void should_skip_workers_rejected_by_resource_measure() {
        Map<String, WorkerStatus> decodeStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus unavailableWorker = createWorkerStatus("127.0.0.1");
        WorkerStatus availableWorker = createWorkerStatus("127.0.0.2");
        decodeStatusMap.put("127.0.0.1:8080", unavailableWorker);
        decodeStatusMap.put("127.0.0.2:8080", availableWorker);
        registerDecode("127.0.0.1:8080", unavailableWorker);
        registerDecode("127.0.0.2:8080", availableWorker);

        Mockito.when(resourceMeasure.isResourceAvailable(
                Mockito.argThat(ep -> ep != null && "127.0.0.1".equals(ep.getIp())))).thenReturn(false);
        Mockito.when(resourceMeasure.isResourceAvailable(
                Mockito.argThat(ep -> ep != null && "127.0.0.2".equals(ep.getIp())))).thenReturn(true);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(12345L);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.DECODE, null);

        assertTrue(result.isSuccess());
        assertEquals("127.0.0.2", result.getServerIp());
    }

    @Test
    void should_properly_set_server_status_fields() {
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-x");
        prefillStatusMap.put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();
        req.setSeqLen(1000);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("127.0.0.1", result.getServerIp());
        assertEquals(8080, result.getHttpPort());
        assertEquals(RoleType.PREFILL, result.getRole());
        assertEquals("group-x", result.getGroup());
    }

    @Test
    void should_handle_null_request_id() {
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillStatusMap.clear();

        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        prefillStatusMap.put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("127.0.0.1", result.getServerIp());
    }

    @Test
    void should_handle_rollback_without_error() {
        randomStrategy.rollBack("127.0.0.1:8080", 0);
    }

    private WorkerStatus createWorkerStatus(String ip) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        workerStatus.setPort(8080);
        workerStatus.setSite("test-site");
        workerStatus.setAlive(true);
        return workerStatus;
    }
}

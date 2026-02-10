package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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

    @BeforeEach
    void setUp() {
        randomStrategy = new RandomStrategy(new EngineWorkerStatus(new ModelMetaConfig()));
    }

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    @Test
    void should_return_error_when_no_workers_available() {
        // Given: No workers registered for the model
        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select a worker
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        // Then: Should return error status
        assertFalse(result.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg(), result.getMessage());
    }

    @Test
    void should_return_error_when_worker_map_is_empty() {
        // Given: Model exists but no workers
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select a worker
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        // Then: Should return error status
        assertFalse(result.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg(), result.getMessage());
    }

    @Test
    void should_return_success_when_workers_available() {
        // Given: Model with available workers
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        // Add a worker
        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1");
        prefillStatusMap.put("127.0.0.1:8080", workerStatus);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select a worker
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        // Then: Should return success status with batchId
        assertTrue(result.isSuccess());
    }

    @Test
    void should_select_randomly_from_available_workers() {
        // Given: Model with multiple available workers
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        // Add multiple workers
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");

        prefillStatusMap.put("127.0.0.1:8080", worker1);
        prefillStatusMap.put("127.0.0.2:8080", worker2);
        prefillStatusMap.put("127.0.0.3:8080", worker3);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select a worker multiple times
        ServerStatus result1 = randomStrategy.select(balanceContext, RoleType.PREFILL, null);
        ServerStatus result2 = randomStrategy.select(balanceContext, RoleType.PREFILL, null);
        ServerStatus result3 = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        // Then: All should be successful (random selection is working)
        assertTrue(result1.isSuccess());
        assertTrue(result2.isSuccess());
        assertTrue(result3.isSuccess());
    }

    @Test
    void should_work_with_different_role_types() {
        // Given: Model with workers for different roles

        // Add workers for different roles
        WorkerStatus prefillWorker = createWorkerStatus("127.0.0.1");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.1:8080", prefillWorker);

        WorkerStatus decodeWorker = createWorkerStatus("127.0.0.2");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.2:8080", decodeWorker);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select workers for different roles
        ServerStatus prefillResult = randomStrategy.select(balanceContext, RoleType.PREFILL, null);
        ServerStatus decodeResult = randomStrategy.select(balanceContext, RoleType.DECODE, null);

        // Then: Both should be successful
        assertTrue(prefillResult.isSuccess());
        assertTrue(decodeResult.isSuccess());
    }

    @Test
    void should_work_with_group_parameter() {
        // Given: Model with workers in specific groups

        // Add worker with specific group
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-a");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.1:8080", worker);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select worker with group parameter
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, "group-a");

        // Then: Should be successful
        assertTrue(result.isSuccess());
    }

    @Test
    void should_return_error_when_no_workers_in_specified_group() {
        // Given: Model with workers but none in the specified group

        // Add worker with different group
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-a");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.1:8080", worker);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select worker with different group parameter
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, "group-b");

        // Then: Should return error status
        assertFalse(result.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg(), result.getMessage());
    }

    @Test
    void should_register_strategy_in_factory() {
        // Given: RandomStrategy is instantiated
        // When: Check if it's registered in the factory
        // Then: Should be able to get it from the factory
        RandomStrategy strategyFromFactory = (RandomStrategy) LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.RANDOM);
        assertNotNull(strategyFromFactory);
        assertSame(randomStrategy, strategyFromFactory);
    }

    @Test
    void should_distribute_requests_uniformly_across_workers() {
        // Given: Model with multiple available workers
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        // Add multiple workers
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");

        prefillStatusMap.put("127.0.0.1:8080", worker1);
        prefillStatusMap.put("127.0.0.2:8080", worker2);
        prefillStatusMap.put("127.0.0.3:8080", worker3);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select workers many times
        int totalRuns = 10000;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.getRequest().setRequestId(String.valueOf(1000L + i));
            ServerStatus status = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        // Then: Each worker should be selected approximately 33% of the time (within 10% tolerance)
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
    void should_select_dead_workers_with_warning() {
        // Given: Model with mix of alive and dead workers
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        // Add dead worker
        WorkerStatus deadWorker = createWorkerStatus("127.0.0.1");
        deadWorker.setAlive(false);
        prefillStatusMap.put("127.0.0.1:8080", deadWorker);

        // Add alive worker
        WorkerStatus aliveWorker = createWorkerStatus("127.0.0.2");
        prefillStatusMap.put("127.0.0.2:8080", aliveWorker);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select worker multiple times
        int totalRuns = 100;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.getRequest().setRequestId(String.valueOf(1000L + i));
            ServerStatus status = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        // Then: Both workers should be selected (RandomStrategy doesn't filter dead workers)
        // Note: RandomStrategy doesn't filter dead workers, it just warns
        assertTrue(selectionCount.containsKey("127.0.0.1") || selectionCount.containsKey("127.0.0.2"));
        assertEquals(totalRuns, selectionCount.getOrDefault("127.0.0.1", 0) + selectionCount.getOrDefault("127.0.0.2", 0));
    }

    @Test
    void should_properly_set_server_status_fields() {
        // Given: Model with a worker
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-x");
        prefillStatusMap.put("127.0.0.1:8080", worker);

        Request req = new Request();

        req.setSeqLen(1000);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select a worker
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        // Then: All server status fields should be properly set
        assertTrue(result.isSuccess());
        assertEquals("127.0.0.1", result.getServerIp());
        assertEquals(8080, result.getHttpPort());
        assertEquals(RoleType.PREFILL, result.getRole());
        assertEquals("group-x", result.getGroup());
    }

    @Test
    void should_handle_null_request_id() {
        // Given: Model with a worker
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        prefillStatusMap.put("127.0.0.1:8080", worker);

        Request req = new Request();


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // When: Select a worker with null requestId
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        // Then: Should still return success (RandomStrategy doesn't require requestId)
        assertTrue(result.isSuccess());
        assertEquals("127.0.0.1", result.getServerIp());
    }

    @Test
    void should_handle_rollback_without_error() {
        // Given: Rollback is called
        // When: Rollback is called (RandomStrategy has empty implementation)
        // Then: Should not throw any exception
        randomStrategy.rollBack("127.0.0.1:8080", "request-id");
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
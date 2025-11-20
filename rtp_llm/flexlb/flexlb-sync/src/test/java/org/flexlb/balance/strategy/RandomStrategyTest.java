package org.flexlb.balance.strategy;

import org.flexlb.balance.LoadBalanceStrategyFactory;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
class RandomStrategyTest {

    private RandomStrategy randomStrategy;

    @BeforeEach
    void setUp() {
        // Create RandomStrategy instance
        randomStrategy = new RandomStrategy(new EngineWorkerStatus(new ModelMetaConfig()));
    }

    @AfterEach
    void tearDown() {
        // Clear the static map after each test
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.clear();
    }

    @Test
    void should_return_error_when_no_workers_available() {
        // Given: No workers registered for the model
        MasterRequest req = new MasterRequest();
        req.setModel("test-model");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

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
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

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
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP
                .get("test-model").getPrefillStatusMap();

        // Add a worker
        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1");
        prefillStatusMap.put("127.0.0.1:8080", workerStatus);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

        // When: Select a worker
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, null);

        // Then: Should return success status with batchId
        assertTrue(result.isSuccess());
    }

    @Test
    void should_select_randomly_from_available_workers() {
        // Given: Model with multiple available workers
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP
                .get("test-model").getPrefillStatusMap();

        // Add multiple workers
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");

        prefillStatusMap.put("127.0.0.1:8080", worker1);
        prefillStatusMap.put("127.0.0.2:8080", worker2);
        prefillStatusMap.put("127.0.0.3:8080", worker3);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

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
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.get("test-model");

        // Add workers for different roles
        WorkerStatus prefillWorker = createWorkerStatus("127.0.0.1");
        modelStatus.getPrefillStatusMap().put("127.0.0.1:8080", prefillWorker);

        WorkerStatus decodeWorker = createWorkerStatus("127.0.0.2");
        modelStatus.getDecodeStatusMap().put("127.0.0.2:8080", decodeWorker);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

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
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.get("test-model");

        // Add worker with specific group
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-a");
        modelStatus.getPrefillStatusMap().put("127.0.0.1:8080", worker);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

        // When: Select worker with group parameter
        ServerStatus result = randomStrategy.select(balanceContext, RoleType.PREFILL, "group-a");

        // Then: Should be successful
        assertTrue(result.isSuccess());
    }

    @Test
    void should_return_error_when_no_workers_in_specified_group() {
        // Given: Model with workers but none in the specified group
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.get("test-model");

        // Add worker with different group
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.setGroup("group-a");
        modelStatus.getPrefillStatusMap().put("127.0.0.1:8080", worker);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

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
        RandomStrategy strategyFromFactory = (RandomStrategy) LoadBalanceStrategyFactory.getLoadBalanceStrategy(LoadBalanceStrategyEnum.RANDOM);
        assertNotNull(strategyFromFactory);
        assertSame(randomStrategy, strategyFromFactory);
    }

    /**
     * Helper method to create a WorkerStatus object for testing
     */
    private WorkerStatus createWorkerStatus(String ip) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        workerStatus.setPort(8080);
        workerStatus.setSite("test-site");
        workerStatus.setAlive(true);
        return workerStatus;
    }
}
package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DefaultRouterTest {

    @Mock
    private ConfigService configService;

    @Mock
    private ModelMetaConfig modelMetaConfig;

    @Mock
    private FlexlbConfig loadBalanceConfig;

    @Mock
    private LoadBalancer prefillLoadBalancer;

    @Mock
    private LoadBalancer decodeLoadBalancer;

    @Mock
    private LoadBalancer vitLoadBalancer;

    @Mock
    private LoadBalancer fusionLoadBalancer;

    @Mock
    private BalanceContext balanceContext;

    @Mock
    private Request request;

    private DefaultRouter defaultRouter;

    @BeforeEach
    void setUp() {
        // Clear all status maps
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();

        // Mock config service
        when(configService.loadBalanceConfig()).thenReturn(loadBalanceConfig);
        lenient().when(loadBalanceConfig.getLoadBalanceStrategy()).thenReturn(LoadBalanceStrategyEnum.SHORTEST_TTFT);
        when(loadBalanceConfig.getStrategyForRoleType(any(RoleType.class))).thenAnswer(inv -> {
            RoleType roleType = inv.getArgument(0);
            if (roleType == RoleType.DECODE) {
                return LoadBalanceStrategyEnum.WEIGHTED_CACHE;
            }
            if (roleType == RoleType.PDFUSION) {
                return LoadBalanceStrategyEnum.RANDOM;
            }
            return LoadBalanceStrategyEnum.SHORTEST_TTFT;
        });
        // batchStrategy is decoupled from regular strategy: defaults to ROUND_ROBIN.
        // mockStaticLoadBalanceStrategyFactory below resets the batch balancer to a plain
        // (non-batch-capable) mock; tests that exercise the batch path swap in a scripted
        // batch-capable impl via replaceBatchLoadBalancer.
        lenient().when(loadBalanceConfig.getBatchLoadBalanceStrategy())
                .thenReturn(LoadBalanceStrategyEnum.ROUND_ROBIN);
        lenient().when(loadBalanceConfig.getBatchScheduleMaxCount()).thenReturn(1000);
        // Default: no configured service routes — role inference falls back to the runtime
        // view, matching the pre-existing single-role tests. Multi-role-config tests
        // override this stub explicitly.
        lenient().when(modelMetaConfig.getConfiguredRoleTypes()).thenReturn(List.of());

        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, prefillLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, decodeLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, vitLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, fusionLoadBalancer);
        // batchStrategy default is ROUND_ROBIN; the constructor resolves it via factory before
        // mockStaticLoadBalanceStrategyFactory swaps the map, so a placeholder must be registered.
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.ROUND_ROBIN, fusionLoadBalancer);

        // Create scheduler instance
        defaultRouter = new DefaultRouter(configService, modelMetaConfig);

        // Mock LoadBalanceStrategyFactory to return our mock load balancers
        mockStaticLoadBalanceStrategyFactory();

        // Mock balance context
        lenient().when(balanceContext.getRequest()).thenReturn(request);
        lenient().when(balanceContext.getRequestId()).thenReturn(12345L);
    }

    @org.junit.jupiter.api.AfterEach
    void tearDown() {
        // Clear all status maps after each test
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    // Helper method to mock the static LoadBalanceStrategyFactory
    private void mockStaticLoadBalanceStrategyFactory() {
        try {
            // Use reflection to set the loadBalancerMap in DefaultRouter
            Field loadBalancerMapField = DefaultRouter.class.getDeclaredField("loadBalancerMap");
            loadBalancerMapField.setAccessible(true);

            @SuppressWarnings("unchecked")
            Map<RoleType, LoadBalancer> loadBalancerMap = (Map<RoleType, LoadBalancer>) loadBalancerMapField.get(defaultRouter);

            // Put mocked LoadBalancer instances into the map
            loadBalancerMap.put(RoleType.PREFILL, prefillLoadBalancer);
            loadBalancerMap.put(RoleType.DECODE, decodeLoadBalancer);
            loadBalancerMap.put(RoleType.VIT, vitLoadBalancer);
            loadBalancerMap.put(RoleType.PDFUSION, fusionLoadBalancer);

            // Reset the batch balancer to a plain mock so tests start from a known state.
            // Tests that exercise the batch path with a scripted batch-capable LB call
            // replaceBatchLoadBalancer.
            replaceBatchLoadBalancer(fusionLoadBalancer);
        } catch (Exception e) {
            fail("Failed to mock LoadBalanceStrategyFactory: " + e.getMessage());
        }
    }

    @Test
    void should_return_response_with_no_available_worker_error_when_worker_status_is_null() {
        // Setup - clear role type list
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertNotNull(response, "Response should not be null");
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode(), "Error code should match NO_AVAILABLE_WORKER");
        // Note: The method logs an error but doesn't fail when status is null
    }

    @Test
    void should_return_response_with_no_available_worker_error_when_model_not_in_worker_status_map() {
        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertNotNull(response, "Response should not be null");
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode(), "Error code should match NO_AVAILABLE_WORKER");
        // Note: The method logs an error but doesn't fail when model is missing
    }

    @Test
    void should_return_success_response_with_prefill_and_decode_servers_when_prefill_selection_succeeds() {
        // Setup - add dummy workers to trigger role types
        org.flexlb.dao.master.WorkerStatus dummyPrefillWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyPrefillWorker.setIp("192.168.1.1");
        dummyPrefillWorker.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("192.168.1.1:8080", dummyPrefillWorker);

        org.flexlb.dao.master.WorkerStatus dummyDecodeWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyDecodeWorker.setIp("192.168.1.2");
        dummyDecodeWorker.setPort(8081);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("192.168.1.2:8081", dummyDecodeWorker);

        ServerStatus prefillServerStatus = new ServerStatus();
        prefillServerStatus.setSuccess(true);
        prefillServerStatus.setServerIp("192.168.1.1");
        prefillServerStatus.setHttpPort(8080);
        prefillServerStatus.setGroup("group1");
        prefillServerStatus.setRole(RoleType.PREFILL);
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), isNull())).thenReturn(prefillServerStatus);

        ServerStatus decodeServerStatus = new ServerStatus();
        decodeServerStatus.setSuccess(true);
        decodeServerStatus.setServerIp("192.168.1.2");
        decodeServerStatus.setHttpPort(8081);
        decodeServerStatus.setRole(RoleType.DECODE);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any())).thenReturn(decodeServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(2, response.getServerStatus().size(), "Should have 2 server statuses");
    }

    @Test
    void should_return_response_with_no_prefill_worker_error_when_prefill_selection_fails() {
        // Setup - add dummy worker to trigger role type
        org.flexlb.dao.master.WorkerStatus dummyPrefillWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyPrefillWorker.setIp("192.168.1.1");
        dummyPrefillWorker.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("192.168.1.1:8080", dummyPrefillWorker);

        ServerStatus prefillServerStatus = new ServerStatus();
        prefillServerStatus.setSuccess(false);
        prefillServerStatus.setMessage("No prefill worker available");
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), isNull())).thenReturn(prefillServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode(), "Error code should match");
        assertNotNull(response.getErrorMessage(), "Error message should not be null");
    }

    @Test
    void should_return_success_response_with_fusion_server_when_pdfusion_selection_succeeds() {
        // Setup - add dummy worker to trigger role type
        org.flexlb.dao.master.WorkerStatus dummyFusionWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyFusionWorker.setIp("192.168.1.3");
        dummyFusionWorker.setPort(8082);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.3:8082", dummyFusionWorker);

        ServerStatus fusionServerStatus = new ServerStatus();
        fusionServerStatus.setSuccess(true);
        fusionServerStatus.setServerIp("192.168.1.3");
        fusionServerStatus.setHttpPort(8082);
        fusionServerStatus.setGroup("group2");
        fusionServerStatus.setRequestId(54321L);
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull())).thenReturn(fusionServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(1, response.getServerStatus().size(), "Should have 1 server status");
    }

    @Test
    void should_batch_schedule_success_when_single_role_registered_and_strategy_supports_batch() {
        // Setup - single role with one dummy worker so getRoleTypeList returns 1 role
        org.flexlb.dao.master.WorkerStatus dummy = new org.flexlb.dao.master.WorkerStatus();
        dummy.setIp("192.168.1.10");
        dummy.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.10:8080", dummy);

        ScriptedBatchLoadBalancer scripted = new ScriptedBatchLoadBalancer(List.of(
                target("192.168.1.10", 8080),
                target("192.168.1.11", 8080)
        ));
        replaceBatchLoadBalancer(scripted);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertTrue(response.isSuccess());
        assertEquals(2, response.getServerStatus().size());
        assertEquals(1, scripted.selectBatchCalls);
        assertEquals(2, scripted.lastRequestedCount);
        for (BatchScheduleTarget target : response.getServerStatus()) {
            assertEquals(RoleType.PDFUSION, target.getRole(),
                    "every target in a single-role batch_schedule response must carry the role "
                            + "so the dispatcher can stamp generate_config.role_addrs 1:1 without "
                            + "a second master round-trip; works for any single-role cluster "
                            + "(PDFUSION, PREFILL-only, DECODE-only, VIT) — not just PDFUSION");
        }
    }

    @Test
    void should_reject_batch_schedule_when_batch_strategy_does_not_support_batch() {
        // Setup - single role registered. fusionLoadBalancer (plain mock) sits in the batch map
        // by default and does not impl BatchLoadBalancer. This is the "operator explicitly set
        // batchStrategy to a non-batch-capable strategy (e.g., SHORTEST_TTFT)" case — reject
        // loudly, never silently fall back, so the operator notices the misconfiguration.
        org.flexlb.dao.master.WorkerStatus dummy = new org.flexlb.dao.master.WorkerStatus();
        dummy.setIp("192.168.1.10");
        dummy.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.10:8080", dummy);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        assertNotNull(response.getErrorMessage());
        assertTrue(response.getErrorMessage().contains("does not support batch_schedule"),
                "error message should mention batch_schedule support: " + response.getErrorMessage());
    }

    @Test
    void should_batch_schedule_succeed_when_role_strategy_is_non_batch_but_batch_strategy_is_RR() {
        // Setup - simulates the typical production case: operator configured PDFUSION's regular
        // strategy as SHORTEST_TTFT (non-batch-capable), but batchStrategy defaults to RR.
        // /schedule still uses ST for single requests; /batch_schedule uses RR independently.
        // Decoupling means flipping DISPATCH_PRE_ASSIGN_BE on does not require sacrificing
        // /schedule's smart routing for the role.
        org.flexlb.dao.master.WorkerStatus dummy = new org.flexlb.dao.master.WorkerStatus();
        dummy.setIp("192.168.1.10");
        dummy.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.10:8080", dummy);

        // Regular map keeps the non-batch-capable mock (fusionLoadBalancer) — /schedule's job.
        // Batch map gets a scripted batch-capable LB — what /batch_schedule consults.
        ScriptedBatchLoadBalancer scripted = new ScriptedBatchLoadBalancer(List.of(
                target("192.168.1.10", 8080),
                target("192.168.1.11", 8080)
        ));
        replaceBatchLoadBalancer(scripted);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertTrue(response.isSuccess(),
                "batch_schedule must succeed when batchStrategy is batch-capable, regardless "
                        + "of what /schedule's strategy for the same role is");
        assertEquals(2, response.getServerStatus().size());
        // Confirm the scripted batch LB was actually consulted, not the regular map's mock.
        assertEquals(1, scripted.selectBatchCalls,
                "batch_schedule must query batchLoadBalancer, not loadBalancerMap");
    }

    @Test
    void should_fail_startup_when_batch_schedule_max_count_is_not_positive() {
        // A non-positive upper bound would make `count > batchScheduleMaxCount` true for
        // every request — /batch_schedule silently rejecting 100% of traffic. Misconfig
        // must fail at boot, mirroring validateEngineTypeConfig.
        when(loadBalanceConfig.getBatchScheduleMaxCount()).thenReturn(0);

        org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                () -> new DefaultRouter(configService, modelMetaConfig),
                "batchScheduleMaxCount=0 must fail startup");
    }

    @Test
    void should_reject_batch_schedule_when_batch_count_is_zero() {
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(0);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("batch_count"),
                "error should mention batch_count: " + response.getErrorMessage());
    }

    @Test
    void should_reject_batch_schedule_when_batch_count_is_negative() {
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(-1);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
    }

    @Test
    void should_reject_batch_schedule_when_batch_count_exceeds_max() {
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(10_001);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
    }

    @Test
    void should_reject_batch_schedule_when_sub_requests_length_mismatches_batch_count() {
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(3);
        Request r1 = new Request();
        r1.setRequestId(1001);
        Request r2 = new Request();
        r2.setRequestId(1002);
        batchRequest.setSubRequests(List.of(r1, r2));

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("sub_requests length 2 != batch_count 3"),
                "error should report exact length mismatch: " + response.getErrorMessage());
    }

    @Test
    void should_accept_batch_schedule_when_sub_requests_length_matches_batch_count() {
        org.flexlb.dao.master.WorkerStatus dummy = new org.flexlb.dao.master.WorkerStatus();
        dummy.setIp("192.168.1.10");
        dummy.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.10:8080", dummy);

        ScriptedBatchLoadBalancer scripted = new ScriptedBatchLoadBalancer(List.of(
                target("192.168.1.10", 8080),
                target("192.168.1.11", 8080)
        ));
        replaceBatchLoadBalancer(scripted);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        Request r1 = new Request();
        r1.setRequestId(2001);
        Request r2 = new Request();
        r2.setRequestId(2002);
        batchRequest.setSubRequests(List.of(r1, r2));

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertTrue(response.isSuccess(),
                "matching length should pass validation; phase 1 RR ignores sub_requests contents");
        assertEquals(2, response.getServerStatus().size());
    }

    @Test
    void should_accept_batch_schedule_when_sub_requests_is_null() {
        org.flexlb.dao.master.WorkerStatus dummy = new org.flexlb.dao.master.WorkerStatus();
        dummy.setIp("192.168.1.10");
        dummy.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.10:8080", dummy);

        ScriptedBatchLoadBalancer scripted = new ScriptedBatchLoadBalancer(List.of(
                target("192.168.1.10", 8080),
                target("192.168.1.11", 8080)
        ));
        replaceBatchLoadBalancer(scripted);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        // sub_requests left null intentionally (Level 0 caller)

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertTrue(response.isSuccess(),
                "null sub_requests must skip length check entirely");
        assertEquals(2, response.getServerStatus().size());
    }

    @Test
    void should_reject_batch_schedule_when_no_role_registered() {
        // All role maps are cleared in @BeforeEach -- no role registered
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("master not ready"),
                "error should mention master not ready: " + response.getErrorMessage());
    }

    @Test
    void should_reject_batch_schedule_when_multiple_roles_registered() {
        // Setup - two role maps populated (multi-role deployment)
        org.flexlb.dao.master.WorkerStatus prefillWorker = new org.flexlb.dao.master.WorkerStatus();
        prefillWorker.setIp("192.168.1.1");
        prefillWorker.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("192.168.1.1:8080", prefillWorker);

        org.flexlb.dao.master.WorkerStatus decodeWorker = new org.flexlb.dao.master.WorkerStatus();
        decodeWorker.setIp("192.168.1.2");
        decodeWorker.setPort(8081);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("192.168.1.2:8081", decodeWorker);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("single-role"),
                "error should mention single-role: " + response.getErrorMessage());
        assertTrue(response.getErrorMessage().contains("/schedule"),
                "error should point caller to /schedule: " + response.getErrorMessage());
    }

    @Test
    void should_reject_batch_schedule_when_config_declares_multiple_roles_but_only_one_has_synced_workers() {
        // Config declares a disaggregated PD deployment, but DECODE workers are all down
        // (or the master just started and has not synced them yet) — the runtime view
        // shows a single role. batch_schedule must still reject: pre-assigning targets
        // for only one stage of a multi-stage deployment would be wrong.
        when(modelMetaConfig.getConfiguredRoleTypes()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));

        org.flexlb.dao.master.WorkerStatus prefillWorker = new org.flexlb.dao.master.WorkerStatus();
        prefillWorker.setIp("192.168.1.1");
        prefillWorker.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("192.168.1.1:8080", prefillWorker);

        ScriptedBatchLoadBalancer scripted = new ScriptedBatchLoadBalancer(List.of(
                target("192.168.1.1", 8080),
                target("192.168.1.2", 8080)
        ));
        replaceBatchLoadBalancer(scripted);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess(),
                "multi-role config with one role alive must not be treated as single-role");
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("single-role"),
                "error should mention single-role: " + response.getErrorMessage());
        assertEquals(0, scripted.selectBatchCalls, "no targets may be pre-assigned");
    }

    @Test
    void should_batch_schedule_success_when_config_declares_single_role() {
        when(modelMetaConfig.getConfiguredRoleTypes()).thenReturn(List.of(RoleType.PDFUSION));

        org.flexlb.dao.master.WorkerStatus dummy = new org.flexlb.dao.master.WorkerStatus();
        dummy.setIp("192.168.1.10");
        dummy.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.10:8080", dummy);

        ScriptedBatchLoadBalancer scripted = new ScriptedBatchLoadBalancer(List.of(
                target("192.168.1.10", 8080),
                target("192.168.1.11", 8080)
        ));
        replaceBatchLoadBalancer(scripted);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertTrue(response.isSuccess(),
                "single-role config must keep batch_schedule available: " + response.getErrorMessage());
        assertEquals(2, response.getServerStatus().size());
    }

    @Test
    void should_return_role_specific_error_when_strategy_returns_no_alive_workers() {
        org.flexlb.dao.master.WorkerStatus dummy = new org.flexlb.dao.master.WorkerStatus();
        dummy.setIp("192.168.1.10");
        dummy.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.10:8080", dummy);

        ScriptedBatchLoadBalancer scripted = new ScriptedBatchLoadBalancer(List.of());
        replaceBatchLoadBalancer(scripted);

        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);

        BatchScheduleResponse response = defaultRouter.batchSchedule(batchRequest);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PDFUSION_WORKER.getErrorCode(), response.getCode());
    }

    @Test
    void should_return_response_with_no_pdfusion_worker_error_when_pdfusion_selection_fails() {
        // Setup - add dummy worker to trigger role type
        org.flexlb.dao.master.WorkerStatus dummyFusionWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyFusionWorker.setIp("192.168.1.3");
        dummyFusionWorker.setPort(8082);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.3:8082", dummyFusionWorker);

        ServerStatus fusionServerStatus = new ServerStatus();
        fusionServerStatus.setSuccess(false);
        fusionServerStatus.setMessage("No fusion worker available");
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull())).thenReturn(fusionServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_PDFUSION_WORKER.getErrorCode(), response.getCode(), "Error code should match");
        assertNotNull(response.getErrorMessage(), "Error message should not be null");
    }

    @Test
    void should_return_success_response_with_fusion_and_vit_servers_when_both_selections_succeed() {
        // Setup - add dummy workers to trigger role types
        org.flexlb.dao.master.WorkerStatus dummyFusionWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyFusionWorker.setIp("192.168.1.3");
        dummyFusionWorker.setPort(8082);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.3:8082", dummyFusionWorker);

        org.flexlb.dao.master.WorkerStatus dummyVitWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyVitWorker.setIp("192.168.1.4");
        dummyVitWorker.setPort(8083);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().put("192.168.1.4:8083", dummyVitWorker);

        ServerStatus fusionServerStatus = new ServerStatus();
        fusionServerStatus.setSuccess(true);
        fusionServerStatus.setServerIp("192.168.1.3");
        fusionServerStatus.setHttpPort(8082);
        fusionServerStatus.setGroup("group2");
        fusionServerStatus.setRole(RoleType.PDFUSION);
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull())).thenReturn(fusionServerStatus);

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(true);
        vitServerStatus.setServerIp("192.168.1.4");
        vitServerStatus.setHttpPort(8083);
        vitServerStatus.setRole(RoleType.VIT);
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any())).thenReturn(vitServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(2, response.getServerStatus().size(), "Should have 2 server statuses");
    }

    @Test
    void should_return_response_with_no_vit_worker_error_when_vit_selection_fails() {
        // Setup - add dummy workers to trigger role types
        org.flexlb.dao.master.WorkerStatus dummyFusionWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyFusionWorker.setIp("192.168.1.3");
        dummyFusionWorker.setPort(8082);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.3:8082", dummyFusionWorker);

        org.flexlb.dao.master.WorkerStatus dummyVitWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyVitWorker.setIp("192.168.1.4");
        dummyVitWorker.setPort(8083);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().put("192.168.1.4:8083", dummyVitWorker);

        ServerStatus fusionServerStatus = new ServerStatus();
        fusionServerStatus.setSuccess(true);
        fusionServerStatus.setServerIp("192.168.1.3");
        fusionServerStatus.setHttpPort(8082);
        fusionServerStatus.setGroup("group2");
        fusionServerStatus.setRole(RoleType.PDFUSION);
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull())).thenReturn(fusionServerStatus);

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(false);
        vitServerStatus.setMessage("No vit worker available");
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any())).thenReturn(vitServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_VIT_WORKER.getErrorCode(), response.getCode(), "Error code should match");
        assertNotNull(response.getErrorMessage(), "Error message should not be null");
    }

    @Test
    void should_log_error_when_master_request_is_null() {
        // Setup
        when(balanceContext.getRequest()).thenReturn(null);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertNotNull(response, "Response should not be null");
    }

    @Test
    void should_return_response_with_no_decode_worker_error_when_decode_selection_fails() {
        // Setup - add dummy workers to trigger role types
        org.flexlb.dao.master.WorkerStatus dummyDecodeWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyDecodeWorker.setIp("192.168.1.2");
        dummyDecodeWorker.setPort(8081);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("192.168.1.2:8081", dummyDecodeWorker);

        ServerStatus decodeServerStatus = new ServerStatus();
        decodeServerStatus.setSuccess(false);
        decodeServerStatus.setMessage("No decode worker available");
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any())).thenReturn(decodeServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_DECODE_WORKER.getErrorCode(), response.getCode(), "Error code should match NO_DECODE_WORKER");
    }

    @Test
    void should_return_response_with_no_prefill_worker_error_and_release_decode_cache_when_prefill_selection_fails_after_decode() {
        // Setup - add dummy workers to trigger role types (decode comes before prefill)
        org.flexlb.dao.master.WorkerStatus dummyDecodeWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyDecodeWorker.setIp("192.168.1.2");
        dummyDecodeWorker.setPort(8081);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("192.168.1.2:8081", dummyDecodeWorker);

        org.flexlb.dao.master.WorkerStatus dummyPrefillWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyPrefillWorker.setIp("192.168.1.1");
        dummyPrefillWorker.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("192.168.1.1:8080", dummyPrefillWorker);

        ServerStatus decodeServerStatus = new ServerStatus();
        decodeServerStatus.setSuccess(true);
        decodeServerStatus.setServerIp("192.168.1.2");
        decodeServerStatus.setHttpPort(8081);
        decodeServerStatus.setGroup("group1");
        decodeServerStatus.setRole(RoleType.DECODE);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any())).thenReturn(decodeServerStatus);

        ServerStatus prefillServerStatus = new ServerStatus();
        prefillServerStatus.setSuccess(false);
        prefillServerStatus.setMessage("No prefill worker available");
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), any())).thenReturn(prefillServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode(), "Error code should match NO_PREFILL_WORKER");
        verify(decodeLoadBalancer).rollBack(eq("192.168.1.2:8081"), anyLong());
    }

    @Test
    void should_return_success_response_with_vit_server_when_only_vit_role_exists_and_selection_succeeds() {
        // Setup - add dummy worker to trigger role type
        org.flexlb.dao.master.WorkerStatus dummyVitWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyVitWorker.setIp("192.168.1.5");
        dummyVitWorker.setPort(8084);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().put("192.168.1.5:8084", dummyVitWorker);

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(true);
        vitServerStatus.setServerIp("192.168.1.5");
        vitServerStatus.setHttpPort(8084);
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), isNull())).thenReturn(vitServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(1, response.getServerStatus().size(), "Should have 1 server status");
    }

    @Test
    void should_return_response_with_no_vit_worker_error_when_only_vit_role_exists_and_selection_fails() {
        // Setup - add dummy worker to trigger role type
        org.flexlb.dao.master.WorkerStatus dummyVitWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyVitWorker.setIp("192.168.1.5");
        dummyVitWorker.setPort(8084);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().put("192.168.1.5:8084", dummyVitWorker);

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(false);
        vitServerStatus.setMessage("No vit worker available");
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), isNull())).thenReturn(vitServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_VIT_WORKER.getErrorCode(), response.getCode(), "Error code should match");
        assertNotNull(response.getErrorMessage(), "Error message should not be null");
    }

    @Test
    void should_return_success_response_with_pdfusion_and_vit_servers_when_both_selections_succeed() {
        // Setup - add dummy workers to trigger role types
        org.flexlb.dao.master.WorkerStatus dummyFusionWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyFusionWorker.setIp("192.168.1.3");
        dummyFusionWorker.setPort(8082);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put("192.168.1.3:8082", dummyFusionWorker);

        org.flexlb.dao.master.WorkerStatus dummyVitWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyVitWorker.setIp("192.168.1.4");
        dummyVitWorker.setPort(8083);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().put("192.168.1.4:8083", dummyVitWorker);

        ServerStatus fusionServerStatus = new ServerStatus();
        fusionServerStatus.setSuccess(true);
        fusionServerStatus.setServerIp("192.168.1.3");
        fusionServerStatus.setHttpPort(8082);
        fusionServerStatus.setGroup("group2");
        fusionServerStatus.setRequestId(54321L);
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull())).thenReturn(fusionServerStatus);

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(true);
        vitServerStatus.setServerIp("192.168.1.4");
        vitServerStatus.setHttpPort(8083);
        vitServerStatus.setRole(RoleType.VIT);
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any())).thenReturn(vitServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(2, response.getServerStatus().size(), "Should have 2 server statuses");
    }

    @Test
    void should_return_success_response_with_prefill_decode_and_vit_servers_when_all_selections_succeed() {
        // Setup - add dummy workers to trigger role types
        org.flexlb.dao.master.WorkerStatus dummyPrefillWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyPrefillWorker.setIp("192.168.1.1");
        dummyPrefillWorker.setPort(8080);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("192.168.1.1:8080", dummyPrefillWorker);

        org.flexlb.dao.master.WorkerStatus dummyDecodeWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyDecodeWorker.setIp("192.168.1.2");
        dummyDecodeWorker.setPort(8081);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("192.168.1.2:8081", dummyDecodeWorker);

        org.flexlb.dao.master.WorkerStatus dummyVitWorker = new org.flexlb.dao.master.WorkerStatus();
        dummyVitWorker.setIp("192.168.1.5");
        dummyVitWorker.setPort(8084);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().put("192.168.1.5:8084", dummyVitWorker);

        ServerStatus prefillServerStatus = new ServerStatus();
        prefillServerStatus.setSuccess(true);
        prefillServerStatus.setServerIp("192.168.1.1");
        prefillServerStatus.setHttpPort(8080);
        prefillServerStatus.setGroup("group1");
        prefillServerStatus.setRole(RoleType.PREFILL);
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), any())).thenReturn(prefillServerStatus);

        ServerStatus decodeServerStatus = new ServerStatus();
        decodeServerStatus.setSuccess(true);
        decodeServerStatus.setServerIp("192.168.1.2");
        decodeServerStatus.setHttpPort(8081);
        decodeServerStatus.setRole(RoleType.DECODE);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any())).thenReturn(decodeServerStatus);

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(true);
        vitServerStatus.setServerIp("192.168.1.5");
        vitServerStatus.setHttpPort(8084);
        vitServerStatus.setRole(RoleType.VIT);
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any())).thenReturn(vitServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(3, response.getServerStatus().size(), "Should have 3 server statuses");
    }

    private BatchScheduleTarget target(String ip, int port) {
        return new BatchScheduleTarget(ip, port, port + 1);
    }

    private void replaceRoleLoadBalancer(RoleType roleType, LoadBalancer loadBalancer) {
        try {
            Field loadBalancerMapField = DefaultRouter.class.getDeclaredField("loadBalancerMap");
            loadBalancerMapField.setAccessible(true);

            @SuppressWarnings("unchecked")
            Map<RoleType, LoadBalancer> loadBalancerMap = (Map<RoleType, LoadBalancer>) loadBalancerMapField.get(defaultRouter);
            loadBalancerMap.put(roleType, loadBalancer);
        } catch (Exception e) {
            fail("Failed to replace load balancer: " + e.getMessage());
        }
    }

    /**
     * Replace the batch-path LoadBalancer. Mirrors {@link #replaceRoleLoadBalancer} but
     * operates on the {@code batchLoadBalancer} that {@code /batch_schedule} consults —
     * independent of the regular {@code loadBalancerMap} that {@code /schedule} uses.
     */
    private void replaceBatchLoadBalancer(LoadBalancer loadBalancer) {
        try {
            Field batchField = DefaultRouter.class.getDeclaredField("batchLoadBalancer");
            batchField.setAccessible(true);
            batchField.set(defaultRouter, loadBalancer);
        } catch (Exception e) {
            fail("Failed to replace batch load balancer: " + e.getMessage());
        }
    }

    private static class ScriptedBatchLoadBalancer implements org.flexlb.balance.strategy.BatchLoadBalancer {
        private final List<BatchScheduleTarget> scriptedResponses;
        int selectBatchCalls;
        int lastRequestedCount;

        ScriptedBatchLoadBalancer(List<BatchScheduleTarget> scriptedResponses) {
            this.scriptedResponses = scriptedResponses;
        }

        @Override
        public ServerStatus select(BalanceContext context, RoleType roleType, String group) {
            throw new UnsupportedOperationException("scripted impl only supports selectBatch");
        }

        @Override
        public List<BatchScheduleTarget> selectBatch(int count, RoleType roleType, String group) {
            selectBatchCalls++;
            lastRequestedCount = count;
            List<BatchScheduleTarget> stamped = new java.util.ArrayList<>(scriptedResponses.size());
            for (BatchScheduleTarget t : scriptedResponses) {
                BatchScheduleTarget copy = new BatchScheduleTarget(
                        t.getServerIp(), t.getHttpPort(), t.getGrpcPort(), roleType);
                stamped.add(copy);
            }
            return stamped;
        }

        @Override
        public void rollBack(String ipPort, long requestId) {
            // No rollback in batch path; route() uses this for /schedule rollback only.
        }
    }
}

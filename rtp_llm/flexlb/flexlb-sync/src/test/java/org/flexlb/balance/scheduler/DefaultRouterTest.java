package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.trace.WhaleSpan;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.any;
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
    private WhaleMasterConfig loadBalanceConfig;

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

    @Mock
    private WhaleSpan span;

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
        when(loadBalanceConfig.getLoadBalanceStrategy()).thenReturn(LoadBalanceStrategyEnum.SHORTEST_TTFT);

        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, prefillLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, decodeLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, vitLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, fusionLoadBalancer);

        // Create scheduler instance
        defaultRouter = new DefaultRouter(configService);

        // Mock LoadBalanceStrategyFactory to return our mock load balancers
        mockStaticLoadBalanceStrategyFactory();

        // Mock balance context
        lenient().when(balanceContext.getRequest()).thenReturn(request);
        lenient().when(balanceContext.getSpan()).thenReturn(span);
        lenient().when(balanceContext.getRequestId()).thenReturn("12345");
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
        verify(span).setAttribute("RoleType.PREFILL.ip", "192.168.1.1");
        verify(span).setAttribute("RoleType.DECODE.ip", "192.168.1.2");
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
        fusionServerStatus.setInterRequestId("54321");
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull())).thenReturn(fusionServerStatus);

        // Execute
        Response response = defaultRouter.route(balanceContext);

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(1, response.getServerStatus().size(), "Should have 1 server status");
        assertEquals("12345", response.getInterRequestId(), "Inter request ID should match");
        verify(span).setAttribute("RoleType.PDFUSION.ip", "192.168.1.3");
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
        verify(span).setAttribute("RoleType.PDFUSION.ip", "192.168.1.3");
        verify(span).setAttribute("RoleType.VIT.ip", "192.168.1.4");
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
        verify(decodeLoadBalancer).rollBack(eq("192.168.1.2:8081"), any());
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
        verify(span).setAttribute("RoleType.VIT.ip", "192.168.1.5");
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
        fusionServerStatus.setInterRequestId("54321");
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
        assertEquals("12345", response.getInterRequestId(), "Inter request ID should match");
        verify(span).setAttribute("RoleType.PDFUSION.ip", "192.168.1.3");
        verify(span).setAttribute("RoleType.VIT.ip", "192.168.1.4");
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
        verify(span).setAttribute("RoleType.PREFILL.ip", "192.168.1.1");
        verify(span).setAttribute("RoleType.DECODE.ip", "192.168.1.2");
        verify(span).setAttribute("RoleType.VIT.ip", "192.168.1.5");
    }
}
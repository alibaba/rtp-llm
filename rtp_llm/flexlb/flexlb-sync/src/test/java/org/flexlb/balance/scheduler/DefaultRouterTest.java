package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DefaultRouterTest {

    @Mock
    private ConfigService configService;

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
    private RoutingQueueReporter routingQueueReporter;

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

        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, prefillLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, decodeLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, vitLoadBalancer);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, fusionLoadBalancer);

        // Create scheduler instance
        defaultRouter = new DefaultRouter(configService, routingQueueReporter);

        // Mock LoadBalanceStrategyFactory to return our mock load balancers
        mockStaticLoadBalanceStrategyFactory();

        // Mock balance context
        lenient().when(balanceContext.getRequest()).thenReturn(request);
        lenient().when(balanceContext.getRequestId()).thenReturn("request-12345");
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
        Response response = defaultRouter.route(balanceContext).block();

        // Verify
        assertNotNull(response, "Response should not be null");
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode(), "Error code should match NO_AVAILABLE_WORKER");
        // Note: The method logs an error but doesn't fail when status is null
    }

    @Test
    void should_return_response_with_no_available_worker_error_when_model_not_in_worker_status_map() {
        // Execute
        Response response = defaultRouter.route(balanceContext).block();

        // Verify
        assertNotNull(response, "Response should not be null");
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode(), "Error code should match NO_AVAILABLE_WORKER");
        // Note: The method logs an error but doesn't fail when model is missing
    }

    @Test
    void shouldReturnNoAvailableWorkerWhenWorkerStatusBecomesEmptyDuringRouting() {
        Map<String, WorkerStatus> originalDecodeStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();
        @SuppressWarnings("unchecked")
        Map<String, WorkerStatus> changingDecodeStatusMap = mock(Map.class);
        when(changingDecodeStatusMap.isEmpty()).thenReturn(false, true);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.setDecodeStatusMap(changingDecodeStatusMap);

        try {
            Response response = defaultRouter.route(balanceContext).block();

            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
            verifyNoInteractions(prefillLoadBalancer, decodeLoadBalancer, vitLoadBalancer, fusionLoadBalancer);
        } finally {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.setDecodeStatusMap(originalDecodeStatusMap);
        }
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
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), isNull()))
                .thenReturn(Mono.just(prefillServerStatus));

        ServerStatus decodeServerStatus = new ServerStatus();
        decodeServerStatus.setSuccess(true);
        decodeServerStatus.setServerIp("192.168.1.2");
        decodeServerStatus.setHttpPort(8081);
        decodeServerStatus.setRole(RoleType.DECODE);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any()))
                .thenReturn(Mono.just(decodeServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), isNull()))
                .thenReturn(Mono.just(prefillServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        fusionServerStatus.setRequestId("request-54321");
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull()))
                .thenReturn(Mono.just(fusionServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(1, response.getServerStatus().size(), "Should have 1 server status");
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
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull()))
                .thenReturn(Mono.just(fusionServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull()))
                .thenReturn(Mono.just(fusionServerStatus));

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(true);
        vitServerStatus.setServerIp("192.168.1.4");
        vitServerStatus.setHttpPort(8083);
        vitServerStatus.setRole(RoleType.VIT);
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any()))
                .thenReturn(Mono.just(vitServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull()))
                .thenReturn(Mono.just(fusionServerStatus));

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(false);
        vitServerStatus.setMessage("No vit worker available");
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any()))
                .thenReturn(Mono.just(vitServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        Response response = defaultRouter.route(balanceContext).block();

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
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any()))
                .thenReturn(Mono.just(decodeServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any()))
                .thenReturn(Mono.just(decodeServerStatus));

        ServerStatus prefillServerStatus = new ServerStatus();
        prefillServerStatus.setSuccess(false);
        prefillServerStatus.setMessage("No prefill worker available");
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), any()))
                .thenReturn(Mono.just(prefillServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

        // Verify
        assertFalse(response.isSuccess(), "Response should not be successful");
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode(), "Error code should match NO_PREFILL_WORKER");
        verify(decodeLoadBalancer).rollBack(eq("192.168.1.2:8081"), anyString());
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
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), isNull()))
                .thenReturn(Mono.just(vitServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), isNull()))
                .thenReturn(Mono.just(vitServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        fusionServerStatus.setRequestId("request-54321");
        when(fusionLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PDFUSION), isNull()))
                .thenReturn(Mono.just(fusionServerStatus));

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(true);
        vitServerStatus.setServerIp("192.168.1.4");
        vitServerStatus.setHttpPort(8083);
        vitServerStatus.setRole(RoleType.VIT);
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any()))
                .thenReturn(Mono.just(vitServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

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
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), any()))
                .thenReturn(Mono.just(prefillServerStatus));

        ServerStatus decodeServerStatus = new ServerStatus();
        decodeServerStatus.setSuccess(true);
        decodeServerStatus.setServerIp("192.168.1.2");
        decodeServerStatus.setHttpPort(8081);
        decodeServerStatus.setRole(RoleType.DECODE);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), any()))
                .thenReturn(Mono.just(decodeServerStatus));

        ServerStatus vitServerStatus = new ServerStatus();
        vitServerStatus.setSuccess(true);
        vitServerStatus.setServerIp("192.168.1.5");
        vitServerStatus.setHttpPort(8084);
        vitServerStatus.setRole(RoleType.VIT);
        when(vitLoadBalancer.select(any(BalanceContext.class), eq(RoleType.VIT), any()))
                .thenReturn(Mono.just(vitServerStatus));

        // Execute
        Response response = defaultRouter.route(balanceContext).block();

        // Verify
        assertTrue(response.isSuccess(), "Response should be successful");
        assertNotNull(response.getServerStatus(), "Server status list should not be null");
        assertEquals(3, response.getServerStatus().size(), "Should have 3 server statuses");
    }

    @Test
    void shouldWaitForAsyncRoleBeforeSelectingNextRoleAndRollbackOnFailure() {
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        Sinks.One<ServerStatus> pendingDecode = Sinks.one();
        ServerStatus decode = successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1");
        ServerStatus failedPrefill = ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        failedPrefill.setMessage("No prefill worker available");
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(pendingDecode.asMono());
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1")))
                .thenReturn(Mono.just(failedPrefill));

        defaultRouter.route(balanceContext).subscribe();

        verify(prefillLoadBalancer, never()).select(any(BalanceContext.class), eq(RoleType.PREFILL), any());
        pendingDecode.tryEmitValue(decode);

        verify(prefillLoadBalancer, timeout(1_000))
                .select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1"));
        verify(decodeLoadBalancer, timeout(1_000)).rollBack("192.168.1.2:8081", "request-12345");
        verify(routingQueueReporter).reportRoutingRollback("route_failure", 1);
    }

    @Test
    void shouldRollbackSelectedWorkersWhenRouteSubscriptionIsCancelled() {
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        ServerStatus decode = successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1");
        Sinks.One<ServerStatus> pendingPrefill = Sinks.one();
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.just(decode));
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1")))
                .thenReturn(pendingPrefill.asMono());

        Disposable routeSubscription = defaultRouter.route(balanceContext).subscribe();
        verify(prefillLoadBalancer, timeout(1_000))
                .select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1"));

        routeSubscription.dispose();

        verify(decodeLoadBalancer, timeout(1_000)).rollBack("192.168.1.2:8081", "request-12345");
        verify(routingQueueReporter).reportRoutingRollback("route_cancelled", 1);
        pendingPrefill.tryEmitValue(successfulStatus(RoleType.PREFILL, "192.168.1.1", 8080, "group-1"));
    }

    @Test
    void shouldRollbackSelectionArrivingDuringRouteCancellation() throws InterruptedException {
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        CountDownLatch selectionHandlingStarted = new CountDownLatch(1);
        CountDownLatch allowSelectionHandlingToContinue = new CountDownLatch(1);
        ExecutorService selectionExecutor = Executors.newSingleThreadExecutor();
        ServerStatus decode = mock(ServerStatus.class);
        when(decode.isSuccess()).thenAnswer(invocation -> {
            selectionHandlingStarted.countDown();
            allowSelectionHandlingToContinue.await(1, TimeUnit.SECONDS);
            return true;
        });
        when(decode.getServerIp()).thenReturn("192.168.1.2");
        when(decode.getHttpPort()).thenReturn(8081);
        when(decode.getRole()).thenReturn(RoleType.DECODE);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.create(sink -> selectionExecutor.execute(() -> sink.success(decode))));

        Disposable routeSubscription = defaultRouter.route(balanceContext).subscribe();
        try {
            assertTrue(selectionHandlingStarted.await(1, TimeUnit.SECONDS));

            routeSubscription.dispose();
            allowSelectionHandlingToContinue.countDown();

            verify(decodeLoadBalancer, timeout(1_000).times(1))
                    .rollBack("192.168.1.2:8081", "request-12345");
            verify(routingQueueReporter).reportRoutingRollback("late_selection_after_rollback", 1);
            verify(prefillLoadBalancer, never()).select(any(BalanceContext.class), eq(RoleType.PREFILL), any());
        } finally {
            allowSelectionHandlingToContinue.countDown();
            routeSubscription.dispose();
            selectionExecutor.shutdownNow();
            selectionExecutor.awaitTermination(1, TimeUnit.SECONDS);
        }
    }

    @Test
    void shouldNotSelectNextRoleWhenCancellationOccursAfterWorkerIsRecorded() throws InterruptedException {
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        CountDownLatch groupLookupStarted = new CountDownLatch(1);
        CountDownLatch allowGroupLookupToContinue = new CountDownLatch(1);
        ExecutorService selectionExecutor = Executors.newSingleThreadExecutor();
        ServerStatus decode = mock(ServerStatus.class);
        when(decode.isSuccess()).thenReturn(true);
        when(decode.getGroup()).thenAnswer(invocation -> {
            groupLookupStarted.countDown();
            allowGroupLookupToContinue.await(1, TimeUnit.SECONDS);
            return "group-1";
        });
        when(decode.getServerIp()).thenReturn("192.168.1.2");
        when(decode.getHttpPort()).thenReturn(8081);
        when(decode.getRole()).thenReturn(RoleType.DECODE);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.create(sink -> selectionExecutor.execute(() -> sink.success(decode))));

        Disposable routeSubscription = defaultRouter.route(balanceContext).subscribe();
        try {
            assertTrue(groupLookupStarted.await(1, TimeUnit.SECONDS));

            routeSubscription.dispose();
            allowGroupLookupToContinue.countDown();

            verify(decodeLoadBalancer, timeout(1_000).times(1))
                    .rollBack("192.168.1.2:8081", "request-12345");
            verify(prefillLoadBalancer, never()).select(any(BalanceContext.class), eq(RoleType.PREFILL), any());
        } finally {
            allowGroupLookupToContinue.countDown();
            routeSubscription.dispose();
            selectionExecutor.shutdownNow();
            selectionExecutor.awaitTermination(1, TimeUnit.SECONDS);
        }
    }

    @Test
    void shouldRollbackWorkerOnlyOnceWhenBusinessFailureAndCancellationOverlap() throws InterruptedException {
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        CountDownLatch rollbackStarted = new CountDownLatch(1);
        CountDownLatch allowRollbackToComplete = new CountDownLatch(1);
        ExecutorService selectionExecutor = Executors.newSingleThreadExecutor();
        ServerStatus decode = successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1");
        ServerStatus failedPrefill = ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.just(decode));
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1")))
                .thenReturn(Mono.create(sink -> selectionExecutor.execute(() -> sink.success(failedPrefill))));
        doAnswer(invocation -> {
                    rollbackStarted.countDown();
                    allowRollbackToComplete.await(1, TimeUnit.SECONDS);
                    return null;
                })
                .when(decodeLoadBalancer)
                .rollBack("192.168.1.2:8081", "request-12345");

        Disposable routeSubscription = defaultRouter.route(balanceContext).subscribe();
        try {
            assertTrue(rollbackStarted.await(1, TimeUnit.SECONDS));

            routeSubscription.dispose();
            allowRollbackToComplete.countDown();

            verify(decodeLoadBalancer, timeout(1_000).times(1))
                    .rollBack("192.168.1.2:8081", "request-12345");
        } finally {
            allowRollbackToComplete.countDown();
            routeSubscription.dispose();
            selectionExecutor.shutdownNow();
            selectionExecutor.awaitTermination(1, TimeUnit.SECONDS);
        }
    }

    @Test
    void shouldIsolateCancellationCleanupForEachRouteSubscription() {
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        ServerStatus decode = successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1");
        Sinks.One<ServerStatus> pendingPrefill = Sinks.one();
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.just(decode));
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1")))
                .thenReturn(pendingPrefill.asMono());

        Mono<Response> route = defaultRouter.route(balanceContext);
        Disposable firstSubscription = route.subscribe();
        Disposable secondSubscription = route.subscribe();
        verify(prefillLoadBalancer, timeout(1_000).times(2))
                .select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1"));

        firstSubscription.dispose();

        verify(decodeLoadBalancer, timeout(1_000).times(1)).rollBack("192.168.1.2:8081", "request-12345");

        secondSubscription.dispose();
        verify(decodeLoadBalancer, timeout(1_000).times(2)).rollBack("192.168.1.2:8081", "request-12345");
    }

    @Test
    void shouldRollbackSelectedWorkersWhenLaterRoleEmitsAnError() {
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        ServerStatus decode = successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1");
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.just(decode));
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1")))
                .thenReturn(Mono.error(new IllegalStateException("prefill selection failed")));

        assertThrows(IllegalStateException.class, () -> defaultRouter.route(balanceContext).block());

        verify(decodeLoadBalancer).rollBack("192.168.1.2:8081", "request-12345");
        verify(routingQueueReporter).reportRoutingRollback("route_error", 1);
    }

    @Test
    void shouldRollbackWorkersFromSuccessfulResponse() {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1")));

        defaultRouter.rollBack(balanceContext, response);

        verify(decodeLoadBalancer).rollBack("192.168.1.2:8081", "request-12345");
        verify(routingQueueReporter).reportRoutingRollback("response_rollback", 1);
    }

    @Test
    void shouldRollbackRoutedResponseOnlyOnce() {
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.just(successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1")));
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1")))
                .thenReturn(Mono.just(successfulStatus(RoleType.PREFILL, "192.168.1.1", 8080, "group-1")));

        Response response = defaultRouter.route(balanceContext).block();
        defaultRouter.rollBack(balanceContext, response);
        defaultRouter.rollBack(balanceContext, response);

        verify(decodeLoadBalancer, times(1)).rollBack("192.168.1.2:8081", "request-12345");
        verify(prefillLoadBalancer, times(1)).rollBack("192.168.1.1:8080", "request-12345");
        verify(routingQueueReporter).reportRoutingRollback("response_rollback", 2);
    }

    @Test
    void shouldRollbackEarlierSelectionsWhenALaterLoadBalancerCompletesEmpty() {
        addWorker(RoleType.PREFILL, "192.168.1.1", 8080);
        addWorker(RoleType.DECODE, "192.168.1.2", 8081);
        when(decodeLoadBalancer.select(any(BalanceContext.class), eq(RoleType.DECODE), isNull()))
                .thenReturn(Mono.just(successfulStatus(RoleType.DECODE, "192.168.1.2", 8081, "group-1")));
        when(prefillLoadBalancer.select(any(BalanceContext.class), eq(RoleType.PREFILL), eq("group-1")))
                .thenReturn(Mono.empty());

        Response response = defaultRouter.route(balanceContext).block();

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        verify(decodeLoadBalancer).rollBack("192.168.1.2:8081", "request-12345");
    }

    @Test
    void shouldNotRollbackWorkersForInvalidResponses() {
        Response failedResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        Response emptySuccessResponse = new Response();
        emptySuccessResponse.setSuccess(true);
        emptySuccessResponse.setServerStatus(List.of());

        defaultRouter.rollBack(balanceContext, null);
        defaultRouter.rollBack(balanceContext, failedResponse);
        defaultRouter.rollBack(balanceContext, emptySuccessResponse);

        verifyNoInteractions(prefillLoadBalancer, decodeLoadBalancer, vitLoadBalancer, fusionLoadBalancer);
    }

    private void addWorker(RoleType roleType, String ip, int port) {
        org.flexlb.dao.master.WorkerStatus worker = new org.flexlb.dao.master.WorkerStatus();
        worker.setIp(ip);
        worker.setPort(port);
        String ipPort = ip + ":" + port;
        if (roleType == RoleType.PREFILL) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(ipPort, worker);
        } else if (roleType == RoleType.DECODE) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort, worker);
        }
    }

    private ServerStatus successfulStatus(RoleType roleType, String ip, int port, String group) {
        ServerStatus serverStatus = new ServerStatus();
        serverStatus.setSuccess(true);
        serverStatus.setRole(roleType);
        serverStatus.setServerIp(ip);
        serverStatus.setHttpPort(port);
        serverStatus.setGroup(group);
        return serverStatus;
    }
}

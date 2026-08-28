package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.config.ConfigService;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
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
    private DecodeResourceMeasure resourceMeasure;
    private EndpointRegistry endpointRegistry;
    private WorkerDirectory workerDirectory;
    private ConfigService configService;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        configService = Mockito.mock(ConfigService.class);
        config = new FlexlbConfig();
        endpointRegistry = StrategyTestSupport.endpointRegistry(configService);
        resourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        PrefillResourceMeasure prefillResourceMeasure =
                Mockito.mock(PrefillResourceMeasure.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config);
        Mockito.when(prefillResourceMeasure.isResourceAvailable(Mockito.anyLong()))
                .thenReturn(true);
        Mockito.when(resourceMeasure.isResourceAvailable(
                Mockito.any(DecodeEndpoint.DecodeRoutingView.class))).thenReturn(true);
        Mockito.when(resourceMeasure.isEngineDispatchAvailable(
                Mockito.any(DecodeEndpoint.DecodeRoutingView.class))).thenReturn(true);
        workerDirectory = new WorkerDirectory(endpointRegistry);
        randomStrategy = new RandomStrategy(workerDirectory,
                prefillResourceMeasure, resourceMeasure);
    }

    @AfterEach
    void tearDown() {
        endpointRegistry.close();
        workerDirectory.statusMap(RoleType.PREFILL).clear();
        workerDirectory.statusMap(RoleType.DECODE).clear();
        workerDirectory.statusMap(RoleType.PDFUSION).clear();
        workerDirectory.statusMap(RoleType.VIT).clear();
    }

    private org.flexlb.balance.endpoint.WorkerEndpoint registerPrefill(
            String ipPort, WorkerStatus ws) {
        return endpointRegistry.registerPreinitializedEndpoint(
                RoleType.PREFILL, ipPort, ws);
    }

    private org.flexlb.balance.endpoint.WorkerEndpoint registerDecode(
            String ipPort, WorkerStatus ws) {
        return endpointRegistry.registerPreinitializedEndpoint(
                RoleType.DECODE, ipPort, ws);
    }

    @Test
    void should_return_error_when_no_workers_available() {
        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.PREFILL, null);

        assertNull(result);
    }

    @Test
    void should_return_error_when_worker_map_is_empty() {
        workerDirectory.statusMap(RoleType.PREFILL).clear();

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.PREFILL, null);

        assertNull(result);
    }

    @Test
    void should_return_success_when_workers_available() {
        Map<String, WorkerStatus> prefillStatusMap = workerDirectory.statusMap(RoleType.PREFILL);

        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1");
        prefillStatusMap.put("127.0.0.1:8080", workerStatus);
        registerPrefill("127.0.0.1:8080", workerStatus);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.PREFILL, null);

        assertNotNull(result);
        assertTrue(result.isSuccess());
    }

    @Test
    void should_select_randomly_from_available_workers() {
        Map<String, WorkerStatus> prefillStatusMap = workerDirectory.statusMap(RoleType.PREFILL);

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

        ServerStatus result1 = selectStatus(balanceContext, RoleType.PREFILL, null);
        ServerStatus result2 = selectStatus(balanceContext, RoleType.PREFILL, null);
        ServerStatus result3 = selectStatus(balanceContext, RoleType.PREFILL, null);

        assertNotNull(result1);
        assertNotNull(result2);
        assertNotNull(result3);
    }

    @Test
    void should_work_with_different_role_types() {
        WorkerStatus prefillWorker = createWorkerStatus("127.0.0.1");
        workerDirectory.statusMap(RoleType.PREFILL).put("127.0.0.1:8080", prefillWorker);
        registerPrefill("127.0.0.1:8080", prefillWorker);

        WorkerStatus decodeWorker = createWorkerStatus(
                "127.0.0.2", RoleType.DECODE, null, true, 1_000L, 1_000L);
        workerDirectory.statusMap(RoleType.DECODE).put("127.0.0.2:8080", decodeWorker);
        registerDecode("127.0.0.2:8080", decodeWorker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus prefillResult = selectStatus(balanceContext, RoleType.PREFILL, null);
        ServerStatus decodeResult = selectStatus(balanceContext, RoleType.DECODE, null);

        assertNotNull(prefillResult);
        assertNotNull(decodeResult);
    }

    @Test
    void should_select_vit_from_role_specific_registry() {
        WorkerStatus vitWorker = createWorkerStatus(
                "127.0.0.3", RoleType.VIT, null, true, 0L, 0L);
        workerDirectory.statusMap(RoleType.VIT)
                .put("127.0.0.3:8080", vitWorker);
        endpointRegistry.registerPreinitializedEndpoint(RoleType.VIT, "127.0.0.3:8080", vitWorker);

        BalanceContext context = new BalanceContext();
        context.setRequest(new Request());

        assertTrue(selectStatus(context, RoleType.VIT, null).isSuccess());
    }

    @Test
    void should_work_with_group_parameter() {
        WorkerStatus worker = createWorkerStatus(
                "127.0.0.1", RoleType.PREFILL, "group-a", true, 0L, 0L);
        workerDirectory.statusMap(RoleType.PREFILL).put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.PREFILL, "group-a");

        assertNotNull(result);
    }

    @Test
    void should_return_error_when_no_workers_in_specified_group() {
        WorkerStatus worker = createWorkerStatus(
                "127.0.0.1", RoleType.PREFILL, "group-a", true, 0L, 0L);
        workerDirectory.statusMap(RoleType.PREFILL).put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.PREFILL, "group-b");

        assertNull(result);
    }

    @Test
    void configured_selector_routes_to_the_random_leaf() {
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        registerPrefill("127.0.0.1:8080", worker);
        config.getRouter().getRoles().getPrefill().setSelector(
                new org.flexlb.config.RoutingConfig.RandomPrefillSelectorConfig());
        RandomStrategy exactLeaf = Mockito.spy(randomStrategy);
        ConfiguredLoadBalanceSelector selector =
                new ConfiguredLoadBalanceSelector(java.util.List.of(exactLeaf));
        BalanceContext context = new BalanceContext();
        context.setRequest(new Request());
        context.setConfig(config);

        try (SelectedRole selected = selector.select(
                context, RoleType.PREFILL, null)) {
            assertNotNull(selected);
            assertSame(worker,
                    endpointRegistry.getPrefill("127.0.0.1:8080").getStatus());
            assertEquals("127.0.0.1", selected.serverStatus().getServerIp());
        }
        Mockito.verify(exactLeaf).select(context, RoleType.PREFILL, null);
    }

    @Test
    void should_distribute_requests_uniformly_across_workers() {
        Map<String, WorkerStatus> prefillStatusMap = workerDirectory.statusMap(RoleType.PREFILL);

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
            ServerStatus status = selectStatus(balanceContext, RoleType.PREFILL, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        double tolerance = 0.10;

        for (String ip : selectionCount.keySet()) {
            int count = selectionCount.get(ip);
            double ratio = (double) count / totalRuns;
            assertTrue(ratio >= 0.33 - tolerance && ratio <= 0.33 + tolerance,
                    "Worker " + ip + " selection ratio " + ratio + " is outside expected range");
        }

    }

    @Test
    void should_skip_dead_workers() {
        Map<String, WorkerStatus> prefillStatusMap = workerDirectory.statusMap(RoleType.PREFILL);

        // A worker becomes non-serviceable only when a not-alive status
        // observation is applied to its endpoint, which closes the admission
        // gate (beginRetirement) so tryPinGeneration() returns null. Merely
        // constructing a WorkerStatus with alive=false does not close the gate,
        // so register a live generation first and then apply the dead status.
        WorkerStatus deadWorker = createWorkerStatus(
                "127.0.0.1", RoleType.PREFILL, null, true, 0L, 0L);
        prefillStatusMap.put("127.0.0.1:8080", deadWorker);
        org.flexlb.balance.endpoint.WorkerEndpoint deadEndpoint =
                registerPrefill("127.0.0.1:8080", deadWorker);
        StrategyTestSupport.apply(
                deadEndpoint,
                StrategyTestSupport.response(RoleType.PREFILL, false, 0L, 0L, 2L));

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
            ServerStatus status = selectStatus(balanceContext, RoleType.PREFILL, null);

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
        config.setScheduler(SchedulerConfig.direct());
        config.setDispatcher(DispatcherConfig.nonBatch());
        Map<String, WorkerStatus> decodeStatusMap = workerDirectory.statusMap(RoleType.DECODE);

        WorkerStatus unavailableWorker = createWorkerStatus(
                "127.0.0.1", RoleType.DECODE, null, true, 1_000L, 1_000L);
        WorkerStatus availableWorker = createWorkerStatus(
                "127.0.0.2", RoleType.DECODE, null, true, 2_000L, 2_000L);
        decodeStatusMap.put("127.0.0.1:8080", unavailableWorker);
        decodeStatusMap.put("127.0.0.2:8080", availableWorker);
        registerDecode("127.0.0.1:8080", unavailableWorker);
        registerDecode("127.0.0.2:8080", availableWorker);

        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any()))
                .thenAnswer(invocation -> {
                    DecodeEndpoint.DecodeRoutingView view = invocation.getArgument(0);
                    return view.totalKv() == 2_000L;
                });

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(12345L);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.DECODE, null);

        assertTrue(result.isSuccess());
        assertEquals("127.0.0.2", result.getServerIp());
    }

    @Test
    void softQueuePrefersDispatchableDecodeBeforeFullFallback() {
        WorkerStatus full = createWorkerStatus(
                "127.0.0.1", RoleType.DECODE, null, true, 1_000L, 1_000L);
        WorkerStatus dispatchable = createWorkerStatus(
                "127.0.0.2", RoleType.DECODE, null, true, 2_000L, 2_000L);
        workerDirectory.statusMap(RoleType.DECODE)
                .put("127.0.0.1:8080", full);
        workerDirectory.statusMap(RoleType.DECODE)
                .put("127.0.0.2:8080", dispatchable);
        registerDecode("127.0.0.1:8080", full);
        registerDecode("127.0.0.2:8080", dispatchable);
        Mockito.when(resourceMeasure.isEngineDispatchAvailable(Mockito.any()))
                .thenAnswer(invocation -> {
                    DecodeEndpoint.DecodeRoutingView view = invocation.getArgument(0);
                    return view.totalKv() == 2_000L;
                });

        Request request = new Request();
        request.setRequestId(12_345L);
        request.setSeqLen(100L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);

        ServerStatus selected = selectStatus(context, RoleType.DECODE, null);

        assertNotNull(selected);
        assertEquals("127.0.0.2", selected.getServerIp());
    }

    @Test
    void softQueueFallsBackWhenEveryDecodeIsTransientlyFull() {
        WorkerStatus full = createWorkerStatus(
                "127.0.0.3", RoleType.DECODE, null, true, 1_000L, 1_000L);
        workerDirectory.statusMap(RoleType.DECODE)
                .put("127.0.0.3:8080", full);
        registerDecode("127.0.0.3:8080", full);
        Mockito.when(resourceMeasure.isEngineDispatchAvailable(Mockito.any()))
                .thenReturn(false);

        Request request = new Request();
        request.setRequestId(12_346L);
        request.setSeqLen(100L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);

        ServerStatus selected = selectStatus(context, RoleType.DECODE, null);

        assertNotNull(selected,
                "all-full fallback must retain non-preemptive queue liveness");
        assertEquals("127.0.0.3", selected.getServerIp());
    }

    @Test
    void decode_random_should_return_exact_capacity_in_direct_and_queue_modes() {
        WorkerStatus worker = createWorkerStatus(
                "127.0.0.4", RoleType.DECODE, null, true, 1_000L, 1_000L);
        workerDirectory.statusMap(RoleType.DECODE)
                .put("127.0.0.4:8080", worker);
        registerDecode("127.0.0.4:8080", worker);
        DecodeEndpoint endpoint = endpointRegistry.getDecode("127.0.0.4:8080");

        config.setScheduler(SchedulerConfig.direct());
        config.setDispatcher(DispatcherConfig.nonBatch());
        assertDecodeSelection(endpoint, 41L, 73);

        config.setScheduler(new SchedulerConfig());
        assertDecodeSelection(endpoint, 42L, 81);
    }

    @Test
    void should_properly_set_server_status_fields() {
        Map<String, WorkerStatus> prefillStatusMap = workerDirectory.statusMap(RoleType.PREFILL);

        WorkerStatus worker = createWorkerStatus(
                "127.0.0.1", RoleType.PREFILL, "group-x", true, 0L, 0L);
        prefillStatusMap.put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();
        req.setSeqLen(1000);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("127.0.0.1", result.getServerIp());
        assertEquals(8080, result.getHttpPort());
        assertEquals(RoleType.PREFILL, result.getRole());
        assertEquals("group-x", result.getGroup());
    }

    @Test
    void should_handle_null_request_id() {
        Map<String, WorkerStatus> prefillStatusMap = workerDirectory.statusMap(RoleType.PREFILL);
        prefillStatusMap.clear();

        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        prefillStatusMap.put("127.0.0.1:8080", worker);
        registerPrefill("127.0.0.1:8080", worker);

        Request req = new Request();

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus result = selectStatus(balanceContext, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("127.0.0.1", result.getServerIp());
    }

    private void assertDecodeSelection(DecodeEndpoint endpoint,
                                       long requestId,
                                       int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(600L);
        request.setMaxNewTokens(900);
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority, System.currentTimeMillis() + 60_000L));

        try (SelectedRole selected = randomStrategy.select(
                context, RoleType.DECODE, null)) {
            assertNotNull(selected);
            assertEquals(1_000L, selected.decodeTotalKv());
            assertEquals(requestId, selected.serverStatus().getRequestId());
            try (var pin = selected.takeGenerationPin()) {
                assertSame(endpoint, pin.endpoint());
                assertEquals(
                        endpoint.getStatus().getGenerationId(),
                        pin.generationId());
            }
        }
        // Selection is now pure ownership capture; exact reservation belongs
        // to QueueRouteAdmission and must not leak back into strategy code.
        assertNull(endpoint.reservationHandle(requestId));
    }

    private WorkerStatus createWorkerStatus(String ip) {
        return createWorkerStatus(
                ip, RoleType.PREFILL, null, true, 0L, 0L);
    }

    private WorkerStatus createWorkerStatus(
            String ip,
            RoleType role,
            String group,
            boolean alive,
            long availableKv,
            long totalKv) {
        return StrategyTestSupport.workerStatus(
                role, group, ip, 8080, 8081,
                alive, availableKv, totalKv);
    }

    private ServerStatus selectStatus(
            BalanceContext context, RoleType role, String group) {
        context.setConfig(config);
        try (SelectedRole selected = randomStrategy.select(
                context, role, group)) {
            return selected == null ? null : selected.serverStatus();
        }
    }
}

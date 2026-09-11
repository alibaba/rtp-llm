package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RoundRobinLoadBalancerTest {
    private final FlexlbConfig config = new FlexlbConfig();
    private final ModelMetaConfig model = mock(ModelMetaConfig.class);
    private final MasterEngineSynchronizer synchronizer = mock(MasterEngineSynchronizer.class);
    private EndpointRegistry endpoints;
    private WorkerDirectory directory;
    private RoundRobinLoadBalancer scheduler;

    @BeforeEach
    void setUp() {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
        endpoints = StrategyTestSupport.endpointRegistry(service);
        directory = new WorkerDirectory(endpoints);
        scheduler = new RoundRobinLoadBalancer(directory, synchronizer, model, service);
    }

    @AfterEach
    void close() {
        endpoints.close();
    }

    @Test
    void rotatesAcrossPublishedWorkersAndUsesGrpcPorts() {
        publish("10.0.0.2", true);
        publish("10.0.0.1", true);
        publish("10.0.0.3", false);
        assertEquals("10.0.0.1", schedule(1).getServerStatus().getFirst().getServerIp());
        List<BatchScheduleTarget> targets = schedule(3).getServerStatus();
        assertEquals(List.of("10.0.0.2", "10.0.0.1", "10.0.0.2"),
                targets.stream().map(BatchScheduleTarget::getServerIp).toList());
        for (BatchScheduleTarget target : targets) {
            assertEquals(8081, target.getGrpcPort());
            assertNull(target.getArpcPort());
            assertEquals(RoleType.PDFUSION, target.getRole());
        }
        assertNotSame(targets.get(0), targets.get(2));
        targets.get(0).setFeUrl("http://fe-a");
        assertNull(targets.get(2).getFeUrl());
    }

    @Test
    void discoveryAloneDoesNotMakeAnLlmWorkerRoutable() {
        directory.currentOrDiscover(RoleType.PDFUSION, "10.0.0.1:8080",
                () -> WorkerStatus.createDiscovered(RoleType.PDFUSION, "", "10.0.0.1", 8080, 8081, ""));
        assertFalse(schedule(1).isSuccess());
    }

    @Test
    void retiringEndpointIsNoLongerSelected() {
        WorkerStatus status = publish("10.0.0.1", true);
        assertTrue(schedule(1).isSuccess());
        EndpointRegistry.DetachedGeneration detached;
        status.lock.lock();
        try {
            detached = directory.beginRetirement(RoleType.PDFUSION, status.getIpPort(), status);
        } finally {
            status.lock.unlock();
        }
        detached.retireAndAwait();
        assertFalse(schedule(1).isSuccess());
    }

    @Test
    void embeddingUsesDiscoveryAndOnlyAdvertisesArpc() {
        config.getWorkerRegistry().setEngineType(EngineType.EMBEDDING);
        when(synchronizer.embeddingWorkerSnapshot(RoleType.PDFUSION))
                .thenReturn(List.of(new WorkerHost("10.0.0.4", 8000)));
        BatchScheduleTarget target = schedule(1).getServerStatus().getFirst();
        assertEquals("10.0.0.4", target.getServerIp());
        assertEquals(8001, target.getArpcPort());
        assertNull(target.getGrpcPort());
        when(synchronizer.embeddingWorkerSnapshot(RoleType.PDFUSION)).thenReturn(List.of());
        assertFalse(schedule(1).isSuccess());
    }

    @Test
    void validatesCountAndAssignmentFlags() {
        config.getRouter().setBatchScheduleMaxCount(2);
        for (int count : new int[]{-1, 0, 3, Integer.MAX_VALUE}) {
            assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), schedule(count).getCode());
        }
        BatchScheduleRequest request = request(1);
        request.setAssignBe(false);
        request.setAssignFe(false);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), scheduler.schedule(request).block().getCode());
    }

    @Test
    void multiRoleTopologyRequiresPerRequestBackendPlacementButAllowsFeOnly() {
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), schedule(1).getCode());
        BatchScheduleRequest request = request(2);
        request.setAssignBe(false);
        request.setAssignFe(true);
        BatchScheduleResponse response = scheduler.schedule(request).block();
        assertTrue(response.isSuccess());
        assertEquals(2, response.getServerStatus().size());
        assertNull(response.getServerStatus().getFirst().getServerIp());
    }

    @Test
    void weightedGroupPolicyCannotBeBypassedByBatchPlacement() {
        TrafficPolicyConfig.Target target = new TrafficPolicyConfig.Target();
        target.setGroup("tenant-a");
        target.setWeight(1);
        TrafficPolicyConfig policy = new TrafficPolicyConfig();
        policy.setDefaultTargets(List.of(target));
        config.getRouter().setGroupSelector(policy);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), schedule(1).getCode());
        policy.setDefaultTargets(List.of());
        publish("10.0.0.1", true);
        assertTrue(schedule(1).isSuccess());
    }

    @Test
    void concurrentBatchesShareOneRotation() {
        publish("10.0.0.1", true);
        publish("10.0.0.2", true);
        publish("10.0.0.3", true);
        List<BatchScheduleTarget> targets = Flux.range(0, 100)
                .flatMap(i -> scheduler.schedule(request(3)), 16)
                .flatMapIterable(BatchScheduleResponse::getServerStatus).collectList().block();
        Map<String, Long> counts = targets.stream().collect(Collectors.groupingBy(
                BatchScheduleTarget::getServerIp, Collectors.counting()));
        assertEquals(Map.of("10.0.0.1", 100L, "10.0.0.2", 100L, "10.0.0.3", 100L), counts);
    }

    private WorkerStatus publish(String ip, boolean alive) {
        WorkerStatus source = StrategyTestSupport.workerStatus(
                RoleType.PDFUSION, "", ip, 8080, 8081, alive, 1000, 1000);
        WorkerEndpoint endpoint = StrategyTestSupport.publishEndpoint(
                endpoints, RoleType.PDFUSION, ip + ":8080", source);
        WorkerStatus status = endpoint.getStatus();
        status.lock.lock();
        try {
            status.recordSuccessfulPoll(alive);
        } finally {
            status.lock.unlock();
        }
        return status;
    }

    private BatchScheduleResponse schedule(int count) {
        return scheduler.schedule(request(count)).block();
    }

    private static BatchScheduleRequest request(int count) {
        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setBatchCount(count);
        return request;
    }
}

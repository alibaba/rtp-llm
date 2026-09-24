package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
        scheduler = new RoundRobinLoadBalancer(directory, model, service);
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
        targets.get(0).setHttpPort(9000);
        assertEquals(8080, targets.get(2).getHttpPort());
    }

    @Test
    void removingThePreviousWorkerContinuesAtTheNextAddress() {
        publish("10.0.0.1", true);
        WorkerStatus previous = publish("10.0.0.2", true);
        publish("10.0.0.3", true);
        assertEquals(List.of("10.0.0.1", "10.0.0.2"), schedule(2).getServerStatus().stream()
                .map(BatchScheduleTarget::getServerIp).toList());
        previous.lock.lock();
        try {
            previous.recordSuccessfulPoll(false);
        } finally {
            previous.lock.unlock();
        }
        assertEquals("10.0.0.3", schedule(1).getServerStatus().getFirst().getServerIp());
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
        WorkerStatus status = directory.currentOrDiscover(RoleType.PDFUSION, "10.0.0.4:8000",
                () -> WorkerStatus.createDiscovered(RoleType.PDFUSION, "", "10.0.0.4", 8000, 8001, ""));
        BatchScheduleTarget target = schedule(1).getServerStatus().getFirst();
        assertEquals("10.0.0.4", target.getServerIp());
        assertEquals(8001, target.getArpcPort());
        assertNull(target.getGrpcPort());
        assertFalse(status.pollHealth().reportedAlive());
        assertEquals(0, directory.routingCapacity(RoleType.PDFUSION));
        status.lock.lock();
        try {
            directory.beginRetirement(RoleType.PDFUSION, status.getIpPort(), status);
        } finally {
            status.lock.unlock();
        }
        assertFalse(schedule(1).isSuccess());
    }

    @Test
    void multiRoleTopologyRequiresPerRequestBackendPlacement() {
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), schedule(1).getCode());
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
        List<BatchScheduleResponse> batches = IntStream.range(0, 150).parallel()
                .mapToObj(i -> scheduler.schedule(2).block())
                .toList();
        batches.forEach(batch -> assertEquals(2L, batch.getServerStatus().stream()
                .map(BatchScheduleTarget::getServerIp).distinct().count()));
        Map<String, Long> counts = batches.stream().flatMap(batch -> batch.getServerStatus().stream()).collect(Collectors.groupingBy(
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
        return scheduler.schedule(count).block();
    }
}

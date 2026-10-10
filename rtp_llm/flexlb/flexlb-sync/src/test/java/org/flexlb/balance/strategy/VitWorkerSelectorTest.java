package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VitWorkerSelectorTest {

    private EndpointRegistry endpoints;
    private VitWorkerSelector strategy;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(new FlexlbConfig());
        endpoints = StrategyTestSupport.endpointRegistry(configService);
        strategy = new VitWorkerSelector(endpoints);
    }

    @AfterEach
    void tearDown() {
        endpoints.close();
    }

    @Test
    void rejectsNonVitSelection() {
        assertThrows(IllegalArgumentException.class,
                () -> strategy.select(1L, RoleType.PREFILL, null));
    }

    @Test
    void returnsNullWithoutVitWorkers() {
        assertNull(strategy.select(2L, RoleType.VIT, null));
    }

    @Test
    void returnsPinnedVitMetadata() {
        registerVit("127.0.0.3", 8080, "group-a");

        try (WorkerAssignment selected = strategy.select(
                3L, RoleType.VIT, "group-a")) {
            assertNotNull(selected);
            assertEquals(RoleType.VIT, selected.serverStatus().getRole());
            assertEquals(3L, selected.serverStatus().getRequestId());
            assertEquals("127.0.0.3", selected.serverStatus().getServerIp());
            assertEquals(8080, selected.serverStatus().getHttpPort());
            assertEquals(8081, selected.serverStatus().getGrpcPort());
            assertEquals("group-a", selected.serverStatus().getGroup());
        }
    }

    @Test
    void skipsOtherGroups() {
        registerVit("127.0.0.4", 8080, "group-a");
        assertNull(strategy.select(
                4L, RoleType.VIT, "group-b"));
    }

    @Test
    void distributesAcrossRegisteredVitWorkers() {
        registerVit("127.0.0.1", 8080, null);
        registerVit("127.0.0.2", 8080, null);
        registerVit("127.0.0.3", 8080, null);
        Map<String, Integer> counts = new HashMap<>();

        for (int request = 0; request < 3_000; request++) {
            try (WorkerAssignment selected = strategy.select(
                    10_000L + request, RoleType.VIT, null)) {
                assertNotNull(selected);
                counts.merge(
                        selected.serverStatus().getServerIp(), 1, Integer::sum);
            }
        }

        assertEquals(3, counts.size());
        counts.forEach((worker, count) -> assertTrue(
                count > 750 && count < 1_250,
                worker + " was selected " + count + " times"));
    }

    @Test
    void matchingWorkersRemainUniformWhenOtherGroupsSeparateTheirAddresses() {
        registerVit("127.0.0.1", 8080, "other");
        registerVit("127.0.0.2", 8080, "target");
        registerVit("127.0.0.3", 8080, "target");
        registerVit("127.0.0.4", 8080, "other");
        EndpointRegistry directory = Mockito.spy(endpoints);
        Mockito.doReturn(List.of("127.0.0.1:8080", "127.0.0.2:8080",
                "127.0.0.3:8080", "127.0.0.4:8080"))
                .when(directory).endpointAddressSnapshot(RoleType.VIT);
        VitWorkerSelector grouped = new VitWorkerSelector(directory);
        Map<String, Integer> counts = new HashMap<>();
        for (int request = 0; request < 4_000; request++) {
            try (WorkerAssignment selected = grouped.select(request, RoleType.VIT, "target")) {
                assertNotNull(selected);
                assertEquals("target", selected.serverStatus().getGroup());
                counts.merge(selected.serverStatus().getServerIp(), 1, Integer::sum);
            }
        }
        assertEquals(2, counts.size());
        counts.forEach((worker, count) -> assertTrue(count > 1_600 && count < 2_400,
                worker + " was selected " + count + " times"));
    }

    private void registerVit(String ip, int port, String group) {
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.VIT, group, ip, port, port + 1,
                true, 0L, 0L);
        StrategyTestSupport.publishEndpoint(endpoints,
                RoleType.VIT, ip + ":" + port, status);
    }
}

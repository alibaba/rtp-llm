package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RandomStrategyTest {

    private EndpointRegistry endpoints;
    private RandomStrategy strategy;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(new FlexlbConfig());
        endpoints = StrategyTestSupport.endpointRegistry(configService);
        strategy = new RandomStrategy(new WorkerDirectory(endpoints));
    }

    @AfterEach
    void tearDown() {
        endpoints.close();
    }

    @Test
    void rejectsNonVitSelection() {
        assertThrows(IllegalArgumentException.class,
                () -> strategy.select(context(1L), RoleType.PREFILL, null));
    }

    @Test
    void returnsNullWithoutVitWorkers() {
        assertNull(strategy.select(context(2L), RoleType.VIT, null));
    }

    @Test
    void returnsPinnedVitMetadata() {
        registerVit("127.0.0.3", 8080, "group-a");

        try (SelectedRole selected = strategy.select(
                context(3L), RoleType.VIT, "group-a")) {
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
                context(4L), RoleType.VIT, "group-b"));
    }

    @Test
    void distributesAcrossRegisteredVitWorkers() {
        registerVit("127.0.0.1", 8080, null);
        registerVit("127.0.0.2", 8080, null);
        registerVit("127.0.0.3", 8080, null);
        Map<String, Integer> counts = new HashMap<>();

        for (int request = 0; request < 3_000; request++) {
            try (SelectedRole selected = strategy.select(
                    context(10_000L + request), RoleType.VIT, null)) {
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

    private void registerVit(String ip, int port, String group) {
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.VIT, group, ip, port, port + 1,
                true, 0L, 0L);
        StrategyTestSupport.publishEndpoint(endpoints,
                RoleType.VIT, ip + ":" + port, status);
    }

    private static BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext context = new BalanceContext();
        context.setConfig(new FlexlbConfig());
        context.setRequest(request);
        return context;
    }
}

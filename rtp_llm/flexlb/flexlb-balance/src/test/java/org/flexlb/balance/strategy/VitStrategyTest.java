package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Request;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks VIT selection, cache affinity and rejection of stale or invalid candidates.
 */
class VitStrategyTest {

    private EndpointRegistry endpoints;
    private VitStrategy strategy;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        endpoints = StrategyTestSupport.endpointRegistry(configService);
        strategy =
                new VitStrategy(
                        org.mockito.Mockito.mock(org.flexlb.cache.VitWorkerCacheManager.class),
                        endpoints);
    }

    @AfterEach
    void tearDown() {
        endpoints.close();
    }

    @Test
    void returnsNullWithoutVitWorkers() {
        assertNull(select(strategy, 2L, null));
    }

    @Test
    void returnsPinnedVitMetadata() {
        registerVit("127.0.0.3", 8080, "group-a");

        var selected = select(strategy, 3L, "group-a");
        assertNotNull(selected);
        assertEquals(RoleType.VIT, selected.getRole());
        assertEquals("3", selected.getRequestId());
        assertEquals("127.0.0.3", selected.getServerIp());
        assertEquals(8080, selected.getHttpPort());
        assertEquals(8081, selected.getGrpcPort());
        assertEquals("group-a", selected.getGroup());
    }

    @Test
    void skipsOtherGroups() {
        registerVit("127.0.0.4", 8080, "group-a");
        assertNull(select(strategy, 4L, "group-b"));
    }

    @Test
    void distributesAcrossRegisteredVitWorkers() {
        registerVit("127.0.0.1", 8080, null);
        registerVit("127.0.0.2", 8080, null);
        registerVit("127.0.0.3", 8080, null);
        Map<String, Integer> counts = new HashMap<>();

        for (int request = 0; request < 3_000; request++) {
            var selected = select(strategy, 10_000L + request, null);
            assertNotNull(selected);
            counts.merge(selected.getServerIp(), 1, Integer::sum);
        }

        assertEquals(3, counts.size());
        counts.forEach(
                (worker, count) ->
                        assertTrue(
                                count > 750 && count < 1_250,
                                worker + " was selected " + count + " times"));
    }

    @Test
    void matchingWorkersRemainUniformWhenOtherGroupsSeparateTheirAddresses() {
        registerVit("127.0.0.1", 8080, "other");
        registerVit("127.0.0.2", 8080, "target");
        registerVit("127.0.0.3", 8080, "target");
        registerVit("127.0.0.4", 8080, "other");
        EndpointRegistry groupedEndpoints = Mockito.spy(endpoints);
        Mockito.doReturn(
                        List.of(
                                endpoints.getEndpoint(RoleType.VIT, "127.0.0.1:8080@0"),
                                endpoints.getEndpoint(RoleType.VIT, "127.0.0.2:8080@0"),
                                endpoints.getEndpoint(RoleType.VIT, "127.0.0.3:8080@0"),
                                endpoints.getEndpoint(RoleType.VIT, "127.0.0.4:8080@0")))
                .when(groupedEndpoints)
                .endpoints(
                        org.mockito.ArgumentMatchers.eq(RoleType.VIT),
                        org.mockito.ArgumentMatchers.any());
        VitStrategy grouped =
                new VitStrategy(
                        org.mockito.Mockito.mock(org.flexlb.cache.VitWorkerCacheManager.class),
                        groupedEndpoints);
        Map<String, Integer> counts = new HashMap<>();
        for (int request = 0; request < 4_000; request++) {
            var selected = select(grouped, request, "target");
            assertNotNull(selected);
            assertEquals("target", selected.getGroup());
            counts.merge(selected.getServerIp(), 1, Integer::sum);
        }
        assertEquals(2, counts.size());
        counts.forEach(
                (worker, count) ->
                        assertTrue(
                                count > 1_600 && count < 2_400,
                                worker + " was selected " + count + " times"));
    }

    private ServerStatus select(VitStrategy selector, long id, String group) {
        var request = Request.builder().requestId(String.valueOf(id)).build();
        var context = StrategyTestSupport.scope(request).context();
        var result = selector.selectAndRegister(context, group);
        if (result.status() == SelectionResult.Status.SUCCESS)
            return result.value().toServerStatus(context.getRequestId());
        assertEquals(SelectionResult.Status.BLOCKED, result.status());
        return null;
    }

    private void registerVit(String ip, int port, String group) {
        WorkerStatus status =
                StrategyTestSupport.workerStatus(
                        RoleType.VIT, group, ip, port, port + 1, true, 0L, 0L);
        StrategyTestSupport.publishEndpoint(endpoints, status);
    }
}

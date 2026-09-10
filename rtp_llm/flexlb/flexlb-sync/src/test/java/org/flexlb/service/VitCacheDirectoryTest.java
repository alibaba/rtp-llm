package org.flexlb.service;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class VitCacheDirectoryTest {
    private VitCacheDirectory directory;
    private final Map<String, WorkerStatus> live = new HashMap<>();
    private final Map<String, WorkerEndpoint> endpoints = new HashMap<>();
    private WorkerStatus a;
    private WorkerStatus b;
    private GeneralHttpNettyService http;
    private LBStatusConsistencyService consistency;

    @BeforeEach
    void setUp() {
        a = worker("10.0.0.1", "one");
        b = worker("10.0.0.2", "two");
        WorkerDirectory workers = mock(WorkerDirectory.class);
        when(workers.statusSnapshot(RoleType.VIT))
                .thenAnswer(ignored -> Map.copyOf(live));
        when(workers.endpointAddressSnapshot(RoleType.VIT))
                .thenAnswer(ignored -> List.copyOf(endpoints.keySet()));
        when(workers.isCurrentStatus(eq(RoleType.VIT), anyString(), any()))
                .thenAnswer(call -> live.get(call.getArgument(1))
                        == call.getArgument(2));
        when(workers.captureEndpoint(eq(RoleType.VIT), anyString()))
                .thenAnswer(call -> {
                    WorkerEndpoint endpoint = endpoints.get(call.getArgument(1));
                    return endpoint == null ? null : endpoint.tryPinGeneration();
                });
        http = mock(GeneralHttpNettyService.class);
        consistency = mock(LBStatusConsistencyService.class);
        directory = new VitCacheDirectory(workers, http, consistency);
    }

    @AfterEach
    void tearDown() {
        directory.stop();
    }

    private WorkerStatus worker(String ip, String group) {
        WorkerStatus worker = WorkerStatus.createDiscovered(
                RoleType.VIT, group, ip, 8000, 8001, "test");
        live.put(worker.getIpPort(), worker);
        endpoints.put(worker.getIpPort(), new WorkerEndpoint(worker));
        return worker;
    }

    private void makeUnavailable(WorkerStatus worker) {
        endpoints.remove(worker.getIpPort());
    }

    private BalanceContext context(String... keys) {
        Request request = new Request();
        request.setRequestId(123);
        request.setMediaKeys(List.of(keys));
        request.setGenerateTimeout(30000);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return context;
    }

    private void snapshot(WorkerStatus worker, String epoch, String... keys) {
        var response = new VitCacheDirectory.CacheKeys();
        response.setFeatureHashVersion(1);
        response.setWorkerInstance(epoch);
        response.setKeys(List.of(keys));
        directory.replace(worker, response);
    }

    private ServerStatus select(BalanceContext context, String group) {
        var result = directory.select(context, group);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (SelectedRole selected = result.value()) {
            return selected.serverStatus();
        }
    }

    @Test
    void prefersMostHitsAndReplacesEvictedKeys() {
        snapshot(a, "a1", "image1", "image2");
        snapshot(b, "b1", "image1");
        assertEquals(a.getIp(), select(context("image1", "image2"), null).getServerIp());
        snapshot(a, "a1");
        assertEquals(b.getIp(), select(context("image1"), null).getServerIp());
    }

    @Test
    void cacheAffinityNeverOverridesGroupOrRoutability() {
        snapshot(a, "a1", "image");
        assertEquals(b.getIp(), select(context("image"), "two").getServerIp());
        makeUnavailable(a);
        assertEquals(b.getIp(), select(context("image"), null).getServerIp());
        makeUnavailable(b);
        assertEquals(PlacementResult.Status.BLOCKED,
                directory.select(context("image"), null).status());
    }

    @Test
    void concurrentColdRequestsSharePlacementButNotConfirmedOwnership() throws Exception {
        var executor = Executors.newFixedThreadPool(8);
        try {
            List<Future<String>> selections = IntStream.range(0, 64)
                    .mapToObj(i -> executor.submit(
                            () -> select(context("cold"), null).getServerIp()))
                    .toList();
            String selected = selections.get(0).get();
            for (Future<String> selection : selections) {
                assertEquals(selected, selection.get());
            }
            WorkerStatus other = selected.equals(a.getIp()) ? b : a;
            snapshot(other, "real", "cold");
            assertEquals(other.getIp(), select(context("cold"), null).getServerIp());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void rejectsStaleSnapshotAndInvalidSelectedWorker() {
        snapshot(a, "a1", "image");
        WorkerStatus replacement = worker(a.getIp(), "one");
        snapshot(a, "old", "image");
        snapshot(b, "b1", "image");
        assertEquals(b.getIp(), select(context("image"), null).getServerIp());
        var ctx = context("image");
        ctx.getRequest().setSelectedVit(select(ctx, "two"));
        try (SelectedRole validated = directory.validate(ctx, "two")) {
            assertTrue(validated != null);
        }
        assertNull(directory.validate(ctx, "one"));
        makeUnavailable(b);
        assertNull(directory.validate(ctx, null));
        assertTrue(live.get(replacement.getIpPort()) == replacement);
    }

    @Test
    void snapshotFailureKeepsKnownKeysAndLeaderChangeRefreshesImmediately() {
        snapshot(a, "a1", "image");
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        directory.refresh();
        verifyNoInteractions(http);
        when(http.request(any(), any(), eq("/mm_cache/keys"), eq(VitCacheDirectory.CacheKeys.class)))
                .thenReturn(Mono.error(new RuntimeException("unavailable")));
        when(consistency.isMaster()).thenReturn(true);
        directory.refresh();
        assertEquals(a.getIp(), select(context("image"), null).getServerIp());
        directory.refresh();
        verify(http, times(2)).request(any(), any(), eq("/mm_cache/keys"), eq(VitCacheDirectory.CacheKeys.class));
        when(consistency.isMaster()).thenReturn(false);
        directory.refresh();
        when(consistency.isMaster()).thenReturn(true);
        directory.refresh();
        verify(http, times(4)).request(any(), any(), eq("/mm_cache/keys"), eq(VitCacheDirectory.CacheKeys.class));
    }
}

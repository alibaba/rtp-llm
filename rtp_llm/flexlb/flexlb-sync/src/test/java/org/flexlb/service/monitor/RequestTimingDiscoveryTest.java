package org.flexlb.service.monitor;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.ModelServiceConfiguration;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.discovery.StaticServiceDiscoveryProvider;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestTimingDiscoveryTest {
    @Test
    void validatedStaticFrontendAndMultiEnginePrefillUseSharedGrpcPorts() {
        Endpoint frontend = endpoint("fe", "10.0.0.1:8600");
        Endpoint prefill = endpoint("p", "10.0.0.2:8700");
        prefill.setMultiEngineNum(4);
        prefill.setWorkerStatusPort(9500);
        Endpoint fusion = endpoint("fusion", "10.0.0.3:8800");
        GroupRoleEndPoint first = new GroupRoleEndPoint();
        first.setGroup("first");
        first.setFrontendEndpoint(frontend);
        first.setPrefillEndpoint(prefill);
        GroupRoleEndPoint second = new GroupRoleEndPoint();
        second.setGroup("second");
        second.setPrefillEndpoint(endpoint("p-copy", "10.0.0.2:8700"));
        second.setPdFusionEndpoint(fusion);
        ServiceRoute route = new ServiceRoute();
        route.setRoleEndpoints(List.of(first, second));
        ConfigService config = mock(ConfigService.class);
        when(config.modelServiceConfig()).thenReturn(route);
        RoutingServiceDiscovery provider = new RoutingServiceDiscovery(List.of(new StaticServiceDiscoveryProvider()));
        ModelMetaConfig model = new ModelServiceConfiguration().modelMetaConfig(config, provider);
        ManualExecutor executor = new ManualExecutor();
        RequestTimingDiscovery discovery = new RequestTimingDiscovery(model, provider, executor, () -> 0);
        assertTrue(discovery.sources().isEmpty());
        executor.runAll();
        assertEquals(Set.of(source("10.0.0.1", 8600, "frontend"), source("10.0.0.2", 8700, "prefill"),
                source("10.0.0.3", 8800, "prefill")), discovery.sources());
        discovery.shutdown();
    }

    @Test
    void successfulEmptyAndFailureAreIsolatedPerEndpoint() {
        Endpoint fe = endpoint("fe", "fe:8600");
        Endpoint p = endpoint("p", "p:8700");
        ModelMetaConfig model = model(fe, p);
        ServiceDiscovery provider = mock(ServiceDiscovery.class);
        when(provider.getHosts(fe)).thenReturn(List.of(WorkerHost.of("fe-old", 8599)))
                .thenThrow(new IllegalStateException("failed fe refresh"));
        when(provider.getHosts(p)).thenReturn(List.of(WorkerHost.of("p-old", 8699))).thenReturn(List.of());
        AtomicLong now = new AtomicLong();
        ManualExecutor executor = new ManualExecutor();
        RequestTimingDiscovery discovery = new RequestTimingDiscovery(model, provider, executor, now::get);
        discovery.sources();
        executor.runAll();
        assertEquals(2, discovery.sources().size());
        now.addAndGet(TimeUnit.SECONDS.toNanos(2));
        discovery.sources();
        executor.runAll();
        assertEquals(Set.of(source("fe-old", 8600, "frontend")), discovery.sources());
        verify(provider, never()).listen(any(Endpoint.class), any());
        verify(provider, never()).validate(any());
        discovery.shutdown();
        verify(provider, never()).shutdown();
    }

    @Test
    void readsServiceEntriesOnceWhileRefreshingInstancesAndClearingStateOnReset() {
        Endpoint fe = endpoint("fe", "fe:8600");
        Endpoint p = endpoint("p", "p:8700");
        ModelMetaConfig model = model(fe, p);
        ServiceDiscovery provider = mock(ServiceDiscovery.class);
        when(provider.getHosts(fe)).thenReturn(List.of(WorkerHost.of("fe", 8599)), List.of());
        when(provider.getHosts(p)).thenReturn(List.of(WorkerHost.of("p", 8699)), List.of(WorkerHost.of("new-p", 8699)));
        ManualExecutor executor = new ManualExecutor();
        AtomicLong now = new AtomicLong();
        RequestTimingDiscovery discovery = new RequestTimingDiscovery(model, provider, executor, now::get);
        discovery.sources();
        executor.runAll();
        assertEquals(Set.of(source("fe", 8600, "frontend"), source("p", 8700, "prefill")), discovery.sources());
        // Changing the mocked topology after construction cannot change the startup service entries.
        when(model.endpointsWithGroup(RoleType.FRONTEND)).thenReturn(List.of());
        when(model.endpointsWithGroup(RoleType.PREFILL)).thenReturn(List.of());
        now.addAndGet(TimeUnit.SECONDS.toNanos(2));
        discovery.sources();
        executor.runAll();
        assertEquals(Set.of(source("new-p", 8700, "prefill")), discovery.sources());
        discovery.reset();
        assertTrue(discovery.sources().isEmpty());
        executor.runAll();
        assertEquals(Set.of(source("new-p", 8700, "prefill")), discovery.sources());
        verify(model, times(1)).endpointsWithGroup(RoleType.FRONTEND);
        verify(model, times(1)).endpointsWithGroup(RoleType.PREFILL);
        verify(model, times(1)).endpointsWithGroup(RoleType.PDFUSION);
        discovery.shutdown();
        assertTrue(discovery.sources().isEmpty());
        assertTrue(executor.isShutdown());
    }

    @Test
    void blockedProviderDoesNotBlockCollectorAndOldGenerationCannotOverwriteNewHosts() throws Exception {
        Endpoint p = endpoint("p", "p:8700");
        ModelMetaConfig model = model(null, p);
        ServiceDiscovery provider = mock(ServiceDiscovery.class);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger calls = new AtomicInteger();
        when(provider.getHosts(p)).thenAnswer(invocation -> {
            if (calls.getAndIncrement() == 0) {
                entered.countDown();
                boolean waiting = true;
                while (waiting) {
                    try {
                        release.await();
                        waiting = false;
                    } catch (InterruptedException ignored) {
                        // An external provider may return after cancellation instead of honoring it.
                    }
                }
                return List.of(WorkerHost.of("stale", 8699));
            }
            return List.of(WorkerHost.of("fresh", 8699));
        });
        ThreadPoolExecutor executor = new ThreadPoolExecutor(2, 2, 1, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
        RequestTimingDiscovery discovery = new RequestTimingDiscovery(model, provider, executor, System::nanoTime);
        try {
            assertTimeoutPreemptively(Duration.ofSeconds(1), () -> assertTrue(discovery.sources().isEmpty()));
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            assertTrue(discovery.sources().isEmpty());
            assertEquals(1, calls.get());
            discovery.reset();
            discovery.sources();
            awaitSources(discovery, Set.of(source("fresh", 8700, "prefill")));
            release.countDown();
            assertTimeoutPreemptively(Duration.ofSeconds(2), () -> {
                while (executor.getActiveCount() != 0) {
                    Thread.sleep(5);
                }
            });
            assertEquals(Set.of(source("fresh", 8700, "prefill")), discovery.sources());
        } finally {
            release.countDown();
            discovery.shutdown();
        }
    }

    @Test
    void nullAndEntirelyInvalidSnapshotsDoNotEraseKnownHosts() {
        Endpoint p = endpoint("p", "p:8700");
        ServiceDiscovery provider = mock(ServiceDiscovery.class);
        List<WorkerHost> invalid = new ArrayList<>();
        invalid.add(null);
        invalid.add(new WorkerHost("", 8600));
        when(provider.getHosts(p)).thenReturn(List.of(WorkerHost.of("p", 8699))).thenReturn(null).thenReturn(invalid);
        AtomicLong now = new AtomicLong();
        ManualExecutor executor = new ManualExecutor();
        RequestTimingDiscovery discovery = new RequestTimingDiscovery(model(null, p), provider, executor, now::get);
        for (int i = 0; i < 3; i++) {
            discovery.sources();
            executor.runAll();
            assertEquals(Set.of(source("p", 8700, "prefill")), discovery.sources());
            now.addAndGet(TimeUnit.SECONDS.toNanos(2));
        }
        discovery.shutdown();
    }

    @Test
    void vipServerRequiresTwoEmptyRefreshRoundsAndRecoveryResetsTheCount() {
        Endpoint p = endpoint("vip-p", "p:8700");
        p.getDiscovery().setType(ServiceDiscoveryType.VIPSERVER);
        ServiceDiscovery provider = mock(ServiceDiscovery.class);
        List<WorkerHost> hosts = List.of(WorkerHost.of("p", 8699));
        when(provider.getHosts(p)).thenReturn(hosts, List.of(), hosts, List.of(), List.of());
        AtomicLong now = new AtomicLong();
        ManualExecutor executor = new ManualExecutor();
        RequestTimingDiscovery discovery = new RequestTimingDiscovery(model(null, p), provider, executor, now::get);
        for (int round = 0; round < 5; round++) {
            discovery.sources();
            executor.runAll();
            Set<RequestTimingSettings.Source> expected = round == 4 ? Set.of() : Set.of(source("p", 8700, "prefill"));
            // Re-reading the snapshot in one collection round cannot confirm an empty VIP result twice.
            for (int read = 0; read < 5; read++) {
                assertEquals(expected, discovery.sources());
                executor.runAll();
            }
            now.addAndGet(TimeUnit.SECONDS.toNanos(2));
        }
        discovery.shutdown();
    }

    private static void awaitSources(RequestTimingDiscovery discovery, Set<RequestTimingSettings.Source> expected) {
        assertTimeoutPreemptively(Duration.ofSeconds(2), () -> {
            while (!discovery.sources().equals(expected)) {
                Thread.sleep(5);
            }
        });
    }

    private static ModelMetaConfig model(Endpoint fe, Endpoint p) {
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.endpointsWithGroup(any())).thenReturn(List.of());
        if (fe != null) {
            when(model.endpointsWithGroup(RoleType.FRONTEND)).thenReturn(List.of(Pair.of("g", fe)));
        }
        if (p != null) {
            when(model.endpointsWithGroup(RoleType.PREFILL)).thenReturn(List.of(Pair.of("g", p)));
        }
        return model;
    }

    private static Endpoint endpoint(String address, String host) {
        DiscoveryConfig config = new DiscoveryConfig();
        config.setType(ServiceDiscoveryType.STATIC_ENV);
        config.setHosts(List.of(host));
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol("grpc");
        endpoint.setDiscovery(config);
        return endpoint;
    }

    private static RequestTimingSettings.Source source(String host, int port, String role) {
        return new RequestTimingSettings.Source(new GrpcTarget(host, port), role);
    }

    private static final class ManualExecutor extends AbstractExecutorService {
        private final ArrayDeque<Runnable> tasks = new ArrayDeque<>();
        private boolean closed;

        void runAll() {
            while (!tasks.isEmpty()) {
                tasks.removeFirst().run();
            }
        }

        @Override
        public void execute(Runnable task) {
            tasks.addLast(task);
        }

        @Override
        public void shutdown() {
            closed = true;
        }

        @Override
        public List<Runnable> shutdownNow() {
            closed = true;
            List<Runnable> pending = List.copyOf(tasks);
            tasks.clear();
            return pending;
        }

        @Override
        public boolean isShutdown() {
            return closed;
        }

        @Override
        public boolean isTerminated() {
            return closed;
        }

        @Override
        public boolean awaitTermination(long timeout, TimeUnit unit) {
            return closed;
        }
    }
}

package org.flexlb.service.address;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class WorkerAddressServiceTest {

    @Mock
    private EngineHealthReporter engineHealthReporter;
    @Mock
    private ModelMetaConfig modelMetaConfig;
    @Mock
    private ServiceDiscovery serviceDiscovery;
    @Mock
    private ConfigService configService;
    private WorkerAddressService workerAddressService;

    @BeforeEach
    void setUp() {
        when(configService.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        workerAddressService = new WorkerAddressService(engineHealthReporter, modelMetaConfig,
                serviceDiscovery, configService);
    }

    @AfterEach
    void tearDown() {
        workerAddressService.destroy();
    }

    @Test
    void firstDiscoveryWithoutCacheReturnsNoWorkers() {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of())
                .thenThrow(new IllegalStateException("unreachable"));
        assertTrue(refresh().isEmpty());
        assertTrue(refresh().isEmpty());
    }

    @Test
    void emptyResultsRetainSnapshotAndNonEmptyRecoveryReplacesIt() {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        WorkerHost first = WorkerHost.of("10.0.0.1", 8080);
        WorkerHost second = WorkerHost.of("10.0.0.2", 8080);
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of(first, second), List.of());

        assertEquals(List.of(first, second), refresh());
        for (int i = 0; i < 100; i++) {
            assertEquals(List.of(first, second), refresh());
        }
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of(second), List.of());
        assertEquals(List.of(second), refresh());
        assertEquals(List.of(second), refresh());
    }

    @Test
    void exceptionRetainsCachedWorkersAndReportsFailureOnce() {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<WorkerHost> hosts = List.of(WorkerHost.of("10.0.0.1", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(hosts)
                .thenThrow(new IllegalStateException("unreachable"));

        assertEquals(hosts, refresh());
        assertEquals(hosts, refresh());
        verify(engineHealthReporter).reportStatusCheckerFail(
                "model", BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
    }

    @Test
    void timeoutRetainsCacheAndLateResultCannotOverwriteRecovery() throws Exception {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<WorkerHost> oldHosts = List.of(WorkerHost.of("10.0.0.1", 8080));
        List<WorkerHost> lateHosts = List.of(WorkerHost.of("10.0.0.2", 8080));
        List<WorkerHost> recoveredHosts = List.of(WorkerHost.of("10.0.0.3", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(oldHosts);
        assertEquals(oldHosts, refresh());
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch interrupted = new CountDownLatch(1);
        when(serviceDiscovery.getHosts(endpoint)).thenAnswer(ignored -> {
            try {
                assertTrue(release.await(5, TimeUnit.SECONDS));
            } catch (InterruptedException expected) {
                interrupted.countDown();
                assertTrue(release.await(5, TimeUnit.SECONDS));
            }
            return lateHosts;
        });
        try {
            assertEquals(oldHosts, refresh());
            assertTrue(interrupted.await(5, TimeUnit.SECONDS));
            verify(engineHealthReporter).reportStatusCheckerFail(
                    "model", BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null);
            doReturn(recoveredHosts).when(serviceDiscovery).getHosts(endpoint);
            assertEquals(recoveredHosts, refresh());
        } finally {
            release.countDown();
            workerAddressService.destroy();
            ThreadPoolExecutor executor = (ThreadPoolExecutor) ReflectionTestUtils.getField(
                    workerAddressService, "serviceDiscoveryExecutor");
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
        // A rejected query reads the cache after the late task has finished.
        assertEquals(recoveredHosts, refresh());
    }

    @Test
    void saturatedExecutorRetainsCacheWithoutRunningDiscoveryOnCaller() throws Exception {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<WorkerHost> hosts = List.of(WorkerHost.of("10.0.0.1", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(hosts);
        assertEquals(hosts, refresh());
        ThreadPoolExecutor executor = (ThreadPoolExecutor) ReflectionTestUtils.getField(
                workerAddressService, "serviceDiscoveryExecutor");
        executor.setCorePoolSize(1);
        executor.setMaximumPoolSize(1);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        executor.submit(() -> {
            started.countDown();
            assertTrue(release.await(5, TimeUnit.SECONDS));
            return null;
        });
        try {
            assertTrue(started.await(5, TimeUnit.SECONDS));
            while (executor.getQueue().offer(() -> { })) {
                // Fill the bounded queue while its only worker is occupied.
            }
            assertEquals(hosts, refresh());
            verify(serviceDiscovery).getHosts(endpoint);
            verify(engineHealthReporter).reportStatusCheckerFail(
                    "model", BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
        } finally {
            executor.getQueue().clear();
            release.countDown();
        }
    }

    @Test
    void interruptedQueryRetainsCacheAndInterruptFlag() {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<WorkerHost> hosts = List.of(WorkerHost.of("10.0.0.1", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(hosts);
        assertEquals(hosts, refresh());
        Thread refreshThread = Thread.currentThread();
        when(serviceDiscovery.getHosts(endpoint)).thenAnswer(ignored -> {
            refreshThread.interrupt();
            new CountDownLatch(1).await(5, TimeUnit.SECONDS);
            return List.of();
        });
        try {
            assertEquals(hosts, refresh());
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    void oneEmptyEndpointRetainsItsWorkersWhileAnotherUpdates() {
        Endpoint first = endpoint("vip-a");
        Endpoint second = endpoint("vip-b");
        configure(first, second);
        WorkerHost a = WorkerHost.of("10.0.0.1", 8080);
        WorkerHost b = WorkerHost.of("10.0.0.2", 8080);
        WorkerHost c = WorkerHost.of("10.0.0.3", 8080);
        when(serviceDiscovery.getHosts(first)).thenReturn(List.of(a), List.of());
        when(serviceDiscovery.getHosts(second)).thenReturn(List.of(b), List.of(c));

        assertEquals(List.of(a, b), refresh());
        assertEquals(List.of(a, c), refresh());
    }

    @Test
    void sameAddressWithDifferentEndpointConfigurationHasSeparateCache() {
        Endpoint first = endpoint("shared-vip");
        first.setGroup("group-a");
        Endpoint second = endpoint("shared-vip");
        second.setGroup("group-b");
        second.setWorkerStatusPort(18002);
        configure(first, second);
        WorkerHost a = new WorkerHost("10.0.0.1", 8080, 8081, 8085, "site", "group-a");
        WorkerHost b = new WorkerHost("10.0.0.1", 8080, 8081, 8085, 18002, "site", "group-b", "");
        when(serviceDiscovery.getHosts(first)).thenReturn(List.of(a), List.of());
        when(serviceDiscovery.getHosts(second)).thenReturn(List.of(b), List.of());

        assertEquals(List.of(a, b), refresh());
        assertEquals(List.of(a, b), refresh());
    }

    @Test
    void callerCannotModifyCachedWorkerList() {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        WorkerHost host = WorkerHost.of("10.0.0.1", 8080);
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of(host), List.of());
        List<WorkerHost> returnedHosts = refresh();
        returnedHosts.clear();

        assertEquals(List.of(host), refresh());
    }

    @Test
    void explicitlyRemovedEndpointDoesNotReuseDiscoveryCache() {
        Endpoint endpoint = endpoint("vip");
        configure(endpoint);
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of(WorkerHost.of("10.0.0.1", 8080)));
        assertEquals(1, refresh().size());
        configure();
        assertTrue(refresh().isEmpty());
    }

    private void configure(Endpoint... endpoints) {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(Arrays.stream(endpoints).map(endpoint -> Pair.of(endpoint.getGroup(), endpoint)).toList());
    }

    private List<WorkerHost> refresh() {
        return workerAddressService.getEngineWorkerList("model", RoleType.PREFILL);
    }

    private static Endpoint endpoint(String address) {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        return endpoint;
    }
}

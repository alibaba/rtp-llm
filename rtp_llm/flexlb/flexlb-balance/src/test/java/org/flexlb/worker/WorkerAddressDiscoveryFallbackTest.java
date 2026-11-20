package org.flexlb.worker;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.config.service.WorkerDiscoveryEndpoint;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.util.discovery.ServiceDiscovery;
import org.flexlb.worker.status.BalanceStatusEnum;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.flexlb.worker.status.WorkerAddressService;
import org.flexlb.worker.status.WorkerHost;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks Worker discovery fallback and address changes across configured sources.
 */
@ExtendWith(MockitoExtension.class)
class WorkerAddressDiscoveryFallbackTest {

    @Mock private ServiceMetricsReporter serviceMetricsReporter;
    @Mock private ModelMetaConfig modelMetaConfig;
    @Mock private ServiceDiscovery serviceDiscovery;
    @Mock private ConfigService configService;
    private WorkerAddressService workerAddressService;

    @BeforeEach
    void setUp() {
        when(configService.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        workerAddressService =
                new WorkerAddressService(
                        serviceMetricsReporter, modelMetaConfig, serviceDiscovery, configService);
    }

    @AfterEach
    void tearDown() {
        workerAddressService.destroy();
    }

    @Test
    void firstDiscoveryDistinguishesEmptyResultFromFailureWithoutCache() {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of())
                .thenThrow(new IllegalStateException("unreachable"));
        assertTrue(refresh().isEmpty());
        org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, this::refresh);
    }

    @Test
    void emptyResultsRetainSnapshotAndNonEmptyRecoveryReplacesIt() {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        DiscoveredHost first = DiscoveredHost.of("10.0.0.1", 8080);
        DiscoveredHost second = DiscoveredHost.of("10.0.0.2", 8080);
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of(first, second), List.of());

        assertAddresses(List.of(first, second), refresh());
        for (int i = 0; i < 100; i++) {
            assertAddresses(List.of(first, second), refresh());
        }
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of(second), List.of());
        assertAddresses(List.of(second), refresh());
        assertAddresses(List.of(second), refresh());
    }

    @Test
    void exceptionRetainsCachedWorkersAndReportsFailureOnce() {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<DiscoveredHost> hosts = List.of(DiscoveredHost.of("10.0.0.1", 8080));
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(hosts)
                .thenThrow(new IllegalStateException("unreachable"));

        assertAddresses(hosts, refresh());
        assertAddresses(hosts, refresh());
        verify(serviceMetricsReporter)
                .reportStatusCheckerFail(BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
        verify(serviceMetricsReporter)
                .reportRawServiceDiscoveryHostCount("model", RoleType.PREFILL, 1);
        verify(serviceMetricsReporter, never())
                .reportRawServiceDiscoveryHostCount("model", RoleType.PREFILL, 0);
    }

    @Test
    void reportsRawEmptyDiscoveryEvenWhenCachedHostsRemainAvailable() {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<DiscoveredHost> hosts = List.of(DiscoveredHost.of("10.0.0.1", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(hosts, List.of());

        assertAddresses(hosts, refresh());
        assertAddresses(hosts, refresh());

        verify(serviceMetricsReporter)
                .reportRawServiceDiscoveryHostCount("model", RoleType.PREFILL, 1);
        verify(serviceMetricsReporter)
                .reportRawServiceDiscoveryHostCount("model", RoleType.PREFILL, 0);
    }

    @Test
    void timeoutRetainsCacheAndLateResultCannotOverwriteRecovery() throws Exception {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<DiscoveredHost> oldHosts = List.of(DiscoveredHost.of("10.0.0.1", 8080));
        List<DiscoveredHost> lateHosts = List.of(DiscoveredHost.of("10.0.0.2", 8080));
        List<DiscoveredHost> recoveredHosts = List.of(DiscoveredHost.of("10.0.0.3", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(oldHosts);
        assertAddresses(oldHosts, refresh());
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch interrupted = new CountDownLatch(1);
        when(serviceDiscovery.getHosts(endpoint))
                .thenAnswer(
                        ignored -> {
                            try {
                                assertTrue(release.await(5, TimeUnit.SECONDS));
                            } catch (InterruptedException expected) {
                                interrupted.countDown();
                                assertTrue(release.await(5, TimeUnit.SECONDS));
                            }
                            return lateHosts;
                        });
        try {
            assertAddresses(oldHosts, refresh());
            assertTrue(interrupted.await(5, TimeUnit.SECONDS));
            verify(serviceMetricsReporter)
                    .reportStatusCheckerFail(BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null);
            doReturn(recoveredHosts).when(serviceDiscovery).getHosts(endpoint);
            assertAddresses(recoveredHosts, refresh());
        } finally {
            release.countDown();
            workerAddressService.destroy();
            ThreadPoolExecutor executor =
                    (ThreadPoolExecutor)
                            ReflectionTestUtils.getField(
                                    workerAddressService, "serviceDiscoveryExecutor");
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
        // A rejected query reads the cache after the late task has finished.
        assertAddresses(recoveredHosts, refresh());
    }

    @Test
    void saturatedExecutorRetainsCacheWithoutRunningDiscoveryOnCaller() throws Exception {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<DiscoveredHost> hosts = List.of(DiscoveredHost.of("10.0.0.1", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(hosts);
        assertAddresses(hosts, refresh());
        ThreadPoolExecutor executor =
                (ThreadPoolExecutor)
                        ReflectionTestUtils.getField(
                                workerAddressService, "serviceDiscoveryExecutor");
        executor.setCorePoolSize(1);
        executor.setMaximumPoolSize(1);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        executor.submit(
                () -> {
                    started.countDown();
                    assertTrue(release.await(5, TimeUnit.SECONDS));
                    return null;
                });
        try {
            assertTrue(started.await(5, TimeUnit.SECONDS));
            while (executor.getQueue().offer(() -> {})) {
                // Fill the bounded queue while its only worker is occupied.
            }
            assertAddresses(hosts, refresh());
            verify(serviceDiscovery).getHosts(endpoint);
            verify(serviceMetricsReporter)
                    .reportStatusCheckerFail(BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
        } finally {
            executor.getQueue().clear();
            release.countDown();
        }
    }

    @Test
    void interruptedQueryRetainsCacheAndInterruptFlag() {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        List<DiscoveredHost> hosts = List.of(DiscoveredHost.of("10.0.0.1", 8080));
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(hosts);
        assertAddresses(hosts, refresh());
        Thread refreshThread = Thread.currentThread();
        when(serviceDiscovery.getHosts(endpoint))
                .thenAnswer(
                        ignored -> {
                            refreshThread.interrupt();
                            new CountDownLatch(1).await(5, TimeUnit.SECONDS);
                            return List.of();
                        });
        try {
            assertAddresses(hosts, refresh());
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    void oneEmptyEndpointRetainsItsWorkersWhileAnotherUpdates() {
        WorkerDiscoveryEndpoint first = endpoint("vip-a");
        WorkerDiscoveryEndpoint second = endpoint("vip-b");
        configure(first, second);
        DiscoveredHost a = DiscoveredHost.of("10.0.0.1", 8080);
        DiscoveredHost b = DiscoveredHost.of("10.0.0.2", 8080);
        DiscoveredHost c = DiscoveredHost.of("10.0.0.3", 8080);
        when(serviceDiscovery.getHosts(first)).thenReturn(List.of(a), List.of());
        when(serviceDiscovery.getHosts(second)).thenReturn(List.of(b), List.of(c));

        assertAddresses(List.of(a, b), refresh());
        assertAddresses(List.of(a, c), refresh());
        verify(serviceMetricsReporter)
                .reportRawServiceDiscoveryHostCount("model", RoleType.PREFILL, 2);
        verify(serviceMetricsReporter)
                .reportRawServiceDiscoveryHostCount("model", RoleType.PREFILL, 1);
    }

    @Test
    void sameAddressWithDifferentEndpointConfigurationHasSeparateCache() {
        WorkerDiscoveryEndpoint first = endpoint("shared-vip");
        first.setGroup("group-a");
        WorkerDiscoveryEndpoint second = endpoint("shared-vip");
        second.setGroup("group-b");
        second.setWorkerStatusPort(18002);
        configure(first, second);
        DiscoveredHost a = new DiscoveredHost("10.0.0.1", 8080, "site");
        DiscoveredHost b = new DiscoveredHost("10.0.0.1", 8080, "site");
        when(serviceDiscovery.getHosts(first)).thenReturn(List.of(a), List.of());
        when(serviceDiscovery.getHosts(second)).thenReturn(List.of(b), List.of());

        var initial = refresh();
        assertEquals(
                List.of("group-a", "group-b"), initial.stream().map(WorkerHost::getGroup).toList());
        assertEquals(
                List.of(8081, 18002),
                initial.stream().map(WorkerHost::getWorkerStatusPort).toList());
        assertEquals(initial, refresh());
    }

    @Test
    void sameAddressWithDifferentEndpointConfigurationHasSeparateAvailabilityWarnings() {
        WorkerDiscoveryEndpoint first = endpoint("shared-vip");
        first.setGroup("group-a");
        WorkerDiscoveryEndpoint second = endpoint("shared-vip");
        second.setGroup("group-b");
        second.setWorkerStatusPort(18002);
        configure(first, second);
        DiscoveredHost a = new DiscoveredHost("10.0.0.1", 8080, "site");
        DiscoveredHost b = new DiscoveredHost("10.0.0.1", 8080, "site");
        when(serviceDiscovery.getHosts(first))
                .thenReturn(List.of(), List.of(), List.of(), List.of(a));
        when(serviceDiscovery.getHosts(second)).thenReturn(List.of(), List.of(b));

        Logger logger = (Logger) LoggerFactory.getLogger("syncLogger");
        Level originalLevel = logger.getLevel();
        ListAppender<ILoggingEvent> events = new ListAppender<>();
        events.start();
        logger.addAppender(events);
        logger.setLevel(Level.INFO);
        try {
            assertTrue(refresh().isEmpty());
            assertEquals(
                    List.of("group-a", "group-b"),
                    events.list.stream()
                            .filter(event -> event.getLevel() == Level.WARN)
                            .map(event -> event.getArgumentArray()[2])
                            .toList());

            assertAddresses(List.of(b), refresh());
            assertAddresses(List.of(b), refresh());
            assertEquals(
                    2L,
                    events.list.stream().filter(event -> event.getLevel() == Level.WARN).count(),
                    "recovering another endpoint must not reset an empty endpoint's warning deadline");
            assertEquals(
                    List.of("group-b"),
                    events.list.stream()
                            .filter(
                                    event ->
                                            event.getFormattedMessage()
                                                    .startsWith("Worker discovery recovered"))
                            .map(event -> event.getArgumentArray()[2])
                            .toList());

            assertAddresses(List.of(a, b), refresh());
            assertEquals(
                    List.of("group-b", "group-a"),
                    events.list.stream()
                            .filter(
                                    event ->
                                            event.getFormattedMessage()
                                                    .startsWith("Worker discovery recovered"))
                            .map(event -> event.getArgumentArray()[2])
                            .toList());
        } finally {
            logger.detachAppender(events);
            logger.setLevel(originalLevel);
            events.stop();
        }
    }

    @Test
    void callerCannotModifyCachedWorkerList() {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        DiscoveredHost host = DiscoveredHost.of("10.0.0.1", 8080);
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of(host), List.of());
        List<WorkerHost> returnedHosts = refresh();
        returnedHosts.clear();

        assertAddresses(List.of(host), refresh());
    }

    @Test
    void explicitlyRemovedEndpointDoesNotReuseDiscoveryCache() {
        WorkerDiscoveryEndpoint endpoint = endpoint("vip");
        configure(endpoint);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(DiscoveredHost.of("10.0.0.1", 8080)));
        assertEquals(1, refresh().size());
        configure();
        assertTrue(refresh().isEmpty());
    }

    private void configure(WorkerDiscoveryEndpoint... endpoints) {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(
                        Arrays.stream(endpoints)
                                .map(endpoint -> Pair.of(endpoint.getGroup(), endpoint))
                                .toList());
    }

    private static void assertAddresses(List<DiscoveredHost> expected, List<WorkerHost> actual) {
        assertEquals(
                expected.stream().map(DiscoveredHost::getIpPort).toList(),
                actual.stream().map(WorkerHost::getIpPort).toList());
    }

    private List<WorkerHost> refresh() {
        return workerAddressService.getEngineWorkerList("model", RoleType.PREFILL);
    }

    private static WorkerDiscoveryEndpoint endpoint(String address) {
        WorkerDiscoveryEndpoint endpoint = new WorkerDiscoveryEndpoint();
        endpoint.setAddress(address);
        return endpoint;
    }
}

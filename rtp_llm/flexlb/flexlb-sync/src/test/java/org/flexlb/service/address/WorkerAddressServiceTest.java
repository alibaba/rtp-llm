package org.flexlb.service.address;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.springframework.test.util.ReflectionTestUtils;

import static org.mockito.Mockito.anyString;
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
        Mockito.lenient().when(configService.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        workerAddressService = new WorkerAddressService(engineHealthReporter, modelMetaConfig,
                serviceDiscovery, configService);
    }

    @AfterEach
    void tearDown() {
        workerAddressService.destroy();
    }

    @Test
    void discoveryFailureReturnsNoWorkers() {
        String modelName = "TestModel";
        String address = "TestAddress";
        when(modelMetaConfig.endpointsWithGroup(modelName, RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group1", endpoint(address))));
        when(serviceDiscovery.getHosts(anyString()))
                .thenThrow(new IllegalStateException("discovery unavailable"));

        List<WorkerHost> actualHosts = workerAddressService.getEngineWorkerList(
                modelName, RoleType.PREFILL);

        Assertions.assertTrue(actualHosts.isEmpty());
    }

    @Test
    void discoveredWorkersAreConvertedThroughThePublicBoundary() {
        String modelName = "TestModel";
        String address = "TestAddress";
        List<WorkerHost> expectedHosts = List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site1", "group1"));
        when(modelMetaConfig.endpointsWithGroup(modelName, RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group1", endpoint(address))));
        when(serviceDiscovery.getHosts(anyString())).thenReturn(expectedHosts);

        List<WorkerHost> actualHosts = workerAddressService.getEngineWorkerList(
                modelName, RoleType.PREFILL);

        Assertions.assertFalse(actualHosts.isEmpty());
    }

    @Test
    void saturatedDiscoveryPoolDoesNotRunDiscoveryOnTheCaller() throws Exception {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group", endpoint("address"))));
        CountDownLatch release = new CountDownLatch(1);
        ThreadPoolExecutor executor = (ThreadPoolExecutor)
                ReflectionTestUtils.getField(workerAddressService, "serviceDiscoveryExecutor");
        executor.setMaximumPoolSize(executor.getCorePoolSize());
        CountDownLatch occupied = new CountDownLatch(executor.getCorePoolSize());
        Runnable blocker = () -> {
            occupied.countDown();
            try { release.await(); }
            catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        };
        Mockito.lenient().when(serviceDiscovery.getHosts("address")).thenAnswer(call -> {
            release.await();
            return List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site", "group"));
        });
        try {
            for (int i = 0; i < executor.getCorePoolSize(); i++) { executor.execute(blocker); }
            Assertions.assertTrue(occupied.await(2, TimeUnit.SECONDS));
            while (executor.getQueue().offer(blocker)) { }
            Assertions.assertTimeoutPreemptively(Duration.ofSeconds(2), () ->
                    Assertions.assertTrue(workerAddressService.getEngineWorkerList("model", RoleType.PREFILL).isEmpty()));
            Mockito.verify(serviceDiscovery, Mockito.never()).getHosts("address");
        } finally {
            executor.getQueue().clear();
            release.countDown();
        }
    }

    @Test
    void interruptedDiscoveryCancelsWorkEvenWhenFailureReportingThrows() throws Exception {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group", endpoint("address"))));
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch exited = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(serviceDiscovery.getHosts("address")).thenAnswer(call -> {
            entered.countDown();
            try { release.await(); return List.of(); }
            finally { exited.countDown(); }
        });
        RuntimeException telemetryFailure = new IllegalStateException("telemetry failed");
        Mockito.doThrow(telemetryFailure).when(engineHealthReporter).reportStatusCheckerFail(
                "model", org.flexlb.enums.BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
        AtomicReference<Throwable> result = new AtomicReference<>();
        AtomicBoolean interrupted = new AtomicBoolean();
        Thread caller = new Thread(() -> {
            try { workerAddressService.getEngineWorkerList("model", RoleType.PREFILL); }
            catch (Throwable failure) { result.set(failure); }
            finally { interrupted.set(Thread.currentThread().isInterrupted()); }
        });
        try {
            caller.start();
            Assertions.assertTrue(entered.await(2, TimeUnit.SECONDS));
            caller.interrupt();
            caller.join(2000);
            Assertions.assertFalse(caller.isAlive());
            Assertions.assertAll(
                    () -> Assertions.assertSame(telemetryFailure, result.get()),
                    () -> Assertions.assertTrue(interrupted.get()),
                    () -> Assertions.assertTrue(exited.await(2, TimeUnit.SECONDS),
                            "discovery must receive cancellation"));
        } finally {
            release.countDown();
            caller.interrupt();
            caller.join(2000);
        }
    }

    private static Endpoint endpoint(String address) {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol(BackendServiceProtocolEnum.GRPC.getName());
        return endpoint;
    }
}

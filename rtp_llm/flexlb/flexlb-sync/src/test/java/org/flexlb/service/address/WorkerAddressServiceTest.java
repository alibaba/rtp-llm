package org.flexlb.service.address;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.enums.BalanceStatusEnum;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

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
    void discoveryFailureIsNotAnEmptySuccessfulResult() {
        String modelName = "TestModel";
        String address = "TestAddress";
        when(modelMetaConfig.endpointsWithGroup(modelName, RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group1", endpoint(address))));
        when(serviceDiscovery.getHosts(anyString()))
                .thenThrow(new IllegalStateException("discovery unavailable"));

        Assertions.assertThrows(IllegalStateException.class,
                () -> workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL));
        Mockito.verify(engineHealthReporter).reportStatusCheckerFail(
                modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
    }

    @Test
    void aFailedGroupPreventsReturningPartialDiscoveryResults() {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("first", endpoint("first")), Pair.of("second", endpoint("second"))));
        when(serviceDiscovery.getHosts("first")).thenReturn(List.of(new WorkerHost("127.0.0.1", 8080)));
        when(serviceDiscovery.getHosts("second")).thenThrow(new IllegalStateException("discovery unavailable"));

        Assertions.assertThrows(IllegalStateException.class,
                () -> workerAddressService.getEngineWorkerList("model", RoleType.PREFILL));
        Mockito.verify(serviceDiscovery).getHosts("first");
        Mockito.verify(serviceDiscovery).getHosts("second");
    }

    @Test
    void successfulEmptyDiscoveryRemainsEmpty() {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group", endpoint("address"))));
        when(serviceDiscovery.getHosts("address")).thenReturn(List.of());

        Assertions.assertTrue(workerAddressService.getEngineWorkerList("model", RoleType.PREFILL).isEmpty());
        Mockito.verifyNoInteractions(engineHealthReporter);
    }

    @Test
    void saturatedDiscoveryPoolFailsWithoutCallingSdkOnTheSyncThread() throws Exception {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group", endpoint("address"))));
        ThreadPoolExecutor executor = (ThreadPoolExecutor)
                org.springframework.test.util.ReflectionTestUtils.getField(workerAddressService, "serviceDiscoveryExecutor");
        executor.setCorePoolSize(1);
        executor.setMaximumPoolSize(1);
        CountDownLatch running = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        executor.execute(() -> {
            running.countDown();
            try {
                release.await();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        });
        try {
            Assertions.assertTrue(running.await(1, TimeUnit.SECONDS));
            Runnable queued = () -> { };
            while (executor.getQueue().remainingCapacity() > 0) {
                executor.execute(queued);
            }
            IllegalStateException failure = Assertions.assertThrows(IllegalStateException.class,
                    () -> workerAddressService.getEngineWorkerList("model", RoleType.PREFILL));
            Assertions.assertInstanceOf(RejectedExecutionException.class, failure.getCause());
            Mockito.verifyNoInteractions(serviceDiscovery);
            Mockito.verify(engineHealthReporter).reportStatusCheckerFail(
                    "model", BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
        } finally {
            release.countDown();
        }
    }

    @Test
    void timeoutIsReportedAndPropagated() throws Exception {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group", endpoint("address"))));
        CountDownLatch interrupted = new CountDownLatch(1);
        when(serviceDiscovery.getHosts("address")).thenAnswer(invocation -> {
            try {
                new CountDownLatch(1).await();
                return List.of();
            } catch (InterruptedException failure) {
                interrupted.countDown();
                throw failure;
            }
        });

        IllegalStateException failure = Assertions.assertThrows(IllegalStateException.class,
                () -> workerAddressService.getEngineWorkerList("model", RoleType.PREFILL));
        Assertions.assertInstanceOf(TimeoutException.class, failure.getCause());
        Assertions.assertTrue(interrupted.await(1, TimeUnit.SECONDS));
        Mockito.verify(engineHealthReporter).reportStatusCheckerFail(
                "model", BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null);
    }

    @Test
    void interruptedDiscoveryPreservesTheCallerInterruptFlag() {
        when(modelMetaConfig.endpointsWithGroup("model", RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group", endpoint("address"))));
        Mockito.lenient().when(serviceDiscovery.getHosts("address")).thenAnswer(invocation -> {
            new CountDownLatch(1).await();
            return List.of();
        });
        Thread.currentThread().interrupt();
        try {
            IllegalStateException failure = Assertions.assertThrows(IllegalStateException.class,
                    () -> workerAddressService.getEngineWorkerList("model", RoleType.PREFILL));
            Assertions.assertInstanceOf(InterruptedException.class, failure.getCause());
            Assertions.assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
        }
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

    private static Endpoint endpoint(String address) {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol(BackendServiceProtocolEnum.GRPC.getName());
        return endpoint;
    }
}

package org.flexlb.sync;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;

import static org.mockito.ArgumentMatchers.any;
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
    private ExecutorService serviceDiscoveryExecutor;

    @InjectMocks
    private WorkerAddressService workerAddressService;

    @Test
    void testGetHosts_Empty() {
        String modelName = "TestModel";
        Endpoint endpoint = endpoint("TestAddress");
        when(serviceDiscovery.getHosts(any(Endpoint.class))).thenReturn(Collections.emptyList());

        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, endpoint);

        Assertions.assertTrue(actualHosts.isEmpty());
    }

    @Test
    void testGetHosts_Success() {
        String modelName = "TestModel";
        Endpoint endpoint = endpoint("TestAddress");
        List<WorkerHost> expectedHosts =
                List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site1", "group1"));
        when(serviceDiscovery.getHosts(any(Endpoint.class))).thenReturn(expectedHosts);

        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, endpoint);

        Assertions.assertFalse(actualHosts.isEmpty());
    }

    @Test
    void testEmptyWorkersWarnOnceAndRecoveryIsLogged() {
        String modelName = "TestModel";
        Endpoint endpoint = endpoint("TestAddress");
        List<WorkerHost> recoveredHosts =
                List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site1", "group1"));
        when(serviceDiscovery.getHosts(any(Endpoint.class)))
                .thenReturn(Collections.emptyList(), Collections.emptyList(), recoveredHosts);

        Logger syncLogger = (Logger) LoggerFactory.getLogger("syncLogger");
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        syncLogger.addAppender(appender);
        try {
            workerAddressService.getServiceHosts(modelName, endpoint);
            workerAddressService.getServiceHosts(modelName, endpoint);
            workerAddressService.getServiceHosts(modelName, endpoint);
        } finally {
            syncLogger.detachAppender(appender);
            appender.stop();
        }

        long emptyWarnings = appender.list.stream()
                .filter(event -> event.getFormattedMessage().contains("No workers discovered"))
                .count();
        long recoveryLogs = appender.list.stream()
                .filter(event -> event.getFormattedMessage().contains("Worker discovery recovered"))
                .count();
        Assertions.assertEquals(1, emptyWarnings);
        Assertions.assertEquals(1, recoveryLogs);
    }

    private Endpoint endpoint(String address) {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        return endpoint;
    }
}

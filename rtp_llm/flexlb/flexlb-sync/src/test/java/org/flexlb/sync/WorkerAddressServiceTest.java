package org.flexlb.sync;

import com.taobao.vipserver.client.core.Host;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@SuppressWarnings("unchecked")
class WorkerAddressServiceTest {

    @Mock
    private WorkerAddressService.VipserverRunner vipserverRunner;

    @Mock
    private ExecutorService vipServerExecutor;

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @InjectMocks
    private WorkerAddressService workerAddressService;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetHosts_Timeout() throws Exception {
        // Arrange
        String modelName = "TestModel";
        String address = "TestAddress";
        Future<List<Host>> future = mock(Future.class);
        when(vipServerExecutor.submit(any(WorkerAddressService.VipserverRunner.class))).thenReturn(future);
        when(future.get(500, TimeUnit.MILLISECONDS)).thenThrow(new TimeoutException());
        Mockito.doNothing().when(engineHealthReporter).reportStatusCheckerFail(any(), any());

        // Act
        List<Host> actualHosts = workerAddressService.agetVIPHosts(modelName, address);

        // Assertions
        Assertions.assertTrue(actualHosts.isEmpty());
        verify(vipserverRunner, times(0)).call();
        verify(vipServerExecutor, times(0)).submit(any(WorkerAddressService.VipserverRunner.class));
        verify(future, times(0)).get(500, TimeUnit.MILLISECONDS);

    }
}
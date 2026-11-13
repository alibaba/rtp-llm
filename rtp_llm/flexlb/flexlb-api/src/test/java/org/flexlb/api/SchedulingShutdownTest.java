package org.flexlb.api;

import org.flexlb.api.server.FlexlbGrpcServer;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.master.MasterStatusService;
import org.flexlb.util.ApplicationWarmupState;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

/**
 * The application drains its ingress before delegating shutdown to the scheduling module.
 */
class SchedulingShutdownTest {
    @Test
    void drainsIngressBeforeClosingSchedulingOnce() {
        var scheduling = mock(AbstractRequestScheduler.class);
        var grpc = mock(FlexlbGrpcServer.class);
        var application = application(scheduling, grpc);
        application.offline();
        application.offline();
        var order = inOrder(grpc, scheduling);
        order.verify(scheduling).checkShutdownThread();
        order.verify(grpc).drain();
        order.verify(scheduling).shutdown(org.mockito.Mockito.anyLong());
        verify(scheduling).shutdown(org.mockito.Mockito.anyLong());
    }

    @Test
    void reportsSchedulingShutdownFailure() {
        var scheduling = mock(AbstractRequestScheduler.class);
        var failure = new IllegalStateException("scheduling cleanup failed");
        doThrow(failure).when(scheduling).shutdown(org.mockito.Mockito.anyLong());
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () -> application(scheduling, mock(FlexlbGrpcServer.class)).offline()));
    }

    @Test
    void rejectsUnsafeShutdownBeforeWaitingForApplicationMonitor() throws Exception {
        var scheduling = mock(AbstractRequestScheduler.class);
        var grpc = mock(FlexlbGrpcServer.class);
        doThrow(new IllegalStateException("owned callback")).when(scheduling).checkShutdownThread();
        var application = application(scheduling, grpc);
        synchronized (application) {
            java.util.concurrent.CompletableFuture.runAsync(
                            () -> assertThrows(IllegalStateException.class, application::offline))
                    .get(2, java.util.concurrent.TimeUnit.SECONDS);
        }
        verifyNoInteractions(grpc);
    }

    private static ApplicationLifecycle application(
            AbstractRequestScheduler scheduling, FlexlbGrpcServer grpc) {
        return new ApplicationLifecycle(
                mock(MasterStatusService.class),
                grpc,
                mock(ServiceMetricsReporter.class),
                new MockEnvironment(),
                new ApplicationWarmupState(),
                scheduling,
                0L);
    }
}

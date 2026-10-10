package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class SchedulerRuntimeTest {
    @Test
    void directStartupUsesOneSchedulerAndSharedConfiguration() {
        try (var f = new Fixture(false)) {
            var scheduler = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(f.runtime);
            assertInstanceOf(DirectRequestScheduler.class, scheduler);
            assertSame(scheduler, SchedulerTestSupport.initializedScheduler(f.runtime));
            assertSame(f.config, scheduler.config);
            assertSame(f.config, new RequestContext(f.config).getConfig());
            assertThrows(IllegalStateException.class,
                    () -> f.runtime.initializeScheduler(mock(AbstractRequestScheduler.class)));
        }
    }

    @Test
    void queueStartupUsesOneScheduler() {
        try (var f = new Fixture(true)) {
            assertInstanceOf(QueuedRequestScheduler.class, SchedulerTestSupport.initializedScheduler(f.runtime));
            assertSame(SchedulerTestSupport.initializedScheduler(f.runtime), SchedulerTestSupport.initializedScheduler(f.runtime));
            assertThrows(IllegalStateException.class,
                    () -> f.runtime.initializeScheduler(mock(AbstractRequestScheduler.class)));
        }
    }

    @Test
    void stoppedRuntimeCannotBeInitializedAgain() {
        var f = new Fixture(false);
        f.close();
        assertThrows(IllegalStateException.class,
                () -> f.runtime.initializeScheduler(mock(AbstractRequestScheduler.class)));
    }

    @Test
    void shutdownUsesTheSchedulerLifecycleHookAndContinuesAfterItsFailure() {
        var config = SchedulingTestConfig.newConfig();
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var endpoints = mock(EndpointRegistry.class);
        var dispatcher = mock(DefaultBatchDispatcher.class);
        var runtime = new SchedulerRuntime(new RequestRepository(), endpoints,
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class), dispatcher,
                service, mock(RecentCacheKeyTraceReporter.class), mock(EngineCancelChannel.class));
        var scheduler = mock(AbstractRequestScheduler.class);
        var expected = new IllegalStateException("placement close failed");
        doThrow(expected).when(scheduler).closePlacement();
        runtime.initializeScheduler(scheduler);

        assertSame(expected, assertThrows(IllegalStateException.class, runtime::shutdown));

        var order = inOrder(scheduler, dispatcher, endpoints);
        order.verify(scheduler).closePlacement();
        order.verify(scheduler).awaitAdmissionMutations();
        order.verify(dispatcher).shutdownAndAwait();
        order.verify(endpoints).close();
        order.verify(scheduler).closeOutstandingAndTerminalize();
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final EndpointRegistry endpoints = mock(EndpointRegistry.class);
        final SchedulerRuntime runtime;
        Fixture() { this(false); }
        Fixture(boolean queued) {
            config.setScheduler(queued ? new SchedulerConfig() : SchedulerConfig.direct());
            config.setDispatcher(DispatcherConfig.nonBatch());
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            runtime = new SchedulerRuntime(new RequestRepository(), endpoints, reporter,
                    mock(RequestSchedulerReporter.class), mock(DefaultBatchDispatcher.class), service,
                    mock(RecentCacheKeyTraceReporter.class), mock(EngineCancelChannel.class));
            runtime.initializeScheduler(PlacementConfiguration.create(runtime, config,
                    mock(RequestWorkerSelector.class), reporter, mock(DecodeCapacityAcquirer.class), new PlacementAvailability()));
        }
        public void close() { runtime.shutdown(); }
    }
}

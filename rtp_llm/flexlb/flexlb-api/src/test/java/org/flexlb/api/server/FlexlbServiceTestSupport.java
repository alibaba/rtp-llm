package org.flexlb.api.server;

import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.config.ConfigService;
import org.flexlb.master.MasterStatusView;
import org.flexlb.worker.status.ServiceMetricsReporter;

/**
 * Assembles API fixtures through the same scheduler and configuration as production.
 */
public final class FlexlbServiceTestSupport {
    public static FlexlbServiceImpl create(
            AbstractRequestScheduler scheduler,
            RequestRepository requests,
            MasterStatusView leadership,
            ServiceMetricsReporter health,
            FlexlbGrpcForwarder forwarder,
            ConfigService config,
            DeliveryMetricsReporter batches,
            ServerScheduleLatencyRecorder latency,
            RequestSchedulerReporter reporter) {
        if (org.mockito.Mockito.mockingDetails(leadership).isMock()) {
            org.mockito.Mockito.doCallRealMethod().when(leadership).shouldForwardToMaster();
        }
        if (org.mockito.Mockito.mockingDetails(scheduler).isMock()
                && !org.mockito.Mockito.mockingDetails(scheduler).isSpy()) {
            org.mockito.Mockito.lenient()
                    .doAnswer(
                            call -> {
                                var result =
                                        scheduler.submit(call.getArgument(0, RequestContext.class));
                                call.getArgument(1, Runnable.class).run();
                                return result;
                            })
                    .when(scheduler)
                    .submit(
                            org.mockito.ArgumentMatchers.any(RequestContext.class),
                            org.mockito.ArgumentMatchers.any(Runnable.class));
        }
        var service =
                new FlexlbServiceImpl(
                        scheduler,
                        config.loadBalanceConfig(),
                        requests,
                        leadership,
                        health,
                        forwarder,
                        batches,
                        latency,
                        reporter);
        if (scheduler instanceof org.flexlb.balance.schedule.DirectRequestScheduler
                || scheduler instanceof org.flexlb.balance.schedule.QueuedRequestScheduler) {
            service.configureRouting(
                    (org.flexlb.balance.strategy.RequestRouteSelector)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    scheduler, "router"));
        }
        return service;
    }
}

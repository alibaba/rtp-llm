package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.MasterStatusView;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;

/** Assembles API fixtures through the same scheduler and configuration as production. */
final class FlexlbServiceTestSupport {
    static FlexlbServiceImpl create(
            AbstractRequestScheduler scheduler,
            RequestRepository requests,
            MasterStatusView leadership,
            EngineHealthReporter health,
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
        if (scheduler instanceof org.flexlb.balance.scheduler.DirectRequestScheduler
                || scheduler instanceof org.flexlb.balance.scheduler.QueuedRequestScheduler) {
            service.configureRouting(
                    (org.flexlb.balance.scheduler.RequestRouteSelector)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    scheduler, "router"));
        }
        return service;
    }
}

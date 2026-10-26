package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.MasterStatusView;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;

/**
 * Assembles API fixtures through the same scheduler and configuration as production.
 */
final class FlexlbServiceTestSupport {
    static FlexlbServiceImpl create(
            RequestScheduler scheduler,
            RequestRepository requests,
            MasterStatusView leadership,
            EngineHealthReporter health,
            FlexlbGrpcForwarder forwarder,
            ConfigService config,
            DeliveryMetricsReporter batches,
            ServerScheduleLatencyRecorder latency,
            RequestSchedulerReporter reporter) {
        return create(
                scheduler,
                requests,
                leadership,
                health,
                forwarder,
                config,
                batches,
                latency,
                reporter,
                null);
    }

    static FlexlbServiceImpl create(
            RequestScheduler scheduler,
            RequestRepository requests,
            MasterStatusView leadership,
            EngineHealthReporter health,
            FlexlbGrpcForwarder forwarder,
            ConfigService config,
            DeliveryMetricsReporter batches,
            ServerScheduleLatencyRecorder latency,
            RequestSchedulerReporter reporter,
            org.flexlb.cache.match.CacheAwareService cache) {
        if (org.mockito.Mockito.mockingDetails(scheduler).isMock()
                && !(scheduler instanceof AbstractRequestScheduler)) {
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
        return new FlexlbServiceImpl(
                scheduler,
                requests,
                leadership,
                health,
                forwarder,
                config,
                batches,
                latency,
                reporter,
                cache);
    }
}

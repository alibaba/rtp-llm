package org.flexlb.httpserver;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.balance.scheduler.*;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.MasterStatusView;
import org.flexlb.service.monitor.*;

/** Assembles API fixtures through the same scheduler and configuration as production. */
final class FlexlbServiceTestSupport {
    static FlexlbServiceImpl create(RequestScheduler scheduler, RequestRepository requests,
            MasterStatusView leadership, EngineHealthReporter health, FlexlbGrpcForwarder forwarder,
            ConfigService config, DeliveryMetricsReporter batches, ServerScheduleLatencyRecorder latency,
            RequestSchedulerReporter reporter) {
        if (org.mockito.Mockito.mockingDetails(leadership).isMock()) {
            org.mockito.Mockito.doCallRealMethod().when(leadership).shouldForwardToMaster();
        }
        if (org.mockito.Mockito.mockingDetails(scheduler).isMock()
                && !(scheduler instanceof AbstractRequestScheduler)) {
            org.mockito.Mockito.lenient().doAnswer(call -> {
                var result = scheduler.submit(call.getArgument(0, RequestContext.class));
                call.getArgument(1, Runnable.class).run();
                return result;
            }).when(scheduler).submit(org.mockito.ArgumentMatchers.any(RequestContext.class),
                    org.mockito.ArgumentMatchers.any(Runnable.class));
        }
        return new FlexlbServiceImpl(
                scheduler, config.loadBalanceConfig(), requests, leadership, health, forwarder, batches, latency, reporter);
    }
}

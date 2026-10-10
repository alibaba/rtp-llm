package org.flexlb.balance.scheduler;

import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;

/** Builds the fixed-mode scheduler selected at startup. */
@Configuration(proxyBeanMethods = false)
public class PlacementConfiguration {
    @Bean(destroyMethod = "")
    public RequestScheduler requestScheduler(@Lazy RequestWorkerSelector router,
                                             DeliveryMetricsReporter reporter, @Lazy DecodeCapacityAcquirer decodeCapacity,
                                             PlacementAvailability availability, SchedulerRuntime runtime,
                                             ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        AbstractRequestScheduler scheduler = create(runtime, config, router, reporter, decodeCapacity, availability);
        runtime.initializeScheduler(scheduler);
        return scheduler;
    }

    static AbstractRequestScheduler create(SchedulerRuntime runtime, FlexlbConfig config, RequestWorkerSelector router,
                                   DeliveryMetricsReporter reporter, DecodeCapacityAcquirer decodeCapacity,
                                   PlacementAvailability availability) {
        if (config.isQueue()) {
            QueuedRequestScheduler queue = new QueuedRequestScheduler(config, router, reporter,
                    decodeCapacity, runtime, availability);
            queue.start();
            return queue;
        }
        return new DirectRequestScheduler(router, runtime, config);
    }
}

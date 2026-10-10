package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;

/** Assembles the scheduler, Prefill delivery, and Decode admission from startup config. */
@Configuration(proxyBeanMethods = false)
public class PlacementConfiguration {

    @Bean
    public AdmissionCapacity decodeAdmissionCapacity(ConfigService configService) {
        var availability = configService.loadBalanceConfig().getRouter().getRoles().getDecode().getAvailability();
        return new AdmissionCapacity(availability.getMaxEngineRequests() == null ? 0L : availability.getMaxEngineRequests(),
                availability.getMaxKvUsagePercent());
    }

    @Bean(destroyMethod = "")
    public RequestScheduler requestScheduler(@Lazy RequestWorkerSelector router,
                                             DeliveryMetricsReporter reporter, @Lazy DecodeCapacityAcquirer decodeCapacity,
                                             PlacementAvailability availability, SchedulerRuntime runtime,
                                             ConfigService configService, AdmissionCapacity capacity, RecentCacheKeyTraceReporter traceReporter) {
        FlexlbConfig config = configService.loadBalanceConfig();
        AbstractRequestScheduler scheduler = create(runtime, config, router, reporter, decodeCapacity, availability, capacity, traceReporter);
        runtime.registerScheduler(scheduler);
        return scheduler;
    }

    static AbstractRequestScheduler create(SchedulerRuntime runtime, FlexlbConfig config, RequestWorkerSelector router,
                                   DeliveryMetricsReporter reporter, DecodeCapacityAcquirer decodeCapacity,
                                   PlacementAvailability availability, AdmissionCapacity capacity, RecentCacheKeyTraceReporter traceReporter) {
        if (config.isQueue()) {
            QueuedRequestScheduler queue = new QueuedRequestScheduler(config, router, reporter,
                    decodeCapacity, runtime, availability, capacity, traceReporter);
            queue.start();
            return queue;
        }
        return new DirectRequestScheduler(router, runtime, config, capacity, traceReporter);
    }

}

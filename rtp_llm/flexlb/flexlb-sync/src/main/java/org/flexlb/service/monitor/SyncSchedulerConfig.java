package org.flexlb.service.monitor;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.ScheduledThreadPoolExecutor;

@Slf4j
@Configuration
public class SyncSchedulerConfig {
    @Bean(name = "taskMetricScheduler")
    public ScheduledThreadPoolExecutor taskMetricScheduler() {
        return new ScheduledThreadPoolExecutor(1, new NamedThreadFactory("task-metric-scheduler"));
    }

    @Bean(name = "engineStatusSyncScheduler")
    public ScheduledThreadPoolExecutor engineStatusSyncScheduler() {
        return new ScheduledThreadPoolExecutor(1, new NamedThreadFactory("engine-status-scheduler"));
    }
}

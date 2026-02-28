package org.flexlb.service.monitor;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;

import java.util.concurrent.ScheduledThreadPoolExecutor;

@Slf4j
@Configuration
@EnableScheduling
public class SyncSchedulerConfig {

    /**
     * Custom thread pool for Spring Boot scheduled tasks {@link org.springframework.scheduling.annotation.Scheduled}
     *
     * <p><b>NOTE:</b> Name must be {@code taskScheduler}</p>
     */
    @Bean(name = "taskScheduler")
    public ScheduledThreadPoolExecutor taskScheduler() {
        return new ScheduledThreadPoolExecutor(4, new NamedThreadFactory("task-scheduler"));
    }

    @Bean(name = "taskMetricScheduler")
    public ScheduledThreadPoolExecutor taskMetricScheduler() {
        return new ScheduledThreadPoolExecutor(1, new NamedThreadFactory("task-metric-scheduler"));
    }
}

package org.flexlb.config;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Configuration
public class AsyncConfiguration {

    @Bean(name = "doFinallyExecutor", destroyMethod = "shutdown")
    public ThreadPoolExecutor doFinallyExecutor() {
        int processors = Runtime.getRuntime().availableProcessors();
        int poolSize = Math.max(processors, 4);
        int maxPoolSize = Math.max(2 * processors, 8);
        return new ThreadPoolExecutor(
                poolSize,
                maxPoolSize,
                0L,
                TimeUnit.MILLISECONDS,
                new SynchronousQueue<>(),
                new NamedThreadFactory("do-finally-executor"),
                // Finish requests on the caller during transient saturation; shutdown may discard late tasks.
                new ThreadPoolExecutor.CallerRunsPolicy());
    }
}

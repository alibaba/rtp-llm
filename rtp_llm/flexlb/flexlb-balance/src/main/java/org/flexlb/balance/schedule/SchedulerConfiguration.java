package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;
import org.springframework.context.annotation.Primary;

/**
 * Assembles the scheduler, Prefill delivery, and Decode preemption from startup config.
 */
@org.springframework.scheduling.annotation.EnableScheduling
@Configuration(proxyBeanMethods = false)
public class SchedulerConfiguration {

    @Bean(name = "taskScheduler")
    public java.util.concurrent.ScheduledThreadPoolExecutor taskScheduler() {
        return new java.util.concurrent.ScheduledThreadPoolExecutor(
                4, new io.micrometer.core.instrument.util.NamedThreadFactory("task-scheduler"));
    }

    @Lazy
    @Bean(destroyMethod = "")
    public RequestTaskPool requestTasks() {
        return new RequestTaskPool();
    }

    @Lazy
    @Bean(destroyMethod = "close")
    public java.util.concurrent.ScheduledThreadPoolExecutor cleanupTimerExecutor() {
        return newTimerExecutor(2, "flexlb-request-cleanup-");
    }

    @Lazy
    @Bean(destroyMethod = "close")
    public java.util.concurrent.ScheduledThreadPoolExecutor deadlineTimerExecutor() {
        return newTimerExecutor(1, "request-scheduler-expiration-");
    }

    @Lazy
    @Bean
    @Primary
    public TaskTimer taskTimer(
            @Qualifier("cleanupTimerExecutor")
                    java.util.concurrent.ScheduledThreadPoolExecutor executor) {
        return (task, delayMs) ->
                executor.schedule(task, delayMs, java.util.concurrent.TimeUnit.MILLISECONDS);
    }

    @Lazy
    @Bean
    public TaskTimer deadlineTaskTimer(
            @Qualifier("deadlineTimerExecutor")
                    java.util.concurrent.ScheduledThreadPoolExecutor executor) {
        return (task, delayMs) ->
                executor.schedule(task, delayMs, java.util.concurrent.TimeUnit.MILLISECONDS);
    }

    @Lazy
    @Bean(destroyMethod = "")
    public ExpirationTimer expirationTimer(
            RequestRepository requests, @Qualifier("deadlineTaskTimer") TaskTimer timer) {
        return new ExpirationTimer(requests, timer);
    }

    @Lazy
    @Bean(destroyMethod = "")
    public ResponseCompletionExecutor responseCompletions(ConfigService config) {
        return new ResponseCompletionExecutor(
                config.loadBalanceConfig().getInternalRuntime().getResponseCompletionThreads());
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(CostBasedPrefillStrategy.class)
    public CostBasedPrefillStrategy prefillStrategy(
            ConfigService config,
            EndpointRegistry endpoints,
            org.flexlb.cache.match.CacheAwareService cache,
            org.flexlb.cache.metrics.CacheMetricsReporter metrics) {
        return new CostBasedPrefillStrategy(config, endpoints, cache, metrics);
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(DecodeStrategy.class)
    public DecodeStrategy decodeStrategy(
            ConfigService config, AdmissionCapacity capacity, EndpointRegistry endpoints) {
        return new DecodeStrategy(config, capacity, endpoints);
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(VitStrategy.class)
    public VitStrategy vitStrategy(
            org.flexlb.cache.VitCacheDirectory directory, EndpointRegistry endpoints) {
        return new VitStrategy(directory, endpoints);
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(SchedulingQueue.class)
    public SchedulingQueue schedulingQueue(ConfigService config) {
        return new OrderedRequestQueue(config.loadBalanceConfig().isPriorityOrdering());
    }

    @Bean
    @ConditionalOnMissingBean(DecodePreemptionPolicy.class)
    public DecodePreemptionPolicy decodePreemptionPolicy() {
        return new PriorityDecodePreemptionPolicy();
    }

    @Bean
    @ConditionalOnMissingBean(DecodePreemption.class)
    @Lazy
    public DecodePreemption decodePreemption(
            RequestRepository requests,
            RequestSchedulerReporter reporter,
            ConfigService configService,
            AdmissionCapacity capacity,
            DecodePreemptionPolicy policy) {
        return new DefaultDecodePreemption(requests, reporter, configService, capacity, policy);
    }

    @Bean
    public AdmissionCapacity decodeAdmissionCapacity(ConfigService configService) {
        var availability =
                configService
                        .loadBalanceConfig()
                        .getRouter()
                        .getRoles()
                        .getDecode()
                        .getAvailability();
        return new AdmissionCapacity(
                availability.getMaxEngineRequests() == null
                        ? 0L
                        : availability.getMaxEngineRequests(),
                availability.getMaxKvUsagePercent());
    }

    @Bean(initMethod = "start", destroyMethod = "")
    @ConditionalOnMissingBean(RequestScheduler.class)
    public AbstractRequestScheduler requestScheduler(
            @Lazy RequestRouteSelector router,
            @Lazy DecodePreemption decodePreemption,
            CapacityNotifier availability,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            ConfigService configService,
            @Lazy SchedulingQueue queue,
            EndpointRegistry endpoints) {
        FlexlbConfig config = configService.loadBalanceConfig();
        if (config.isQueue()) {
            return new QueuedRequestScheduler(
                    config,
                    router,
                    decodePreemption,
                    requests,
                    tasks,
                    deadlines,
                    engineCancellation,
                    responseCompletions,
                    deliveryReporter,
                    cacheTrace,
                    availability,
                    queue,
                    endpoints);
        }
        return new DirectRequestScheduler(
                router,
                requests,
                tasks,
                deadlines,
                engineCancellation,
                responseCompletions,
                deliveryReporter,
                cacheTrace,
                endpoints);
    }

    static java.util.concurrent.ScheduledThreadPoolExecutor newTimerExecutor(
            int threads, String threadName) {
        var executor =
                new java.util.concurrent.ScheduledThreadPoolExecutor(
                        threads, Thread.ofPlatform().daemon().name(threadName, 1).factory());
        executor.setRemoveOnCancelPolicy(true);
        executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        executor.setContinueExistingPeriodicTasksAfterShutdownPolicy(false);
        return executor;
    }
}

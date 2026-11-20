package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.TokenBalancedPrefillStrategy;
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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
    public ExecutorService requestWorkExecutor() {
        return Executors.newThreadPerTaskExecutor(
                Thread.ofVirtual().name("flexlb-request-cleanup-", 1).factory());
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
    @ConditionalOnMissingBean(PrefillSelectionStrategy.class)
    public PrefillSelectionStrategy prefillStrategy(
            ConfigService config,
            EndpointRegistry endpoints,
            org.flexlb.cache.match.CacheAwareService cache,
            org.flexlb.cache.metrics.CacheMetricsReporter metrics) {
        return config.loadBalanceConfig()
                                .getRouter()
                                .getRoles()
                                .getPrefill()
                                .getSelection()
                                .getType()
                        == org.flexlb.config.routing.RoutingConfig.PrefillSelectionType
                                .TOKEN_BALANCE
                ? new TokenBalancedPrefillStrategy(config, endpoints, cache, metrics)
                : new CostBasedPrefillStrategy(config, endpoints, cache, metrics);
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
            org.flexlb.cache.VitWorkerCacheManager cacheManager, EndpointRegistry endpoints) {
        return new VitStrategy(cacheManager, endpoints);
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
            @Lazy RequestPreemption preemption,
            CapacityNotifier availability,
            RequestRepository requests,
            @Qualifier("requestWorkExecutor") ExecutorService requestWork,
            ExpirationTimer deadlines,
            EngineRequestCanceller engineRequestCanceller,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            ConfigService configService,
            @Lazy SchedulingQueue queue) {
        FlexlbConfig config = configService.loadBalanceConfig();
        if (config.isQueue()) {
            return new QueuedRequestScheduler(
                    config,
                    router,
                    preemption,
                    requests,
                    requestWork,
                    deadlines,
                    engineRequestCanceller,
                    responseCompletions,
                    deliveryReporter,
                    cacheTrace,
                    availability,
                    queue);
        }
        return new DirectRequestScheduler(
                router,
                requests,
                requestWork,
                deadlines,
                engineRequestCanceller,
                responseCompletions,
                deliveryReporter,
                cacheTrace);
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(SchedulingQueue.class)
    SchedulingQueue schedulingQueue(ConfigService config) {
        return new OrderedRequestQueue(
                config.loadBalanceConfig().isPriorityOrdering(),
                config.loadBalanceConfig().queueScheduler().getOrdering().getPolicy().getType()
                        == org.flexlb.config.scheduler.QueueOrderingConfig.PolicyType.PREFILL_COST);
    }

    @Bean
    @ConditionalOnMissingBean(RequestPreemption.class)
    @Lazy
    RequestPreemption requestPreemption(
            RequestRepository requests,
            RequestSchedulerReporter reporter,
            ConfigService configService,
            AdmissionCapacity capacity) {
        return new DefaultRequestPreemption(
                requests, reporter, configService, capacity, new DecodePreemptionPlanner());
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

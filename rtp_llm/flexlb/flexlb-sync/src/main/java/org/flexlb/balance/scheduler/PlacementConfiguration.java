package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.OrderedPrefillRequestQueue;
import org.flexlb.balance.endpoint.PrefillRequestQueue;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.eviction.DecodePreemptionPolicy;
import org.flexlb.balance.eviction.DefaultDecodePreemption;
import org.flexlb.balance.eviction.PriorityDecodePreemptionPolicy;
import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelectionStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.VitSelectionStrategy;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
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
public class PlacementConfiguration {

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
    @Primary
    public ScheduledTaskTimer taskTimer() {
        return new ScheduledTaskTimer();
    }

    @Lazy
    @Bean(destroyMethod = "close")
    public ScheduledTaskTimer deadlineTaskTimer() {
        return new ScheduledTaskTimer(1, "request-scheduler-expiration-");
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
                config.loadBalanceConfig()
                        .getInternalRuntime()
                        .getBatchDispatchCompletionThreads());
    }

    @Lazy
    @Bean(destroyMethod = "")
    public DeliveryExecution deliveryExecution(
            ConfigService config, RequestTaskPool tasks, DeliveryMetricsReporter reporter) {
        var deliveries = new DeliveryExecution(config, tasks::recordFailure);
        reporter.registerExecutor(deliveries);
        return deliveries;
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(PrefillSelectionStrategy.class)
    public PrefillSelectionStrategy prefillSelectionStrategy(
            CacheAwareService cache, ConfigService config) {
        return new CostBasedPrefillStrategy(cache, config);
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(DecodeSelectionStrategy.class)
    public DecodeSelectionStrategy decodeSelectionStrategy(
            ConfigService config, AdmissionCapacity capacity) {
        return new DecodeSelector(config, capacity);
    }

    @Bean
    @ConditionalOnMissingBean(VitSelectionStrategy.class)
    public VitSelectionStrategy vitSelectionStrategy() {
        return new VitWorkerSelector();
    }

    @Bean
    @ConditionalOnMissingBean(GroupPlanner.class)
    public GroupPlanner groupPlanner(ConfigService config) {
        return BuiltinGroupPlanner.from(config.loadBalanceConfig());
    }

    @Bean
    @ConditionalOnMissingBean(PrefillRequestQueue.Factory.class)
    public PrefillRequestQueue.Factory workerQueues() {
        return order -> OrderedPrefillRequestQueue.ordered(16, order);
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean(PlacementQueue.class)
    public PlacementQueue placementQueue(ConfigService config) {
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
            DeliveryDispatcher dispatcher,
            RequestRepository requests,
            RequestTaskPool tasks,
            TaskTimer timer,
            RequestSchedulerReporter reporter,
            ConfigService configService,
            AdmissionCapacity capacity,
            DecodePreemptionPolicy policy) {
        return new DefaultDecodePreemption(
                dispatcher, requests, tasks, timer, reporter, configService, capacity, policy);
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
            PlacementAvailability availability,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            DeliveryDispatcher dispatcher,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            ConfigService configService,
            GroupPlanner grouping,
            @Lazy PlacementQueue queue,
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
                    dispatcher,
                    responseCompletions,
                    deliveryReporter,
                    cacheTrace,
                    availability,
                    queue,
                    grouping,
                    endpoints);
        }
        return new DirectRequestScheduler(
                router,
                requests,
                tasks,
                deadlines,
                dispatcher,
                responseCompletions,
                deliveryReporter,
                cacheTrace,
                endpoints);
    }
}

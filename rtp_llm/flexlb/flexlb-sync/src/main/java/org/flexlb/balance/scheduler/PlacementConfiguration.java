package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.WorkerQueue;
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
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;

/** Assembles the scheduler, Prefill delivery, and Decode preemption from startup config. */
@Configuration(proxyBeanMethods = false)
public class PlacementConfiguration {

    @Lazy
    @Bean(destroyMethod = "")
    public RequestTaskPool requestTasks() {
        return new RequestTaskPool();
    }

    @Lazy
    @Bean(destroyMethod = "")
    public ScheduledTaskTimer taskTimer() {
        return new ScheduledTaskTimer();
    }

    @Lazy
    @Bean
    public RequestSignals requestSignals() {
        return new RequestSignals();
    }

    @Lazy
    @Bean(destroyMethod = "")
    public ExpirationTimer expirationTimer(RequestRepository requests) {
        return new ExpirationTimer(requests);
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

    @Lazy
    @Bean
    public RequestLifecycle requestLifecycle(
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            RequestSignals signals,
            DeliveryDispatcher dispatcher,
            ResponseCompletionExecutor completions,
            DeliveryMetricsReporter reporter,
            TaskTimer timer,
            AdmissionCapacity capacity,
            RecentCacheKeyTraceReporter trace) {
        return new RequestLifecycle(
                requests,
                tasks,
                deadlines,
                signals,
                dispatcher,
                completions,
                reporter,
                timer,
                capacity,
                trace);
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
    @ConditionalOnMissingBean(WorkerQueue.Factory.class)
    public WorkerQueue.Factory workerQueues() {
        return order -> PrefillActiveIndex.ordered(16, order);
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
            RequestSignals signals,
            RequestSchedulerReporter reporter,
            ConfigService configService,
            AdmissionCapacity capacity,
            DecodePreemptionPolicy policy) {
        return new DefaultDecodePreemption(
                dispatcher,
                requests,
                tasks,
                timer,
                signals,
                reporter,
                configService,
                capacity,
                policy);
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

    @Bean(destroyMethod = "")
    @ConditionalOnMissingBean(RequestScheduler.class)
    public RequestScheduler requestScheduler(
            @Lazy RequestWorkerSelector router,
            DeliveryMetricsReporter reporter,
            @Lazy DecodePreemption decodePreemption,
            PlacementAvailability availability,
            RequestLifecycle lifecycle,
            ConfigService configService,
            GroupPlanner grouping,
            @Lazy PlacementQueue queue,
            DeliveryExecution deliveries,
            RequestTaskPool tasks) {
        FlexlbConfig config = configService.loadBalanceConfig();
        RequestScheduler scheduler =
                create(
                        lifecycle,
                        config,
                        router,
                        reporter,
                        decodePreemption,
                        availability,
                        grouping,
                        queue,
                        new PrefillQueueRunner.Dependencies(
                                deliveries, reporter, tasks::recordFailure));
        return scheduler;
    }

    static RequestScheduler create(
            RequestLifecycle lifecycle,
            FlexlbConfig config,
            RequestWorkerSelector router,
            DeliveryMetricsReporter reporter,
            DecodePreemption decodePreemption,
            PlacementAvailability availability,
            GroupPlanner grouping,
            PlacementQueue placementQueue,
            PrefillQueueRunner.Dependencies queueDependencies) {
        if (config.isQueue()) {
            QueuedRequestScheduler queue =
                    new QueuedRequestScheduler(
                            config,
                            router,
                            reporter,
                            decodePreemption,
                            lifecycle,
                            availability,
                            placementQueue,
                            grouping,
                            queueDependencies);
            queue.start();
            return queue;
        }
        return new DirectRequestScheduler(router, lifecycle);
    }
}

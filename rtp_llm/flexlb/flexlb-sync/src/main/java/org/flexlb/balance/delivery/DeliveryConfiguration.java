package org.flexlb.balance.delivery;

import org.flexlb.balance.estimate.BatchTtftEstimator;
import org.flexlb.balance.estimate.EndpointPrefillLoadProvider;
import org.flexlb.balance.estimate.NonBatchTtftEstimator;
import org.flexlb.balance.estimate.PrefillLoadProvider;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.prefill.QueuePrefillLoadProvider;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

/**
 * 按启动配置里的 dispatcher.type 选择投递策略和对应的 TTFT 估算器。
 * 两者必须同时是 BATCH 或同时是 NON_BATCH，所以都在这里创建。
 * 选 Prefill 时读负载视图的方式也按 dispatcher.type 在这里选择。
 */
@Configuration(proxyBeanMethods = false)
public class DeliveryConfiguration {

    private static final int DEFAULT_SERVER_PORT = 7001;

    @Bean
    public DeliveryStrategy deliveryStrategy(ConfigService configService,
                                             RequestRegistry requests,
                                             DefaultBatchDispatcher batchDispatcher,
                                             BatchSchedulerReporter reporter,
                                             Environment environment) {
        DeliveryMetrics metrics = new DeliveryMetrics(reporter);
        return switch (dispatcherType(configService)) {
            case BATCH -> {
                int port = environment.getProperty("server.port", Integer.class, DEFAULT_SERVER_PORT);
                yield new BatchDeliveryStrategy(batchDispatcher::tryPrepareSubmission,
                                                BatchIdGenerator.create(port),
                                                requests,
                                                metrics);
            }
            // NON_BATCH_V2 不经过老的投递流程，这里只是让老组件能正常创建
            case NON_BATCH, NON_BATCH_V2 -> new NonBatchDeliveryStrategy(requests, metrics);
        };
    }

    @Bean
    public PrefillTtftEstimator prefillTtftEstimator(ConfigService configService) {
        return switch (dispatcherType(configService)) {
            case BATCH -> new BatchTtftEstimator();
            case NON_BATCH, NON_BATCH_V2 -> new NonBatchTtftEstimator();
        };
    }

    @Bean
    public PrefillLoadProvider prefillLoadProvider(ConfigService configService, PrefillQueueRegistry prefillQueues) {
        return switch (dispatcherType(configService)) {
            case BATCH, NON_BATCH -> new EndpointPrefillLoadProvider();
            case NON_BATCH_V2 -> new QueuePrefillLoadProvider(prefillQueues, configService);
        };
    }

    private static DispatcherConfig.Type dispatcherType(ConfigService configService) {
        return configService.loadBalanceConfig().getDispatcher().getType();
    }
}

package org.flexlb.balance.delivery;

import org.flexlb.balance.estimate.estimator.BatchTtftEstimator;
import org.flexlb.balance.estimate.estimator.NonBatchTtftEstimator;
import org.flexlb.balance.estimate.estimator.PrefillTtftEstimator;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

/**
 * 按启动配置里的 dispatcher.type 选择投递策略和对应的 TTFT 估算器。
 * 两者必须同时是 BATCH 或同时是 NON_BATCH，所以都在这里创建。
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
            case NON_BATCH -> new NonBatchDeliveryStrategy(requests, metrics);
        };
    }

    @Bean
    public PrefillTtftEstimator prefillTtftEstimator(ConfigService configService) {
        return switch (dispatcherType(configService)) {
            case BATCH -> new BatchTtftEstimator();
            case NON_BATCH -> new NonBatchTtftEstimator();
        };
    }

    private static DispatcherConfig.Type dispatcherType(ConfigService configService) {
        return configService.loadBalanceConfig().getDispatcher().getType();
    }
}

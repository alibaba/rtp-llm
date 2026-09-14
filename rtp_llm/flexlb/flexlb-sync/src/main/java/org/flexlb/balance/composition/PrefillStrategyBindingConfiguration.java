package org.flexlb.balance.composition;

import org.flexlb.balance.strategy.CostBasedBatchedPrefillStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.PrefillStrategy;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.GlobalDecisionConfig;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Objects;

/**
 * Binds the immutable Prefill decision topology from the startup configuration.
 */
@Configuration(proxyBeanMethods = false)
public class PrefillStrategyBindingConfiguration {

    /**
     * Global FIXED_WINDOW owns joint placement, while every other startup mode
     * retains the independent cost-based selector. Runtime updates change only
     * active numeric policy parameters and never replace this bean.
     */
    @Bean
    public PrefillStrategy activePrefillStrategy(ConfigService configService,
                                                 WorkerDirectory workerDirectory,
                                                 CacheAwareService cacheAwareService,
                                                 EngineHealthReporter engineHealthReporter) {
        FlexlbConfig config = Objects.requireNonNull(
                configService.loadBalanceConfig(), "startup FlexLB configuration");
        if (config.isQueue()
                && config.queueScheduler().getGlobalDecision().getType()
                == GlobalDecisionConfig.Type.FIXED_WINDOW) {
            return new CostBasedBatchedPrefillStrategy(
                    workerDirectory, cacheAwareService, engineHealthReporter);
        }
        return new CostBasedPrefillStrategy(
                workerDirectory, cacheAwareService, engineHealthReporter);
    }
}
